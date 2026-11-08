"""FastAPI backend for local text-to-image, image-to-image, and ingestion workflows."""

from __future__ import annotations

import hashlib
import io
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from image_search.embedder import ClipEmbedder
from image_search.store import VectorStore

ROOT = Path(os.getenv("IMAGE_SEARCH_HOME", Path.cwd())).resolve()
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DEMO_DIR = DATA_DIR / "demo" / "images"
STATIC_DIR = Path(__file__).resolve().parent / "static"
PUBLIC_DEMO = os.getenv("IMAGE_SEARCH_PUBLIC_DEMO", "false").strip().lower() in {"1", "true", "yes", "on"}
GALLERY_INGEST_ENABLED = not PUBLIC_DEMO and os.getenv("IMAGE_SEARCH_ENABLE_INGEST", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MAX_UPLOAD_BYTES = int(os.getenv("IMAGE_SEARCH_MAX_UPLOAD_BYTES", "15000000"))
MAX_IMAGE_PIXELS = int(os.getenv("IMAGE_SEARCH_MAX_PIXELS", "50000000"))
if GALLERY_INGEST_ENABLED:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DEMO_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = Path(os.getenv("IMAGE_SEARCH_DB", DATA_DIR / "index.sqlite3"))
MODEL_ID = os.getenv("CLIP_MODEL_ID", "openai/clip-vit-base-patch32")

store = VectorStore(DB_PATH)
embedder = ClipEmbedder(model_id=MODEL_ID)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(
    title="FrameFind API",
    version="0.1.0",
    description="Local-first multimodal image retrieval using CLIP embeddings and exact cosine similarity.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/demo/images", StaticFiles(directory=DEMO_DIR), name="demo-images")
if GALLERY_INGEST_ENABLED:
    app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


class TextSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=512)
    limit: int = Field(default=24, ge=1, le=100)


class ImageSearchResponse(BaseModel):
    query_type: str
    query: str | None = None
    count: int
    results: list[dict[str, Any]]


def _results(query_vector, limit: int, exclude_id: str | None = None) -> list[dict[str, Any]]:
    if store.count() == 0:
        raise HTTPException(
            status_code=409,
            detail="The gallery is empty. Run `uv run image-search seed-demo` to download the licensed sample gallery and build its embeddings.",
        )
    try:
        ranked = store.search(query_vector, limit=limit, exclude_id=exclude_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return [record.public_dict(score=score) for record, score in ranked]


def _asset_path(image_url: str) -> Path | None:
    if image_url.startswith("/demo/images/"):
        candidate = (DEMO_DIR / Path(image_url).name).resolve()
        return candidate if candidate.is_relative_to(DEMO_DIR.resolve()) else None
    if image_url.startswith("/uploads/"):
        candidate = (UPLOAD_DIR / Path(image_url).name).resolve()
        return candidate if candidate.is_relative_to(UPLOAD_DIR.resolve()) else None
    return None


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "indexed_images": store.count(),
        "model_id": MODEL_ID,
        "model_loaded": embedder._model is not None,
        "device": str(embedder._device) if embedder._model is not None else "not loaded yet",
        "search_mode": "exact cosine similarity over normalized float32 vectors",
        "public_demo": PUBLIC_DEMO,
        "gallery_ingest_enabled": GALLERY_INGEST_ENABLED,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
    }


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    images = store.list_images(limit=500)
    dimensions = store.dimensions()
    return {
        "images": store.count(),
        "model_id": MODEL_ID,
        "embedding_dimensions": dimensions[0] if len(dimensions) == 1 else dimensions or None,
        "vector_storage": "SQLite BLOB (float32), normalized",
        "similarity": "cosine (dot product of normalized vectors)",
        "search_complexity": "exact linear scan over indexed embeddings",
        "license_groups": sorted({item.license_name for item in images if item.license_name}),
    }


@app.get("/api/images")
def list_images(limit: int = 120, offset: int = 0) -> dict[str, Any]:
    if not 1 <= limit <= 500 or offset < 0:
        raise HTTPException(status_code=422, detail="limit must be 1–500 and offset must be non-negative")
    images = store.list_images(limit=limit, offset=offset)
    return {"count": store.count(), "results": [item.public_dict() for item in images]}


@app.post("/api/search/text", response_model=ImageSearchResponse)
def search_text(payload: TextSearchRequest) -> ImageSearchResponse:
    try:
        vector = embedder.encode_text(payload.query)
    except (ValueError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"Text embedding failed: {exc}") from exc
    results = _results(vector, payload.limit)
    return ImageSearchResponse(query_type="text", query=payload.query, count=len(results), results=results)


@app.post("/api/search/image", response_model=ImageSearchResponse)
async def search_image(
    file: Annotated[UploadFile, File(description="PNG, JPEG, or WebP image")],
    limit: Annotated[int, Form(ge=1, le=100)] = 24,
) -> ImageSearchResponse:
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_UPLOAD_BYTES:,} bytes")
    try:
        vector = embedder.encode_image(payload)
    except (ValueError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=f"Could not embed image: {exc}") from exc
    results = _results(vector, limit)
    return ImageSearchResponse(query_type="image", count=len(results), results=results)


@app.get("/api/similar/{image_id}", response_model=ImageSearchResponse)
def similar_images(image_id: str, limit: int = 24) -> ImageSearchResponse:
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=422, detail="limit must be 1–100")
    record = store.get_image(image_id)
    if not record:
        raise HTTPException(status_code=404, detail="Image not found")
    path = _asset_path(record.image_url)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Indexed image file is missing")
    try:
        vector = embedder.encode_image(path)
    except (ValueError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"Could not embed image: {exc}") from exc
    results = _results(vector, limit, exclude_id=image_id)
    return ImageSearchResponse(query_type="similarity", query=record.title, count=len(results), results=results)


@app.post("/api/ingest")
async def ingest_image(
    file: Annotated[UploadFile, File(description="Local image file")],
    title: Annotated[str, Form(max_length=160)] = "",
    description: Annotated[str, Form(max_length=1200)] = "",
) -> dict[str, Any]:
    if not GALLERY_INGEST_ENABLED:
        raise HTTPException(status_code=404, detail="Adding images to the gallery is disabled in this demo")
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_UPLOAD_BYTES:,} bytes")
    try:
        original = Image.open(io.BytesIO(payload))
        image_format = original.format
        original.verify()
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(payload))).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=415, detail="Upload a valid PNG, JPEG, or WebP image") from exc
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_IMAGE_PIXELS:,} pixels")
    if image_format not in {"JPEG", "PNG", "WEBP"}:
        raise HTTPException(status_code=415, detail="Upload a PNG, JPEG, or WebP image")
    safe_title = re.sub(r"\s+", " ", title).strip()[:160] or Path(file.filename or "image").stem[:160]
    image_id = "upload-" + hashlib.sha256(payload).hexdigest()[:20]
    suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}[image_format]
    file_name = f"{image_id}{suffix}"
    destination = UPLOAD_DIR / file_name
    if not destination.exists():
        destination.write_bytes(payload)
    try:
        vector = embedder.encode_image(payload)
        store.add_image(
            image_id=image_id,
            title=safe_title,
            description=description.strip(),
            creator="Local upload",
            license_name="User-provided · rights not verified",
            image_url=f"/uploads/{file_name}",
            source_page="",
            width=image.width,
            height=image.height,
            sha256=hashlib.sha256(payload).hexdigest(),
            embedding=vector,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail=f"Image ingestion failed: {exc}") from exc
    record = store.get_image(image_id)
    return {"status": "indexed", "image": record.public_dict() if record else None}
