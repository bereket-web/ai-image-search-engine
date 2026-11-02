"""Download the curated Commons gallery and generate its CLIP vector index."""

from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image

from image_search.commons import download_demo_assets
from image_search.embedder import ClipEmbedder
from image_search.store import VectorStore

PROJECT_ROOT = Path(os.getenv("IMAGE_SEARCH_HOME", Path.cwd())).resolve()
CATALOG_PATH = PROJECT_ROOT / "data" / "demo" / "catalog.json"
IMAGES_PATH = PROJECT_ROOT / "data" / "demo" / "images"
ASSETS_PATH = PROJECT_ROOT / "data" / "demo" / "assets.json"
DB_PATH = Path(os.getenv("IMAGE_SEARCH_DB", PROJECT_ROOT / "data" / "index.sqlite3"))


def seed_demo_gallery(batch_size: int = 8) -> dict[str, int | str]:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Curated Commons catalog not found: {CATALOG_PATH}")
    assets = download_demo_assets(CATALOG_PATH, IMAGES_PATH)
    embedder = ClipEmbedder(model_id=os.getenv("CLIP_MODEL_ID", "openai/clip-vit-base-patch32"))
    store = VectorStore(DB_PATH)
    keep_ids = {row["id"] for row in assets}
    store.remove_missing_ids("commons-", keep_ids)
    keep_files = {row["local_file"] for row in assets}
    for stale_path in IMAGES_PATH.glob("commons-*"):
        if stale_path.name not in keep_files:
            stale_path.unlink(missing_ok=True)
    indexed = 0
    embedding_dimension = 0
    for start in range(0, len(assets), batch_size):
        batch = assets[start : start + batch_size]
        paths = [IMAGES_PATH / row["local_file"] for row in batch]
        vectors = embedder.encode_images(paths)
        embedding_dimension = int(vectors.shape[1])
        for row, vector, path in zip(batch, vectors, paths, strict=True):
            with Image.open(path) as image:
                width, height = image.size
            store.add_image(
                image_id=row["id"],
                title=row["title"],
                description=" ".join([row["description"], " ".join(row["tags"])]).strip(),
                creator=row["creator"],
                license_name=row["license_name"],
                license_url=row["license_url"],
                source_page=row["source_page"],
                image_url=row["image_url"],
                width=width,
                height=height,
                sha256=row["image_sha256"],
                embedding=vector,
            )
            indexed += 1
    ASSETS_PATH.write_text(json.dumps(assets, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"indexed": indexed, "total": store.count(), "dimension": embedding_dimension, "device": embedder.device}
