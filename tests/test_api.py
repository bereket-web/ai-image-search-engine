from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from image_search import api
from image_search.store import VectorStore


def png_bytes(color: tuple[int, int, int] = (100, 140, 180)) -> bytes:
    image = Image.new("RGB", (32, 24), color)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    store = VectorStore(tmp_path / "api.sqlite3")
    upload_source = api.UPLOAD_DIR / "test-fixture.png"
    upload_source.parent.mkdir(parents=True, exist_ok=True)
    upload_source.write_bytes(png_bytes())
    store.add_image(
        image_id="target",
        title="Blue mountain",
        description="A mountain reflected in water",
        creator="Test creator",
        license_name="CC0",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        source_page="https://commons.wikimedia.org/wiki/File:Test.jpg",
        image_url="/uploads/test-fixture.png",
        width=32,
        height=24,
        sha256="fixture",
        embedding=[1, 0, 0],
    )
    store.add_image(
        image_id="second",
        title="Another scene",
        image_url="/uploads/test-fixture.png",
        width=32,
        height=24,
        sha256="fixture-2",
        embedding=[0, 1, 0],
    )
    monkeypatch.setattr(api, "store", store)
    monkeypatch.setattr(api.embedder, "encode_text", lambda text: np.array([1, 0, 0], dtype=np.float32))
    monkeypatch.setattr(api.embedder, "encode_image", lambda image: np.array([1, 0, 0], dtype=np.float32))
    with TestClient(api.app) as test_client:
        yield test_client, store
    upload_source.unlink(missing_ok=True)


def test_gallery_and_text_search(client):
    test_client, _ = client
    assert test_client.get("/api/health").json()["indexed_images"] == 2
    gallery = test_client.get("/api/images").json()
    assert gallery["count"] == 2
    response = test_client.post("/api/search/text", json={"query": "mountain reflected in water", "limit": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["id"] == "target"
    assert body["results"][0]["score"] == pytest.approx(1.0)
    assert body["results"][0]["license_name"] == "CC0"


def test_image_search_and_similar_endpoint(client):
    test_client, _ = client
    response = test_client.post(
        "/api/search/image",
        files={"file": ("query.png", png_bytes(), "image/png")},
        data={"limit": 2},
    )
    assert response.status_code == 200
    assert response.json()["results"][0]["id"] == "target"
    similar = test_client.get("/api/similar/target?limit=2")
    assert similar.status_code == 200
    assert all(item["id"] != "target" for item in similar.json()["results"])


def test_ingestion_is_local_and_metadata_is_cautious(client):
    test_client, store = client
    response = test_client.post(
        "/api/ingest",
        files={"file": ("new-scene.png", png_bytes((30, 80, 120)), "image/png")},
        data={"title": "New scene", "description": "A local test image"},
    )
    assert response.status_code == 200
    image = response.json()["image"]
    assert image["title"] == "New scene"
    assert image["license_name"] == "User-provided · rights not verified"
    assert store.get_image(image["id"]) is not None
    path = api._asset_path(image["image_url"])
    assert path is not None and path.is_file()
    path.unlink(missing_ok=True)


def test_invalid_upload_is_rejected(client):
    test_client, _ = client
    response = test_client.post(
        "/api/ingest",
        files={"file": ("not-an-image.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415
