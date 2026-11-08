from __future__ import annotations

from fastapi.testclient import TestClient

from image_search import api


def test_public_demo_disables_gallery_ingestion(monkeypatch):
    monkeypatch.setattr(api, "PUBLIC_DEMO", True)
    monkeypatch.setattr(api, "GALLERY_INGEST_ENABLED", False)
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 5_000_000)

    with TestClient(api.app) as client:
        health = client.get("/api/health").json()
        response = client.post(
            "/api/ingest",
            files={"file": ("image.png", b"not-read-in-public-mode", "image/png")},
        )

    assert health["public_demo"] is True
    assert health["gallery_ingest_enabled"] is False
    assert health["max_upload_bytes"] == 5_000_000
    assert response.status_code == 404
    assert response.json()["detail"] == "Adding images to the gallery is disabled in this demo"
