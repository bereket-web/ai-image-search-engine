from __future__ import annotations

from image_search.embedder import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, ClipEmbedder


def test_default_model_uses_pinned_revision():
    embedder = ClipEmbedder()
    assert embedder.model_id == DEFAULT_MODEL_ID
    assert embedder.revision == DEFAULT_MODEL_REVISION


def test_custom_model_does_not_inherit_clip_revision():
    assert ClipEmbedder(model_id="example/custom-model").revision is None
    assert ClipEmbedder(model_id="example/custom-model", revision="revision-123").revision == "revision-123"
