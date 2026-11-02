from __future__ import annotations

import pytest

from image_search.commons import clean_metadata, is_allowed_license
from image_search.store import VectorStore


def add(store: VectorStore, image_id: str, vector: list[float]) -> None:
    store.add_image(
        image_id=image_id,
        title=image_id,
        image_url=f"/demo/images/{image_id}.jpg",
        width=32,
        height=24,
        sha256=f"sha-{image_id}",
        embedding=vector,
    )


def test_exact_cosine_order_and_exclusion(tmp_path):
    store = VectorStore(tmp_path / "vectors.sqlite3")
    add(store, "same", [1, 0, 0])
    add(store, "near", [0.8, 0.6, 0])
    add(store, "opposite", [-1, 0, 0])

    ranked = store.search([1, 0, 0], limit=3)
    assert [row.id for row, _ in ranked] == ["same", "near", "opposite"]
    assert [score for _, score in ranked] == pytest.approx([1.0, 0.8, -1.0])
    assert [row.id for row, _ in store.search([1, 0, 0], exclude_id="same")] == ["near", "opposite"]


def test_vectors_are_normalized_and_persisted(tmp_path):
    path = tmp_path / "vectors.sqlite3"
    store = VectorStore(path)
    add(store, "image", [10, 0])
    reopened = VectorStore(path)
    result, score = reopened.search([3, 0], limit=1)[0]
    assert result.id == "image"
    assert score == pytest.approx(1.0)
    assert reopened.count() == 1


def test_rejects_zero_nonfinite_and_dimension_mismatch(tmp_path):
    store = VectorStore(tmp_path / "vectors.sqlite3")
    with pytest.raises(ValueError, match="zero"):
        add(store, "zero", [0, 0])
    with pytest.raises(ValueError, match="finite"):
        add(store, "nan", [float("nan"), 0])
    add(store, "one", [1, 0])
    with pytest.raises(ValueError, match="dimension"):
        store.search([1, 0, 0])


def test_exact_index_limit_is_enforced(tmp_path):
    store = VectorStore(tmp_path / "vectors.sqlite3", max_images=1)
    add(store, "one", [1, 0])
    with pytest.raises(ValueError, match="limit reached"):
        add(store, "two", [0, 1])


def test_removes_stale_ids_only_within_namespace(tmp_path):
    store = VectorStore(tmp_path / "vectors.sqlite3")
    add(store, "commons-old", [1, 0])
    add(store, "commons-keep", [0, 1])
    add(store, "upload-private", [1, 1])
    assert store.remove_missing_ids("commons-", {"commons-keep"}) == 1
    assert {item.id for item in store.list_images()} == {"commons-keep", "upload-private"}


def test_commons_license_allowlist_is_conservative():
    assert is_allowed_license("CC BY-SA 4.0")
    assert is_allowed_license("Public domain")
    assert is_allowed_license("CC0")
    assert not is_allowed_license("CC BY-NC-SA 4.0")
    assert not is_allowed_license("All rights reserved")
    assert not is_allowed_license("")


def test_commons_metadata_html_is_cleaned():
    assert clean_metadata('<a href="https://example.test">Ada &amp; Bob</a><br>Credit') == "Ada & Bob Credit"
