from __future__ import annotations

import httpx
import pytest

from image_search.commons import clean_metadata, fetch_file_metadata, is_allowed_license
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


def test_search_can_be_scoped_to_a_curated_gallery(tmp_path):
    store = VectorStore(tmp_path / "vectors.sqlite3")
    add(store, "gallery-near", [0.9, 0.1])
    add(store, "upload-nearest", [1, 0])
    add(store, "gallery-other", [0, 1])

    ranked = store.search([1, 0], limit=3, include_ids={"gallery-near", "gallery-other"})
    assert [row.id for row, _ in ranked] == ["gallery-near", "gallery-other"]


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


def commons_payload(metadata: dict[str, dict[str, str]]) -> dict:
    return {
        "query": {
            "pages": {
                "123": {
                    "pageid": 123,
                    "title": "File:sample.jpg",
                    "imageinfo": [
                        {
                            "url": "https://upload.wikimedia.org/full/sample.jpg",
                            "thumburl": "https://upload.wikimedia.org/thumb/sample.jpg",
                            "sha1": "source-hash",
                            "width": 640,
                            "height": 480,
                            "descriptionurl": "https://commons.wikimedia.org/wiki/File:sample.jpg",
                            "extmetadata": metadata,
                        }
                    ],
                }
            }
        }
    }


def test_fetch_file_metadata_preserves_attribution_fields():
    metadata = {
        "LicenseShortName": {"value": "CC BY 4.0"},
        "LicenseUrl": {"value": "https://creativecommons.org/licenses/by/4.0/"},
        "Artist": {"value": '<a href="https://example.test/artist">Ada &amp; Bob</a>'},
        "ImageDescription": {"value": "A sample landscape"},
        "ObjectName": {"value": "Sample landscape"},
    }
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=commons_payload(metadata)))
    )
    try:
        record = fetch_file_metadata("File:sample.jpg", client=client)
    finally:
        client.close()

    assert record["file_title"] == "File:sample.jpg"
    assert record["creator"] == "Ada & Bob"
    assert record["license_name"] == "CC BY 4.0"
    assert record["license_url"] == "https://creativecommons.org/licenses/by/4.0/"
    assert record["source_page"] == "https://commons.wikimedia.org/wiki/File:sample.jpg"


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"LicenseShortName": {"value": "CC BY-NC-SA 4.0"}}, "unapproved or missing license"),
        (
            {"LicenseShortName": {"value": "CC BY 4.0"}, "Restrictions": {"value": "No commercial use"}},
            "reuse restrictions",
        ),
        (
            {"LicenseShortName": {"value": "CC BY 4.0"}, "DeletionReason": {"value": "Pending review"}},
            "marks .* for deletion",
        ),
    ],
)
def test_fetch_file_metadata_refuses_unapproved_or_restricted_assets(metadata, message):
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=commons_payload(metadata)))
    )
    try:
        with pytest.raises(ValueError, match=message):
            fetch_file_metadata("File:sample.jpg", client=client)
    finally:
        client.close()
