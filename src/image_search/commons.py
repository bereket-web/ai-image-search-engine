"""Rights-aware sample-image retrieval from Wikimedia Commons' public API."""

from __future__ import annotations

import hashlib
import html
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from PIL import Image, UnidentifiedImageError

API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "AIImageSearchPortfolio/0.1 (local research demo; https://commons.wikimedia.org/wiki/Commons:Contact_us)"
MAX_IMAGE_BYTES = 12_000_000
MAX_IMAGE_PIXELS = 50_000_000
_ALLOWED_LICENSES = {
    "public domain",
    "cc0",
    "cc by 2.0",
    "cc by 2.5",
    "cc by 3.0",
    "cc by 4.0",
    "cc by-sa 2.0",
    "cc by-sa 2.5",
    "cc by-sa 3.0",
    "cc by-sa 4.0",
}


def clean_metadata(value: str | None) -> str:
    """Convert Commons' HTML-formatted metadata into readable plain text."""
    if not value:
        return ""
    text = re.sub(r"<br\s*/?>", " ", value, flags=re.IGNORECASE)
    text = re.sub(r"</(?:p|div|li)\s*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def metadata_value(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key, {})
    return clean_metadata(value.get("value", "") if isinstance(value, dict) else str(value))


def is_allowed_license(name: str) -> bool:
    """Allow only a small, explicit set of free or public-domain license labels."""
    return name.strip().lower() in _ALLOWED_LICENSES


def fetch_file_metadata(file_title: str, client: httpx.Client | None = None) -> dict[str, Any]:
    """Fetch current file info, license metadata, and a 960px Commons thumbnail."""
    owns_client = client is None
    http = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30, follow_redirects=True)
    try:
        response = http.get(
            API_URL,
            params={
                "action": "query",
                "format": "json",
                "prop": "imageinfo",
                "iiprop": "url|size|sha1|extmetadata",
                "iiurlwidth": 960,
                "titles": file_title if file_title.startswith("File:") else f"File:{file_title}",
            },
        )
        response.raise_for_status()
        payload = response.json()
        page = next(iter(payload.get("query", {}).get("pages", {}).values()), None)
        if not page or "missing" in page:
            raise ValueError(f"Commons file not found: {file_title}")
        info = (page.get("imageinfo") or [None])[0]
        if not info:
            raise ValueError(f"No image metadata returned for {file_title}")
        ext = info.get("extmetadata", {})
        license_name = metadata_value(ext, "LicenseShortName")
        if not is_allowed_license(license_name):
            raise ValueError(f"Refusing unapproved or missing license for {file_title}: {license_name or 'unknown'}")
        if metadata_value(ext, "DeletionReason"):
            raise ValueError(f"Commons marks {file_title} for deletion; refusing it")
        restrictions = metadata_value(ext, "Restrictions")
        if restrictions:
            raise ValueError(f"Commons lists reuse restrictions for {file_title}: {restrictions}")
        thumb_url = info.get("thumburl") or info.get("url")
        if not thumb_url:
            raise ValueError(f"Commons did not return a download URL for {file_title}")
        description = metadata_value(ext, "ImageDescription")
        title = metadata_value(ext, "ObjectName")
        if not title or title.lower() in {"file", "image"}:
            title = page["title"].removeprefix("File:").rsplit(".", 1)[0].replace("_", " ")
        creator = metadata_value(ext, "Attribution") or metadata_value(ext, "Artist")
        credit = metadata_value(ext, "Credit")
        if credit and credit not in creator:
            creator = f"{creator} · {credit}" if creator else credit
        return {
            "page_id": int(page["pageid"]),
            "file_title": page["title"],
            "title": title,
            "description": description,
            "creator": creator,
            "license_name": license_name,
            "license_url": metadata_value(ext, "LicenseUrl"),
            "source_page": info.get("descriptionurl", ""),
            "source_image_url": info.get("url", ""),
            "download_url": thumb_url,
            "source_sha1": info.get("sha1", ""),
            "source_width": int(info.get("width", 0)),
            "source_height": int(info.get("height", 0)),
            "attribution_required": metadata_value(ext, "AttributionRequired").lower() == "true",
            "copyrighted": metadata_value(ext, "Copyrighted"),
            "restrictions": restrictions,
        }
    finally:
        if owns_client:
            http.close()


def download_demo_assets(catalog_path: str | Path, images_dir: str | Path) -> list[dict[str, Any]]:
    """Recheck Commons licensing and download only manifest-selected free images.

    The manifest contains Commons file titles and human-authored search tags. Each
    accepted asset is revalidated against the current `extmetadata` response before
    any bytes are written to the local gallery.
    """
    catalog_file = Path(catalog_path)
    images_path = Path(images_dir)
    images_path.mkdir(parents=True, exist_ok=True)
    catalog = json.loads(catalog_file.read_text(encoding="utf-8"))
    if not isinstance(catalog, list) or not catalog:
        raise ValueError("The Commons catalog must be a non-empty JSON array")
    client = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=45, follow_redirects=True)
    assets: list[dict[str, Any]] = []
    try:
        for index, item in enumerate(catalog, start=1):
            if not isinstance(item, dict) or not item.get("file_title"):
                raise ValueError(f"Catalog row {index} is missing file_title")
            record = fetch_file_metadata(str(item["file_title"]), client=client)
            response = client.get(record["download_url"])
            response.raise_for_status()
            payload = response.content
            if not payload or len(payload) > MAX_IMAGE_BYTES:
                raise ValueError(
                    f"Image download is empty or exceeds {MAX_IMAGE_BYTES:,} bytes: {record['file_title']}"
                )
            digest = hashlib.sha256(payload).hexdigest()
            try:
                image = Image.open(__import__("io").BytesIO(payload))
                image.verify()
                image = Image.open(__import__("io").BytesIO(payload))
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError(f"Image exceeds {MAX_IMAGE_PIXELS:,} pixels: {record['file_title']}")
                width, height = image.size
                image_format = image.format or "JPEG"
            except (UnidentifiedImageError, OSError) as exc:
                raise ValueError(f"Commons returned an unreadable image: {record['file_title']}") from exc
            extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif"}.get(
                image_format.upper(), ".img"
            )
            file_name = f"commons-{record['page_id']}-{digest[:10]}{extension}"
            destination = images_path / file_name
            destination.write_bytes(payload)
            tags = item.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]
            record["api_title"] = record["title"]
            record["api_description"] = record["description"]
            record["title"] = str(item.get("display_title") or record["title"])
            record["description"] = str(item.get("display_description") or record["description"])
            record.update(
                {
                    "id": f"commons-{record['page_id']}",
                    "image_sha256": digest,
                    "local_file": file_name,
                    "width": width,
                    "height": height,
                    "tags": [str(tag) for tag in tags],
                    "image_url": f"/demo/images/{quote(file_name)}",
                }
            )
            assets.append(record)
            if index < len(catalog):
                time.sleep(0.15)
    finally:
        client.close()
    return assets
