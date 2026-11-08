"""SQLite metadata store and exact cosine-similarity index for small local galleries."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

MAX_EXACT_INDEX_IMAGES = 25_000


@dataclass(frozen=True)
class ImageRecord:
    id: str
    title: str
    description: str
    creator: str
    license_name: str
    license_url: str
    source_page: str
    image_url: str
    width: int
    height: int
    created_at: str = ""

    def public_dict(self, score: float | None = None) -> dict[str, Any]:
        item = asdict(self)
        if score is not None:
            item["score"] = round(float(score), 6)
        return item


class VectorStore:
    """Persist float32 vectors in SQLite and rank them by exact normalized dot product.

    This implementation deliberately favors a portable, inspectable local index over
    approximate nearest-neighbor infrastructure. Search cost grows linearly with the
    number of indexed images; the built-in limit prevents accidental large scans.
    """

    def __init__(self, database_path: str | Path, max_images: int = MAX_EXACT_INDEX_IMAGES):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_images = max_images
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS images (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    creator TEXT NOT NULL DEFAULT '',
                    license_name TEXT NOT NULL DEFAULT '',
                    license_url TEXT NOT NULL DEFAULT '',
                    source_page TEXT NOT NULL DEFAULT '',
                    image_url TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_images_created ON images(created_at DESC)")

    @staticmethod
    def _validate_embedding(embedding: np.ndarray | list[float]) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("Embedding must be a finite, non-empty vector")
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError("Embedding cannot be the zero vector")
        return np.ascontiguousarray(vector / norm, dtype=np.float32)

    def add_image(
        self,
        *,
        image_id: str,
        title: str,
        description: str = "",
        creator: str = "",
        license_name: str = "",
        license_url: str = "",
        source_page: str = "",
        image_url: str,
        width: int,
        height: int,
        sha256: str,
        embedding: np.ndarray | list[float],
    ) -> None:
        vector = self._validate_embedding(embedding)
        with self._connect() as connection:
            existing = connection.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            if (
                existing >= self.max_images
                and not connection.execute("SELECT 1 FROM images WHERE id = ?", (image_id,)).fetchone()
            ):
                raise ValueError(f"Exact local index limit reached ({self.max_images:,} images)")
            dimensions = connection.execute("SELECT DISTINCT dimension FROM images").fetchall()
            if dimensions and any(row[0] != vector.size for row in dimensions):
                raise ValueError("Embedding dimension does not match the existing index")
            connection.execute(
                """
                INSERT INTO images (
                    id, title, description, creator, license_name, license_url,
                    source_page, image_url, width, height, sha256, dimension, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title, description=excluded.description,
                    creator=excluded.creator, license_name=excluded.license_name,
                    license_url=excluded.license_url, source_page=excluded.source_page,
                    image_url=excluded.image_url, width=excluded.width, height=excluded.height,
                    sha256=excluded.sha256, dimension=excluded.dimension, embedding=excluded.embedding
                """,
                (
                    image_id,
                    title.strip() or image_id,
                    description,
                    creator,
                    license_name,
                    license_url,
                    source_page,
                    image_url,
                    int(width),
                    int(height),
                    sha256,
                    int(vector.size),
                    vector.tobytes(),
                ),
            )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ImageRecord:
        return ImageRecord(
            id=row["id"],
            title=row["title"],
            description=row["description"],
            creator=row["creator"],
            license_name=row["license_name"],
            license_url=row["license_url"],
            source_page=row["source_page"],
            image_url=row["image_url"],
            width=row["width"],
            height=row["height"],
            created_at=row["created_at"],
        )

    def list_images(self, limit: int = 500, offset: int = 0) -> list[ImageRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM images ORDER BY created_at DESC, title COLLATE NOCASE LIMIT ? OFFSET ?",
                (max(0, min(limit, self.max_images)), max(0, offset)),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get_image(self, image_id: str) -> ImageRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM images").fetchone()[0])

    def dimensions(self) -> list[int]:
        with self._connect() as connection:
            rows = connection.execute("SELECT DISTINCT dimension FROM images ORDER BY dimension").fetchall()
        return [int(row[0]) for row in rows]

    def remove_missing_ids(self, prefix: str, keep_ids: set[str]) -> int:
        with self._connect() as connection:
            rows = connection.execute("SELECT id FROM images").fetchall()
            stale = [row[0] for row in rows if row[0].startswith(prefix) and row[0] not in keep_ids]
            connection.executemany("DELETE FROM images WHERE id = ?", [(image_id,) for image_id in stale])
        return len(stale)

    def search(
        self,
        query_embedding: np.ndarray | list[float],
        limit: int = 24,
        exclude_id: str | None = None,
        include_ids: set[str] | None = None,
    ) -> list[tuple[ImageRecord, float]]:
        query = self._validate_embedding(query_embedding)
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM images ORDER BY id").fetchall()
        if not rows:
            return []
        if len(rows) > self.max_images:
            raise ValueError(f"Index has more than the configured exact-search limit ({self.max_images:,})")
        valid_rows: list[sqlite3.Row] = []
        vectors: list[np.ndarray] = []
        for row in rows:
            if include_ids is not None and row["id"] not in include_ids:
                continue
            if exclude_id is not None and row["id"] == exclude_id:
                continue
            if row["dimension"] != query.size:
                raise ValueError("Query embedding dimension does not match the index")
            valid_rows.append(row)
            vectors.append(np.frombuffer(row["embedding"], dtype=np.float32))
        if not vectors:
            return []
        matrix = np.stack(vectors).astype(np.float32, copy=False)
        scores = matrix @ query
        order = np.argsort(-scores, kind="stable")[:limit]
        return [(self._row_to_record(valid_rows[int(i)]), float(scores[int(i)])) for i in order]
