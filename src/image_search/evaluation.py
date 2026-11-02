"""Evaluate the curated paraphrase queries against known sample targets."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from image_search.embedder import ClipEmbedder
from image_search.store import VectorStore

ROOT = Path(os.getenv("IMAGE_SEARCH_HOME", Path.cwd())).resolve()
QUERIES_PATH = ROOT / "reports" / "evaluation_queries.json"
ASSETS_PATH = ROOT / "data" / "demo" / "assets.json"
OUTPUT_PATH = ROOT / "reports" / "evaluation_summary.json"


def evaluate_queries() -> dict[str, Any]:
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    assets = json.loads(ASSETS_PATH.read_text(encoding="utf-8"))
    targets = {row["file_title"].removeprefix("File:"): row["id"] for row in assets}
    embedder = ClipEmbedder(model_id=os.getenv("CLIP_MODEL_ID", "openai/clip-vit-base-patch32"))
    store = VectorStore(Path(os.getenv("IMAGE_SEARCH_DB", ROOT / "data" / "index.sqlite3")))
    rows: list[dict[str, Any]] = []
    for item in queries:
        expected_id = targets[item["target_file"]]
        ranking = store.search(embedder.encode_text(item["query"]), limit=min(store.count(), 100))
        target_rank = next((rank for rank, (record, _) in enumerate(ranking, 1) if record.id == expected_id), None)
        rows.append(
            {
                "query": item["query"],
                "target_file": item["target_file"],
                "target_rank": target_rank,
                "top1_file": next((asset["file_title"] for asset in assets if asset["id"] == ranking[0][0].id), None)
                if ranking
                else None,
                "target_cosine": next((round(score, 6) for record, score in ranking if record.id == expected_id), None),
            }
        )
    reciprocal = [1.0 / row["target_rank"] if row["target_rank"] else 0.0 for row in rows]
    summary = {
        "model_id": embedder.model_id,
        "gallery_images": len(assets),
        "queries": len(rows),
        "hit_at_1": float(np.mean([row["target_rank"] == 1 for row in rows])) if rows else 0.0,
        "hit_at_5": float(np.mean([row["target_rank"] is not None and row["target_rank"] <= 5 for row in rows]))
        if rows
        else 0.0,
        "mrr": float(np.mean(reciprocal)) if rows else 0.0,
        "mean_target_rank": float(np.mean([row["target_rank"] for row in rows if row["target_rank"]]))
        if any(row["target_rank"] for row in rows)
        else None,
        "results": rows,
        "evaluation_note": "Small, hand-curated, single-gallery prompt test; targets and paraphrased queries were authored for this demo. Not an independent or representative benchmark.",
    }
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary
