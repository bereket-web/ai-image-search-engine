from __future__ import annotations

import json
from types import SimpleNamespace

from image_search import evaluation


class FakeEmbedder:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.revision = "test-revision"

    def encode_text(self, text: str) -> str:
        return text


class FakeStore:
    def __init__(self, database_path):
        self.records = [
            (SimpleNamespace(id="user-upload"), 0.99),
            (SimpleNamespace(id="gallery-target"), 0.80),
        ]

    def search(self, query, limit: int, include_ids: set[str]):
        return [(record, score) for record, score in self.records if record.id in include_ids][:limit]


def test_evaluation_ranks_only_the_fixed_gallery(tmp_path, monkeypatch):
    queries_path = tmp_path / "queries.json"
    assets_path = tmp_path / "assets.json"
    output_path = tmp_path / "summary.json"
    queries_path.write_text(json.dumps([{"query": "a target scene", "target_file": "target.jpg"}]), encoding="utf-8")
    assets_path.write_text(json.dumps([{"id": "gallery-target", "file_title": "File:target.jpg"}]), encoding="utf-8")
    monkeypatch.setattr(evaluation, "QUERIES_PATH", queries_path)
    monkeypatch.setattr(evaluation, "ASSETS_PATH", assets_path)
    monkeypatch.setattr(evaluation, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(evaluation, "ClipEmbedder", FakeEmbedder)
    monkeypatch.setattr(evaluation, "VectorStore", FakeStore)

    summary = evaluation.evaluate_queries()

    assert summary["hit_at_1"] == 1.0
    assert summary["results"][0]["target_rank"] == 1
    assert summary["results"][0]["top1_file"] == "File:target.jpg"
    assert json.loads(output_path.read_text(encoding="utf-8")) == summary
