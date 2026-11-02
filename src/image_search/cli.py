"""CLI helpers for seeding the gallery, evaluating retrieval, and launching the local service."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(prog="image-search", description="Local-first CLIP image retrieval")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="run the local search interface and API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    seed = commands.add_parser("seed-demo", help="download the licensed sample gallery and create embeddings")
    seed.add_argument("--batch-size", type=int, default=8)
    commands.add_parser("stats", help="show local index metadata without loading the model")
    commands.add_parser("evaluate", help="run the small curated prompt-retrieval evaluation")
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn

        uvicorn.run("image_search.api:app", host=args.host, port=args.port, reload=args.reload)
    elif args.command == "seed-demo":
        from image_search.seed import seed_demo_gallery

        print(json.dumps(seed_demo_gallery(batch_size=args.batch_size), indent=2))
    elif args.command == "stats":
        from image_search.api import store

        print(json.dumps({"indexed_images": store.count()}, indent=2))
    elif args.command == "evaluate":
        from image_search.evaluation import evaluate_queries

        print(json.dumps(evaluate_queries(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
