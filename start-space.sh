#!/bin/sh
set -eu

uv run --locked --no-sync image-search seed-demo
exec uv run --locked --no-sync uvicorn image_search.api:app \
  --host 0.0.0.0 \
  --port "${PORT:-7860}" \
  --workers 1 \
  --limit-concurrency 2 \
  --timeout-keep-alive 10
