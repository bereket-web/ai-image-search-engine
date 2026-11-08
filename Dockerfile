FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/tmp/huggingface \
    IMAGE_SEARCH_HOME=/app \
    IMAGE_SEARCH_PUBLIC_DEMO=true \
    IMAGE_SEARCH_ENABLE_INGEST=false \
    IMAGE_SEARCH_MAX_UPLOAD_BYTES=5000000 \
    IMAGE_SEARCH_MAX_PIXELS=20000000 \
    CLIP_CPU_THREADS=2

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /uvx /bin/

COPY pyproject.toml uv.lock LICENSE README.md ./
COPY src/ ./src/
COPY data/demo/catalog.json ./data/demo/catalog.json
COPY start-space.sh ./start-space.sh

RUN uv sync --locked --no-dev \
    && uv cache clean \
    && chmod +x ./start-space.sh

EXPOSE 7860
CMD ["./start-space.sh"]
