# FrameFind architecture

```mermaid
flowchart LR
    U[Browser: local search UI] -->|Text / image query| API[FastAPI]
    U -->|Image ingestion| API
    API --> EMB[CLIP ViT-B/32 encoder\nimage and English text]
    EMB -->|L2-normalized 512-d float32 vectors| DB[(SQLite metadata + vectors)]
    API -->|Exact dot product = cosine| DB
    API -->|ranked results + source metadata| U
    SEED[Commons seed command] -->|license recheck + local files| IMG[Curated demo gallery]
    IMG --> EMB
    EMB --> DB
    UP[User uploads] -->|local only, git-ignored| EMB
    UP --> DB
```

Search, upload, image content, and index data flow between the browser and the local FastAPI process. The page references Google Fonts, so a browser with network access may make a separate font request; no prompt or image content is sent to Google. For strict offline use, self-host or replace those font links. The seed command reads a fixed Commons manifest, rechecks each file's current `LicenseShortName`, `Restrictions`, and deletion metadata through the official API, downloads the thumbnail, computes an embedding, and persists both vector and provenance fields. User uploads are placed under `data/uploads/`, excluded from version control, and are not sent to third-party services.

The public Hugging Face Space runs with `IMAGE_SEARCH_PUBLIC_DEMO=true`: it disables gallery ingestion and does not mount `/uploads`, so visitors cannot persistently add files to the shared collection. Text and reference-image queries are sent to the hosted Space; reference images are processed for search and are not added to the gallery. The UI warns users not to submit sensitive content, and the hosted upload limit is 5 MB. The public demo has no user authentication or application-level rate limiting and is not a production service. Its pinned upstream CLIP checkpoint is downloaded at runtime rather than bundled into the Space repository or Docker image.

The search index is an intentionally transparent SQLite store plus a NumPy exact cosine scan. Cost is O(ND) per request for N images and D=512 dimensions. This design is suited to a small local gallery; it is not an ANN/vector-database scale-out architecture. A configurable hard limit defaults to 25,000 records. Larger libraries need a measured ANN backend and recall/latency tests before the limit is raised.
