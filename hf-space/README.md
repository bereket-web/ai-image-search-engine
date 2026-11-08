---
title: FrameFind — AI Image Search
sdk: docker
app_port: 7860
license: mit
short_description: Search a licensed Commons gallery with CLIP
fullWidth: true
tags:
  - image-search
  - computer-vision
  - clip
  - fastapi
  - docker
---

# FrameFind: public image-search demo

Search a small, rights-documented Wikimedia Commons gallery with English text or a reference image. Results use exact cosine ranking over frozen CLIP image embeddings; scores are similarity rankings, not probabilities.

## Privacy and safe use

This is a public demo, not a private workspace or production service. Text queries and reference-image bytes are sent to this Hugging Face Space for inference. Reference images are processed for search and are not added to the shared gallery; the public deployment disables persistent gallery ingestion. Do not submit sensitive or confidential images. Hugging Face may process service-level access or operational logs under its own policies.

There is no user authentication or application-level rate limiting. Reference images are limited to 5 MB and 20 megapixels. The service uses CPU inference and may be slow after a cold start. Do not rely on it for availability or sensitive work.

## Data and model

The gallery is downloaded from Wikimedia Commons at startup, after the app rechecks the current file license and restriction metadata. Each result card shows the source page, creator credit, and license. The repository's MIT license applies to original code only; Commons images retain their separate licenses and attribution requirements.

The upstream `openai/clip-vit-base-patch32` checkpoint is pinned to revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` and downloaded at runtime. The Space repository and Docker image do not contain redistributed model weights, user uploads, or a prebuilt vector index.
