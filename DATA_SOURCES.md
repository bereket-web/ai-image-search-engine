# Data and model sources

## Demo gallery

The fixed sample manifest is `data/demo/catalog.json`. It selects 16 Wikimedia Commons files spanning landscapes, waterfalls, northern lights, architecture, paintings, animals, plants, ceramics, and food. `image-search seed-demo` contacts the official Commons Action API at run time and checks the current `LicenseShortName`, `Restrictions`, and deletion metadata before downloading the API's 960-pixel thumbnail. The resulting `data/demo/assets.json` records the file page, creator/credit, license name and URL, source checksum, thumbnail checksum, local file, and the curated tags. Do not remove or weaken the allowlist in `src/image_search/commons.py` without reviewing the corresponding licenses.

The collector relies on Wikimedia's documented [Commons machine-readable data/API](https://commons.wikimedia.org/wiki/Commons:Machine-readable_data) and [CommonsMetadata fields](https://www.mediawiki.org/wiki/Extension:CommonsMetadata), including `LicenseShortName`, `LicenseUrl`, `Artist`, `Attribution`, `Restrictions`, and `DeletionReason`.

The gallery's source images are listed by file page in the generated `assets.json`; each demo card links directly to that page. Commons metadata is community-maintained and may be inaccurate or changed. The collector accepts only Public domain, CC0, CC BY 2.0–4.0, and CC BY-SA 2.0–4.0 labels, and refuses files with missing licenses, deletion notices, or reuse restrictions. Respect attribution and ShareAlike conditions when redistributing images; verify the linked file page and local jurisdictional rights before any public reuse. The repo's MIT license covers only original source code, not these image assets.

User uploads are stored locally in ignored `data/uploads/`, are never transmitted to an external model provider by the app, and are not covered by Commons licensing checks. The UI marks their rights as unverified. Do not redistribute uploads without permission.

## Encoder

The encoder defaults to [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32) at immutable Hub revision [`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`](https://huggingface.co/openai/clip-vit-base-patch32/tree/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268), a frozen ViT-B/32 CLIP image encoder and text encoder. The revision can be overridden with `CLIP_MODEL_REVISION`; when selecting a different `CLIP_MODEL_ID`, set an explicit revision for reproducibility. The public model card describes intended research use, warns that untested deployment is out of scope, documents English-only training/evaluation limitations, and notes known weaknesses including fine-grained recognition, counting, and bias. The upstream [OpenAI CLIP source repository](https://github.com/openai/CLIP) is MIT-licensed. This project neither trains nor republishes model weights. Model weights are downloaded and cached locally by Transformers on first use. Confirm the checkpoint's current repository terms before reuse or redistribution.

## No raw copyrighted corpora

The project does not ingest any bulk copyrighted stock-photo or web-crawl dataset. Only the explicitly listed, rights-filtered Commons demo thumbnails are downloaded. No embeddings, API tokens, or user images are published by this local build.
