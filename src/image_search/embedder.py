"""Lazy CLIP image/text encoders. Model files are cached by Hugging Face Transformers locally."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"


class ClipEmbedder:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str | None = None):
        self.model_id = model_id
        self.device_name = device
        self._model = None
        self._processor = None
        self._torch = None

    @property
    def device(self) -> str:
        self._load()
        return str(self._device)

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor

        torch.set_num_threads(max(1, int(os.getenv("CLIP_CPU_THREADS", "4"))))
        chosen = self.device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(chosen)
        self._processor = CLIPProcessor.from_pretrained(self.model_id)
        self._model = CLIPModel.from_pretrained(self.model_id).to(self._device).eval()
        self._torch = torch

    @staticmethod
    def _normalize(features) -> np.ndarray:
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        elif hasattr(features, "image_embeds"):
            features = features.image_embeds
        elif hasattr(features, "text_embeds"):
            features = features.text_embeds
        if hasattr(features, "detach"):
            features = features.detach().float().cpu().numpy()
        array = np.asarray(features, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if not np.isfinite(array).all() or np.any(norms == 0):
            raise ValueError("The encoder produced an invalid embedding")
        return np.ascontiguousarray(array / norms, dtype=np.float32)

    @staticmethod
    def _open_image(image: Image.Image | str | Path | bytes | BinaryIO) -> Image.Image:
        try:
            if isinstance(image, Image.Image):
                opened = image.copy()
            elif isinstance(image, (str, Path)):
                opened = Image.open(image)
            elif isinstance(image, bytes):
                opened = Image.open(io.BytesIO(image))
            else:
                opened = Image.open(image)
            if opened.width * opened.height > 50_000_000:
                raise ValueError("Images larger than 50 megapixels are not supported")
            return ImageOps.exif_transpose(opened).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Please provide a valid PNG, JPEG, or WebP image") from exc

    def encode_images(self, images: list[Image.Image | str | Path | bytes | BinaryIO]) -> np.ndarray:
        if not images:
            raise ValueError("At least one image is required")
        self._load()
        opened = [self._open_image(image) for image in images]
        inputs = self._processor(images=opened, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            features = self._model.get_image_features(**inputs)
        return self._normalize(features)

    def encode_image(self, image: Image.Image | str | Path | bytes | BinaryIO) -> np.ndarray:
        return self.encode_images([image])[0]

    def encode_text(self, text: str) -> np.ndarray:
        clean = text.strip()
        if not clean or len(clean) > 512:
            raise ValueError("Text query must contain 1–512 characters")
        self._load()
        inputs = self._processor(text=[clean], return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            features = self._model.get_text_features(**inputs)
        return self._normalize(features)[0]
