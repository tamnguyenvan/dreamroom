"""fal.ai client for Gemini Nano Banana Lite image editing."""

from __future__ import annotations

import base64
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "google/nano-banana-lite/edit"


@dataclass
class GeminiEditResult:
    """Downloaded Gemini output and the provider response metadata."""

    image_bytes: bytes
    image_url: str
    response: dict
    elapsed_seconds: float

    def to_dict(self) -> dict:
        images = self.response.get("images") or []
        image = images[0] if images and isinstance(images[0], dict) else {}
        return {
            "model": self.response.get("model", DEFAULT_MODEL),
            "image_url": self.image_url,
            "description": self.response.get("description", ""),
            "size": [image.get("width"), image.get("height")],
            "elapsed_seconds": round(float(self.elapsed_seconds), 3),
        }


class GeminiClient:
    """Call Gemini Nano Banana Lite through the fal.ai Python client."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self.timeout = timeout

    def remove_object(self, image_bgr: np.ndarray, prompt: str) -> GeminiEditResult:
        """Remove the selected object from one uploaded square image."""

        fal_client = self._load_fal_client()
        image_url = self._upload_image(fal_client, image_bgr)
        started = time.perf_counter()
        response = fal_client.subscribe(
            self.model,
            arguments={
                "prompt": prompt,
                "image_urls": [image_url],
                "aspect_ratio": "1:1",
                "output_format": "png",
                "num_images": 1,
                "limit_generations": True,
            },
            client_timeout=self.timeout,
        )
        if not isinstance(response, dict):
            raise RuntimeError("Gemini returned an invalid response")
        images = response.get("images") or []
        image = images[0] if images and isinstance(images[0], dict) else {}
        output_url = image.get("url")
        if not output_url:
            raise RuntimeError("Gemini response is missing images[0].url")
        image_url = str(output_url)
        image_bytes = self._read_url(image_url)
        elapsed = time.perf_counter() - started
        logger.info("Gemini object removal completed in %.1fs", elapsed)
        return GeminiEditResult(
            image_bytes=image_bytes,
            image_url=image_url,
            response=response,
            elapsed_seconds=elapsed,
        )

    @staticmethod
    def _load_fal_client():
        if not os.getenv("FAL_KEY"):
            raise RuntimeError("FAL_KEY is required for Gemini object removal")
        try:
            import fal_client
        except ImportError as exc:
            raise RuntimeError(
                "fal-client is required; install dependencies from requirements.txt"
            ) from exc
        return fal_client

    @staticmethod
    def _upload_image(fal_client, image_bgr: np.ndarray) -> str:
        ok, encoded = cv2.imencode(".png", image_bgr)
        if not ok:
            raise ValueError("failed to PNG-encode the Gemini input image")
        path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                handle.write(encoded.tobytes())
                path = Path(handle.name)
            return str(fal_client.upload_file(path))
        finally:
            if path is not None:
                path.unlink(missing_ok=True)

    def _read_url(self, url: str) -> bytes:
        if url.startswith("data:"):
            try:
                return base64.b64decode(url.split(",", 1)[1])
            except (IndexError, ValueError) as exc:
                raise RuntimeError("invalid Gemini image data URI") from exc
        response = requests.get(url, timeout=self.timeout)
        if response.status_code != 200:
            raise RuntimeError(
                f"Gemini image download HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
        return response.content
