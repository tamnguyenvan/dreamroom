"""BytePlus ModelArk Seedream 5.0 Pro image-generation client."""

from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"
DEFAULT_MODEL = "dola-seedream-5-0-pro-260628"


@dataclass
class SeedreamResult:
    """Downloaded Seedream output and the response metadata."""

    image_bytes: bytes
    image_url: str
    response: dict
    elapsed_seconds: float

    def to_dict(self) -> dict:
        data = self.response.get("data") or [{}]
        item = data[0] if isinstance(data[0], dict) else {}
        return {
            "model": self.response.get("model"),
            "image_url": self.image_url,
            "size": item.get("size"),
            "usage": self.response.get("usage"),
            "elapsed_seconds": round(float(self.elapsed_seconds), 3),
            "request": {
                "response_format": "url",
                "output_format": "jpeg",
                "size": "1K",
                "optimize_prompt_options": {"mode": "fast"},
            },
        }


class SeedreamClient:
    """Call Seedream with two local images encoded as data URLs."""

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str = DEFAULT_ENDPOINT,
        model: str = DEFAULT_MODEL,
        timeout: float = 300.0,
    ) -> None:
        self.api_key = api_key or os.getenv("ARK_API_KEY")
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        room_image_bgr: np.ndarray,
        furniture_image_bgr: np.ndarray,
        prompt: str,
    ) -> SeedreamResult:
        """Generate one render from room image 1 and furniture image 2."""

        if not self.api_key:
            raise RuntimeError("ARK_API_KEY is required for Seedream rendering")
        payload = {
            "model": self.model,
            "prompt": prompt,
            "image": [
                self._image_data_url(room_image_bgr),
                self._image_data_url(furniture_image_bgr),
            ],
            "size": "1K",
            "output_format": "jpeg",
            "response_format": "url",
            "watermark": False,
            "optimize_prompt_options": {"mode": "fast"},
        }
        started = time.perf_counter()
        response = requests.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Seedream HTTP {response.status_code}: {response.text[:500]}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise RuntimeError("Seedream returned invalid JSON") from exc
        data = body.get("data") or []
        if not data or not isinstance(data[0], dict) or not data[0].get("url"):
            raise RuntimeError("Seedream response is missing data[0].url")
        image_url = str(data[0]["url"])
        image_response = requests.get(image_url, timeout=self.timeout)
        if image_response.status_code != 200:
            raise RuntimeError(
                f"Seedream image download HTTP {image_response.status_code}: "
                f"{image_response.text[:200]}"
            )
        elapsed = time.perf_counter() - started
        logger.info("Seedream render completed in %.1fs", elapsed)
        return SeedreamResult(
            image_bytes=image_response.content,
            image_url=image_url,
            response=body,
            elapsed_seconds=elapsed,
        )

    @staticmethod
    def _image_data_url(image_bgr: np.ndarray) -> str:
        ok, encoded = cv2.imencode(
            ".jpg",
            image_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )
        if not ok:
            raise ValueError("failed to JPEG-encode a Seedream input image")
        payload = base64.b64encode(encoded.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{payload}"
