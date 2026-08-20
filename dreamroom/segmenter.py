"""Remote SimpleClick interactive segmentation client."""

from __future__ import annotations

import base64
import binascii
import logging
import time

import cv2
import numpy as np
import requests

from .config import Settings

logger = logging.getLogger(__name__)


def sample_points(points: list[list[float]], limit: int) -> list[list[int]]:
    """Keep at most ``limit`` points, evenly spread over the full stroke."""

    if len(points) <= limit:
        return [[int(round(p[0])), int(round(p[1]))] for p in points]
    step = (len(points) - 1) / (limit - 1)
    indices = [round(i * step) for i in range(limit)]
    return [[int(round(points[i][0])), int(round(points[i][1]))] for i in indices]


def _validate_points(
    points: list[list[int]], width: int, height: int, name: str, required: bool
) -> list[list[int]]:
    if not points:
        if required:
            raise ValueError(f"{name} must contain at least one [x, y] point")
        return []
    for index, point in enumerate(points):
        if len(point) != 2:
            raise ValueError(f"{name}[{index}] must be [x, y]")
        x, y = float(point[0]), float(point[1])
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"{name}[{index}] is outside the image bounds")
    return [[int(p[0]), int(p[1])] for p in points]


class SimpleClickSegmenter:
    """Call the deployed SimpleClick service for each confirmed stroke set."""

    def __init__(self, settings: Settings) -> None:
        self.endpoint = settings.simpleclick_endpoint.rstrip("/")
        self.timeout = settings.simpleclick_timeout
        self.threshold = settings.threshold
        self.max_points = settings.max_points

    def segment(
        self,
        image_rgb: np.ndarray,
        positive_points: list[list[int]],
        negative_points: list[list[int]] | None = None,
        threshold: float | None = None,
    ) -> np.ndarray:
        """Send an RGB image and clicks; return a boolean mask at image size."""

        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb must be an HxWx3 RGB array")
        threshold = self.threshold if threshold is None else float(threshold)
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1")

        height, width = image_rgb.shape[:2]
        positive = sample_points(
            _validate_points(positive_points, width, height, "positive_points", True),
            self.max_points,
        )
        negative = sample_points(
            _validate_points(
                negative_points or [], width, height, "negative_points", False
            ),
            self.max_points,
        )
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".png", image_bgr)
        if not ok:
            raise ValueError("failed to PNG-encode the SimpleClick input image")

        payload = {
            "image": base64.b64encode(encoded.tobytes()).decode("ascii"),
            "positive_points": positive,
            "negative_points": negative,
            "threshold": threshold,
        }
        logger.info("calling remote SimpleClick at %s", self.endpoint)
        started = time.perf_counter()
        response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        if response.status_code != 200:
            raise RuntimeError(
                f"SimpleClick HTTP {response.status_code}: {response.text[:300]}"
            )
        try:
            result = response.json()
        except ValueError as exc:
            raise RuntimeError("SimpleClick returned invalid JSON") from exc
        mask = self._decode_mask(result.get("mask"), (height, width))
        logger.info(
            "remote SimpleClick completed in %.1fs", time.perf_counter() - started
        )
        return mask

    @staticmethod
    def _decode_mask(value: object, image_shape: tuple[int, int]) -> np.ndarray:
        if not isinstance(value, str) or not value:
            raise RuntimeError("SimpleClick response is missing a mask")
        encoded = value.split(",", 1)[1] if value.startswith("data:") else value
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise RuntimeError("SimpleClick returned an invalid base64 mask") from exc
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError("SimpleClick returned an undecodable mask")
        if image.shape != image_shape:
            image = cv2.resize(
                image,
                (image_shape[1], image_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        return image > 0
