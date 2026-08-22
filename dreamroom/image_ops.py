"""Image loading, saving, and resize helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_image_bgr(path: str | Path) -> np.ndarray:
    """Load an image as BGR; raises a clear error on failure."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"image does not exist: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode image: {path}")
    return image


def decode_image_bgr(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes as BGR; raises a clear error on failure."""

    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode image bytes")
    return image


def resize_max_side(image: np.ndarray, max_side: int = 1280) -> tuple[np.ndarray, float]:
    """Resize so the longest side is at most ``max_side``.

    Returns the resized image and the scale factor that maps resized
    coordinates back to original coordinates (``original = resized * scale``).
    """

    if max_side <= 0:
        raise ValueError("max_side must be positive")
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image.copy(), 1.0

    scale = max_side / longest
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized, 1.0 / scale


def save_image(path: str | Path, image_bgr: np.ndarray) -> Path:
    """Write an image, creating parent directories."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image_bgr):
        raise IOError(f"failed to write image: {path}")
    return path


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    """Convert a boolean/float mask to a 0-255 single-channel image."""

    return np.where(mask > 0, 255, 0).astype(np.uint8)
