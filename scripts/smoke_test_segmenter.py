#!/usr/bin/env python
"""Non-interactive smoke test for the remote SimpleClick service.

Builds a synthetic image (dark circle on a light background), runs one
segmentation with a positive point on the circle and a negative point on the
background, then verifies that the mask covers the circle and excludes the
background.

Usage:
    python scripts/smoke_test_segmenter.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from dreamroom.config import Settings  # noqa: E402
from dreamroom.image_ops import resize_max_side  # noqa: E402
from dreamroom.segmenter import SimpleClickSegmenter, sample_points  # noqa: E402


def make_synthetic_image() -> np.ndarray:
    """RGB image: light gradient background, dark circle 'object'."""

    height, width = 480, 640
    y, x = np.mgrid[0:height, 0:width]
    background = (200 + (x / width * 40 - 20)).astype(np.uint8)
    image = np.stack([background, background, background], axis=-1)
    cv2.circle(image, (width // 2, height // 2), 80, (40, 40, 40), -1)
    return image


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    assert sample_points([[i, i] for i in range(100)], 24) == [
        [round(i * 99 / 23), round(i * 99 / 23)] for i in range(24)
    ]

    image, scale = resize_max_side(make_synthetic_image(), Settings().max_side)
    assert max(image.shape[:2]) <= 1280 and scale == 1.0

    segmenter = SimpleClickSegmenter(Settings())
    positive = [[320, 240]]  # circle center [x, y]
    negative = [[20, 20]]  # background corner

    mask = segmenter.segment(image, positive, negative)
    assert mask.shape == image.shape[:2]
    assert mask.dtype == bool
    assert mask[240, 320], "circle center must be inside the mask"
    assert not mask[20, 20], "background corner must be outside the mask"
    area_fraction = float(mask.mean())
    assert 0.001 < area_fraction < 0.9, f"implausible mask area {area_fraction:.4f}"

    mask2 = segmenter.segment(image, positive + [[290, 240]], negative)
    assert mask2.shape == mask.shape

    print(f"synthetic remote segmentation OK: area={area_fraction:.3f}")


if __name__ == "__main__":
    main()
