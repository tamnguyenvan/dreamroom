"""Input-image preparation for the final furniture render."""

from __future__ import annotations

import cv2
import numpy as np

from .placement_geometry import TargetBoxPlacement
from .viz3d import project_points

TARGET_BOX_COLOR = (0, 0, 255)  # red in BGR
TARGET_BOX_EDGES = [
    (0, 1),
    (1, 3),
    (3, 2),
    (2, 0),
    (4, 5),
    (5, 7),
    (7, 6),
    (6, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


def draw_target_box_2d(
    image_bgr: np.ndarray,
    target: TargetBoxPlacement,
    k_px: np.ndarray,
    pixel_scale: tuple[float, float],
) -> np.ndarray:
    """Draw the target box as a red wireframe on a clean room-image copy."""

    frame = image_bgr.copy()
    pixels, valid = project_points(target.box.corners(), k_px)
    pixels *= np.asarray(pixel_scale, dtype=np.float64)
    for start, end in TARGET_BOX_EDGES:
        if valid[start] and valid[end]:
            cv2.line(
                frame,
                tuple(np.round(pixels[start]).astype(int)),
                tuple(np.round(pixels[end]).astype(int)),
                TARGET_BOX_COLOR,
                3,
                cv2.LINE_AA,
            )
    for point, is_valid in zip(pixels, valid):
        if is_valid:
            cv2.circle(
                frame,
                tuple(np.round(point).astype(int)),
                5,
                TARGET_BOX_COLOR,
                -1,
                cv2.LINE_AA,
            )
    return frame
