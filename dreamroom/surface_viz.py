"""Separate 2D diagnostics for room-surface masks."""

from __future__ import annotations

import cv2
import numpy as np

from .sam3_client import SurfaceSegmentation

FLOOR_COLOR = (70, 210, 90)
RUG_COLOR = (40, 210, 240)
WALL_COLORS = (
    (255, 120, 40),
    (220, 70, 220),
    (50, 180, 255),
    (180, 90, 255),
    (255, 190, 70),
    (80, 220, 220),
)


def draw_surface_debug_2d(
    image_bgr: np.ndarray,
    segmentation: SurfaceSegmentation,
) -> np.ndarray:
    """Draw SAM masks on a clean copy, independent of geometry diagnostics."""

    result = image_bgr.copy()
    tint = result.copy()
    floor_mask = segmentation.combined_mask("floor")
    rug_mask = segmentation.combined_mask("rug")
    tint[floor_mask] = FLOOR_COLOR
    tint[rug_mask] = RUG_COLOR
    for index, item in enumerate(segmentation.instances("wall")):
        tint[item.mask] = WALL_COLORS[index % len(WALL_COLORS)]
    result = cv2.addWeighted(result, 0.58, tint, 0.42, 0.0)

    _draw_mask_outline(result, floor_mask, FLOOR_COLOR, "floor")
    _draw_mask_outline(result, rug_mask, RUG_COLOR, "rug")
    for index, item in enumerate(segmentation.instances("wall"), start=1):
        score = "" if item.score is None else f" {item.score:.2f}"
        _draw_mask_outline(
            result,
            item.mask,
            WALL_COLORS[(index - 1) % len(WALL_COLORS)],
            f"wall {index}{score}",
        )

    provider = "OneFormer" if segmentation.provider == "oneformer" else "SAM3"
    summary = (
        f"{provider}: walls={len(segmentation.instances('wall'))}  "
        f"floor={len(segmentation.instances('floor'))}  "
        f"rug={len(segmentation.instances('rug'))}"
    )
    cv2.rectangle(result, (0, 0), (result.shape[1], 36), (20, 20, 20), -1)
    cv2.putText(
        result,
        summary,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return result


def _draw_mask_outline(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    label: str,
) -> None:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return
    cv2.drawContours(image, contours, -1, color, 2, cv2.LINE_AA)
    contour = max(contours, key=cv2.contourArea)
    x, y, _, _ = cv2.boundingRect(contour)
    cv2.putText(
        image,
        label,
        (x + 4, max(22, y + 22)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
        cv2.LINE_AA,
    )
