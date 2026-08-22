"""Square cropping and compositing helpers for selected-object removal."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SquareObjectCrop:
    """A direct, unpadded square crop containing the selected object."""

    image_bgr: np.ndarray
    x: int
    y: int
    size: int
    object_bbox_xyxy: tuple[int, int, int, int]

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "size": self.size,
            "bounds_xyxy": [self.x, self.y, self.x + self.size, self.y + self.size],
            "object_bbox_xyxy": list(self.object_bbox_xyxy),
            "size_hw": [self.size, self.size],
        }


def crop_selected_object(image_bgr: np.ndarray, mask: np.ndarray) -> SquareObjectCrop:
    """Crop a square of ``min(height, width)`` around the selected mask.

    The crop is always a direct image slice. Its origin is chosen to contain
    the complete mask bounding box whenever that is possible, while staying as
    close as possible to the mask center. No padding is introduced.
    """

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must have shape (height, width, 3)")
    if mask.shape != image_bgr.shape[:2]:
        raise ValueError("selected mask must match the image height and width")

    selected_y, selected_x = np.where(mask > 0)
    if selected_x.size == 0:
        raise ValueError("selected mask is empty; cannot crop the selected object")

    height, width = image_bgr.shape[:2]
    size = min(height, width)
    x_min = int(selected_x.min())
    y_min = int(selected_y.min())
    x_max = int(selected_x.max())
    y_max = int(selected_y.max())
    bbox = (x_min, y_min, x_max + 1, y_max + 1)
    if x_max - x_min + 1 > size or y_max - y_min + 1 > size:
        raise ValueError(
            "selected object is larger than the required square crop; "
            f"object bbox={bbox}, crop_size={size}"
        )

    x_lower = max(0, x_max - size + 1)
    x_upper = min(x_min, width - size)
    y_lower = max(0, y_max - size + 1)
    y_upper = min(y_min, height - size)
    if x_lower > x_upper or y_lower > y_upper:
        raise ValueError("could not place an unpadded square crop around the selected object")

    object_center_x = (x_min + x_max + 1) / 2.0
    object_center_y = (y_min + y_max + 1) / 2.0
    x = _clamp(round(object_center_x - size / 2), x_lower, x_upper)
    y = _clamp(round(object_center_y - size / 2), y_lower, y_upper)
    return SquareObjectCrop(
        image_bgr=image_bgr[y : y + size, x : x + size].copy(),
        x=x,
        y=y,
        size=size,
        object_bbox_xyxy=bbox,
    )


def resize_patch(patch_bgr: np.ndarray, size: int) -> np.ndarray:
    """Resize an edited patch to the exact square crop size."""

    if size <= 0:
        raise ValueError("patch size must be positive")
    if patch_bgr.ndim != 3 or patch_bgr.shape[2] != 3:
        raise ValueError("patch_bgr must have shape (height, width, 3)")
    if patch_bgr.shape[:2] == (size, size):
        return patch_bgr.copy()
    return cv2.resize(patch_bgr, (size, size), interpolation=cv2.INTER_AREA)


def annotate_selected_object(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mark the selected object with a translucent red mask and contour."""

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must have shape (height, width, 3)")
    if mask.shape != image_bgr.shape[:2]:
        raise ValueError("selected mask must match the crop height and width")

    frame = image_bgr.copy()
    color_layer = np.zeros_like(frame)
    color_layer[:] = (0, 0, 255)
    tinted = cv2.addWeighted(frame, 0.65, color_layer, 0.35, 0)
    selected = mask > 0
    frame[selected] = tinted[selected]
    contours, _ = cv2.findContours(
        selected.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(frame, contours, -1, (0, 0, 255), 3, cv2.LINE_AA)
    return frame


def stitch_patch(
    image_bgr: np.ndarray,
    patch_bgr: np.ndarray,
    crop: SquareObjectCrop,
) -> np.ndarray:
    """Replace the crop bounds with an edited patch and return a new image."""

    if patch_bgr.shape[:2] != (crop.size, crop.size):
        raise ValueError("patch must match the square crop size before stitching")
    result = image_bgr.copy()
    result[crop.y : crop.y + crop.size, crop.x : crop.x + crop.size] = patch_bgr
    return result


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))
