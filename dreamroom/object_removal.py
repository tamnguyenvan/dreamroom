"""Aspect-ratio cropping and compositing helpers for selected-object removal."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

SUPPORTED_ASPECT_RATIOS = (
    "21:9",
    "16:9",
    "3:2",
    "4:3",
    "5:4",
    "1:1",
    "4:5",
    "3:4",
    "2:3",
    "9:16",
    "4:1",
    "1:4",
    "8:1",
    "1:8",
)


@dataclass(frozen=True)
class ObjectCrop:
    """A direct, unpadded crop containing the selected object."""

    image_bgr: np.ndarray
    x: int
    y: int
    width: int
    height: int
    aspect_ratio: str
    object_bbox_xyxy: tuple[int, int, int, int]

    @property
    def size(self) -> int:
        """Return the edge length for a square crop."""

        if self.width != self.height:
            raise ValueError("rectangular crops do not have a single size")
        return self.width

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "size": self.width if self.width == self.height else None,
            "width": self.width,
            "height": self.height,
            "aspect_ratio": self.aspect_ratio,
            "bounds_xyxy": [
                self.x,
                self.y,
                self.x + self.width,
                self.y + self.height,
            ],
            "object_bbox_xyxy": list(self.object_bbox_xyxy),
            "size_hw": [self.height, self.width],
        }


def crop_selected_object(image_bgr: np.ndarray, mask: np.ndarray) -> ObjectCrop:
    """Crop the selected object using the best supported direct aspect ratio.

    1:1 keeps the existing behavior when the complete selection fits in a
    square whose edge is the shorter image side. If it does not fit, choose
    the supported ratio closest to the selection bounding box and use the
    largest direct crop of that ratio that contains the selection. No padding
    is introduced.
    """

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must have shape (height, width, 3)")
    if mask.shape != image_bgr.shape[:2]:
        raise ValueError("selected mask must match the image height and width")

    selected_y, selected_x = np.where(mask > 0)
    if selected_x.size == 0:
        raise ValueError("selected mask is empty; cannot crop the selected object")

    height, width = image_bgr.shape[:2]
    x_min = int(selected_x.min())
    y_min = int(selected_y.min())
    x_max = int(selected_x.max())
    y_max = int(selected_y.max())
    bbox = (x_min, y_min, x_max + 1, y_max + 1)
    object_width = x_max - x_min + 1
    object_height = y_max - y_min + 1

    try:
        return _make_crop(
            image_bgr,
            bbox,
            width=min(height, width),
            height=min(height, width),
            aspect_ratio="1:1",
        )
    except ValueError as square_error:
        object_ratio = object_width / object_height
        candidates: list[tuple[float, str, int, int]] = []
        for aspect_ratio in SUPPORTED_ASPECT_RATIOS:
            if aspect_ratio == "1:1":
                continue
            crop_width, crop_height = _largest_crop_size(
                width,
                height,
                aspect_ratio,
            )
            if crop_width < object_width or crop_height < object_height:
                continue
            try:
                _crop_origin(width, height, crop_width, crop_height, bbox)
            except ValueError:
                continue
            ratio = _ratio_value(aspect_ratio)
            candidates.append(
                (
                    abs(math.log(ratio / object_ratio)),
                    aspect_ratio,
                    crop_width,
                    crop_height,
                )
            )

        if not candidates:
            raise ValueError(
                "selected object does not fit any supported crop aspect ratio; "
                f"object bbox={bbox}, image_size_hw={[height, width]}"
            ) from square_error
        _, aspect_ratio, crop_width, crop_height = min(candidates)
        return _make_crop(
            image_bgr,
            bbox,
            width=crop_width,
            height=crop_height,
            aspect_ratio=aspect_ratio,
        )


def _make_crop(
    image_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    aspect_ratio: str,
) -> ObjectCrop:
    image_height, image_width = image_bgr.shape[:2]
    x, y = _crop_origin(image_width, image_height, width, height, bbox)
    return ObjectCrop(
        image_bgr=image_bgr[y : y + height, x : x + width].copy(),
        x=x,
        y=y,
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        object_bbox_xyxy=bbox,
    )


def _crop_origin(
    image_width: int,
    image_height: int,
    crop_width: int,
    crop_height: int,
    bbox: tuple[int, int, int, int],
) -> tuple[int, int]:
    x_min, y_min, x_max, y_max = bbox
    if crop_width > image_width or crop_height > image_height:
        raise ValueError("crop is larger than the source image")
    if x_max - x_min > crop_width or y_max - y_min > crop_height:
        raise ValueError("selected object is larger than the crop")

    x_lower = max(0, x_max - crop_width)
    x_upper = min(x_min, image_width - crop_width)
    y_lower = max(0, y_max - crop_height)
    y_upper = min(y_min, image_height - crop_height)
    if x_lower > x_upper or y_lower > y_upper:
        raise ValueError("could not place an unpadded crop around the selected object")

    object_center_x = (x_min + x_max) / 2.0
    object_center_y = (y_min + y_max) / 2.0
    x = _clamp(round(object_center_x - crop_width / 2), x_lower, x_upper)
    y = _clamp(round(object_center_y - crop_height / 2), y_lower, y_upper)
    return x, y


def _largest_crop_size(
    image_width: int,
    image_height: int,
    aspect_ratio: str,
) -> tuple[int, int]:
    ratio_width, ratio_height = _parse_aspect_ratio(aspect_ratio)
    crop_width = min(image_width, math.floor(image_height * ratio_width / ratio_height))
    crop_height = min(image_height, math.floor(image_width * ratio_height / ratio_width))
    return max(1, crop_width), max(1, crop_height)


def _ratio_value(aspect_ratio: str) -> float:
    width, height = _parse_aspect_ratio(aspect_ratio)
    return width / height


def _parse_aspect_ratio(aspect_ratio: str) -> tuple[int, int]:
    try:
        width, height = (int(value) for value in aspect_ratio.split(":", 1))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid aspect ratio: {aspect_ratio!r}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid aspect ratio: {aspect_ratio!r}")
    return width, height


# Backward-compatible name for callers that only use the square path.
SquareObjectCrop = ObjectCrop


def resize_patch(patch_bgr: np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    """Resize an edited patch to the exact crop size."""

    if patch_bgr.ndim != 3 or patch_bgr.shape[2] != 3:
        raise ValueError("patch_bgr must have shape (height, width, 3)")
    if isinstance(size, int):
        target_width = target_height = size
    else:
        target_width, target_height = size
    if target_width <= 0 or target_height <= 0:
        raise ValueError("patch dimensions must be positive")
    if patch_bgr.shape[:2] == (target_height, target_width):
        return patch_bgr.copy()
    return cv2.resize(
        patch_bgr,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )


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
    crop: ObjectCrop,
) -> np.ndarray:
    """Replace the crop bounds with an edited patch and return a new image."""

    if patch_bgr.shape[:2] != (crop.height, crop.width):
        raise ValueError("patch must match the crop size before stitching")
    result = image_bgr.copy()
    result[crop.y : crop.y + crop.height, crop.x : crop.x + crop.width] = patch_bgr
    return result


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))
