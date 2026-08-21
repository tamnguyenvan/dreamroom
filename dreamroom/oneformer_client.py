"""Client for the remote OneFormer semantic room-surface segmenter."""

from __future__ import annotations

import base64
import binascii
import logging

import cv2
import numpy as np
import requests

from .sam3_client import Sam3Mask, SurfaceSegmentation

logger = logging.getLogger(__name__)

SURFACE_LABELS = ("wall", "floor", "rug")


class OneFormerClient:
    """Call the deployed OneFormer semantic-segmentation endpoint."""

    def __init__(self, endpoint: str, timeout: float = 300.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def segment_surfaces(self, image_bgr: np.ndarray) -> SurfaceSegmentation:
        """Return OneFormer wall, floor, and rug masks at image resolution."""

        response = requests.post(
            self.endpoint,
            json={"image": self._encode_image(image_bgr)},
            timeout=self.timeout,
        )
        if response.status_code != 200:
            detail = response.text[:300]
            raise RuntimeError(
                f"OneFormer request failed with HTTP {response.status_code}: {detail}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("OneFormer returned invalid JSON") from exc

        image_shape = tuple(payload.get("image_size_hw", image_bgr.shape[:2]))
        if len(image_shape) != 2:
            raise RuntimeError("OneFormer response has an invalid image shape")
        image_shape = (int(image_shape[0]), int(image_shape[1]))
        masks: dict[str, list[Sam3Mask]] = {}
        raw_masks = payload.get("masks") or {}
        for label in SURFACE_LABELS:
            parsed: list[Sam3Mask] = []
            for item in raw_masks.get(label, []) or []:
                if not isinstance(item, dict) or not item.get("mask"):
                    raise RuntimeError(f"OneFormer {label} mask has no image payload")
                mask = self._decode_mask(item["mask"], image_shape)
                if mask.any():
                    score = item.get("score")
                    parsed.append(
                        Sam3Mask(
                            label=label,
                            mask=mask,
                            score=None if score is None else float(score),
                            box=item.get("box"),
                        )
                    )
            masks[label] = parsed

        logger.info(
            "OneFormer segmented surfaces: %s",
            ", ".join(f"{label}={len(masks[label])}" for label in SURFACE_LABELS),
        )
        return SurfaceSegmentation(
            masks=masks,
            image_shape=image_shape,
            model=str(payload.get("model", "shi-labs/oneformer_ade20k_swin_large")),
            provider="oneformer",
        )

    @staticmethod
    def _encode_image(image_bgr: np.ndarray) -> str:
        ok, encoded = cv2.imencode(".png", image_bgr)
        if not ok:
            raise ValueError("failed to PNG-encode the OneFormer input image")
        return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode(
            "ascii"
        )

    @staticmethod
    def _decode_mask(value: object, image_shape: tuple[int, int]) -> np.ndarray:
        if not isinstance(value, str) or not value:
            raise RuntimeError("OneFormer mask payload is not a non-empty string")
        encoded = value.split(",", 1)[1] if value.startswith("data:") else value
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise RuntimeError("OneFormer mask is not valid base64") from exc
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError("failed to decode a OneFormer mask image")
        if image.ndim == 3:
            image = image[:, :, 0]
        mask = image > 127
        if mask.shape != image_shape:
            mask = (
                cv2.resize(
                    mask.astype(np.uint8),
                    (image_shape[1], image_shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                > 0
            )
        return mask
