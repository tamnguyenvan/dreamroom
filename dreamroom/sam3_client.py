"""fal.ai SAM 3 client for text-prompted room-surface segmentation."""

from __future__ import annotations

import base64
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "fal-ai/sam-3/image"
SURFACE_PROMPTS = ("wall", "floor", "rug")


@dataclass
class Sam3Mask:
    """One binary SAM 3 mask and its optional model metadata."""

    label: str
    mask: np.ndarray
    score: float | None = None
    box: list[float] | None = None

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "score": None if self.score is None else round(float(self.score), 4),
            "box": self.box,
            "area_px": int(self.mask.sum()),
        }


@dataclass
class SurfaceSegmentation:
    """Provider-generated masks grouped by room-surface label."""

    masks: dict[str, list[Sam3Mask]]
    image_shape: tuple[int, int]
    model: str
    provider: str = "sam3"

    def instances(self, label: str) -> list[Sam3Mask]:
        return self.masks.get(label, [])

    def combined_mask(self, label: str) -> np.ndarray:
        combined = np.zeros(self.image_shape, dtype=bool)
        for item in self.instances(label):
            combined |= item.mask
        return combined

    def floor_and_rug_mask(self) -> np.ndarray:
        return self.combined_mask("floor") | self.combined_mask("rug")

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "image_size_hw": list(self.image_shape),
            "prompts": {
                label: [item.to_dict() for item in self.instances(label)]
                for label in SURFACE_PROMPTS
            },
        }


class Sam3Client:
    """Upload one image and segment room surfaces with three text prompts."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        timeout: float = 300.0,
        min_score: float = 0.25,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.min_score = min_score

    def segment_surfaces(self, image_bgr: np.ndarray) -> SurfaceSegmentation:
        """Return wall, floor, and rug masks at the input-image resolution."""

        fal_client = self._load_fal_client()
        image_url = self._upload_image(fal_client, image_bgr)
        with ThreadPoolExecutor(max_workers=len(SURFACE_PROMPTS)) as executor:
            futures = {
                label: executor.submit(self._segment_prompt, fal_client, image_url, label)
                for label in SURFACE_PROMPTS
            }
            responses = {label: future.result() for label, future in futures.items()}

        image_shape = image_bgr.shape[:2]
        masks = {
            label: self._parse_masks(label, response, image_shape)
            for label, response in responses.items()
        }
        logger.info(
            "SAM 3 segmented surfaces: %s",
            ", ".join(f"{label}={len(items)}" for label, items in masks.items()),
        )
        return SurfaceSegmentation(masks=masks, image_shape=image_shape, model=self.model)

    @staticmethod
    def _load_fal_client():
        if not os.getenv("FAL_KEY"):
            raise RuntimeError("FAL_KEY is required for SAM 3 surface segmentation")
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
            raise ValueError("failed to PNG-encode the SAM 3 input image")
        path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                handle.write(encoded.tobytes())
                path = Path(handle.name)
            return str(fal_client.upload_file(path))
        finally:
            if path is not None:
                path.unlink(missing_ok=True)

    def _segment_prompt(self, fal_client, image_url: str, label: str) -> dict:
        max_masks = 12 if label == "wall" else 6
        return fal_client.subscribe(
            self.model,
            arguments={
                "image_url": image_url,
                "prompt": label,
                "apply_mask": False,
                "output_format": "png",
                "return_multiple_masks": True,
                "max_masks": max_masks,
                "include_scores": True,
                "include_boxes": True,
            },
            client_timeout=self.timeout,
        )

    def _parse_masks(
        self,
        label: str,
        response: dict,
        image_shape: tuple[int, int],
    ) -> list[Sam3Mask]:
        media = response.get("masks") or []
        metadata = response.get("metadata") or []
        scores = response.get("scores") or []
        boxes = response.get("boxes") or []
        parsed: list[Sam3Mask] = []
        for index, item in enumerate(media):
            meta = metadata[index] if index < len(metadata) else {}
            score = meta.get("score")
            if score is None and index < len(scores):
                score = scores[index]
            if score is not None and float(score) < self.min_score:
                continue
            box = meta.get("box")
            if box is None and index < len(boxes):
                box = boxes[index]
            url = item.get("url") if isinstance(item, dict) else None
            if not url:
                raise RuntimeError(f"SAM 3 {label} mask {index} has no URL")
            mask = self._download_mask(str(url), image_shape)
            if mask.any():
                parsed.append(
                    Sam3Mask(
                        label=label,
                        mask=mask,
                        score=None if score is None else float(score),
                        box=None if box is None else [float(value) for value in box],
                    )
                )
        return parsed

    def _download_mask(self, url: str, image_shape: tuple[int, int]) -> np.ndarray:
        data = self._read_url(url)
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError("failed to decode a SAM 3 mask image")
        mask = self._binary_mask(image)
        if mask.shape != image_shape:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (image_shape[1], image_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        return mask

    def _read_url(self, url: str) -> bytes:
        if url.startswith("data:"):
            try:
                return base64.b64decode(url.split(",", 1)[1])
            except (IndexError, ValueError) as exc:
                raise RuntimeError("invalid SAM 3 mask data URI") from exc
        response = requests.get(url, timeout=self.timeout)
        if response.status_code != 200:
            raise RuntimeError(
                f"SAM 3 mask download HTTP {response.status_code}: {response.text[:200]}"
            )
        return response.content

    @staticmethod
    def _binary_mask(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image > 127
        if image.shape[2] == 4:
            alpha = image[:, :, 3]
            if alpha.min() != alpha.max():
                return alpha > 127
            image = image[:, :, :3]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray > 127
