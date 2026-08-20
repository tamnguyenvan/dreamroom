"""Local SimpleClick interactive segmentation.

Ported from ``modal_app.py`` (simpleclick-modal project): same checkpoint,
same NoBRS predictor configuration, same point sampling. Runs on GPU when
available, otherwise on CPU. The image features are cached, so repeated
segmentations with new strokes on the same image are much faster than the
first one.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import sys
import time
from pathlib import Path

import numpy as np

from .config import Settings

logger = logging.getLogger(__name__)


def _load_checkpoint(path: Path):
    """Load the checkpoint with the lowest peak memory possible.

    ``mmap=True`` keeps the state dict on disk-backed pages, so loading the
    model peaks at roughly the model size instead of state-dict + model.
    """

    import torch

    try:
        return torch.load(str(path), map_location="cpu", mmap=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        logger.warning("mmap checkpoint load failed (%s); using standard load", exc)
        return torch.load(path, map_location="cpu")


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
    """Keeps the SimpleClick model loaded and segments from click points."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._predictor = None
        self._device = None
        self._image_key: str | None = None

    def load(self) -> None:
        """Load the checkpoint and build the predictor (idempotent)."""

        if self._predictor is not None:
            return
        if not self.settings.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"checkpoint missing: {self.settings.checkpoint_path} "
                "(run scripts/setup_simpleclick.sh)"
            )
        root = str(self.settings.simpleclick_root)
        if root not in sys.path:
            sys.path.insert(0, root)

        import torch
        from isegm.inference import utils
        from isegm.inference.predictors import get_predictor

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            "loading SimpleClick checkpoint %s on %s",
            self.settings.checkpoint_path,
            self._device,
        )
        tic = time.time()
        state_dict = _load_checkpoint(self.settings.checkpoint_path)
        model = utils.load_is_model(
            state_dict,
            self._device,
            eval_ritm=False,
            cpu_dist_maps=True,
        )
        del state_dict
        gc.collect()
        input_size = self.settings.model_input_size
        self._predictor = get_predictor(
            model,
            "NoBRS",
            self._device,
            prob_thresh=self.settings.threshold,
            with_flip=self.settings.with_flip,
            zoom_in_params={
                "skip_clicks": -1,
                "target_size": (input_size, input_size),
                "expansion_ratio": self.settings.zoom_in_expansion,
            },
            predictor_params={"max_size": self.settings.max_longest_size},
        )
        logger.info("SimpleClick ready in %.1fs", time.time() - tic)

    def segment(
        self,
        image_rgb: np.ndarray,
        positive_points: list[list[int]],
        negative_points: list[list[int]] | None = None,
        threshold: float | None = None,
    ) -> np.ndarray:
        """Segment ``image_rgb`` from positive/negative ``[x, y]`` points.

        Returns a boolean mask with the image dimensions.
        """

        self.load()
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb must be an HxWx3 RGB array")
        height, width = image_rgb.shape[:2]
        threshold = self.settings.threshold if threshold is None else float(threshold)
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1")

        positive = _validate_points(positive_points, width, height, "positive_points", True)
        negative = _validate_points(negative_points or [], width, height, "negative_points", False)
        positive = sample_points(positive, self.settings.max_points)
        negative = sample_points(negative, self.settings.max_points)

        self._set_image(image_rgb)

        import torch
        from isegm.inference.clicker import Click, Clicker

        clicker = Clicker(
            init_clicks=[
                *[Click(is_positive=True, coords=(p[1], p[0])) for p in positive],
                *[Click(is_positive=False, coords=(p[1], p[0])) for p in negative],
            ]
        )
        tic = time.time()
        with torch.inference_mode():
            probabilities = self._predictor.get_prediction(clicker)
        logger.info(
            "segmented with %d positive / %d negative points in %.1fs",
            len(positive),
            len(negative),
            time.time() - tic,
        )
        return probabilities > threshold

    def _set_image(self, image_rgb: np.ndarray) -> None:
        """Run the encoder only when the input image actually changed."""

        key = f"{image_rgb.shape}:{hashlib.md5(image_rgb.tobytes()).hexdigest()}"
        if key == self._image_key:
            return
        import torch

        tic = time.time()
        with torch.inference_mode():
            self._predictor.set_input_image(np.ascontiguousarray(image_rgb))
        logger.info("image features computed in %.1fs", time.time() - tic)
        self._image_key = key
