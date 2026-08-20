"""Load and resize the input image."""

from __future__ import annotations

from ..models import PipelineContext
from .base import PipelineStage, StageStatus
from ...image_ops import load_image_bgr, resize_max_side


class ResizeStage(PipelineStage):
    name = "resize"

    def run(self, context: PipelineContext) -> StageStatus:
        original = load_image_bgr(context.image_path)
        context.original_size = original.shape[:2]
        context.image_bgr, context.resize_scale = resize_max_side(
            original, context.settings.max_side
        )
        print(
            f"[resize] {original.shape[1]}x{original.shape[0]} -> "
            f"{context.image_bgr.shape[1]}x{context.image_bgr.shape[0]} "
            f"(max side {context.settings.max_side})"
        )
        return StageStatus.COMPLETED
