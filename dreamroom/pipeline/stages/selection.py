"""Interactive object selection and segmentation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ...image_ops import bgr_to_rgb
from ...ui import select_object
from ..models import PipelineContext
from .base import PipelineStage, StageStatus


SegmentFn = Callable[[list[list[int]], list[list[int]]], np.ndarray]


class SelectionStage(PipelineStage):
    name = "object_selection"
    dependencies = ("resize",)

    def __init__(self, segmenter_factory: Callable[[np.ndarray], SegmentFn]) -> None:
        self._segmenter_factory = segmenter_factory

    def run(self, context: PipelineContext) -> StageStatus:
        if context.image_bgr is None:
            raise RuntimeError("resize must finish before object selection")
        image_rgb = bgr_to_rgb(context.image_bgr)
        segment_fn = self._segmenter_factory(image_rgb)

        print("[object] draw strokes on the object, then press Enter to segment")
        context.selection = select_object(
            context.image_bgr,
            segment_fn,
            max_points=context.settings.max_points,
            max_display_width=context.settings.max_display_width,
        )
        if context.selection is None:
            print("[object] aborted by user")
            return StageStatus.ABORTED
        print(f"[object] confirmed: {int(context.selection.mask.sum())} px")
        return StageStatus.COMPLETED
