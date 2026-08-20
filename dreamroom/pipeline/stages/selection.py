"""Step 1: interactive object selection and segmentation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ...image_ops import bgr_to_rgb
from ...ui import select_object
from ..models import PipelineContext
from .base import PipelineStage, StageStatus


SegmentFn = Callable[[list[list[int]], list[list[int]]], np.ndarray]


class SelectionStage(PipelineStage):
    name = "step_1_segment"

    def __init__(self, segmenter_factory: Callable[[np.ndarray], SegmentFn]) -> None:
        self._segmenter_factory = segmenter_factory

    def run(self, context: PipelineContext) -> StageStatus:
        if context.image_bgr is None:
            raise RuntimeError("Step 0 must run before object selection")
        image_rgb = bgr_to_rgb(context.image_bgr)
        segment_fn = self._segmenter_factory(image_rgb)

        print("[step 1] draw strokes on the object, then press Enter to segment")
        context.selection = select_object(
            context.image_bgr,
            segment_fn,
            max_points=context.settings.max_points,
            max_display_width=context.settings.max_display_width,
        )
        if context.selection is None:
            print("[step 1] aborted by user")
            return StageStatus.ABORTED
        print(f"[step 1] object confirmed: {int(context.selection.mask.sum())} px")
        return StageStatus.COMPLETED
