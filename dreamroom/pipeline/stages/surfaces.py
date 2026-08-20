"""Step 4: text-prompted floor, rug, and wall segmentation with SAM 3."""

from __future__ import annotations

from collections.abc import Callable

from ...geometry3d import resize_mask
from ...sam3_client import Sam3Client
from ...surface_viz import draw_surface_debug_2d
from ..models import PipelineContext
from .base import PipelineStage, StageStatus


class SurfaceStage(PipelineStage):
    name = "step_4_sam3_surfaces"

    def __init__(self, client_factory: Callable[[PipelineContext], Sam3Client] | None = None):
        self._client_factory = client_factory or self._default_client

    @staticmethod
    def _default_client(context: PipelineContext) -> Sam3Client:
        return Sam3Client(
            model=context.settings.sam3_model,
            timeout=context.settings.sam3_timeout,
            min_score=context.settings.sam3_min_score,
        )

    def run(self, context: PipelineContext) -> StageStatus:
        if not context.settings.moge_enabled:
            print("[step 4] skipped (moge disabled)")
            return StageStatus.SKIPPED
        if context.image_bgr is None or context.moge is None:
            raise RuntimeError("Steps 0-3 must run before SAM 3 segmentation")

        print("[step 4] segmenting walls, floor, and rug with SAM 3...")
        context.surface_segmentation = self._client_factory(context).segment_surfaces(
            context.image_bgr
        )
        pm_w, pm_h = context.moge.image_size
        context.floor_surface_mask_pm = resize_mask(
            context.surface_segmentation.combined_mask("floor"), pm_w, pm_h
        )
        context.rug_surface_mask_pm = resize_mask(
            context.surface_segmentation.combined_mask("rug"), pm_w, pm_h
        )
        context.wall_surface_masks_pm = [
            resize_mask(item.mask, pm_w, pm_h)
            for item in context.surface_segmentation.instances("wall")
        ]
        print(
            f"[step 4] masks: {len(context.wall_surface_masks_pm)} wall, "
            f"{len(context.surface_segmentation.instances('floor'))} floor, "
            f"{len(context.surface_segmentation.instances('rug'))} rug"
        )
        if context.settings.debug:
            context.debug_surfaces_2d = draw_surface_debug_2d(
                context.image_bgr, context.surface_segmentation
            )
        return StageStatus.COMPLETED
