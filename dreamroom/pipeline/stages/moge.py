"""Step 3: MoGe-2 inference and calibrated point-map preparation."""

from __future__ import annotations

from ...geometry3d import calibrate_scale, resize_mask
from ...moge_client import MogeClient
from ..models import PipelineContext
from .base import PipelineStage, StageStatus


class MogeStage(PipelineStage):
    name = "step_3_moge"

    def run(self, context: PipelineContext) -> StageStatus:
        if not context.settings.moge_enabled:
            print("[step 3] skipped (moge disabled)")
            return StageStatus.SKIPPED
        if context.image_bgr is None or context.selection is None or context.reference is None:
            raise RuntimeError("Steps 0-2 must run before MoGe inference")

        print("[step 3] calling MoGe-2 API (production)...")
        context.moge = MogeClient(
            context.settings.moge_endpoint, context.settings.moge_timeout
        ).predict(
            context.image_bgr,
            include_mesh=context.settings.debug,
            include_debug=context.settings.debug,
        )
        pm_w, pm_h = context.moge.image_size
        sx = pm_w / context.image_bgr.shape[1]
        sy = pm_h / context.image_bgr.shape[0]
        print(
            f"[step 3] point map {pm_w}x{pm_h} "
            f"(working image {context.image_bgr.shape[1]}x{context.image_bgr.shape[0]})"
        )

        context.mask_pm = resize_mask(context.selection.mask, pm_w, pm_h)
        context.scale_correction, context.calibration = calibrate_scale(
            context.moge.point_map,
            [context.reference.start[0] * sx, context.reference.start[1] * sy],
            [context.reference.end[0] * sx, context.reference.end[1] * sy],
            context.reference.meters,
        )
        context.point_map = context.moge.point_map * context.scale_correction
        return StageStatus.COMPLETED
