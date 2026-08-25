"""MoGe-2 inference and calibrated point-map preparation tasks."""

from __future__ import annotations

from ...geometry3d import resize_mask
from ...moge_client import MogeClient
from ..models import PipelineContext
from .base import PipelineStage, StageStatus


class MogeStage(PipelineStage):
    name = "moge_inference"
    dependencies = ("resize",)
    background = True

    def run(self, context: PipelineContext) -> StageStatus:
        if not context.settings.moge_enabled:
            print("[moge] skipped (disabled)")
            return StageStatus.SKIPPED
        if context.image_bgr is None:
            raise RuntimeError("resize must finish before MoGe inference")

        mode = "debug" if context.settings.debug else "production"
        print(f"[moge] calling MoGe-2 API ({mode})...")
        context.moge = MogeClient(
            context.settings.moge_endpoint, context.settings.moge_timeout
        ).predict(
            context.image_bgr,
            include_mesh=context.settings.debug,
            include_debug=context.settings.debug,
        )
        pm_w, pm_h = context.moge.image_size
        print(
            f"[moge] point map {pm_w}x{pm_h} "
            f"(working image {context.image_bgr.shape[1]}x{context.image_bgr.shape[0]})"
        )
        return StageStatus.COMPLETED


class PointMapStage(PipelineStage):
    """Prepare the native point map for object-dimension calibration."""

    name = "prepare_point_map"
    dependencies = ("moge_inference", "object_dimensions")
    background = True

    def run(self, context: PipelineContext) -> StageStatus:
        if not context.settings.moge_enabled:
            print("[point-map] skipped (moge disabled)")
            return StageStatus.SKIPPED
        if (
            context.image_bgr is None
            or context.selection is None
            or context.old_object_dimensions_m is None
            or context.moge is None
        ):
            raise RuntimeError("MoGe, selection, and old-object dimensions are required")

        pm_w, pm_h = context.moge.image_size
        context.mask_pm = resize_mask(context.selection.mask, pm_w, pm_h)
        context.scale_correction = 1.0
        context.calibration = {
            "applied": False,
            "reason": "reference line removed; geometry remains in native MoGe units",
            "coordinate_scale": "native_moge_units",
        }
        context.point_map = context.moge.point_map.copy()
        print("[point-map] using native MoGe coordinates (no reference line)")
        return StageStatus.COMPLETED
