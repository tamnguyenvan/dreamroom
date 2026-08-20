"""Step 4: floor fitting and floor-aligned object-box fitting."""

from __future__ import annotations

from ...geometry3d import (
    extract_object_points,
    fallback_floor_plane,
    fit_box,
    fit_floor_plane,
    floor_candidate_points,
)
from ..models import PipelineContext
from .base import PipelineStage, StageStatus


class GeometryStage(PipelineStage):
    name = "step_4_fit_3d"

    def run(self, context: PipelineContext) -> StageStatus:
        if not context.settings.moge_enabled:
            print("[step 4] skipped (moge disabled)")
            return StageStatus.SKIPPED
        if (
            context.image_bgr is None
            or context.selection is None
            or context.moge is None
            or context.point_map is None
            or context.mask_pm is None
            or context.scale_correction is None
        ):
            raise RuntimeError("Steps 0-3 must run before 3D fitting")

        print("[step 4] fitting floor plane and 3D box...")
        object_points = extract_object_points(context.point_map, context.mask_pm)
        context.floor = fit_floor_plane(
            floor_candidate_points(context.point_map, context.mask_pm)
        )
        if context.floor is None:
            context.floor = fallback_floor_plane(object_points)
            print("[step 4] floor not found; using fallback camera-up plane")
        context.box = fit_box(object_points, context.floor)
        print(
            f"[step 4] box extents (m): {context.box.extents[0]:.2f} x "
            f"{context.box.extents[1]:.2f} x {context.box.extents[2]:.2f}, "
            f"scale correction x{context.calibration.get('factor', 1.0):.3f}"
        )

        return StageStatus.COMPLETED
