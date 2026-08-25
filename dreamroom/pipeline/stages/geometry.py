"""Floor fitting and floor-aligned object-box fitting."""

from __future__ import annotations

from ...geometry3d import (
    extract_object_points,
    fallback_floor_plane,
    fit_box,
    fit_floor_plane,
    floor_candidate_points,
    segmented_surface_points,
)
from ..models import PipelineContext
from .base import PipelineStage, StageStatus


class GeometryStage(PipelineStage):
    name = "fit_geometry"
    dependencies = ("prepare_point_map", "prepare_surface_masks")
    background = True

    def run(self, context: PipelineContext) -> StageStatus:
        if not context.settings.moge_enabled:
            print("[geometry] skipped (moge disabled)")
            return StageStatus.SKIPPED
        if (
            context.image_bgr is None
            or context.selection is None
            or context.moge is None
            or context.point_map is None
            or context.mask_pm is None
            or context.scale_correction is None
        ):
            raise RuntimeError("point-map preparation must finish first")

        print("[geometry] fitting floor plane and 3D box...")
        object_points = extract_object_points(context.point_map, context.mask_pm)
        surface_provider = (
            context.surface_segmentation.provider
            if context.surface_segmentation is not None
            else "sam3"
        )
        floor_points = None
        context.floor = None
        if (
            context.floor_surface_mask_pm is not None
            and context.rug_surface_mask_pm is not None
        ):
            try:
                floor_and_rug = (
                    context.floor_surface_mask_pm | context.rug_surface_mask_pm
                )
                floor_points = segmented_surface_points(
                    context.point_map,
                    floor_and_rug,
                    context.mask_pm,
                )
                context.floor = fit_floor_plane(floor_points)
                if context.floor is not None:
                    context.floor_fit_method = surface_provider
            except Exception as exc:
                print(f"[geometry] {surface_provider} floor fit failed: {exc}")
            if floor_points is not None:
                print(
                    f"[geometry] {surface_provider} floor/rug point candidates: "
                    f"{len(floor_points)}"
                )

        if context.floor is None:
            manual_points = floor_candidate_points(context.point_map, context.mask_pm)
            context.floor = fit_floor_plane(manual_points)
            if context.floor is not None:
                context.floor_fit_method = "manual"
                print(
                    f"[geometry] {surface_provider} floor fit unavailable; "
                    "using manual bottom-image point-cloud fallback"
                )
            else:
                context.floor = fallback_floor_plane(object_points)
                context.floor_fit_method = "camera_up"
                print("[geometry] floor not found; using fallback camera-up plane")
        context.box = fit_box(object_points, context.floor)
        source = context.floor_fit_method or "unknown"
        print(
            f"[geometry] floor source: {source}, "
            f"semantic candidates: {len(floor_points) if floor_points is not None else 0}, "
            f"box extents (MoGe units): {context.box.extents[0]:.2f} x "
            f"{context.box.extents[1]:.2f} x {context.box.extents[2]:.2f}, "
            "native coordinate scale x1.000"
        )

        return StageStatus.COMPLETED
