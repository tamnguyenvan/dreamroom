"""Room-surface wall fitting and final geometry visualization."""

from __future__ import annotations

from ...viz3d import draw_debug_2d, export_debug_glb, intrinsics_px
from ...wall_geometry import fit_segmented_wall_planes, fit_wall_planes
from ..models import PipelineContext
from .base import PipelineStage, StageStatus


class WallStage(PipelineStage):
    name = "fit_walls"
    dependencies = ("fit_geometry",)
    background = True

    def run(self, context: PipelineContext) -> StageStatus:
        if not context.settings.moge_enabled:
            print("[walls] skipped (moge disabled)")
            return StageStatus.SKIPPED
        if (
            context.image_bgr is None
            or context.selection is None
            or context.moge is None
            or context.point_map is None
            or context.mask_pm is None
            or context.scale_correction is None
            or context.floor is None
            or context.box is None
        ):
            raise RuntimeError("geometry fitting must finish before wall fitting")

        context.walls = []
        context.wall_fit_method = None
        surface_provider = (
            context.surface_segmentation.provider
            if context.surface_segmentation is not None
            else "sam3"
        )
        if context.surface_segmentation is not None:
            candidate_counts = [
                int(mask.sum()) for mask in context.wall_surface_masks_pm
            ]
            print(
                f"[walls] fitting planes from {surface_provider}-selected pixels "
                f"({candidate_counts} points per mask)..."
            )
            try:
                context.walls = fit_segmented_wall_planes(
                    context.point_map,
                    context.wall_surface_masks_pm,
                    context.mask_pm,
                    context.floor,
                )
            except Exception as exc:
                print(f"[walls] {surface_provider} wall fit failed: {exc}")
            if context.walls:
                context.wall_fit_method = surface_provider

        if not context.walls:
            print(
                f"[walls] {surface_provider} wall fit unavailable; "
                "using manual global point-cloud fallback"
            )
            context.walls = fit_wall_planes(
                context.point_map,
                context.mask_pm,
                context.floor,
            )
            context.wall_fit_method = "manual"

        print(f"[walls] detected {len(context.walls)} wall plane(s)")
        for index, wall in enumerate(context.walls, start=1):
            print(
                f"  wall {index}: {wall.width:.2f} x {wall.height:.2f} MoGe units, "
                f"{wall.inlier_count} inliers, confidence {wall.confidence:.2f}"
            )

        pm_w, pm_h = context.moge.image_size
        k_px = intrinsics_px(context.moge.metadata, pm_w, pm_h)
        debug_walls = context.walls if context.settings.debug else None
        context.debug_2d = draw_debug_2d(
            context.image_bgr,
            context.box,
            context.floor,
            k_px,
            (context.image_bgr.shape[1] / pm_w, context.image_bgr.shape[0] / pm_h),
            context.selection.mask,
            debug_walls,
        )
        if context.settings.debug:
            context.debug_3d = export_debug_glb(
                context.moge.glb_bytes,
                context.box,
                context.floor,
                context.scale_correction,
                context.walls,
            )
        return StageStatus.COMPLETED
