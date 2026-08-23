"""Heuristic placement orientation and target-box construction."""

from __future__ import annotations

from ...placement_geometry import (
    apply_target_depth_correction,
    build_target_box,
    calculate_view_angle_depth_correction,
    infer_placement_orientation,
)
from ...placement_viz import (
    draw_placement_debug_2d,
    export_placement_debug_glb,
)
from ...viz3d import intrinsics_px
from ..models import PipelineContext
from .base import PipelineStage, StageStatus


class PlacementStage(PipelineStage):
    name = "target_box"
    dependencies = ("fit_walls",)
    background = True

    def run(self, context: PipelineContext) -> StageStatus:
        if not context.settings.moge_enabled:
            print("[target-box] skipped (moge disabled)")
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
            raise RuntimeError("geometry and wall fitting must finish before placement")

        print("[target-box] inferring geometry-only placement orientation...")
        context.placement_orientation = infer_placement_orientation(
            context.box,
            context.floor,
            context.walls,
            context.point_map,
            context.mask_pm,
        )
        orientation = context.placement_orientation
        print(
            f"[target-box] {orientation.mode}: rear "
            f"{orientation.primary_rear_face or 'unknown'}, "
            f"confidence {orientation.confidence:.2f}"
        )

        context.depth_correction = calculate_view_angle_depth_correction(
            context.box,
            orientation,
        )
        if context.depth_correction.get("applied"):
            print(
                "[target-box] depth correction: "
                f"view {context.depth_correction['view_angle_degrees']:.1f}°, "
                f"scale x{context.depth_correction['factor']:.3f}, "
                f"{context.depth_correction['old_depth']:.2f}m -> "
                f"{context.depth_correction['new_depth']:.2f}m"
            )

        dimensions = (
            context.settings.target_width_m,
            context.settings.target_depth_m,
            context.settings.target_height_m,
        )
        if any(value is not None for value in dimensions):
            if not all(value is not None for value in dimensions):
                raise ValueError("target width, depth, and height must be provided together")
            context.target_placement = build_target_box(
                context.box,
                context.floor,
                context.walls,
                orientation,
                dimensions[0],
                dimensions[1],
                dimensions[2],
                wall_snap_distance=context.settings.wall_snap_distance_m,
            )
            if context.target_placement is not None:
                if context.depth_correction.get("applied"):
                    context.depth_correction["target_depth_requested"] = dimensions[1]
                    context.target_placement = apply_target_depth_correction(
                        context.target_placement,
                        context.depth_correction["factor"],
                    )
                    context.depth_correction["target_depth_applied"] = round(
                        context.target_placement.box.extents[1], 4
                    )
                extents = context.target_placement.box.extents
                print(
                    f"[target-box] target box (m): {extents[0]:.2f} x "
                    f"{extents[1]:.2f} x {extents[2]:.2f} "
                    f"[{context.target_placement.anchor_mode}]"
                )

        if context.settings.debug:
            pm_w, pm_h = context.moge.image_size
            k_px = intrinsics_px(context.moge.metadata, pm_w, pm_h)
            context.debug_placement_2d = draw_placement_debug_2d(
                context.image_bgr,
                context.box,
                orientation,
                context.walls,
                k_px,
                (
                    context.image_bgr.shape[1] / pm_w,
                    context.image_bgr.shape[0] / pm_h,
                ),
                context.target_placement,
                context.selection.mask,
            )
            context.debug_placement_3d = export_placement_debug_glb(
                context.moge.glb_bytes,
                context.box,
                orientation,
                context.walls,
                context.scale_correction,
                context.target_placement,
            )
        return StageStatus.COMPLETED
