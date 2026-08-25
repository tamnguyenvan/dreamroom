"""Heuristic placement orientation and target-box construction."""

from __future__ import annotations

import numpy as np

from ...geometry3d import (
    calculate_aspect_ratio_calibration,
    target_dimensions_in_moge_units,
)
from ...placement_geometry import build_target_box, infer_placement_orientation
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

        if context.old_object_dimensions_m is not None:
            if orientation.primary_rear_face is None:
                depth_axis = 1
            else:
                axis_name = orientation.primary_rear_face.split("_", 1)[0]
                if axis_name not in {"axis0", "axis1"}:
                    raise RuntimeError(
                        f"unsupported placement face: {orientation.primary_rear_face}"
                    )
                depth_axis = int(axis_name[-1])
            width_axis = 1 - depth_axis
            old_moge_dimensions = np.array(
                [
                    context.box.extents[width_axis],
                    context.box.extents[depth_axis],
                    context.box.extents[2],
                ],
                dtype=float,
            )
            _, dimension_calibration = calculate_aspect_ratio_calibration(
                old_moge_dimensions,
                context.old_object_dimensions_m,
            )
            dimension_calibration["moge_axis_mapping"] = {
                "width_axis": width_axis,
                "depth_axis": depth_axis,
                "source": "placement_orientation",
            }
            context.calibration["object_ratio_calibration"] = dimension_calibration
            context.calibration["scene_units_per_meter"] = (
                old_moge_dimensions / np.asarray(context.old_object_dimensions_m)
            ).tolist()
            print(
                "[target-box] old-object semantic dimensions (MoGe units): "
                f"width={old_moge_dimensions[0]:.3f}, "
                f"depth={old_moge_dimensions[1]:.3f}, "
                f"height={old_moge_dimensions[2]:.3f}"
            )
            print(
                "[target-box] old-object ratios: "
                f"actual={dimension_calibration['actual_ratio']:.3f}, "
                f"moge={dimension_calibration['moge_ratio']:.3f}, "
                f"depth correction={dimension_calibration['depth_ratio_factor']:.3f}"
            )

        dimensions = (
            context.settings.target_width_m,
            context.settings.target_depth_m,
            context.settings.target_height_m,
        )
        if any(value is not None for value in dimensions):
            if not all(value is not None for value in dimensions):
                raise ValueError("target width, depth, and height must be provided together")
            requested_dimensions = np.asarray(dimensions, dtype=float)
            dimension_calibration = context.calibration.get("object_ratio_calibration")
            if (
                dimension_calibration is None
                or context.old_object_dimensions_m is None
            ):
                raise RuntimeError(
                    "old-object dimensions and ratio calibration are required "
                    "to construct a target box"
                )
            moge_dimensions, target_dimensions_m = target_dimensions_in_moge_units(
                requested_dimensions,
                old_moge_dimensions,
                context.old_object_dimensions_m,
                dimension_calibration["depth_ratio_factor"],
            )
            dimensions = tuple(moge_dimensions.tolist())
            dimension_calibration["requested_target_dimensions_m"] = (
                requested_dimensions.tolist()
            )
            dimension_calibration["calibrated_target_dimensions_m"] = (
                target_dimensions_m.tolist()
            )
            dimension_calibration["target_box_extents_moge"] = list(dimensions)
            print(
                "[target-box] target dimensions: "
                f"{target_dimensions_m[0]:.2f} x {target_dimensions_m[1]:.2f} x "
                f"{target_dimensions_m[2]:.2f} m; "
                f"native extents {dimensions[0]:.2f} x {dimensions[1]:.2f} x "
                f"{dimensions[2]:.2f}"
            )
            scene_units_per_meter = context.calibration.get(
                "scene_units_per_meter", [1.0, 1.0, 1.0]
            )
            wall_snap_distance = context.settings.wall_snap_distance_m * float(
                scene_units_per_meter[1]
            )
            context.target_placement = build_target_box(
                context.box,
                context.floor,
                context.walls,
                orientation,
                dimensions[0],
                dimensions[1],
                dimensions[2],
                wall_snap_distance=wall_snap_distance,
            )
            if context.target_placement is not None:
                extents = context.target_placement.box.extents
                print(
                    f"[target-box] target box (MoGe units): {extents[0]:.2f} x "
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
