"""Furniture preprocessing and Seedream 5.0 Pro rendering tasks."""

from __future__ import annotations

from collections.abc import Callable

from ...image_ops import load_image_bgr, resize_max_side
from ...render_viz import draw_target_box_2d
from ...seedream_client import SeedreamClient
from ...viz3d import intrinsics_px
from ..models import PipelineContext
from .base import PipelineStage, StageStatus


class FurnitureStage(PipelineStage):
    """Load and resize the furniture reference while geometry is computed."""

    name = "prepare_furniture"
    dependencies = ("resize",)
    background = True

    def run(self, context: PipelineContext) -> StageStatus:
        if not context.settings.moge_enabled:
            print("[furniture] skipped (moge disabled)")
            return StageStatus.SKIPPED
        if context.settings.furniture_path is None:
            print("[furniture] skipped (no --furniture supplied)")
            return StageStatus.SKIPPED
        furniture = load_image_bgr(context.settings.furniture_path)
        context.render_furniture, _ = resize_max_side(furniture, max_side=512)
        height, width = context.render_furniture.shape[:2]
        print(f"[furniture] prepared reference at {width}x{height}")
        return StageStatus.COMPLETED


class RenderStage(PipelineStage):
    name = "render_furniture"
    dependencies = ("target_box", "prepare_furniture", "remove_selected_object")
    background = True

    def __init__(
        self,
        client_factory: Callable[[PipelineContext], SeedreamClient] | None = None,
    ) -> None:
        self._client_factory = client_factory or self._default_client

    @staticmethod
    def _default_client(context: PipelineContext) -> SeedreamClient:
        return SeedreamClient(
            endpoint=context.settings.seedream_endpoint,
            model=context.settings.seedream_model,
            timeout=context.settings.seedream_timeout,
        )

    def run(self, context: PipelineContext) -> StageStatus:
        if context.settings.furniture_path is None:
            print("[render] skipped (no --furniture supplied)")
            return StageStatus.SKIPPED
        if not context.settings.moge_enabled:
            print("[render] skipped (moge disabled)")
            return StageStatus.SKIPPED
        if (
            context.image_bgr is None
            or context.moge is None
            or context.target_placement is None
            or context.render_furniture is None
            or context.object_removal_image is None
        ):
            raise RuntimeError(
                "object removal, geometry, and furniture preparation must finish before rendering; "
                "provide --new-dimensions WIDTH DEPTH HEIGHT"
            )

        pm_w, pm_h = context.moge.image_size
        room_with_box = draw_target_box_2d(
            context.object_removal_image,
            context.target_placement,
            intrinsics_px(context.moge.metadata, pm_w, pm_h),
            (
                context.image_bgr.shape[1] / pm_w,
                context.image_bgr.shape[0] / pm_h,
            ),
        )
        dimension_calibration = context.calibration.get("object_ratio_calibration", {})
        target_dimensions_m = dimension_calibration.get(
            "calibrated_target_dimensions_m",
            context.target_placement.box.extents.tolist(),
        )
        prompt = (
            "Replace the selected old furniture in Image 1 with the furniture "
            "shown in Image 2. Image 1 is the room photo with the old furniture "
            "removed and a red wireframe target box marking the exact placement "
            "region. Match "
            f"the target box dimensions of {target_dimensions_m[0]:.2f} m width, "
            f"{target_dimensions_m[1]:.2f} m depth, and "
            f"{target_dimensions_m[2]:.2f} m height. "
            "Place the new furniture on the floor inside that box, matching "
            "its perspective, orientation, scale, lighting, shadows, and room "
            "geometry. Preserve the walls, floor, windows, camera viewpoint, "
            "and every non-target object. Remove the red guide box in the final "
            "photorealistic image. Do not add text, labels, or extra furniture."
        )

        print("[render] calling Seedream 5.0 Pro (fast, 1K)...")
        result = self._client_factory(context).generate(
            room_with_box,
            context.render_furniture,
            prompt,
        )
        context.render_room = room_with_box
        context.rendered_image = result.image_bytes
        context.render_metadata = {
            **result.to_dict(),
            "prompt": prompt,
            "object_removal": context.object_removal_metadata,
            "furniture_source": str(context.settings.furniture_path.resolve()),
            "room_input_size_hw": list(room_with_box.shape[:2]),
            "furniture_input_size_hw": list(context.render_furniture.shape[:2]),
            "input_mode": "two_images",
        }
        print(f"[render] completed in {result.elapsed_seconds:.1f}s")
        return StageStatus.COMPLETED
