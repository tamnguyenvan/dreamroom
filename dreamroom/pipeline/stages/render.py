"""Step 8: render replacement furniture with Seedream 5.0 Pro."""

from __future__ import annotations

from collections.abc import Callable

from ...image_ops import load_image_bgr, resize_max_side
from ...render_viz import draw_target_box_2d
from ...seedream_client import SeedreamClient
from ...viz3d import intrinsics_px
from ..models import PipelineContext
from .base import PipelineStage, StageStatus


class RenderStage(PipelineStage):
    name = "step_8_render"

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
            print("[step 8] skipped (no --furniture supplied)")
            return StageStatus.SKIPPED
        if not context.settings.moge_enabled:
            print("[step 8] skipped (moge disabled)")
            return StageStatus.SKIPPED
        if (
            context.image_bgr is None
            or context.moge is None
            or context.target_placement is None
        ):
            raise RuntimeError(
                "Steps 0-7 must produce a target box before rendering; "
                "provide all three --new-* dimensions"
            )

        furniture = load_image_bgr(context.settings.furniture_path)
        furniture, _ = resize_max_side(furniture, max_side=512)
        pm_w, pm_h = context.moge.image_size
        room_with_box = draw_target_box_2d(
            context.image_bgr,
            context.target_placement,
            intrinsics_px(context.moge.metadata, pm_w, pm_h),
            (
                context.image_bgr.shape[1] / pm_w,
                context.image_bgr.shape[0] / pm_h,
            ),
        )
        extents = context.target_placement.box.extents
        prompt = (
            "Replace the selected old furniture in Image 1 with the furniture "
            "shown in Image 2. Image 1 is the original room photo with a red "
            "wireframe target box marking the exact placement region. Match "
            f"the target box dimensions of {extents[0]:.2f} m width, "
            f"{extents[1]:.2f} m depth, and {extents[2]:.2f} m height. "
            "Place the new furniture on the floor inside that box, matching "
            "its perspective, orientation, scale, lighting, shadows, and room "
            "geometry. Preserve the walls, floor, windows, camera viewpoint, "
            "and every non-target object. Remove the red guide box in the final "
            "photorealistic image. Do not add text, labels, or extra furniture."
        )

        print("[step 8] calling Seedream 5.0 Pro (fast, 1K)...")
        result = self._client_factory(context).generate(
            room_with_box,
            furniture,
            prompt,
        )
        context.render_room = room_with_box
        context.render_furniture = furniture
        context.rendered_image = result.image_bytes
        context.render_metadata = {
            **result.to_dict(),
            "prompt": prompt,
            "furniture_source": str(context.settings.furniture_path.resolve()),
            "room_input_size_hw": list(room_with_box.shape[:2]),
            "furniture_input_size_hw": list(furniture.shape[:2]),
            "input_mode": "two_images",
        }
        print(f"[step 8] rendered furniture in {result.elapsed_seconds:.1f}s")
        return StageStatus.COMPLETED
