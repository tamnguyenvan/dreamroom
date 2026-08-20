"""Step 2: interactive reference-scale measurement."""

from __future__ import annotations

from ...ui import get_reference_scale
from ..models import PipelineContext
from .base import PipelineStage, StageStatus


class ReferenceStage(PipelineStage):
    name = "step_2_reference"

    def run(self, context: PipelineContext) -> StageStatus:
        if context.image_bgr is None or context.selection is None:
            raise RuntimeError("Steps 0 and 1 must run before reference measurement")
        print("[step 2] enter the known reference length, then draw the line and press Enter")
        context.reference = get_reference_scale(
            context.image_bgr,
            context.selection.mask,
            max_display_width=context.settings.max_display_width,
        )
        if context.reference is None:
            print("[step 2] aborted by user")
            return StageStatus.ABORTED
        print(
            f"[step 2] reference: {context.reference.pixel_length:.1f} px = "
            f"{context.reference.meters:g} m "
            f"({context.reference.px_per_meter:.2f} px/m)"
        )
        return StageStatus.COMPLETED
