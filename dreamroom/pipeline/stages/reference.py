"""Interactive old-object dimension input."""

from __future__ import annotations

from ...ui import prompt_object_dimensions
from ..models import PipelineContext
from .base import PipelineStage, StageStatus


class ReferenceStage(PipelineStage):
    name = "object_dimensions"
    dependencies = ("object_selection",)

    def run(self, context: PipelineContext) -> StageStatus:
        if context.image_bgr is None or context.selection is None:
            raise RuntimeError("object selection must finish before dimension input")
        if not context.settings.moge_enabled:
            return StageStatus.COMPLETED
        print("[dimensions] enter the old object's width, depth, and height in meters")
        context.old_object_dimensions_m = prompt_object_dimensions()
        if context.old_object_dimensions_m is None:
            print("[dimensions] aborted by user")
            return StageStatus.ABORTED
        print(
            "[dimensions] old object: "
            f"{context.old_object_dimensions_m[0]:g} x "
            f"{context.old_object_dimensions_m[1]:g} x "
            f"{context.old_object_dimensions_m[2]:g} m"
        )
        return StageStatus.COMPLETED
