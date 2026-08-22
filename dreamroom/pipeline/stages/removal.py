"""Remove the selected object before downstream furniture rendering."""

from __future__ import annotations

from collections.abc import Callable

from ...gemini_client import GeminiClient, GeminiEditResult
from ...image_ops import decode_image_bgr
from ...object_removal import (
    SquareObjectCrop,
    annotate_selected_object,
    crop_selected_object,
    resize_patch,
    stitch_patch,
)
from ..models import PipelineContext
from .base import PipelineStage, StageStatus

OBJECT_REMOVAL_PROMPT = (
    "Remove the selected object and keep everything else in the room unchanged."
)


class RemovalStage(PipelineStage):
    """Remove the confirmed selection while independent pipeline work runs."""

    name = "remove_selected_object"
    dependencies = ("object_selection",)
    background = True

    def __init__(
        self,
        client_factory: Callable[[PipelineContext], GeminiClient] | None = None,
    ) -> None:
        self._client_factory = client_factory or self._default_client

    @staticmethod
    def _default_client(context: PipelineContext) -> GeminiClient:
        return GeminiClient(
            model=context.settings.gemini_model,
            timeout=context.settings.gemini_timeout,
        )

    def run(self, context: PipelineContext) -> StageStatus:
        if context.settings.furniture_path is None:
            print("[removal] skipped (no --furniture supplied)")
            return StageStatus.SKIPPED
        if not context.settings.moge_enabled:
            print("[removal] skipped (moge disabled)")
            return StageStatus.SKIPPED
        if context.image_bgr is None or context.selection is None:
            raise RuntimeError("image resize and object selection must finish before removal")

        object_crop = crop_selected_object(context.image_bgr, context.selection.mask)
        crop_mask = context.selection.mask[
            object_crop.y : object_crop.y + object_crop.size,
            object_crop.x : object_crop.x + object_crop.size,
        ]
        removal_input = annotate_selected_object(object_crop.image_bgr, crop_mask)
        print("[removal] calling Gemini Nano Banana Lite...")
        result = self._client_factory(context).remove_object(
            removal_input,
            OBJECT_REMOVAL_PROMPT,
        )
        removed_patch = resize_patch(
            decode_image_bgr(result.image_bytes),
            object_crop.size,
        )
        context.object_removal_image = stitch_patch(
            context.image_bgr,
            removed_patch,
            object_crop,
        )
        context.object_removal_metadata = self._metadata(
            result,
            object_crop,
            removed_patch.shape[:2],
            context.image_bgr.shape[:2],
        )
        if context.settings.debug:
            context.object_removal_crop = removal_input
            context.object_removal_patch = removed_patch
        print(f"[removal] completed in {result.elapsed_seconds:.1f}s")
        return StageStatus.COMPLETED

    @staticmethod
    def _metadata(
        result: GeminiEditResult,
        object_crop: SquareObjectCrop,
        patch_size_hw: tuple[int, int],
        source_size_hw: tuple[int, int],
    ) -> dict:
        return {
            **result.to_dict(),
            "prompt": OBJECT_REMOVAL_PROMPT,
            "crop": object_crop.to_dict(),
            "source_image_size_hw": list(source_size_hw),
            "edited_patch_input_size_hw": list(object_crop.image_bgr.shape[:2]),
            "edited_patch_output_size_hw": list(patch_size_hw),
            "stitched_image_size_hw": list(source_size_hw),
        }
