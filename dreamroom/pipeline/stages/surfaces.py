"""Independent room-surface inference and point-map mask preparation tasks."""

from __future__ import annotations

from collections.abc import Callable

from ...geometry3d import resize_mask
from ...oneformer_client import OneFormerClient
from ...sam3_client import Sam3Client
from ...surface_viz import draw_surface_debug_2d
from ..models import PipelineContext
from .base import PipelineStage, StageStatus


class SurfaceStage(PipelineStage):
    name = "surface_segmentation"
    dependencies = ("resize",)
    background = True

    def __init__(
        self,
        client_factory: Callable[[PipelineContext], Sam3Client] | None = None,
        *,
        oneformer_factory: Callable[
            [PipelineContext], OneFormerClient | None
        ] | None = None,
        sam3_factory: Callable[[PipelineContext], Sam3Client] | None = None,
    ):
        # ``client_factory`` remains the positional SAM3 test/compatibility hook.
        self._sam3_client_factory = (
            sam3_factory or client_factory or self._default_sam3_client
        )
        self._oneformer_client_factory = (
            oneformer_factory or self._default_oneformer_client
        )

    @staticmethod
    def _default_sam3_client(context: PipelineContext) -> Sam3Client:
        return Sam3Client(
            model=context.settings.sam3_model,
            timeout=context.settings.sam3_timeout,
            min_score=context.settings.sam3_min_score,
        )

    @staticmethod
    def _default_oneformer_client(
        context: PipelineContext,
    ) -> OneFormerClient | None:
        if not context.settings.oneformer_endpoint:
            return None
        return OneFormerClient(
            context.settings.oneformer_endpoint,
            timeout=context.settings.oneformer_timeout,
        )

    @staticmethod
    def _usable(segmentation) -> bool:
        return bool(
            segmentation is not None
            and segmentation.instances("wall")
            and segmentation.instances("floor")
        )

    @staticmethod
    def _print_summary(provider: str, segmentation) -> None:
        print(
            f"[surfaces] {provider} masks: "
            f"{len(segmentation.instances('wall'))} wall, "
            f"{len(segmentation.instances('floor'))} floor, "
            f"{len(segmentation.instances('rug'))} rug"
        )

    def run(self, context: PipelineContext) -> StageStatus:
        if not context.settings.moge_enabled:
            print("[surfaces] skipped (moge disabled)")
            return StageStatus.SKIPPED
        if context.image_bgr is None:
            raise RuntimeError("resize must finish before surface segmentation")

        context.surface_segmentation = None
        if context.settings.oneformer_endpoint:
            print("[surfaces] segmenting walls, floor, and rug with OneFormer...")
            try:
                client = self._oneformer_client_factory(context)
                candidate = (
                    None
                    if client is None
                    else client.segment_surfaces(context.image_bgr)
                )
                if self._usable(candidate):
                    context.surface_segmentation = candidate
                    self._print_summary("OneFormer", candidate)
                    return StageStatus.COMPLETED
                print(
                    "[surfaces] OneFormer did not produce both wall and floor masks; "
                    "trying SAM3"
                )
            except Exception as exc:
                print(f"[surfaces] OneFormer unavailable ({exc}); trying SAM3")

        print("[surfaces] segmenting walls, floor, and rug with SAM3...")
        try:
            context.surface_segmentation = self._sam3_client_factory(
                context
            ).segment_surfaces(context.image_bgr)
        except Exception as exc:
            context.surface_segmentation = None
            print(
                f"[surfaces] SAM3 unavailable ({exc}); "
                "floor/wall fitting will use manual fallbacks"
            )
            return StageStatus.SKIPPED
        segmentation = context.surface_segmentation
        if not self._usable(segmentation):
            context.surface_segmentation = None
            print(
                "[surfaces] SAM3 did not produce both wall and floor masks; "
                "floor/wall fitting will use manual fallbacks"
            )
            return StageStatus.SKIPPED
        self._print_summary("SAM3", segmentation)
        return StageStatus.COMPLETED


class SurfaceMaskStage(PipelineStage):
    """Resize provider masks only after the MoGe point-map size is known."""

    name = "prepare_surface_masks"
    dependencies = ("surface_segmentation", "moge_inference")
    background = True

    def run(self, context: PipelineContext) -> StageStatus:
        if not context.settings.moge_enabled:
            print("[surface-masks] skipped (moge disabled)")
            return StageStatus.SKIPPED
        if (
            context.image_bgr is None
            or context.moge is None
            or context.surface_segmentation is None
        ):
            print("[surface-masks] skipped (surface segmentation unavailable)")
            return StageStatus.SKIPPED

        pm_w, pm_h = context.moge.image_size
        context.floor_surface_mask_pm = resize_mask(
            context.surface_segmentation.combined_mask("floor"), pm_w, pm_h
        )
        context.rug_surface_mask_pm = resize_mask(
            context.surface_segmentation.combined_mask("rug"), pm_w, pm_h
        )
        context.wall_surface_masks_pm = [
            resize_mask(item.mask, pm_w, pm_h)
            for item in context.surface_segmentation.instances("wall")
        ]
        if context.settings.debug:
            context.debug_surfaces_2d = draw_surface_debug_2d(
                context.image_bgr, context.surface_segmentation
            )
        print(f"[surface-masks] prepared at {pm_w}x{pm_h}")
        return StageStatus.COMPLETED
