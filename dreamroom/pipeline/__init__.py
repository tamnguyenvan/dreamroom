"""Extensible dependency-based furniture replacement pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import Settings
from ..segmenter import SimpleClickSegmenter
from .models import PipelineContext, PipelineResult
from .outputs import OutputWriter
from .runner import TaskGraphRunner
from .stages import (
    FurnitureStage,
    GeometryStage,
    MogeStage,
    PointMapStage,
    PlacementStage,
    ReferenceStage,
    RenderStage,
    ResizeStage,
    SelectionStage,
    SurfaceMaskStage,
    SurfaceStage,
    WallStage,
)
from .timing import LatencyTracker, print_latency_stats

__all__ = ["FurniturePipeline", "PipelineContext", "PipelineResult"]


class FurniturePipeline:
    """Build and execute the pipeline task graph."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._segmenter: SimpleClickSegmenter | None = None

    @property
    def segmenter(self) -> SimpleClickSegmenter:
        if self._segmenter is None:
            self._segmenter = SimpleClickSegmenter(self.settings)
        return self._segmenter

    def _segmenter_factory(self, image_rgb: np.ndarray):
        def segment(positive: list[list[int]], negative: list[list[int]]) -> np.ndarray:
            return self.segmenter.segment(image_rgb, positive, negative)

        return segment

    def _tasks(self):
        return [
            ResizeStage(),
            MogeStage(),
            SurfaceStage(),
            FurnitureStage(),
            SelectionStage(self._segmenter_factory),
            ReferenceStage(),
            PointMapStage(),
            SurfaceMaskStage(),
            GeometryStage(),
            WallStage(),
            PlacementStage(),
            RenderStage(),
        ]

    def run(self, image_path: str | Path, output_dir: str | Path | None = None) -> PipelineResult | None:
        """Run all configured tasks; returns ``None`` when the user aborts."""

        context = PipelineContext(Path(image_path), self.settings)
        latency = LatencyTracker()
        if not TaskGraphRunner(self._tasks()).run(context, latency):
            return None

        save_started = latency.start_task(
            "save_outputs",
            ("render_furniture",),
            execution="main",
        )
        try:
            out_dir = OutputWriter().save(context, output_dir)
        except Exception:
            latency.finish_task("save_outputs", save_started, "failed")
            raise
        latency.finish_task("save_outputs", save_started, "completed")
        latency.record_total()
        context.latency_seconds = latency.values
        report = latency.report()
        OutputWriter.write_stats(out_dir, report)
        print(f"[done] outputs written to {out_dir}")
        print_latency_stats(latency.values, report["summary"])
        return PipelineResult.from_context(out_dir, context)
