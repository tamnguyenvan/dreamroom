"""Extensible stage-based furniture replacement pipeline."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..config import Settings
from ..segmenter import SimpleClickSegmenter
from .models import PipelineContext, PipelineResult
from .outputs import OutputWriter
from .runner import StageRunner
from .stages import (
    GeometryStage,
    MogeStage,
    PlacementStage,
    ReferenceStage,
    ResizeStage,
    SelectionStage,
    WallStage,
)
from .timing import LatencyTracker, print_latency_stats

__all__ = ["FurniturePipeline", "PipelineContext", "PipelineResult"]


class FurniturePipeline:
    """Build and execute the ordered pipeline stages."""

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

    def _stages(self):
        return [
            ResizeStage(),
            SelectionStage(self._segmenter_factory),
            ReferenceStage(),
            MogeStage(),
            GeometryStage(),
            WallStage(),
            PlacementStage(),
        ]

    def run(self, image_path: str | Path, output_dir: str | Path | None = None) -> PipelineResult | None:
        """Run all configured stages; returns ``None`` when the user aborts."""

        context = PipelineContext(Path(image_path), self.settings)
        latency = LatencyTracker()
        if not StageRunner(self._stages()).run(context, latency):
            return None

        save_started = time.perf_counter()
        out_dir = OutputWriter().save(context, output_dir)
        latency.record("save_outputs", save_started)
        latency.record_total()
        context.latency_seconds = latency.values
        OutputWriter.write_stats(out_dir, latency.values)
        print(f"[done] outputs written to {out_dir}")
        print_latency_stats(latency.values)
        return PipelineResult.from_context(out_dir, context)
