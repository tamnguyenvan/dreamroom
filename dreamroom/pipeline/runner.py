"""Ordered stage execution and per-stage timing."""

from __future__ import annotations

import time

from .models import PipelineContext
from .stages.base import PipelineStage, StageStatus
from .timing import LatencyTracker


class StageRunner:
    """Run stages in order and stop cleanly on user aborts."""

    def __init__(self, stages: list[PipelineStage]) -> None:
        self.stages = stages

    def run(self, context: PipelineContext, latency: LatencyTracker) -> bool:
        for stage in self.stages:
            started = time.perf_counter()
            status = stage.run(context)
            if status is StageStatus.SKIPPED:
                latency.record_skipped(stage.name)
            else:
                latency.record(stage.name, started)
            if status is StageStatus.ABORTED:
                return False
        return True
