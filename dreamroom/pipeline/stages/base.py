"""Task protocol primitives for the dependency-based pipeline."""

from __future__ import annotations

from enum import Enum, auto

from ..models import PipelineContext


class StageStatus(Enum):
    COMPLETED = auto()
    SKIPPED = auto()
    ABORTED = auto()


class PipelineStage:
    """One dependency-graph task.

    Background tasks must not open UI windows. Independent background tasks may
    run concurrently and therefore must write disjoint context fields.
    """

    name: str
    dependencies: tuple[str, ...] = ()
    background: bool = False

    def run(self, context: PipelineContext) -> StageStatus:
        raise NotImplementedError
