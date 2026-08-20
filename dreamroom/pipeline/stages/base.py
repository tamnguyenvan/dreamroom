"""Stage protocol primitives."""

from __future__ import annotations

from enum import Enum, auto

from ..models import PipelineContext


class StageStatus(Enum):
    COMPLETED = auto()
    SKIPPED = auto()
    ABORTED = auto()


class PipelineStage:
    """Small interface implemented by every ordered stage."""

    name: str

    def run(self, context: PipelineContext) -> StageStatus:
        raise NotImplementedError
