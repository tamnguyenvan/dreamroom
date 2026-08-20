"""Dependency-aware pipeline task scheduling."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

from .models import PipelineContext
from .stages.base import PipelineStage, StageStatus
from .timing import LatencyTracker


class TaskGraphRunner:
    """Run ready background tasks concurrently while UI tasks stay on main."""

    def __init__(self, stages: list[PipelineStage], max_workers: int = 4) -> None:
        self.stages = stages
        self.max_workers = max_workers
        self._validate_graph()

    def _validate_graph(self) -> None:
        names = [stage.name for stage in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("pipeline task names must be unique")
        known = set(names)
        for stage in self.stages:
            missing = set(stage.dependencies) - known
            if missing:
                raise ValueError(
                    f"task {stage.name!r} has unknown dependencies: {sorted(missing)}"
                )

    def run(self, context: PipelineContext, latency: LatencyTracker) -> bool:
        latency.register(self.stages)
        terminal: dict[str, StageStatus] = {}
        running: dict[Future[StageStatus], PipelineStage] = {}
        executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="dreamroom",
        )
        succeeded = False
        try:
            while len(terminal) < len(self.stages):
                aborted = self._collect_finished(running, terminal)
                if aborted:
                    return False

                ready = [
                    stage
                    for stage in self.stages
                    if stage.name not in terminal
                    and stage not in running.values()
                    and all(name in terminal for name in stage.dependencies)
                ]
                for stage in ready:
                    if stage.background:
                        future = executor.submit(self._run_task, stage, context, latency)
                        running[future] = stage

                foreground = next(
                    (stage for stage in ready if not stage.background),
                    None,
                )
                if foreground is not None:
                    status = self._run_task(foreground, context, latency)
                    terminal[foreground.name] = status
                    if status is StageStatus.ABORTED:
                        return False
                    continue

                if len(terminal) == len(self.stages):
                    break
                if running:
                    wait(tuple(running), return_when=FIRST_COMPLETED)
                    continue

                unresolved = [
                    stage.name for stage in self.stages if stage.name not in terminal
                ]
                raise RuntimeError(f"pipeline dependency cycle: {unresolved}")
            succeeded = True
            return True
        finally:
            if not succeeded:
                for future in running:
                    future.cancel()
            executor.shutdown(wait=succeeded, cancel_futures=not succeeded)

    @staticmethod
    def _run_task(
        stage: PipelineStage,
        context: PipelineContext,
        latency: LatencyTracker,
    ) -> StageStatus:
        started = latency.start_task(
            stage.name,
            stage.dependencies,
            execution="background" if stage.background else "main",
        )
        try:
            status = stage.run(context)
        except Exception:
            latency.finish_task(stage.name, started, "failed")
            raise
        latency.finish_task(stage.name, started, status.name.lower())
        return status

    @staticmethod
    def _collect_finished(
        running: dict[Future[StageStatus], PipelineStage],
        terminal: dict[str, StageStatus],
    ) -> bool:
        for future, stage in list(running.items()):
            if not future.done():
                continue
            del running[future]
            status = future.result()
            terminal[stage.name] = status
            if status is StageStatus.ABORTED:
                return True
        return False


# Compatibility alias for callers importing the old runner name.
StageRunner = TaskGraphRunner
