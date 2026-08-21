"""Thread-safe task timing and concurrency reporting."""

from __future__ import annotations

import time
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .stages.base import PipelineStage


class LatencyTracker:
    """Collect named durations without coupling stages to output formats."""

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.values: dict[str, float | None] = {}
        self.tasks: dict[str, dict] = {}
        self._lock = Lock()

    def register(self, stages: list[PipelineStage]) -> None:
        with self._lock:
            for stage in stages:
                self.values.setdefault(stage.name, None)
                self.tasks.setdefault(
                    stage.name,
                    {
                        "dependencies": list(stage.dependencies),
                        "execution": "background" if stage.background else "main",
                        "status": "pending",
                        "started_at_seconds": None,
                        "ended_at_seconds": None,
                        "duration_seconds": None,
                    },
                )

    def start_task(
        self,
        name: str,
        dependencies: tuple[str, ...] = (),
        execution: str = "main",
    ) -> float:
        started = time.perf_counter()
        with self._lock:
            self.values.setdefault(name, None)
            self.tasks[name] = {
                "dependencies": list(dependencies),
                "execution": execution,
                "status": "running",
                "started_at_seconds": started - self.started,
                "ended_at_seconds": None,
                "duration_seconds": None,
            }
        return started

    def finish_task(self, name: str, started: float, status: str) -> None:
        ended = time.perf_counter()
        duration = ended - started
        with self._lock:
            self.values[name] = None if status == "skipped" else duration
            task = self.tasks[name]
            task["status"] = status
            task["ended_at_seconds"] = ended - self.started
            task["duration_seconds"] = duration

    def record(self, name: str, started: float) -> None:
        self.finish_task(name, started, "completed")

    def record_skipped(self, name: str) -> None:
        started = self.start_task(name)
        self.finish_task(name, started, "skipped")

    def record_total(self) -> None:
        with self._lock:
            self.values["total"] = time.perf_counter() - self.started

    def report(self) -> dict:
        with self._lock:
            values = dict(self.values)
            tasks = {name: dict(task) for name, task in self.tasks.items()}
        total = float(values.get("total") or 0.0)
        active = sum(
            float(task["duration_seconds"] or 0.0)
            for task in tasks.values()
            if task["status"] == "completed"
        )
        critical = self._critical_path_seconds(tasks)
        return {
            "latency_seconds": values,
            "tasks": tasks,
            "summary": {
                "wall_clock_seconds": total,
                "active_task_seconds": active,
                "critical_path_seconds": critical,
                "concurrency_saved_seconds": max(0.0, active - total),
            },
        }

    @staticmethod
    def _critical_path_seconds(tasks: dict[str, dict]) -> float:
        longest: dict[str, float] = {}
        remaining = set(tasks)
        while remaining:
            progressed = False
            for name in list(remaining):
                dependencies = tasks[name]["dependencies"]
                if not all(dependency in longest for dependency in dependencies):
                    continue
                duration = (
                    float(tasks[name]["duration_seconds"] or 0.0)
                    if tasks[name]["status"] == "completed"
                    else 0.0
                )
                longest[name] = duration + max(
                    (longest[dependency] for dependency in dependencies),
                    default=0.0,
                )
                remaining.remove(name)
                progressed = True
            if not progressed:
                return 0.0
        return max(longest.values(), default=0.0)


def print_latency_stats(
    latency_seconds: dict[str, float | None],
    summary: dict[str, float] | None = None,
) -> None:
    """Print the user-facing per-task latency and concurrency summary."""

    print("[stats] latency:")
    for name, seconds in latency_seconds.items():
        if seconds is None:
            print(f"  {name}: skipped")
        else:
            print(f"  {name}: {seconds:.3f}s")
    if summary is not None:
        print("[stats] summary:")
        print(f"  wall_clock: {summary.get('wall_clock_seconds', 0.0):.3f}s")
        print(f"  active_tasks: {summary.get('active_task_seconds', 0.0):.3f}s")
        print(f"  critical_path: {summary.get('critical_path_seconds', 0.0):.3f}s")
        print(
            "  concurrency_saved: "
            f"{summary.get('concurrency_saved_seconds', 0.0):.3f}s"
        )
