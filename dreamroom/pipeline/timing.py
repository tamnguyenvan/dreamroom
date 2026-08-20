"""Latency tracking and reporting for pipeline stages."""

from __future__ import annotations

import time


class LatencyTracker:
    """Collect named durations without coupling stages to output formats."""

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.values: dict[str, float | None] = {}

    def record(self, name: str, started: float) -> None:
        self.values[name] = time.perf_counter() - started

    def record_skipped(self, name: str) -> None:
        self.values[name] = None

    def record_total(self) -> None:
        self.values["total"] = time.perf_counter() - self.started


def print_latency_stats(latency_seconds: dict[str, float | None]) -> None:
    """Print the stable, user-facing latency summary."""

    print("[stats] latency:")
    for name in (
        "step_0_resize",
        "step_1_segment",
        "step_2_reference",
        "step_3_moge",
        "step_4_fit_3d",
        "save_outputs",
        "total",
    ):
        seconds = latency_seconds[name]
        if seconds is None:
            print(f"  {name}: skipped")
        else:
            print(f"  {name}: {seconds:.3f}s")
