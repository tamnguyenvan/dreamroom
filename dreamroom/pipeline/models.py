"""Shared state and public result models for the pipeline runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import Settings
from ..geometry3d import Box3D, FloorPlane
from ..moge_client import MogeResult
from ..placement_geometry import PlacementOrientation, TargetBoxPlacement
from ..ui.reference import ReferenceScale
from ..ui.strokes import ObjectSelection
from ..wall_geometry import WallPlane


@dataclass
class PipelineContext:
    """Mutable data passed between ordered pipeline stages."""

    image_path: Path
    settings: Settings
    image_bgr: np.ndarray | None = None
    original_size: tuple[int, int] | None = None
    resize_scale: float = 1.0
    selection: ObjectSelection | None = None
    reference: ReferenceScale | None = None
    moge: MogeResult | None = None
    point_map: np.ndarray | None = None
    mask_pm: np.ndarray | None = None
    scale_correction: float | None = None
    calibration: dict = field(default_factory=dict)
    box: Box3D | None = None
    floor: FloorPlane | None = None
    walls: list[WallPlane] = field(default_factory=list)
    placement_orientation: PlacementOrientation | None = None
    target_placement: TargetBoxPlacement | None = None
    debug_2d: np.ndarray | None = None
    debug_3d: bytes | None = None
    debug_placement_2d: np.ndarray | None = None
    debug_placement_3d: bytes | None = None
    latency_seconds: dict[str, float | None] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Everything produced by steps 0-6 in a completed run."""

    output_dir: Path
    image_bgr: np.ndarray
    selection: ObjectSelection
    reference: ReferenceScale
    resize_scale: float
    original_size: tuple[int, int]
    box: Box3D | None = None
    floor: FloorPlane | None = None
    walls: list[WallPlane] = field(default_factory=list)
    placement_orientation: PlacementOrientation | None = None
    target_placement: TargetBoxPlacement | None = None
    scale_correction: float | None = None
    latency_seconds: dict[str, float | None] = field(default_factory=dict)

    @classmethod
    def from_context(cls, output_dir: Path, context: PipelineContext) -> "PipelineResult":
        assert context.image_bgr is not None
        assert context.selection is not None
        assert context.reference is not None
        assert context.original_size is not None
        return cls(
            output_dir=output_dir,
            image_bgr=context.image_bgr,
            selection=context.selection,
            reference=context.reference,
            resize_scale=context.resize_scale,
            original_size=context.original_size,
            box=context.box,
            floor=context.floor,
            walls=context.walls,
            placement_orientation=context.placement_orientation,
            target_placement=context.target_placement,
            scale_correction=context.scale_correction,
            latency_seconds=context.latency_seconds,
        )
