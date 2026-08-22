"""Tests for geometry-only placement orientation and target-box construction."""

from __future__ import annotations

import io

import numpy as np
import pytest

from dreamroom.geometry3d import Box3D, FloorPlane
from dreamroom.placement_geometry import (
    PlacementOrientation,
    box_vertical_faces,
    build_target_box,
    infer_placement_orientation,
)
from dreamroom.placement_viz import export_placement_debug_glb
from dreamroom.wall_geometry import WallPlane


FLOOR = FloorPlane(
    point=np.zeros(3),
    normal=np.array([0.0, 1.0, 0.0]),
    inlier_ratio=1.0,
    num_candidates=1000,
)


def old_box() -> Box3D:
    return Box3D(
        center=np.array([0.0, 0.5, -3.0]),
        axes=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        extents=np.array([2.0, 1.0, 1.0]),
    )


def wall_from_bottom_edge(
    start: list[float],
    end: list[float],
    normal: list[float],
    height: float = 3.0,
) -> WallPlane:
    start_point = np.asarray(start, dtype=float)
    end_point = np.asarray(end, dtype=float)
    up = height * FLOOR.normal
    return WallPlane(
        point=0.5 * (start_point + end_point),
        normal=np.asarray(normal, dtype=float),
        corners=np.array([start_point, end_point, end_point + up, start_point + up]),
        inlier_count=2000,
        num_candidates=5000,
        rmse=0.01,
        width=float(np.linalg.norm(end_point - start_point)),
        height=height,
        confidence=0.9,
    )


def test_box_vertical_faces_have_stable_opposites():
    faces = {face.face_id: face for face in box_vertical_faces(old_box())}

    assert set(faces) == {
        "axis0_negative",
        "axis0_positive",
        "axis1_negative",
        "axis1_positive",
    }
    assert faces["axis1_positive"].opposite_id == "axis1_negative"
    assert faces["axis1_positive"].normal == pytest.approx([0.0, 0.0, -1.0])
    assert FLOOR.signed_distance(faces["axis1_positive"].bottom_edge) == pytest.approx(
        np.zeros(2)
    )


def test_wall_backed_orientation_and_target_box():
    box = old_box()
    back_wall = wall_from_bottom_edge(
        [-3.0, 0.0, -3.7],
        [3.0, 0.0, -3.7],
        [0.0, 0.0, 1.0],
    )

    orientation = infer_placement_orientation(box, FLOOR, [back_wall])
    target = build_target_box(box, FLOOR, [back_wall], orientation, 3.0, 2.0, 1.5)

    assert orientation.mode == "wall_backed"
    assert orientation.primary_rear_face == "axis1_positive"
    assert target is not None and target.wall_aligned
    assert target.box.extents == pytest.approx([3.0, 2.0, 1.5])
    assert target.rear_anchor == pytest.approx([0.0, 0.0, -3.7])
    assert target.primary_wall_distance == pytest.approx(0.0)
    assert target.primary_wall_distance_before_snap == pytest.approx(0.2)
    assert target.primary_wall_snapped
    assert target.wall_snap_threshold == pytest.approx(0.4)
    assert target.anchor_mode == "wall_snapped"
    half_width = 0.5 * target.box.extents[0] * target.box.axes[0]
    rear_edge = np.array(
        [target.rear_anchor - half_width, target.rear_anchor + half_width]
    )
    assert back_wall.signed_distance(rear_edge) == pytest.approx(
        np.zeros(2), abs=1e-8
    )
    assert FLOOR.signed_distance(target.box.corners()[:4]) == pytest.approx(
        np.zeros(4), abs=1e-8
    )
    assert np.linalg.det(target.box.axes.T) == pytest.approx(1.0)


def test_wall_backed_target_keeps_gap_beyond_snap_threshold():
    wall = wall_from_bottom_edge(
        [-3.0, 0.0, -4.0],
        [3.0, 0.0, -4.0],
        [0.0, 0.0, 1.0],
    )
    orientation = PlacementOrientation(
        mode="wall_backed",
        primary_rear_face="axis1_positive",
        secondary_anchor_face=None,
        primary_wall_index=0,
        secondary_wall_index=None,
        confidence=0.8,
        reason="test",
    )

    target = build_target_box(
        old_box(),
        FLOOR,
        [wall],
        orientation,
        3.0,
        2.0,
        1.5,
        wall_snap_distance=0.4,
    )

    assert target.rear_anchor == pytest.approx([0.0, 0.0, -3.5])
    assert target.primary_wall_distance == pytest.approx(0.5)
    assert target.primary_wall_distance_before_snap == pytest.approx(0.5)
    assert not target.primary_wall_snapped
    assert target.anchor_mode == "rear_face"


def test_corner_orientation_preserves_secondary_wall_clearance():
    box = old_box()
    back_wall = wall_from_bottom_edge(
        [-3.0, 0.0, -3.6],
        [3.0, 0.0, -3.6],
        [0.0, 0.0, 1.0],
    )
    side_wall = wall_from_bottom_edge(
        [-1.25, 0.0, -6.0],
        [-1.25, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    )

    orientation = infer_placement_orientation(box, FLOOR, [back_wall, side_wall])
    target = build_target_box(
        box,
        FLOOR,
        [back_wall, side_wall],
        orientation,
        3.0,
        2.0,
        1.5,
    )

    assert orientation.mode == "corner_backed"
    assert {
        orientation.primary_rear_face,
        orientation.secondary_anchor_face,
    } == {"axis1_positive", "axis0_negative"}
    assert target is not None and target.secondary_clearance_preserved
    assert target.primary_wall_snapped
    assert target.primary_wall_distance == pytest.approx(0.0, abs=1e-8)
    assert FLOOR.signed_distance(target.box.corners()[:4]) == pytest.approx(
        np.zeros(4), abs=1e-8
    )


def test_target_keeps_old_orientation_when_wall_tilt_is_large():
    box = old_box()
    wall = wall_from_bottom_edge(
        [-3.0, 0.0, -3.7],
        [3.0, 0.0, -3.7],
        [0.5, 0.0, 0.8660254],
    )
    orientation = PlacementOrientation(
        mode="angled_wall_backed",
        primary_rear_face="axis1_positive",
        secondary_anchor_face=None,
        primary_wall_index=0,
        secondary_wall_index=None,
        confidence=0.7,
        reason="test",
    )

    target = build_target_box(box, FLOOR, [wall], orientation, 2.0, 1.0, 1.0)

    assert target is not None and not target.wall_aligned
    assert not target.primary_wall_snapped
    assert target.tilt_degrees == pytest.approx(30.0, abs=0.1)
    assert target.box.axes[1] == pytest.approx([0.0, 0.0, 1.0])


def test_no_evidence_is_ambiguous():
    orientation = infer_placement_orientation(old_box(), FLOOR, [])
    target = build_target_box(old_box(), FLOOR, [], orientation, 3.0, 2.0, 1.5)

    assert orientation.mode == "ambiguous"
    assert orientation.primary_rear_face is None
    assert target is not None
    assert target.anchor_mode == "center_fallback"
    assert target.box.axes == pytest.approx(old_box().axes)
    assert FLOOR.signed_distance(target.box.corners()[:4]) == pytest.approx(
        np.zeros(4), abs=1e-8
    )


def test_placement_debug_glb_has_separate_geometry():
    trimesh = pytest.importorskip("trimesh")
    source = trimesh.Scene(trimesh.creation.box(extents=[1.0, 1.0, 1.0]))
    box = old_box()
    wall = wall_from_bottom_edge(
        [-3.0, 0.0, -3.7],
        [3.0, 0.0, -3.7],
        [0.0, 0.0, 1.0],
    )
    orientation = infer_placement_orientation(box, FLOOR, [wall])
    target = build_target_box(box, FLOOR, [wall], orientation, 3.0, 2.0, 1.5)

    exported = export_placement_debug_glb(
        source.export(file_type="glb"),
        box,
        orientation,
        [wall],
        target=target,
    )

    assert exported is not None
    result = trimesh.load(io.BytesIO(exported), file_type="glb", force="scene")
    assert "old_fitted_box" in result.geometry
    assert "candidate_axis1_positive" in result.geometry
    assert "target_box" in result.geometry
