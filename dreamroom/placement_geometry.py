"""Heuristic placement orientation and target-box construction."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry3d import Box3D, FloorPlane
from .wall_geometry import WallPlane, floor_basis

FACE_IDS = (
    "axis0_negative",
    "axis0_positive",
    "axis1_negative",
    "axis1_positive",
)


@dataclass
class BoxFace:
    """One vertical face of a floor-aligned box."""

    face_id: str
    axis_index: int
    sign: int
    center: np.ndarray
    normal: np.ndarray
    horizontal_axis: np.ndarray
    corners: np.ndarray
    bottom_edge: np.ndarray
    width: float
    height: float

    @property
    def opposite_id(self) -> str:
        suffix = "positive" if self.sign < 0 else "negative"
        return f"axis{self.axis_index}_{suffix}"


@dataclass
class FaceEvidence:
    """Best wall association and visible support for one box face."""

    face_id: str
    visible_fraction: float
    wall_index: int | None = None
    score: float = 0.0
    distance: float | None = None
    angle_degrees: float | None = None
    parallelism: float = 0.0
    points_toward_wall: float = 0.0
    overlap_ratio: float = 0.0

    def to_dict(self) -> dict:
        return {
            "face_id": self.face_id,
            "visible_fraction": round(float(self.visible_fraction), 4),
            "wall_index": self.wall_index,
            "score": round(float(self.score), 4),
            "distance": None if self.distance is None else round(float(self.distance), 4),
            "angle_degrees": (
                None
                if self.angle_degrees is None
                else round(float(self.angle_degrees), 2)
            ),
            "parallelism": round(float(self.parallelism), 4),
            "points_toward_wall": round(float(self.points_toward_wall), 4),
            "overlap_ratio": round(float(self.overlap_ratio), 4),
        }


@dataclass
class PlacementOrientation:
    """Geometry-only decision about how the old object is anchored."""

    mode: str
    primary_rear_face: str | None
    secondary_anchor_face: str | None
    primary_wall_index: int | None
    secondary_wall_index: int | None
    confidence: float
    reason: str
    face_evidence: list[FaceEvidence] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "primary_rear_face": self.primary_rear_face,
            "secondary_anchor_face": self.secondary_anchor_face,
            "primary_wall_index": self.primary_wall_index,
            "secondary_wall_index": self.secondary_wall_index,
            "confidence": round(float(self.confidence), 4),
            "reason": self.reason,
            "face_evidence": [item.to_dict() for item in self.face_evidence],
        }


@dataclass
class TargetBoxPlacement:
    """Constructed target box plus placement diagnostics."""

    box: Box3D
    rear_anchor: np.ndarray
    rear_face_id: str | None
    primary_wall_index: int | None
    secondary_wall_index: int | None
    wall_aligned: bool
    tilt_degrees: float | None
    primary_wall_distance: float | None
    secondary_clearance_preserved: bool = False
    anchor_mode: str = "rear_face"

    def to_dict(self) -> dict:
        half_width = 0.5 * self.box.extents[0] * self.box.axes[0]
        depth = self.box.extents[1] * self.box.axes[1]
        rear_edge = np.array(
            [self.rear_anchor - half_width, self.rear_anchor + half_width]
        )
        front_edge = rear_edge + depth
        return {
            "box": self.box.to_dict(),
            "rear_anchor": np.round(self.rear_anchor, 4).tolist(),
            "rear_edge": np.round(rear_edge, 4).tolist(),
            "front_edge": np.round(front_edge, 4).tolist(),
            "rear_face_id": self.rear_face_id,
            "anchor_mode": self.anchor_mode,
            "primary_wall_index": self.primary_wall_index,
            "secondary_wall_index": self.secondary_wall_index,
            "wall_aligned": bool(self.wall_aligned),
            "tilt_degrees": (
                None
                if self.tilt_degrees is None
                else round(float(self.tilt_degrees), 2)
            ),
            "primary_wall_distance": (
                None
                if self.primary_wall_distance is None
                else round(float(self.primary_wall_distance), 4)
            ),
            "secondary_clearance_preserved": bool(
                self.secondary_clearance_preserved
            ),
        }


def box_vertical_faces(box: Box3D) -> list[BoxFace]:
    """Return the four vertical faces with stable semantic identifiers."""

    corners = box.corners()
    definitions = [
        ("axis0_negative", 0, -1, [0, 1, 5, 4], [0, 1]),
        ("axis0_positive", 0, 1, [2, 3, 7, 6], [2, 3]),
        ("axis1_negative", 1, -1, [0, 2, 6, 4], [0, 2]),
        ("axis1_positive", 1, 1, [1, 3, 7, 5], [1, 3]),
    ]
    faces = []
    for face_id, axis_index, sign, face_indices, edge_indices in definitions:
        other_axis = 1 - axis_index
        faces.append(
            BoxFace(
                face_id=face_id,
                axis_index=axis_index,
                sign=sign,
                center=box.center
                + sign * 0.5 * box.extents[axis_index] * box.axes[axis_index],
                normal=sign * box.axes[axis_index],
                horizontal_axis=box.axes[other_axis],
                corners=corners[face_indices],
                bottom_edge=corners[edge_indices],
                width=float(box.extents[other_axis]),
                height=float(box.extents[2]),
            )
        )
    return faces


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _point_segment_distance(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> float:
    segment = end - start
    denominator = float(segment @ segment)
    if denominator < 1e-12:
        return float(np.linalg.norm(point - start))
    amount = float(np.clip((point - start) @ segment / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + amount * segment)))


def _segments_intersect(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> bool:
    first = first_end - first_start
    second = second_end - second_start
    denominator = _cross_2d(first, second)
    if abs(denominator) < 1e-9:
        return False
    offset = second_start - first_start
    first_amount = _cross_2d(offset, second) / denominator
    second_amount = _cross_2d(offset, first) / denominator
    return 0.0 <= first_amount <= 1.0 and 0.0 <= second_amount <= 1.0


def _segment_distance(first: np.ndarray, second: np.ndarray) -> float:
    if _segments_intersect(first[0], first[1], second[0], second[1]):
        return 0.0
    return min(
        _point_segment_distance(first[0], second[0], second[1]),
        _point_segment_distance(first[1], second[0], second[1]),
        _point_segment_distance(second[0], first[0], first[1]),
        _point_segment_distance(second[1], first[0], first[1]),
    )


def _closest_point_on_segment(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> np.ndarray:
    segment = end - start
    denominator = float(segment @ segment)
    if denominator < 1e-12:
        return start.copy()
    amount = float(np.clip((point - start) @ segment / denominator, 0.0, 1.0))
    return start + amount * segment


def _visible_face_fractions(
    point_map: np.ndarray | None,
    object_mask: np.ndarray | None,
    box: Box3D,
) -> dict[str, float]:
    fractions = {face_id: 0.0 for face_id in FACE_IDS}
    if point_map is None or object_mask is None:
        return fractions
    points = point_map[object_mask]
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        return fractions

    local = (points - box.center) @ box.axes.T
    top_margin = max(0.03, 0.08 * box.extents[2])
    side_points = local[:, 2] < 0.5 * box.extents[2] - top_margin
    if side_points.sum() >= 50:
        local = local[side_points]
    half = 0.5 * box.extents[:2]
    residuals = np.column_stack(
        (
            np.abs(local[:, 0] + half[0]) / max(box.extents[0], 1e-6),
            np.abs(local[:, 0] - half[0]) / max(box.extents[0], 1e-6),
            np.abs(local[:, 1] + half[1]) / max(box.extents[1], 1e-6),
            np.abs(local[:, 1] - half[1]) / max(box.extents[1], 1e-6),
        )
    )
    assignments = np.argmin(residuals, axis=1)
    counts = np.bincount(assignments, minlength=4)
    for index, face_id in enumerate(FACE_IDS):
        fractions[face_id] = float(counts[index] / max(len(assignments), 1))
    return fractions


def _face_wall_evidence(
    face: BoxFace,
    wall: WallPlane,
    wall_index: int,
    floor: FloorPlane,
    visible_fraction: float,
    proximity_scale: float,
) -> FaceEvidence:
    axis_u, axis_v = floor_basis(floor)

    def floor_xy(points: np.ndarray) -> np.ndarray:
        relative = points - floor.point
        return np.column_stack((relative @ axis_u, relative @ axis_v))

    face_segment = floor_xy(face.bottom_edge)
    wall_segment = floor_xy(wall.corners[:2])
    face_center = floor_xy(face.center[None, :])[0]
    distance = _segment_distance(face_segment, wall_segment)
    closest_wall = _closest_point_on_segment(
        face_center, wall_segment[0], wall_segment[1]
    )
    toward_2d = closest_wall - face_center
    toward_norm = np.linalg.norm(toward_2d)
    face_normal_2d = np.array([face.normal @ axis_u, face.normal @ axis_v])
    if toward_norm > 1e-9:
        points_toward_wall = max(
            0.0, float(face_normal_2d @ (toward_2d / toward_norm))
        )
    else:
        points_toward_wall = 1.0

    parallelism = abs(float(face.normal @ wall.normal))
    parallelism = float(np.clip(parallelism, 0.0, 1.0))
    angle = float(np.degrees(np.arccos(parallelism)))
    wall_tangent = wall_segment[1] - wall_segment[0]
    wall_length = np.linalg.norm(wall_tangent)
    if wall_length > 1e-9:
        wall_tangent /= wall_length
        face_interval = np.sort(face_segment @ wall_tangent)
        wall_interval = np.sort(wall_segment @ wall_tangent)
        overlap = max(
            0.0,
            min(face_interval[1], wall_interval[1])
            - max(face_interval[0], wall_interval[0]),
        )
        overlap_ratio = float(np.clip(overlap / max(face.width, 1e-6), 0.0, 1.0))
    else:
        overlap_ratio = 0.0

    proximity = float(np.exp(-distance / max(proximity_scale, 1e-6)))
    score = (
        0.45 * proximity
        + 0.25 * parallelism
        + 0.20 * points_toward_wall
        + 0.10 * overlap_ratio
    )
    return FaceEvidence(
        face_id=face.face_id,
        visible_fraction=visible_fraction,
        wall_index=wall_index,
        score=float(score),
        distance=distance,
        angle_degrees=angle,
        parallelism=parallelism,
        points_toward_wall=points_toward_wall,
        overlap_ratio=overlap_ratio,
    )


def infer_placement_orientation(
    box: Box3D,
    floor: FloorPlane,
    walls: list[WallPlane],
    point_map: np.ndarray | None = None,
    object_mask: np.ndarray | None = None,
    *,
    near_wall_distance: float = 0.45,
    parallel_angle_degrees: float = 25.0,
    minimum_wall_score: float = 0.45,
    minimum_score_margin: float = 0.08,
    minimum_visibility_margin: float = 0.12,
) -> PlacementOrientation:
    """Infer wall-backed/corner/free-standing placement without semantics."""

    faces = box_vertical_faces(box)
    visible = _visible_face_fractions(point_map, object_mask, box)
    proximity_scale = max(0.25, 0.35 * min(box.extents[0], box.extents[1]))
    evidence = []
    for face in faces:
        candidates = [
            _face_wall_evidence(
                face,
                wall,
                wall_index,
                floor,
                visible[face.face_id],
                proximity_scale,
            )
            for wall_index, wall in enumerate(walls)
        ]
        if candidates:
            evidence.append(max(candidates, key=lambda item: item.score))
        else:
            evidence.append(
                FaceEvidence(face.face_id, visible_fraction=visible[face.face_id])
            )

    by_face = {face.face_id: face for face in faces}
    strong = [
        item
        for item in evidence
        if item.wall_index is not None
        and item.distance is not None
        and item.angle_degrees is not None
        and item.distance <= near_wall_distance
        and item.angle_degrees <= parallel_angle_degrees
        and item.points_toward_wall >= 0.35
        and item.score >= minimum_wall_score
    ]
    corner_pairs: list[tuple[float, FaceEvidence, FaceEvidence]] = []
    for first_index, first in enumerate(strong):
        for second in strong[first_index + 1 :]:
            if by_face[first.face_id].axis_index == by_face[second.face_id].axis_index:
                continue
            if first.wall_index == second.wall_index:
                continue
            first_wall = walls[first.wall_index]
            second_wall = walls[second.wall_index]
            wall_angle = float(
                np.degrees(
                    np.arccos(
                        np.clip(abs(first_wall.normal @ second_wall.normal), 0.0, 1.0)
                    )
                )
            )
            if 60.0 <= wall_angle <= 90.0:
                corner_pairs.append((first.score + second.score, first, second))
    if corner_pairs:
        _, first, second = max(corner_pairs, key=lambda item: item[0])
        primary, secondary = sorted(
            (first, second), key=lambda item: item.score, reverse=True
        )
        return PlacementOrientation(
            mode="corner_backed",
            primary_rear_face=primary.face_id,
            secondary_anchor_face=secondary.face_id,
            primary_wall_index=primary.wall_index,
            secondary_wall_index=secondary.wall_index,
            confidence=float(min(primary.score, secondary.score)),
            reason="two adjacent faces are strongly associated with perpendicular walls",
            face_evidence=evidence,
        )

    ranked = sorted(evidence, key=lambda item: item.score, reverse=True)
    best = ranked[0]
    second_score = ranked[1].score if len(ranked) > 1 else 0.0
    margin = best.score - second_score
    if best in strong and margin >= minimum_score_margin:
        return PlacementOrientation(
            mode="wall_backed",
            primary_rear_face=best.face_id,
            secondary_anchor_face=None,
            primary_wall_index=best.wall_index,
            secondary_wall_index=None,
            confidence=float(best.score),
            reason="one face has a dominant nearby parallel wall association",
            face_evidence=evidence,
        )

    angled = [
        item
        for item in evidence
        if item.wall_index is not None
        and item.distance is not None
        and item.distance <= near_wall_distance
        and item.points_toward_wall >= 0.35
        and item.score >= 0.35
    ]
    if angled:
        angled_best = max(angled, key=lambda item: item.score)
        angled_second = sorted(
            (item.score for item in angled if item is not angled_best), reverse=True
        )
        angled_margin = angled_best.score - (angled_second[0] if angled_second else 0.0)
        if angled_margin >= minimum_score_margin:
            return PlacementOrientation(
                mode="angled_wall_backed",
                primary_rear_face=angled_best.face_id,
                secondary_anchor_face=None,
                primary_wall_index=angled_best.wall_index,
                secondary_wall_index=None,
                confidence=float(angled_best.score),
                reason="a nearby wall-facing face was found, but it is not parallel",
                face_evidence=evidence,
            )

    visible_ranked = sorted(
        evidence, key=lambda item: item.visible_fraction, reverse=True
    )
    visible_margin = (
        visible_ranked[0].visible_fraction - visible_ranked[1].visible_fraction
    )
    if (
        visible_ranked[0].visible_fraction >= 0.25
        and visible_margin >= minimum_visibility_margin
    ):
        front = by_face[visible_ranked[0].face_id]
        return PlacementOrientation(
            mode="free_standing",
            primary_rear_face=front.opposite_id,
            secondary_anchor_face=None,
            primary_wall_index=None,
            secondary_wall_index=None,
            confidence=float(visible_margin),
            reason="no wall anchor was reliable; rear is opposite the most visible face",
            face_evidence=evidence,
        )

    return PlacementOrientation(
        mode="ambiguous",
        primary_rear_face=None,
        secondary_anchor_face=None,
        primary_wall_index=None,
        secondary_wall_index=None,
        confidence=0.0,
        reason="wall and visibility evidence do not identify a unique placement face",
        face_evidence=evidence,
    )


def build_target_box(
    old_box: Box3D,
    floor: FloorPlane,
    walls: list[WallPlane],
    orientation: PlacementOrientation,
    new_width: float,
    new_depth: float,
    new_height: float,
    *,
    wall_alignment_degrees: float = 10.0,
) -> TargetBoxPlacement:
    """Construct a floor-contact target box from the inferred rear face."""

    dimensions = np.array([new_width, new_depth, new_height], dtype=float)
    if not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        raise ValueError("target width, depth, and height must be positive")
    faces = {face.face_id: face for face in box_vertical_faces(old_box)}
    if orientation.primary_rear_face is None:
        up = floor.normal
        width_direction = old_box.axes[0].copy()
        depth_direction = old_box.axes[1].copy()
        base_center = old_box.base_center()
        base_center = base_center - floor.signed_distance(base_center) * up
        target_box = Box3D(
            center=base_center + 0.5 * new_height * up,
            axes=np.stack((width_direction, depth_direction, up)),
            extents=dimensions,
        )
        rear_anchor = base_center - 0.5 * new_depth * depth_direction
        return TargetBoxPlacement(
            box=target_box,
            rear_anchor=rear_anchor,
            rear_face_id=None,
            primary_wall_index=None,
            secondary_wall_index=None,
            wall_aligned=False,
            tilt_degrees=None,
            primary_wall_distance=None,
            anchor_mode="center_fallback",
        )

    rear_face = faces[orientation.primary_rear_face]
    up = floor.normal
    old_depth = -rear_face.normal
    old_depth -= float(old_depth @ up) * up
    old_depth /= np.linalg.norm(old_depth)
    depth_direction = old_depth.copy()
    primary_wall = (
        walls[orientation.primary_wall_index]
        if orientation.primary_wall_index is not None
        else None
    )
    tilt_degrees = None
    wall_aligned = False
    if primary_wall is not None:
        wall_normal = primary_wall.normal - float(primary_wall.normal @ up) * up
        wall_normal /= np.linalg.norm(wall_normal)
        tilt_degrees = float(
            np.degrees(
                np.arccos(
                    np.clip(abs(rear_face.normal @ wall_normal), 0.0, 1.0)
                )
            )
        )
        if tilt_degrees < wall_alignment_degrees:
            candidate = wall_normal
            if candidate @ old_depth < 0:
                candidate = -candidate
            depth_direction = candidate
            wall_aligned = True

    width_direction = np.cross(depth_direction, up)
    width_direction /= np.linalg.norm(width_direction)
    rear_anchor = rear_face.center - floor.signed_distance(rear_face.center) * up
    target_center = (
        rear_anchor
        + 0.5 * new_depth * depth_direction
        + 0.5 * new_height * up
    )
    target_box = Box3D(
        center=target_center,
        axes=np.stack((width_direction, depth_direction, up)),
        extents=dimensions,
    )

    secondary_preserved = False
    if (
        orientation.secondary_anchor_face is not None
        and orientation.secondary_wall_index is not None
    ):
        secondary_face = faces[orientation.secondary_anchor_face]
        secondary_wall = walls[orientation.secondary_wall_index]
        side_centers = [
            target_box.base_center() - 0.5 * new_width * width_direction,
            target_box.base_center() + 0.5 * new_width * width_direction,
        ]
        target_side = min(
            side_centers,
            key=lambda point: abs(float(secondary_wall.signed_distance(point))),
        )
        desired_distance = float(secondary_wall.signed_distance(secondary_face.center))
        current_distance = float(secondary_wall.signed_distance(target_side))
        denominator = float(width_direction @ secondary_wall.normal)
        if abs(denominator) > 0.5:
            translation = (desired_distance - current_distance) / denominator
            shift = translation * width_direction
            target_box.center = target_box.center + shift
            rear_anchor = rear_anchor + shift
            secondary_preserved = True

    return TargetBoxPlacement(
        box=target_box,
        rear_anchor=rear_anchor,
        rear_face_id=rear_face.face_id,
        primary_wall_index=orientation.primary_wall_index,
        secondary_wall_index=orientation.secondary_wall_index,
        wall_aligned=wall_aligned,
        tilt_degrees=tilt_degrees,
        primary_wall_distance=(
            None
            if primary_wall is None
            else abs(float(primary_wall.signed_distance(rear_anchor)))
        ),
        secondary_clearance_preserved=secondary_preserved,
    )
