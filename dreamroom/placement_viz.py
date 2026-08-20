"""Separate debug views for placement orientation and target-box geometry."""

from __future__ import annotations

import io
import logging

import cv2
import numpy as np

from .geometry3d import Box3D
from .placement_geometry import (
    PlacementOrientation,
    TargetBoxPlacement,
    box_vertical_faces,
)
from .ui.window import draw_banner
from .viz3d import BOX_EDGES, project_points
from .wall_geometry import WallPlane

logger = logging.getLogger(__name__)

FACE_COLOR = (0, 170, 255)
PRIMARY_COLOR = (60, 220, 60)
SECONDARY_COLOR = (255, 210, 40)
TARGET_COLOR = (255, 80, 40)
FACE_LABELS = {
    "axis0_negative": "F0-",
    "axis0_positive": "F0+",
    "axis1_negative": "F1-",
    "axis1_positive": "F1+",
}
MODE_LABELS = {
    "wall_backed": "wall-backed",
    "corner_backed": "corner-backed",
    "angled_wall_backed": "angled",
    "free_standing": "free-standing",
    "ambiguous": "ambiguous",
}


def draw_placement_debug_2d(
    image_bgr: np.ndarray,
    old_box: Box3D,
    orientation: PlacementOrientation,
    walls: list[WallPlane],
    k_px: np.ndarray,
    pixel_scale: tuple[float, float],
    target: TargetBoxPlacement | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Draw placement evidence on a clean copy of the working image."""

    frame = image_bgr.copy()
    sx, sy = pixel_scale

    def image_pixels(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pixels, valid = project_points(points, k_px)
        return pixels * np.array([sx, sy]), valid

    if mask is not None:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(frame, contours, -1, (200, 200, 200), 1, cv2.LINE_AA)

    evidence = {item.face_id: item for item in orientation.face_evidence}
    for face in box_vertical_faces(old_box):
        pixels, valid = image_pixels(face.corners)
        if not valid.all():
            continue
        if face.face_id == orientation.primary_rear_face:
            color = PRIMARY_COLOR
            thickness = 3
        elif face.face_id == orientation.secondary_anchor_face:
            color = SECONDARY_COLOR
            thickness = 3
        else:
            color = FACE_COLOR
            thickness = 1
        polygon = np.round(pixels).astype(np.int32).reshape(-1, 1, 2)
        fill = frame.copy()
        cv2.fillPoly(fill, [polygon], color)
        alpha = 0.18 if thickness > 1 else 0.06
        frame = cv2.addWeighted(fill, alpha, frame, 1.0 - alpha, 0)
        cv2.polylines(frame, [polygon], True, color, thickness, cv2.LINE_AA)
        item = evidence.get(face.face_id)
        label_at = tuple(np.round(pixels.mean(axis=0)).astype(int))
        cv2.putText(
            frame,
            FACE_LABELS[face.face_id],
            label_at,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    associated = [
        (orientation.primary_wall_index, PRIMARY_COLOR),
        (orientation.secondary_wall_index, SECONDARY_COLOR),
    ]
    for wall_index, color in associated:
        if wall_index is None or wall_index >= len(walls):
            continue
        wall_pixels, valid = image_pixels(walls[wall_index].corners)
        if valid.all():
            polygon = np.round(wall_pixels).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [polygon], True, color, 2, cv2.LINE_AA)

    if target is not None:
        corners, valid = image_pixels(target.box.corners())
        for start, end in BOX_EDGES:
            if valid[start] and valid[end]:
                cv2.line(
                    frame,
                    tuple(np.round(corners[start]).astype(int)),
                    tuple(np.round(corners[end]).astype(int)),
                    TARGET_COLOR,
                    3,
                    cv2.LINE_AA,
                )

    lines = [
        f"mode: {MODE_LABELS.get(orientation.mode, orientation.mode)}  "
        f"conf {orientation.confidence:.2f}",
        f"rear: {FACE_LABELS.get(orientation.primary_rear_face, 'unknown')}",
        "scores: "
        + "  ".join(
            f"{FACE_LABELS[face_id]} {evidence[face_id].score:.2f}"
            for face_id in ("axis0_negative", "axis0_positive")
            if face_id in evidence
        ),
        "        "
        + "  ".join(
            f"{FACE_LABELS[face_id]} {evidence[face_id].score:.2f}"
            for face_id in ("axis1_negative", "axis1_positive")
            if face_id in evidence
        ),
    ]
    if orientation.secondary_anchor_face is not None:
        lines.append(
            f"secondary: {FACE_LABELS[orientation.secondary_anchor_face]}"
        )
    if target is not None:
        extents = target.box.extents
        lines.append(
            f"target (m): {extents[0]:.2f} x {extents[1]:.2f} x {extents[2]:.2f}"
        )
    return draw_banner(frame, lines)


def export_placement_debug_glb(
    glb_bytes: bytes | None,
    old_box: Box3D,
    orientation: PlacementOrientation,
    walls: list[WallPlane],
    scale_factor: float = 1.0,
    target: TargetBoxPlacement | None = None,
) -> bytes | None:
    """Export a clean scene containing only placement-related overlays."""

    if glb_bytes is None:
        logger.warning("no MoGe GLB available; skipping placement GLB")
        return None
    try:
        import trimesh
    except ImportError:
        logger.warning("trimesh not installed; skipping placement GLB")
        return None

    scene = trimesh.load(io.BytesIO(glb_bytes), file_type="glb")
    scene.apply_scale(scale_factor)

    old_mesh = trimesh.creation.box(extents=old_box.extents)
    old_transform = np.eye(4)
    old_transform[:3, :3] = old_box.axes.T
    old_transform[:3, 3] = old_box.center
    old_mesh.apply_transform(old_transform)
    old_mesh.visual.face_colors = [220, 160, 40, 45]
    scene.add_geometry(old_mesh, geom_name="old_fitted_box")

    for face in box_vertical_faces(old_box):
        if face.face_id == orientation.primary_rear_face:
            color = [50, 230, 70, 180]
        elif face.face_id == orientation.secondary_anchor_face:
            color = [40, 190, 255, 160]
        else:
            color = [255, 150, 30, 35]
        face_mesh = trimesh.Trimesh(
            vertices=face.corners,
            faces=[[0, 1, 2], [0, 2, 3], [2, 1, 0], [3, 2, 0]],
            process=False,
        )
        face_mesh.visual.face_colors = color
        scene.add_geometry(face_mesh, geom_name=f"candidate_{face.face_id}")

    associated = [
        (orientation.primary_wall_index, "primary_wall", [50, 230, 70, 45]),
        (orientation.secondary_wall_index, "secondary_wall", [40, 190, 255, 45]),
    ]
    for wall_index, name, color in associated:
        if wall_index is None or wall_index >= len(walls):
            continue
        wall = walls[wall_index]
        wall_mesh = trimesh.Trimesh(
            vertices=wall.corners,
            faces=[[0, 1, 2], [0, 2, 3], [2, 1, 0], [3, 2, 0]],
            process=False,
        )
        wall_mesh.visual.face_colors = color
        scene.add_geometry(wall_mesh, geom_name=name)

    if target is not None:
        target_mesh = trimesh.creation.box(extents=target.box.extents)
        target_transform = np.eye(4)
        target_transform[:3, :3] = target.box.axes.T
        target_transform[:3, 3] = target.box.center
        target_mesh.apply_transform(target_transform)
        target_mesh.visual.face_colors = [40, 100, 255, 105]
        scene.add_geometry(target_mesh, geom_name="target_box")

    out = io.BytesIO()
    scene.export(out, file_type="glb")
    return out.getvalue()
