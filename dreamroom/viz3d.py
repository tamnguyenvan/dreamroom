"""Debug visualization for fitted floor, box, and wall geometry.

Projection uses the MoGe normalized intrinsics at the point-map resolution;
pixels are then scaled up to the working-image resolution (which is larger,
because the API clamps inputs to max-side 800 while the pipeline uses 1280).
"""

from __future__ import annotations

import io
import logging

import cv2
import numpy as np

from .geometry3d import Box3D, FloorPlane
from .ui.window import draw_banner
from .wall_geometry import WallPlane

logger = logging.getLogger(__name__)

BOX_COLOR = (0, 220, 230)  # yellow (BGR)
BOX_TOP_COLOR = (230, 160, 0)  # orange-ish top face
FLOOR_COLOR = (180, 80, 180)  # purple
MASK_COLOR = (0, 200, 0)  # green
WALL_COLORS = [
    (255, 120, 40),
    (255, 60, 180),
    (80, 180, 255),
    (200, 120, 255),
    (255, 200, 80),
    (120, 255, 180),
]

# corner order from Box3D.corners(): 0-3 bottom face, 4-7 top face
BOX_EDGES = [
    (0, 1), (1, 3), (3, 2), (2, 0),  # bottom
    (4, 5), (5, 7), (7, 6), (6, 4),  # top
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
]


def intrinsics_px(metadata: dict, width: int, height: int) -> np.ndarray:
    """Denormalize MoGe intrinsics to pixel units at (width, height)."""

    k_norm = np.asarray(metadata["intrinsics"], dtype=np.float64)
    return np.array(
        [
            [k_norm[0, 0] * width, 0.0, 0.5 * width],
            [0.0, k_norm[1, 1] * height, 0.5 * height],
            [0.0, 0.0, 1.0],
        ]
    )


def project_points(points3d: np.ndarray, k_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project MoGe-space points to pixels; returns (pixels, valid_mask)."""

    depth = -points3d[:, 2]  # -Z is forward
    valid = depth > 1e-6
    safe_depth = np.where(valid, depth, 1.0)
    u = k_px[0, 0] * (points3d[:, 0] / safe_depth) + k_px[0, 2]
    v = k_px[1, 2] - k_px[1, 1] * (points3d[:, 1] / safe_depth)
    return np.stack([u, v], axis=1), valid


def floor_quad(box: Box3D, size_factor: float = 1.8, min_size: float = 1.0) -> np.ndarray:
    """A square on the floor plane around the box base (MoGe coords)."""

    base = box.base_center()
    half = max(min_size, size_factor * max(box.extents[0], box.extents[1])) / 2.0
    a1, a2 = box.axes[0], box.axes[1]
    return np.array(
        [
            base - half * a1 - half * a2,
            base + half * a1 - half * a2,
            base + half * a1 + half * a2,
            base - half * a1 + half * a2,
        ]
    )


def draw_debug_2d(
    image_bgr: np.ndarray,
    box: Box3D,
    plane: FloorPlane,
    k_px: np.ndarray,
    pixel_scale: tuple[float, float],
    mask: np.ndarray | None = None,
    walls: list[WallPlane] | None = None,
) -> np.ndarray:
    """Overlay the fitted box, floor plane, and optional walls.

    ``pixel_scale`` = (image_w / pm_w, image_h / pm_h) maps point-map pixels
    back to working-image pixels.
    """

    frame = image_bgr.copy()
    sx, sy = pixel_scale

    def to_image_pixels(points3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pixels, valid = project_points(points3d, k_px)
        pixels = pixels * np.array([sx, sy])
        return pixels, valid

    # mask contour for context
    if mask is not None:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(frame, contours, -1, MASK_COLOR, 1, cv2.LINE_AA)

    # floor quad (translucent fill)
    quad_pixels, quad_valid = to_image_pixels(floor_quad(box))
    if quad_valid.all():
        polygon = quad_pixels.astype(np.int32).reshape(-1, 1, 2)
        fill = frame.copy()
        cv2.fillPoly(fill, [polygon], FLOOR_COLOR)
        frame = cv2.addWeighted(fill, 0.25, frame, 0.75, 0)
        cv2.polylines(frame, [polygon], True, FLOOR_COLOR, 2, cv2.LINE_AA)

    # finite wall patches (debug mode only at the pipeline call site)
    for index, wall in enumerate(walls or []):
        wall_pixels, wall_valid = to_image_pixels(wall.corners)
        if not wall_valid.all():
            continue
        color = WALL_COLORS[index % len(WALL_COLORS)]
        polygon = np.round(wall_pixels).astype(np.int32).reshape(-1, 1, 2)
        fill = frame.copy()
        cv2.fillPoly(fill, [polygon], color)
        frame = cv2.addWeighted(fill, 0.14, frame, 0.86, 0)
        cv2.polylines(frame, [polygon], True, color, 2, cv2.LINE_AA)
        label_at = tuple(np.round(wall_pixels.mean(axis=0)).astype(int))
        cv2.putText(
            frame,
            f"W{index + 1} {wall.confidence:.2f}",
            label_at,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    # box edges
    corners, valid = to_image_pixels(box.corners())
    for start, end in BOX_EDGES:
        if valid[start] and valid[end]:
            color = BOX_TOP_COLOR if start >= 4 and end >= 4 else BOX_COLOR
            cv2.line(
                frame,
                tuple(np.round(corners[start]).astype(int)),
                tuple(np.round(corners[end]).astype(int)),
                color,
                2,
                cv2.LINE_AA,
            )
    for point, ok in zip(corners, valid):
        if ok:
            cv2.circle(frame, tuple(np.round(point).astype(int)), 3, BOX_COLOR, -1)

    lines = [
        f"box (m): {box.extents[0]:.2f} x {box.extents[1]:.2f} x {box.extents[2]:.2f}",
        f"floor inliers: {plane.inlier_ratio * 100:.0f}%{'  [FALLBACK plane]' if plane.fallback else ''}",
    ]
    if walls is not None:
        lines.append(f"walls: {len(walls)}")
    return draw_banner(frame, lines)


def export_debug_glb(
    glb_bytes: bytes | None,
    box: Box3D,
    plane: FloorPlane,
    scale_factor: float = 1.0,
    walls: list[WallPlane] | None = None,
) -> bytes | None:
    """Merge the fitted box, floor plane, and walls into the MoGe GLB scene.

    ``scale_factor`` converts the native MoGe scene to the calibrated metric
    coordinates already used by ``box`` and ``plane``.
    Returns ``None`` when trimesh is unavailable or no GLB was returned.
    """

    if glb_bytes is None:
        logger.warning("no MoGe GLB available; skipping 3D debug export")
        return None
    try:
        import trimesh
    except ImportError:
        logger.warning("trimesh not installed; skipping 3D debug export")
        return None

    scene = trimesh.load(io.BytesIO(glb_bytes), file_type="glb")
    scene.apply_scale(scale_factor)

    box_mesh = trimesh.creation.box(extents=box.extents)
    transform = np.eye(4)
    transform[:3, :3] = box.axes.T  # axes are rows; columns are the local frame
    transform[:3, 3] = box.center
    box_mesh.apply_transform(transform)
    box_mesh.visual.face_colors = [230, 60, 60, 130]
    scene.add_geometry(box_mesh, geom_name="fitted_box")

    quad = floor_quad(box)
    quad_mesh = trimesh.Trimesh(
        vertices=quad,
        faces=[[0, 1, 2], [0, 2, 3], [2, 1, 0], [3, 2, 0]],  # both windings
        process=False,
    )
    quad_mesh.visual.face_colors = [80, 200, 80, 90]
    scene.add_geometry(quad_mesh, geom_name="floor_plane")

    wall_colors = [
        [40, 130, 255, 100],
        [220, 60, 220, 100],
        [255, 170, 40, 100],
        [160, 80, 255, 100],
        [50, 210, 210, 100],
        [80, 220, 130, 100],
    ]
    for index, wall in enumerate(walls or []):
        wall_mesh = trimesh.Trimesh(
            vertices=wall.corners,
            faces=[[0, 1, 2], [0, 2, 3], [2, 1, 0], [3, 2, 0]],
            process=False,
        )
        wall_mesh.visual.face_colors = wall_colors[index % len(wall_colors)]
        scene.add_geometry(wall_mesh, geom_name=f"wall_plane_{index + 1:02d}")

    out = io.BytesIO()
    scene.export(out, file_type="glb")
    return out.getvalue()
