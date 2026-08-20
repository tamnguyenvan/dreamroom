"""Map 2D masks into MoGe space and fit a floor-aligned 3D box.

All 3D coordinates live in the MoGe glb camera frame: +X right, +Y up,
-Z forward, in meters (after reference-scale correction).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

UP = np.array([0.0, 1.0, 0.0])  # +Y is up in the MoGe camera frame
MIN_OBJECT_POINTS = 50


@dataclass
class FloorPlane:
    """A plane in MoGe space; ``normal`` is unit length and points up."""

    point: np.ndarray  # (3,) a point on the plane
    normal: np.ndarray  # (3,) unit normal
    inlier_ratio: float
    num_candidates: int
    fallback: bool = False

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        return (points - self.point) @ self.normal

    def to_dict(self) -> dict:
        return {
            "point": np.round(self.point, 4).tolist(),
            "normal": np.round(self.normal, 4).tolist(),
            "inlier_ratio": round(float(self.inlier_ratio), 4),
            "num_candidates": int(self.num_candidates),
            "fallback": bool(self.fallback),
        }


@dataclass
class Box3D:
    """Oriented 3D box; ``axes`` rows are unit vectors, axes[2] = floor normal."""

    center: np.ndarray  # (3,)
    axes: np.ndarray  # (3, 3) right-handed rows
    extents: np.ndarray  # (3,) full side lengths along the axes

    def corners(self) -> np.ndarray:
        """8 corners; indices 0-3 are the bottom face, 4-7 the top face."""

        half = self.extents / 2.0
        corners = []
        for up_sign in (-1.0, 1.0):
            for s1 in (-1.0, 1.0):
                for s2 in (-1.0, 1.0):
                    offset = s1 * half[0] * self.axes[0] + s2 * half[1] * self.axes[1]
                    corners.append(self.center + offset + up_sign * half[2] * self.axes[2])
        return np.array(corners)

    def base_center(self) -> np.ndarray:
        return self.center - self.axes[2] * (self.extents[2] / 2.0)

    def to_dict(self) -> dict:
        return {
            "center": np.round(self.center, 4).tolist(),
            "axes": np.round(self.axes, 4).tolist(),
            "extents": np.round(self.extents, 4).tolist(),
            "corners": np.round(self.corners(), 4).tolist(),
            "convention": "axes rows: [e1, e2, up]; corners 0-3 bottom, 4-7 top",
        }


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a boolean mask to the point-map resolution (nearest neighbor)."""

    resized = cv2.resize(
        mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    )
    return resized > 0


def _nearest_valid_point(
    point_map: np.ndarray, xy: list[float], max_radius: int = 6
) -> np.ndarray | None:
    """3D point at the pixel nearest to ``xy`` (point-map coords) that is valid."""

    height, width = point_map.shape[:2]
    cx, cy = int(round(xy[0])), int(round(xy[1]))
    for radius in range(max_radius + 1):
        x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
        window = point_map[y0:y1, x0:x1].reshape(-1, 3)
        ys, xs = np.mgrid[y0:y1, x0:x1]
        pixel_distance = (xs.reshape(-1) - cx) ** 2 + (ys.reshape(-1) - cy) ** 2
        valid = np.isfinite(window).all(axis=1)
        if valid.any():
            return window[valid][np.argmin(pixel_distance[valid])]
    return None


def calibrate_scale(
    point_map: np.ndarray,
    ref_start_xy: list[float],
    ref_end_xy: list[float],
    meters: float,
) -> tuple[float, dict]:
    """Correction factor from the reference line (point-map coordinates).

    Returns ``(factor, info)``; multiply the point map by ``factor`` so one
    unit equals one meter. Falls back to 1.0 with a warning in ``info``.
    """

    p1 = _nearest_valid_point(point_map, ref_start_xy)
    p2 = _nearest_valid_point(point_map, ref_end_xy)
    if p1 is None or p2 is None:
        logger.warning("reference endpoints fell on invalid points; no scale correction")
        return 1.0, {"applied": False, "reason": "invalid endpoint points"}
    moge_distance = float(np.linalg.norm(p1 - p2))
    if moge_distance < 1e-6:
        logger.warning("reference 3D distance is degenerate; no scale correction")
        return 1.0, {"applied": False, "reason": "degenerate distance"}
    factor = meters / moge_distance
    info = {
        "applied": True,
        "moge_distance": round(moge_distance, 4),
        "meters": meters,
        "factor": round(factor, 4),
        "point_start": np.round(p1, 4).tolist(),
        "point_end": np.round(p2, 4).tolist(),
    }
    logger.info("scale correction: %.3f moge-units -> %.3f m (x%.3f)", moge_distance, meters, factor)
    return factor, info



def extract_object_points(
    point_map: np.ndarray,
    mask_pm: np.ndarray,
    erode_px: int = 1,
    clip_percentile: float = 1.0,
) -> np.ndarray:
    """Valid 3D points of the masked object, lightly cleaned of edge noise."""

    mask = mask_pm
    if erode_px > 0:
        kernel = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel) > 0
        if eroded.sum() >= MIN_OBJECT_POINTS:
            mask = eroded
    points = point_map[mask]
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < MIN_OBJECT_POINTS:
        raise ValueError(f"too few valid object points: {len(points)}")
    radial = np.linalg.norm(points, axis=1)
    lo, hi = np.percentile(radial, [clip_percentile, 100.0 - clip_percentile])
    return points[(radial >= lo) & (radial <= hi)]


def floor_candidate_points(
    point_map: np.ndarray, mask_pm: np.ndarray, bottom_fraction: float = 0.5
) -> np.ndarray:
    """Non-object points from the bottom part of the image (floor candidates)."""

    height = point_map.shape[0]
    rows = np.zeros(height, dtype=bool)
    rows[int(height * (1.0 - bottom_fraction)):] = True
    candidates = rows[:, None] & ~mask_pm & np.isfinite(point_map).all(axis=2)
    return point_map[candidates]


def segmented_surface_points(
    point_map: np.ndarray,
    surface_mask: np.ndarray,
    exclude_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Valid point-map samples selected by a semantic surface mask."""

    if point_map.shape[:2] != surface_mask.shape:
        raise ValueError("point map and surface mask resolutions must match")
    keep = surface_mask.astype(bool) & np.isfinite(point_map).all(axis=2)
    if exclude_mask is not None:
        if exclude_mask.shape != surface_mask.shape:
            raise ValueError("surface and exclusion mask resolutions must match")
        keep &= ~exclude_mask.astype(bool)
    return point_map[keep]


def fit_floor_plane(
    candidates: np.ndarray,
    iterations: int = 2000,
    threshold: float = 0.02,
    min_inlier_ratio: float = 0.2,
    min_up_cos: float = 0.55,
    seed: int = 0,
) -> FloorPlane | None:
    """RANSAC floor plane; normal constrained to point roughly up (+Y)."""

    if len(candidates) < 100:
        logger.warning("too few floor candidates: %d", len(candidates))
        return None
    rng = np.random.default_rng(seed)
    best_inliers: np.ndarray | None = None
    for _ in range(iterations):
        sample = candidates[rng.choice(len(candidates), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        if normal @ UP < 0:
            normal = -normal
        if normal @ UP < min_up_cos:
            continue
        distances = np.abs((candidates - sample[0]) @ normal)
        inliers = distances < threshold
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    ratio = best_inliers.sum() / len(candidates) if best_inliers is not None else 0.0
    if best_inliers is None or ratio < min_inlier_ratio:
        logger.warning("floor RANSAC failed: inlier ratio %.3f", ratio)
        return None

    inlier_points = candidates[best_inliers]
    centroid = inlier_points.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_points - centroid, full_matrices=False)
    normal = vh[-1]
    if normal @ UP < 0:
        normal = -normal
    return FloorPlane(centroid, normal, ratio, len(candidates), fallback=False)


def fallback_floor_plane(object_points: np.ndarray) -> FloorPlane:
    """Camera-up plane through the object's lowest points (no floor visible)."""

    lowest = np.percentile(object_points[:, 1], 2.0)
    point = np.array([0.0, lowest, float(np.median(object_points[:, 2]))])
    return FloorPlane(point, UP.copy(), inlier_ratio=0.0, num_candidates=0, fallback=True)


def fit_box(
    object_points: np.ndarray, plane: FloorPlane, height_percentile: float = 99.0
) -> Box3D:
    """Floor-aligned box: min-area footprint on the plane + robust height."""

    normal = plane.normal
    forward = np.array([0.0, 0.0, -1.0])
    t1 = np.cross(normal, forward)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)  # right-handed: t1 x t2 = normal

    relative = object_points - plane.point
    heights = relative @ normal
    top = np.percentile(heights, height_percentile)
    keep = heights <= top  # drop flying outliers above the object
    uv = np.stack([relative[keep] @ t1, relative[keep] @ t2], axis=1).astype(np.float32)

    rect = cv2.minAreaRect(uv)
    corners2d = cv2.boxPoints(rect)  # (4, 2), ordered around the rectangle
    e1 = corners2d[1] - corners2d[0]
    e2 = corners2d[3] - corners2d[0]
    len1, len2 = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
    if len1 < 1e-6 or len2 < 1e-6:
        raise ValueError("degenerate object footprint")
    e1, e2 = e1 / len1, e2 / len2
    if e1[0] * e2[1] - e1[1] * e2[0] < 0:  # enforce right-handedness (e1 x e2 = +up)
        e1, e2, len1, len2 = e2, e1, len2, len1

    axis1 = e1[0] * t1 + e1[1] * t2
    axis2 = e2[0] * t1 + e2[1] * t2
    center2d = corners2d.mean(axis=0)
    center = plane.point + center2d[0] * t1 + center2d[1] * t2 + normal * (top / 2.0)
    return Box3D(
        center=center,
        axes=np.stack([axis1, axis2, normal]),
        extents=np.array([len1, len2, top]),
    )
