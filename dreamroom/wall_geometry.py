"""Global wall-plane fitting constrained by an already fitted floor plane."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .geometry3d import FloorPlane

logger = logging.getLogger(__name__)

MIN_WALL_CANDIDATES = 100


@dataclass
class WallPlane:
    """A finite vertical wall patch in calibrated MoGe coordinates."""

    point: np.ndarray  # (3,) point on the wall-floor intersection
    normal: np.ndarray  # (3,) horizontal unit normal, facing the camera
    corners: np.ndarray  # (4, 3): bottom-start, bottom-end, top-end, top-start
    inlier_count: int
    num_candidates: int
    rmse: float
    width: float
    height: float
    confidence: float
    connected_support_ratio: float = 1.0
    image_fill_ratio: float = 1.0
    occupied_image_cells: int = 0

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        return (points - self.point) @ self.normal

    def to_dict(self) -> dict:
        offset = -float(self.normal @ self.point)
        return {
            "point": np.round(self.point, 4).tolist(),
            "normal": np.round(self.normal, 4).tolist(),
            "equation": np.round(np.append(self.normal, offset), 4).tolist(),
            "corners": np.round(self.corners, 4).tolist(),
            "inlier_count": int(self.inlier_count),
            "num_candidates": int(self.num_candidates),
            "inlier_ratio": round(self.inlier_count / max(self.num_candidates, 1), 4),
            "rmse": round(float(self.rmse), 4),
            "width": round(float(self.width), 4),
            "height": round(float(self.height), 4),
            "confidence": round(float(self.confidence), 4),
            "connected_support_ratio": round(
                float(self.connected_support_ratio), 4
            ),
            "image_fill_ratio": round(float(self.image_fill_ratio), 4),
            "occupied_image_cells": int(self.occupied_image_cells),
            "convention": "corners: bottom-start, bottom-end, top-end, top-start",
        }


@dataclass
class _WallCandidates:
    points: np.ndarray
    floor_uv: np.ndarray
    heights: np.ndarray
    normal_uv: np.ndarray
    pixels: np.ndarray
    floor_axis_u: np.ndarray
    floor_axis_v: np.ndarray

    def subset(self, keep: np.ndarray) -> "_WallCandidates":
        return _WallCandidates(
            points=self.points[keep],
            floor_uv=self.floor_uv[keep],
            heights=self.heights[keep],
            normal_uv=self.normal_uv[keep],
            pixels=self.pixels[keep],
            floor_axis_u=self.floor_axis_u,
            floor_axis_v=self.floor_axis_v,
        )

    def __len__(self) -> int:
        return len(self.points)


def floor_basis(plane: FloorPlane) -> tuple[np.ndarray, np.ndarray]:
    """Return orthonormal horizontal axes whose cross product is floor-up."""

    forward = np.array([0.0, 0.0, -1.0])
    axis_v = forward - float(forward @ plane.normal) * plane.normal
    if np.linalg.norm(axis_v) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
        axis_v = right - float(right @ plane.normal) * plane.normal
    axis_v /= np.linalg.norm(axis_v)
    axis_u = np.cross(axis_v, plane.normal)
    axis_u /= np.linalg.norm(axis_u)
    return axis_u, axis_v


def estimate_point_normals(
    point_map: np.ndarray,
    radius: int = 2,
    base_max_neighbor_gap: float = 0.08,
    depth_gap_factor: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate local normals while rejecting depth-discontinuity neighborhoods."""

    height, width = point_map.shape[:2]
    normals = np.full((height, width, 3), np.nan, dtype=np.float64)
    normal_valid = np.zeros((height, width), dtype=bool)
    if height <= 2 * radius or width <= 2 * radius:
        return normals, normal_valid

    center = point_map[radius:-radius, radius:-radius].astype(np.float64)
    left = point_map[radius:-radius, :-2 * radius].astype(np.float64)
    right = point_map[radius:-radius, 2 * radius:].astype(np.float64)
    top = point_map[:-2 * radius, radius:-radius].astype(np.float64)
    bottom = point_map[2 * radius:, radius:-radius].astype(np.float64)
    neighborhood = (center, left, right, top, bottom)
    valid = np.logical_and.reduce([np.isfinite(item).all(axis=2) for item in neighborhood])

    tangent_x = right - left
    tangent_y = bottom - top
    local_normals = np.cross(tangent_x, tangent_y)
    lengths = np.linalg.norm(local_normals, axis=2)
    radial = np.linalg.norm(center, axis=2)
    gap_limit = base_max_neighbor_gap + depth_gap_factor * radial
    gaps = np.maximum.reduce(
        [np.linalg.norm(item - center, axis=2) for item in (left, right, top, bottom)]
    )
    valid &= (lengths > 1e-9) & (gaps < gap_limit)
    local_normals[valid] /= lengths[valid][:, None]

    normals[radius:-radius, radius:-radius][valid] = local_normals[valid]
    normal_valid[radius:-radius, radius:-radius] = valid
    return normals, normal_valid


def _wall_candidates(
    point_map: np.ndarray,
    object_mask: np.ndarray,
    floor: FloorPlane,
    mask_dilation_px: int,
    min_height: float,
    max_height: float,
    max_floor_normal_dot: float,
    sample_stride: int,
    max_candidates: int,
) -> _WallCandidates | None:
    normals, normal_valid = estimate_point_normals(point_map)
    finite = np.isfinite(point_map).all(axis=2)
    heights = np.full(point_map.shape[:2], np.nan, dtype=np.float64)
    heights[finite] = floor.signed_distance(point_map[finite])

    kernel_size = 2 * mask_dilation_px + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    excluded_object = cv2.dilate(object_mask.astype(np.uint8), kernel) > 0
    wall_like = np.zeros(point_map.shape[:2], dtype=bool)
    wall_like[normal_valid] = (
        np.abs(normals[normal_valid] @ floor.normal) <= max_floor_normal_dot
    )
    keep = (
        finite
        & normal_valid
        & wall_like
        & ~excluded_object
        & (heights >= min_height)
        & (heights <= max_height)
    )

    grid = np.zeros_like(keep)
    grid[::sample_stride, ::sample_stride] = True
    keep &= grid
    rows, cols = np.nonzero(keep)
    if len(rows) < MIN_WALL_CANDIDATES:
        logger.warning("too few vertical wall candidates: %d", len(rows))
        return None
    if len(rows) > max_candidates:
        selected = np.linspace(0, len(rows) - 1, max_candidates, dtype=np.int64)
        rows, cols = rows[selected], cols[selected]

    points = point_map[rows, cols].astype(np.float64)
    point_normals = normals[rows, cols]
    axis_u, axis_v = floor_basis(floor)
    relative = points - floor.point
    floor_uv = np.column_stack((relative @ axis_u, relative @ axis_v))
    normal_uv = np.column_stack((point_normals @ axis_u, point_normals @ axis_v))
    normal_uv /= np.linalg.norm(normal_uv, axis=1, keepdims=True)
    return _WallCandidates(
        points=points,
        floor_uv=floor_uv,
        heights=relative @ floor.normal,
        normal_uv=normal_uv,
        pixels=np.column_stack((cols, rows)),
        floor_axis_u=axis_u,
        floor_axis_v=axis_v,
    )


def _segmented_wall_candidates(
    point_map: np.ndarray,
    wall_mask: np.ndarray,
    object_mask: np.ndarray,
    floor: FloorPlane,
    *,
    mask_dilation_px: int,
    min_height: float,
    max_height: float,
    sample_stride: int,
    max_candidates: int,
) -> _WallCandidates | None:
    """Project SAM-selected wall points into the fitted floor frame."""

    finite = np.isfinite(point_map).all(axis=2)
    heights = np.full(point_map.shape[:2], np.nan, dtype=np.float64)
    heights[finite] = floor.signed_distance(point_map[finite])
    kernel_size = 2 * mask_dilation_px + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    excluded_object = cv2.dilate(object_mask.astype(np.uint8), kernel) > 0
    keep = (
        wall_mask.astype(bool)
        & finite
        & ~excluded_object
        & (heights >= min_height)
        & (heights <= max_height)
    )
    grid = np.zeros_like(keep)
    grid[::sample_stride, ::sample_stride] = True
    keep &= grid
    rows, cols = np.nonzero(keep)
    if len(rows) < MIN_WALL_CANDIDATES:
        return None
    if len(rows) > max_candidates:
        selected = np.linspace(0, len(rows) - 1, max_candidates, dtype=np.int64)
        rows, cols = rows[selected], cols[selected]

    points = point_map[rows, cols].astype(np.float64)
    axis_u, axis_v = floor_basis(floor)
    relative = points - floor.point
    floor_uv = np.column_stack((relative @ axis_u, relative @ axis_v))
    return _WallCandidates(
        points=points,
        floor_uv=floor_uv,
        heights=relative @ floor.normal,
        normal_uv=np.zeros((len(points), 2), dtype=np.float64),
        pixels=np.column_stack((cols, rows)),
        floor_axis_u=axis_u,
        floor_axis_v=axis_v,
    )


def _line_inliers(
    candidates: _WallCandidates,
    normal: np.ndarray,
    offset: float,
    threshold: float,
    min_normal_cos: float,
) -> np.ndarray:
    distances = np.abs(candidates.floor_uv @ normal + offset)
    normal_agreement = np.abs(candidates.normal_uv @ normal)
    return (distances <= threshold) & (normal_agreement >= min_normal_cos)


def _ransac_line(
    candidates: _WallCandidates,
    iterations: int,
    threshold: float,
    min_normal_cos: float,
    min_pair_distance: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    best_inliers: np.ndarray | None = None
    best_normal: np.ndarray | None = None
    best_offset = 0.0
    for _ in range(iterations):
        first, second = rng.choice(len(candidates), size=2, replace=False)
        tangent = candidates.floor_uv[second] - candidates.floor_uv[first]
        length = np.linalg.norm(tangent)
        if length < min_pair_distance:
            continue
        tangent /= length
        normal = np.array([-tangent[1], tangent[0]])
        offset = -float(normal @ candidates.floor_uv[first])
        inliers = _line_inliers(
            candidates, normal, offset, threshold, min_normal_cos
        )
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_normal = normal
            best_offset = offset
    if best_inliers is None or best_normal is None:
        return None
    return best_normal, best_offset, best_inliers


def _ransac_segment_line(
    candidates: _WallCandidates,
    iterations: int,
    threshold: float,
    min_pair_distance: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """RANSAC a floor-frame line without requiring estimated point normals."""

    best_inliers: np.ndarray | None = None
    best_normal: np.ndarray | None = None
    best_offset = 0.0
    for _ in range(iterations):
        first, second = rng.choice(len(candidates), size=2, replace=False)
        tangent = candidates.floor_uv[second] - candidates.floor_uv[first]
        length = np.linalg.norm(tangent)
        if length < min_pair_distance:
            continue
        tangent /= length
        normal = np.array([-tangent[1], tangent[0]])
        offset = -float(normal @ candidates.floor_uv[first])
        inliers = np.abs(candidates.floor_uv @ normal + offset) <= threshold
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_normal = normal
            best_offset = offset
    if best_inliers is None or best_normal is None:
        return None
    return best_normal, best_offset, best_inliers


def _refine_segment_line(
    candidates: _WallCandidates,
    initial_inliers: np.ndarray,
    initial_normal: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    inliers = initial_inliers
    normal = initial_normal
    offset = 0.0
    for _ in range(3):
        points = candidates.floor_uv[inliers]
        if len(points) < 2:
            break
        centroid = points.mean(axis=0)
        _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
        refined = vh[-1]
        if refined @ normal < 0:
            refined = -refined
        normal = refined / np.linalg.norm(refined)
        offset = -float(normal @ centroid)
        refined_inliers = (
            np.abs(candidates.floor_uv @ normal + offset) <= threshold
        )
        if refined_inliers.sum() < 2:
            break
        inliers = refined_inliers
    return normal, offset, inliers


def _refine_line(
    candidates: _WallCandidates,
    initial_inliers: np.ndarray,
    initial_normal: np.ndarray,
    initial_offset: float,
    threshold: float,
    min_normal_cos: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    inliers = initial_inliers
    normal = initial_normal
    offset = initial_offset
    for _ in range(3):
        if inliers.sum() < 2:
            break
        points = candidates.floor_uv[inliers]
        centroid = points.mean(axis=0)
        _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
        refined = vh[-1]
        if refined @ normal < 0:
            refined = -refined
        refined /= np.linalg.norm(refined)
        refined_offset = -float(refined @ centroid)
        refined_inliers = _line_inliers(
            candidates, refined, refined_offset, threshold, min_normal_cos
        )
        if refined_inliers.sum() < 2:
            break
        normal, offset, inliers = refined, refined_offset, refined_inliers
    return normal, offset, inliers


def _make_wall(
    candidates: _WallCandidates,
    floor: FloorPlane,
    normal_2d: np.ndarray,
    offset: float,
    inliers: np.ndarray,
    threshold: float,
    num_candidates: int,
    image_cell_size: int,
) -> WallPlane:
    axis_u, axis_v = candidates.floor_axis_u, candidates.floor_axis_v
    tangent_2d = np.array([-normal_2d[1], normal_2d[0]])
    along = candidates.floor_uv[inliers] @ tangent_2d
    start, end = np.percentile(along, [2.0, 98.0])
    top = float(np.percentile(candidates.heights[inliers], 98.0))
    closest = -offset * normal_2d
    start_uv = closest + start * tangent_2d
    end_uv = closest + end * tangent_2d
    base_start = floor.point + start_uv[0] * axis_u + start_uv[1] * axis_v
    base_end = floor.point + end_uv[0] * axis_u + end_uv[1] * axis_v
    wall_normal = normal_2d[0] * axis_u + normal_2d[1] * axis_v
    midpoint = 0.5 * (base_start + base_end)
    if wall_normal @ -midpoint < 0:
        wall_normal = -wall_normal

    distances = np.abs(candidates.floor_uv[inliers] @ normal_2d + offset)
    rmse = float(np.sqrt(np.mean(distances**2)))
    width = float(end - start)
    height_span = float(
        np.percentile(candidates.heights[inliers], 95.0)
        - np.percentile(candidates.heights[inliers], 5.0)
    )
    count_support = min(1.0, inliers.sum() / 800.0)
    global_support = min(1.0, inliers.sum() / max(0.1 * num_candidates, 1.0))
    support = float(np.sqrt(count_support * global_support))
    coverage = min(1.0, width / 1.5) * min(1.0, height_span / 1.5)
    residual_quality = float(np.exp(-rmse / max(threshold, 1e-6)))
    confidence = float(np.clip(support * coverage * residual_quality, 0.0, 1.0))
    connected_ratio, fill_ratio, occupied_cells = _image_support_metrics(
        candidates.pixels[inliers], image_cell_size
    )
    corners = np.array(
        [
            base_start,
            base_end,
            base_end + top * floor.normal,
            base_start + top * floor.normal,
        ]
    )
    return WallPlane(
        point=midpoint,
        normal=wall_normal,
        corners=corners,
        inlier_count=int(inliers.sum()),
        num_candidates=num_candidates,
        rmse=rmse,
        width=width,
        height=top,
        confidence=confidence,
        connected_support_ratio=connected_ratio,
        image_fill_ratio=fill_ratio,
        occupied_image_cells=occupied_cells,
    )


def _image_support_metrics(
    pixels: np.ndarray, cell_size: int
) -> tuple[float, float, int]:
    """Connected support and occupancy of inliers on a coarse image grid."""

    if len(pixels) == 0:
        return 0.0, 0.0, 0
    cells = np.floor_divide(pixels, cell_size).astype(np.int32)
    unique_cells, counts = np.unique(cells, axis=0, return_counts=True)
    cell_lookup = {tuple(cell): index for index, cell in enumerate(unique_cells)}
    visited: set[int] = set()
    largest_support = 0
    for start in range(len(unique_cells)):
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        component_support = 0
        while stack:
            current = stack.pop()
            component_support += int(counts[current])
            x, y = unique_cells[current]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = cell_lookup.get((x + dx, y + dy))
                    if neighbor is not None and neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
        largest_support = max(largest_support, component_support)

    lower = unique_cells.min(axis=0)
    upper = unique_cells.max(axis=0)
    bounding_area = int(np.prod(upper - lower + 1))
    connected_ratio = largest_support / len(pixels)
    fill_ratio = len(unique_cells) / max(bounding_area, 1)
    return float(connected_ratio), float(fill_ratio), len(unique_cells)


def _is_duplicate(first: WallPlane, second: WallPlane) -> bool:
    angle_cos = abs(float(first.normal @ second.normal))
    offset = abs(float((second.point - first.point) @ first.normal))
    return angle_cos >= np.cos(np.deg2rad(5.0)) and offset <= 0.08


def filter_wall_planes(
    walls: list[WallPlane],
    *,
    min_confidence: float = 0.5,
    min_relative_height: float = 0.7,
    parallel_angle_degrees: float = 5.0,
    parallel_offset: float = 0.5,
    weak_parallel_support_ratio: float = 0.5,
) -> list[WallPlane]:
    """Reject weak partial walls and secondary layers near stronger walls."""

    if not walls:
        return []
    confidence_walls = [
        wall for wall in walls if wall.confidence >= min_confidence
    ]
    if not confidence_walls:
        return []
    tallest = max(wall.height for wall in confidence_walls)
    quality_walls = [
        wall
        for wall in confidence_walls
        if wall.height >= min_relative_height * tallest
    ]
    strongest_first = sorted(
        quality_walls, key=lambda wall: wall.inlier_count, reverse=True
    )
    kept: list[WallPlane] = []
    min_parallel_cos = float(np.cos(np.deg2rad(parallel_angle_degrees)))
    for wall in strongest_first:
        shadowed = False
        for stronger in kept:
            parallel = abs(float(wall.normal @ stronger.normal)) >= min_parallel_cos
            offset = abs(float((wall.point - stronger.point) @ stronger.normal))
            much_weaker = (
                wall.inlier_count
                < weak_parallel_support_ratio * stronger.inlier_count
            )
            if parallel and offset <= parallel_offset and much_weaker:
                shadowed = True
                break
        if not shadowed:
            kept.append(wall)
    return sorted(
        kept, key=lambda wall: (wall.confidence, wall.inlier_count), reverse=True
    )


def fit_wall_planes(
    point_map: np.ndarray,
    object_mask: np.ndarray,
    floor: FloorPlane,
    *,
    max_walls: int = 6,
    iterations: int = 600,
    distance_threshold: float = 0.05,
    min_inliers: int = 100,
    min_inlier_ratio: float = 0.01,
    min_width: float = 0.75,
    min_height_span: float = 0.60,
    min_connected_support_ratio: float = 0.25,
    min_image_fill_ratio: float = 0.10,
    min_height: float = 0.05,
    max_height: float = 4.5,
    max_floor_normal_dot: float = 0.35,
    max_normal_angle_degrees: float = 35.0,
    mask_dilation_px: int = 5,
    sample_stride: int = 2,
    max_candidates: int = 25_000,
    image_cell_size: int = 16,
    min_confidence: float = 0.5,
    min_relative_height: float = 0.7,
    parallel_shadow_offset: float = 0.5,
    weak_parallel_support_ratio: float = 0.5,
    seed: int = 0,
) -> list[WallPlane]:
    """Fit multiple global walls as vertical planes in the floor frame."""

    if point_map.shape[:2] != object_mask.shape:
        raise ValueError("point map and object mask resolutions must match")
    candidates = _wall_candidates(
        point_map,
        object_mask,
        floor,
        mask_dilation_px,
        min_height,
        max_height,
        max_floor_normal_dot,
        sample_stride,
        max_candidates,
    )
    if candidates is None:
        return []

    rng = np.random.default_rng(seed)
    min_normal_cos = float(np.cos(np.deg2rad(max_normal_angle_degrees)))
    required_inliers = max(
        min_inliers,
        int(np.ceil(min_inlier_ratio * len(candidates))),
    )
    remaining = np.ones(len(candidates), dtype=bool)
    walls: list[WallPlane] = []
    max_models = max_walls * 3
    for _ in range(max_models):
        remaining_indices = np.flatnonzero(remaining)
        if len(remaining_indices) < max(required_inliers, MIN_WALL_CANDIDATES):
            break
        active = candidates.subset(remaining_indices)
        fit = _ransac_line(
            active,
            iterations,
            distance_threshold,
            min_normal_cos,
            min_pair_distance=0.25,
            rng=rng,
        )
        if fit is None:
            break
        normal, initial_offset, initial_inliers = fit
        if initial_inliers.sum() < required_inliers:
            break
        normal, offset, inliers = _refine_line(
            active,
            initial_inliers,
            normal,
            initial_offset,
            distance_threshold,
            min_normal_cos,
        )
        if inliers.sum() < required_inliers:
            break

        global_inliers = remaining_indices[inliers]
        remaining[global_inliers] = False
        wall = _make_wall(
            active,
            floor,
            normal,
            offset,
            inliers,
            distance_threshold,
            len(candidates),
            image_cell_size,
        )
        wall_heights = active.heights[inliers]
        height_span = float(
            np.percentile(wall_heights, 95.0)
            - np.percentile(wall_heights, 5.0)
        )
        if (
            wall.width < min_width
            or height_span < min_height_span
            or wall.connected_support_ratio < min_connected_support_ratio
            or wall.image_fill_ratio < min_image_fill_ratio
        ):
            continue
        duplicate_index = next(
            (index for index, item in enumerate(walls) if _is_duplicate(item, wall)),
            None,
        )
        if duplicate_index is not None:
            if wall.inlier_count > walls[duplicate_index].inlier_count:
                walls[duplicate_index] = wall
            continue
        walls.append(wall)
        if len(walls) >= max_walls:
            break

    walls = filter_wall_planes(
        walls,
        min_confidence=min_confidence,
        min_relative_height=min_relative_height,
        parallel_offset=parallel_shadow_offset,
        weak_parallel_support_ratio=weak_parallel_support_ratio,
    )
    logger.info("fitted %d wall plane(s) from %d candidates", len(walls), len(candidates))
    return walls


def fit_segmented_wall_planes(
    point_map: np.ndarray,
    wall_masks: list[np.ndarray],
    object_mask: np.ndarray,
    floor: FloorPlane,
    *,
    max_walls: int = 6,
    max_planes_per_mask: int = 2,
    iterations: int = 500,
    distance_threshold: float = 0.05,
    min_inliers: int = 80,
    min_inlier_ratio: float = 0.08,
    min_width: float = 0.75,
    min_height_span: float = 0.60,
    min_connected_support_ratio: float = 0.25,
    min_image_fill_ratio: float = 0.08,
    min_height: float = 0.05,
    max_height: float = 4.5,
    mask_dilation_px: int = 3,
    sample_stride: int = 2,
    max_candidates_per_mask: int = 20_000,
    image_cell_size: int = 16,
    min_confidence: float = 0.35,
    min_relative_height: float = 0.6,
    seed: int = 0,
) -> list[WallPlane]:
    """Fit vertical planes only from SAM-selected wall pixels."""

    if point_map.shape[:2] != object_mask.shape:
        raise ValueError("point map and object mask resolutions must match")
    if any(mask.shape != object_mask.shape for mask in wall_masks):
        raise ValueError("point map and wall mask resolutions must match")

    rng = np.random.default_rng(seed)
    walls: list[WallPlane] = []
    for wall_mask in wall_masks:
        candidates = _segmented_wall_candidates(
            point_map,
            wall_mask,
            object_mask,
            floor,
            mask_dilation_px=mask_dilation_px,
            min_height=min_height,
            max_height=max_height,
            sample_stride=sample_stride,
            max_candidates=max_candidates_per_mask,
        )
        if candidates is None:
            continue
        required_inliers = max(
            min_inliers,
            int(np.ceil(min_inlier_ratio * len(candidates))),
        )
        remaining = np.ones(len(candidates), dtype=bool)
        for _ in range(max_planes_per_mask):
            remaining_indices = np.flatnonzero(remaining)
            if len(remaining_indices) < max(required_inliers, MIN_WALL_CANDIDATES):
                break
            active = candidates.subset(remaining_indices)
            fit = _ransac_segment_line(
                active,
                iterations,
                distance_threshold,
                min_pair_distance=0.25,
                rng=rng,
            )
            if fit is None:
                break
            normal, _, initial_inliers = fit
            if initial_inliers.sum() < required_inliers:
                break
            normal, offset, inliers = _refine_segment_line(
                active,
                initial_inliers,
                normal,
                distance_threshold,
            )
            if inliers.sum() < required_inliers:
                break
            remaining[remaining_indices[inliers]] = False
            wall = _make_wall(
                active,
                floor,
                normal,
                offset,
                inliers,
                distance_threshold,
                len(candidates),
                image_cell_size,
            )
            wall_heights = active.heights[inliers]
            height_span = float(
                np.percentile(wall_heights, 95.0)
                - np.percentile(wall_heights, 5.0)
            )
            if (
                wall.width < min_width
                or height_span < min_height_span
                or wall.connected_support_ratio < min_connected_support_ratio
                or wall.image_fill_ratio < min_image_fill_ratio
            ):
                continue
            duplicate_index = next(
                (index for index, item in enumerate(walls) if _is_duplicate(item, wall)),
                None,
            )
            if duplicate_index is not None:
                if wall.inlier_count > walls[duplicate_index].inlier_count:
                    walls[duplicate_index] = wall
                continue
            walls.append(wall)
            if len(walls) >= max_walls:
                break
        if len(walls) >= max_walls:
            break

    walls = filter_wall_planes(
        walls,
        min_confidence=min_confidence,
        min_relative_height=min_relative_height,
    )
    logger.info(
        "fitted %d wall plane(s) from %d SAM mask(s)", len(walls), len(wall_masks)
    )
    return walls
