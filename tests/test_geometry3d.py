"""Tests for 3D geometry and the MoGe client ZIP parsing.

A synthetic room (floor, back wall, 1x1x1 m box on the floor) is rendered
into a point map with a pinhole camera in the MoGe convention
(+X right, +Y up, -Z forward), so the fitted plane/box can be compared
against ground truth.
"""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

from dreamroom.geometry3d import (
    Box3D,
    calculate_aspect_ratio_calibration,
    FloorPlane,
    calibrate_scale,
    extract_object_points,
    fallback_floor_plane,
    fit_box,
    fit_floor_plane,
    floor_candidate_points,
    resize_mask,
    segmented_surface_points,
    target_dimensions_in_moge_units,
)
from dreamroom.moge_client import DEFAULT_ENDPOINT, MogeClient
from dreamroom.viz3d import export_debug_glb, intrinsics_px, project_points
from dreamroom.wall_geometry import (
    WallPlane,
    filter_wall_planes,
    fit_segmented_wall_planes,
    fit_wall_planes,
)

WIDTH, HEIGHT = 800, 600
FX = FY = 600.0
FLOOR_Y, WALL_Z = -1.5, -6.0
BOX = dict(x0=-0.5, x1=0.5, y0=-1.5, y1=-0.5, z0=-3.5, z1=-2.5)  # 1 m cube


def synthetic_room(noise_std: float = 0.0, seed: int = 0):
    """Return (point_map, object_mask, K) of a synthetic room."""

    us, vs = np.meshgrid(np.arange(WIDTH, dtype=float), np.arange(HEIGHT, dtype=float))
    cx, cy = WIDTH / 2, HEIGHT / 2
    dirs = np.stack([(us - cx) / FX, -(vs - cy) / FY, -np.ones_like(us)], axis=-1)
    dx, dy, dz = dirs[..., 0], dirs[..., 1], dirs[..., 2]

    best_t = np.full((HEIGHT, WIDTH), np.inf)
    kind = np.zeros((HEIGHT, WIDTH), dtype=np.int8)  # 1 floor, 2 wall, 3 box

    def update(t, valid, k):
        nonlocal best_t, kind
        hit = valid & (t > 1e-6) & (t < best_t)
        best_t[hit] = t[hit]
        kind[hit] = k

    update(FLOOR_Y / dy, dy < 0, 1)  # floor
    update(WALL_Z / dz, np.ones((HEIGHT, WIDTH), bool), 2)  # back wall

    def in_box(px, py, pz, axis):
        x_ok = (BOX["x0"] <= px) & (px <= BOX["x1"])
        y_ok = (BOX["y0"] <= py) & (py <= BOX["y1"])
        z_ok = (BOX["z0"] <= pz) & (pz <= BOX["z1"])
        return {"x": y_ok & z_ok, "y": x_ok & z_ok, "z": x_ok & y_ok}[axis]

    t = BOX["z1"] / dz
    update(t, in_box(t * dx, t * dy, t * dz, "z"), 3)  # front face
    t = BOX["y1"] / dy
    update(t, (dy < 0) & in_box(t * dx, t * dy, t * dz, "y"), 3)  # top face
    t = BOX["x0"] / dx
    update(t, (dx < 0) & in_box(t * dx, t * dy, t * dz, "x"), 3)  # left face
    t = BOX["x1"] / dx
    update(t, (dx > 0) & in_box(t * dx, t * dy, t * dz, "x"), 3)  # right face

    point_map = best_t[..., None] * dirs
    rng = np.random.default_rng(seed)
    if noise_std > 0:
        point_map = point_map * (1.0 + noise_std * rng.standard_normal(point_map.shape))
    holes = rng.random((HEIGHT, WIDTH)) < 0.01
    point_map[holes] = np.nan

    mask = kind == 3
    K = np.array([[FX, 0, cx], [0, FY, cy], [0, 0, 1]])
    return point_map.astype(np.float32), mask, K


def test_resize_mask():
    mask = np.zeros((480, 640), dtype=bool)
    mask[100:200, 100:200] = True
    resized = resize_mask(mask, 400, 300)
    assert resized.shape == (300, 400)
    assert resized.dtype == bool
    assert resized[93, 93] and not resized[10, 10]
    assert resized.sum() == pytest.approx(mask.sum() * (400 * 300) / (640 * 480), rel=0.05)


def test_intrinsics_and_projection():
    point_map, mask, K = synthetic_room()
    metadata = {"intrinsics": np.array([[FX / WIDTH, 0, 0.5], [0, FY / HEIGHT, 0.5], [0, 0, 1]])}
    K_pm = intrinsics_px(metadata, WIDTH, HEIGHT)
    assert K_pm == pytest.approx(K)

    # box front-face corners project to known pixels
    pixels, valid = project_points(np.array([[-0.5, -1.0, -2.5], [0.5, -1.0, -2.5]]), K)
    assert valid.all()
    assert pixels[0] == pytest.approx([280.0, 60.0], abs=1.0)
    assert pixels[1] == pytest.approx([520.0, 60.0], abs=1.0)


def test_export_debug_glb_scales_source_scene():
    trimesh = pytest.importorskip("trimesh")
    source_mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    source_mesh.apply_translation([10.0, 0.0, 0.0])
    source = trimesh.Scene()
    source.add_geometry(source_mesh, geom_name="source")

    box = Box3D(np.zeros(3), np.eye(3), np.ones(3))
    plane = FloorPlane(np.zeros(3), np.array([0.0, 1.0, 0.0]), 1.0, 100)
    wall = WallPlane(
        point=np.array([0.0, 0.0, -2.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        corners=np.array(
            [
                [-1.0, 0.0, -2.0],
                [1.0, 0.0, -2.0],
                [1.0, 2.0, -2.0],
                [-1.0, 2.0, -2.0],
            ]
        ),
        inlier_count=500,
        num_candidates=1000,
        rmse=0.01,
        width=2.0,
        height=2.0,
        confidence=0.8,
    )
    exported = export_debug_glb(
        source.export(file_type="glb"),
        box,
        plane,
        scale_factor=0.5,
        walls=[wall],
    )

    assert exported is not None
    result = trimesh.load(io.BytesIO(exported), file_type="glb", force="scene")
    assert result.bounds[1, 0] == pytest.approx(5.5)
    assert result.geometry["fitted_box"].extents == pytest.approx(np.ones(3))
    assert "wall_plane_01" in result.geometry




def test_floor_and_box_fit_clean():
    point_map, mask, _ = synthetic_room()
    object_points = extract_object_points(point_map, mask)
    candidates = floor_candidate_points(point_map, mask)
    assert len(candidates) > 1000

    plane = fit_floor_plane(candidates, seed=0)
    assert plane is not None and not plane.fallback
    assert plane.normal @ np.array([0, 1, 0]) > 0.99
    assert plane.point[1] == pytest.approx(FLOOR_Y, abs=1e-3)

    box = fit_box(object_points, plane)
    assert sorted(box.extents) == pytest.approx([1.0, 1.0, 1.0], abs=0.05)
    assert box.center == pytest.approx([0.0, -1.0, -3.0], abs=0.05)
    assert np.linalg.det(box.axes.T) == pytest.approx(1.0, abs=1e-6)
    bottom = box.corners()[:4]
    assert plane.signed_distance(bottom) == pytest.approx(np.zeros(4), abs=1e-6)


def test_global_wall_fit_clean():
    point_map, mask, _ = synthetic_room()
    floor = fit_floor_plane(floor_candidate_points(point_map, mask), seed=0)
    assert floor is not None

    walls = fit_wall_planes(point_map, mask, floor, iterations=300, seed=0)

    assert walls
    back_wall = min(walls, key=lambda wall: abs(wall.point[2] - WALL_Z))
    assert abs(back_wall.normal @ floor.normal) < 1e-6
    assert abs(back_wall.normal[2]) > 0.99
    assert back_wall.point[2] == pytest.approx(WALL_Z, abs=0.05)
    assert back_wall.width > 2.0
    assert back_wall.height > 1.5
    assert floor.signed_distance(back_wall.corners[:2]) == pytest.approx(
        np.zeros(2), abs=1e-6
    )
    assert back_wall.to_dict()["confidence"] > 0.0


def test_sam_selected_floor_and_wall_fit_clean():
    point_map, object_mask, _ = synthetic_room()
    finite = np.isfinite(point_map).all(axis=2)
    floor_mask = finite & ~object_mask & np.isclose(point_map[:, :, 1], FLOOR_Y)
    wall_mask = finite & ~object_mask & np.isclose(point_map[:, :, 2], WALL_Z)

    floor_points = segmented_surface_points(point_map, floor_mask, object_mask)
    floor = fit_floor_plane(floor_points, seed=0)
    assert floor is not None
    assert floor.point[1] == pytest.approx(FLOOR_Y, abs=1e-3)

    walls = fit_segmented_wall_planes(
        point_map,
        [wall_mask],
        object_mask,
        floor,
        iterations=300,
        seed=0,
    )
    assert len(walls) == 1
    assert walls[0].point[2] == pytest.approx(WALL_Z, abs=0.05)
    assert abs(walls[0].normal[2]) > 0.99


def test_wall_post_filter_rejects_parallel_shadow_and_partial_plane():
    def wall(normal, point, inliers, height, confidence):
        return WallPlane(
            point=np.asarray(point, dtype=float),
            normal=np.asarray(normal, dtype=float),
            corners=np.zeros((4, 3)),
            inlier_count=inliers,
            num_candidates=25_000,
            rmse=0.01,
            width=3.0,
            height=height,
            confidence=confidence,
        )

    back = wall([0.0, 0.0, 1.0], [0.0, -1.0, -4.0], 5_000, 2.2, 0.75)
    shadow = wall([0.04, 0.0, 0.9992], [0.0, -1.0, -3.65], 1_000, 2.3, 0.55)
    side = wall([1.0, 0.0, 0.0], [-2.0, -1.0, -4.0], 3_000, 2.1, 0.70)
    partial = wall([0.7, 0.0, 0.714], [1.0, -1.0, -2.0], 600, 1.4, 0.35)

    filtered = filter_wall_planes([shadow, partial, side, back])

    assert len(filtered) == 2
    assert filtered[0] is back
    assert filtered[1] is side


def test_floor_and_box_fit_noisy():
    point_map, mask, _ = synthetic_room(noise_std=0.01, seed=1)
    object_points = extract_object_points(point_map, mask)
    plane = fit_floor_plane(floor_candidate_points(point_map, mask), seed=0)
    assert plane is not None
    assert plane.normal @ np.array([0, 1, 0]) > 0.98
    assert plane.point[1] == pytest.approx(FLOOR_Y, abs=0.05)

    box = fit_box(object_points, plane)
    assert sorted(box.extents) == pytest.approx([1.0, 1.0, 1.0], abs=0.15)


def test_floor_fallback():
    object_points = np.array(
        [[0.0, -1.5, -3.0], [0.5, -1.4, -3.0], [0.0, -0.5, -2.5], [0.5, -0.6, -2.6]]
    )
    assert fit_floor_plane(object_points) is None  # too few candidates
    plane = fallback_floor_plane(object_points)
    assert plane.fallback
    assert plane.normal.tolist() == [0.0, 1.0, 0.0]
    assert plane.point[1] == pytest.approx(-1.5, abs=0.1)

    box = fit_box(object_points, plane)
    assert box.extents[2] == pytest.approx(1.0, abs=0.15)


def make_zip(with_glb: bool = True) -> bytes:
    buf = io.BytesIO()
    point_map = np.zeros((4, 5, 3), dtype=np.float32)
    metadata = {"image_size": [5, 4], "intrinsics": np.eye(3).tolist()}
    npy_buf = io.BytesIO()
    np.save(npy_buf, point_map, allow_pickle=False)
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("point_map.npy", npy_buf.getvalue())
        archive.writestr("metadata.json", json.dumps(metadata))
        if with_glb:
            archive.writestr("output.glb", b"glb-bytes")
        archive.writestr("depth.png", b"depth-bytes")
    return buf.getvalue()


def test_moge_parse_zip():
    result = MogeClient.parse_zip(make_zip())
    assert result.point_map.shape == (4, 5, 3)
    assert result.image_size == (5, 4)
    assert result.intrinsics == pytest.approx(np.eye(3))
    assert result.glb_bytes == b"glb-bytes"
    assert result.depth_png == b"depth-bytes"
    assert result.normal_png is None


def test_moge_predict_request_flags(monkeypatch):
    requests_seen = []

    class Response:
        status_code = 200
        content = make_zip()
        text = ""

    def fake_post(*args, **kwargs):
        requests_seen.append(kwargs["data"])
        return Response()

    monkeypatch.setattr("dreamroom.moge_client.requests.post", fake_post)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    client = MogeClient("https://example.test/predict")
    client.predict(image)
    client.predict(image, include_mesh=True, include_debug=True)

    assert requests_seen == [
        {"include_mesh": "false", "include_debug": "false"},
        {"include_mesh": "true", "include_debug": "true"},
    ]


def test_moge_parse_zip_missing_members():
    with pytest.raises(RuntimeError):
        MogeClient.parse_zip(make_zip(with_glb=False)[:-10])  # truncated/corrupt


def test_moge_client_endpoint_normalization():
    assert MogeClient(None).endpoint == DEFAULT_ENDPOINT
    assert MogeClient("https://x.modal.run").endpoint == "https://x.modal.run/predict"
    assert MogeClient("https://x.modal.run/predict").endpoint == "https://x.modal.run/predict"

def test_calibrate_scale():
    point_map, _, _ = synthetic_room()
    factor, info = calibrate_scale(point_map, [280.0, 60.0], [520.0, 60.0], meters=2.0)
    assert info["applied"]
    assert factor == pytest.approx(2.0, rel=0.02)

    factor, info = calibrate_scale(point_map, [10.0, 10.0], [20.0, 10.0], meters=1.0)
    assert info["applied"] and factor > 0


def test_calibrate_scale_invalid_points():
    point_map = np.full((10, 10, 3), np.nan, dtype=np.float32)
    factor, info = calibrate_scale(point_map, [1.0, 1.0], [5.0, 5.0], meters=1.0)
    assert factor == 1.0 and not info["applied"]


def test_ratio_calibration_matches_moge_bias_example():
    factor, info = calculate_aspect_ratio_calibration(
        [1.7, 2.7, 1.5], [1.6, 2.0, 1.3]
    )

    expected_factor = (2.0 / 1.6) / (2.7 / 1.7)
    assert factor == pytest.approx(expected_factor)
    assert info["factor_definition"] == (
        "actual_depth_width_ratio / moge_depth_width_ratio"
    )
    moge_target, adjusted_target = target_dimensions_in_moge_units(
        [0.99, 2.0, 0.9],
        [1.7, 2.7, 1.5],
        [1.6, 2.0, 1.3],
        factor,
    )
    assert adjusted_target == pytest.approx([0.99, 2.0, 0.9])
    assert moge_target == pytest.approx(
        [
            1.7 * 0.99 / 1.6,
            1.7 * 2.0 / 1.6 / factor,
            1.5 * 0.9 / 1.3,
        ]
    )
