"""Unit tests for the non-GUI logic of the dreamroom pipeline.

Mouse and key events are simulated by calling the app handlers directly;
``WindowApp.flush`` and terminal ``input`` are monkeypatched, so no window
is opened.
"""

from __future__ import annotations

import builtins
import json

import cv2
import numpy as np
import pytest

from dreamroom.config import Settings
from dreamroom.geometry3d import Box3D, FloorPlane
from dreamroom.image_ops import load_image_bgr, mask_to_uint8, resize_max_side, save_image
from dreamroom.moge_client import MogeResult
from dreamroom.placement_geometry import PlacementOrientation, TargetBoxPlacement
from dreamroom.pipeline import FurniturePipeline
from dreamroom.pipeline.models import PipelineContext
from dreamroom.pipeline.outputs import OutputWriter
from dreamroom.pipeline.stages.placement import PlacementStage
from dreamroom.pipeline.stages.walls import WallStage
from dreamroom.pipeline.timing import print_latency_stats
from dreamroom.segmenter import sample_points
from dreamroom.ui.reference import ReferenceLineApp, ReferenceScale, prompt_meters
from dreamroom.ui.strokes import ObjectSelection, SelectObjectApp, select_object
from dreamroom.ui.window import WindowApp
from dreamroom.wall_geometry import WallPlane


def make_image(width: int = 640, height: int = 480) -> np.ndarray:
    image = np.full((height, width, 3), 180, dtype=np.uint8)
    cv2.circle(image, (width // 2, height // 2), 80, (40, 40, 40), -1)
    return image


def drag(app: WindowApp, button_down: int, points: list[tuple[int, int]]) -> None:
    """Simulate a drag gesture through the app's mouse handler."""

    button_up = cv2.EVENT_LBUTTONUP if button_down == cv2.EVENT_LBUTTONDOWN else cv2.EVENT_RBUTTONUP
    app.on_mouse(button_down, *points[0], 0)
    for point in points[1:]:
        app.on_mouse(cv2.EVENT_MOUSEMOVE, *point, 0)
    app.on_mouse(button_up, *points[-1], 0)


# -- step 0 -------------------------------------------------------------------
def test_resize_downscales_and_reports_scale():
    image = make_image(2560, 1280)
    resized, scale = resize_max_side(image, 1280)
    assert resized.shape[:2] == (640, 1280)
    assert scale == pytest.approx(2.0)


def test_resize_keeps_small_images():
    image = make_image(640, 480)
    resized, scale = resize_max_side(image, 1280)
    assert resized.shape == image.shape
    assert scale == 1.0


def test_load_and_save_roundtrip(tmp_path):
    image = make_image()
    out = save_image(tmp_path / "nested" / "img.png", image)
    loaded = load_image_bgr(out)
    assert loaded.shape == image.shape
    with pytest.raises(FileNotFoundError):
        load_image_bgr(tmp_path / "missing.png")


# -- point sampling -------------------------------------------------------------
def test_sample_points_keeps_short_lists():
    assert sample_points([[1, 2], [3.4, 5.6]], 24) == [[1, 2], [3, 6]]


def test_sample_points_spreads_evenly():
    points = [[i, 2 * i] for i in range(100)]
    sampled = sample_points(points, 24)
    assert len(sampled) == 24
    assert sampled[0] == [0, 0]
    assert sampled[-1] == [99, 198]


# -- step 1 UI logic -------------------------------------------------------------
def test_select_object_flow(monkeypatch):
    monkeypatch.setattr(WindowApp, "flush", lambda self: None)
    image = make_image()
    calls = []

    def fake_segment(positive, negative):
        calls.append((positive, negative))
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[100:200, 100:200] = True
        return mask

    app = SelectObjectApp(image, fake_segment)
    drag(app, cv2.EVENT_LBUTTONDOWN, [(100, 240), (200, 240), (300, 240)])
    drag(app, cv2.EVENT_RBUTTONDOWN, [(10, 10), (20, 20)])
    assert len(app.strokes) == 2

    app.on_key(ord("u"))  # undo the negative stroke
    assert len(app.strokes) == 1
    drag(app, cv2.EVENT_RBUTTONDOWN, [(10, 10), (20, 20)])

    app.on_key(13)  # Enter closes the annotation phase; segmentation runs outside it
    assert app._done and app.mode == "annotate"

    mask = fake_segment(app.points(is_positive=True), app.points(is_positive=False))
    app.mask = mask
    app.mode = "preview"
    app.on_key(ord("y"))  # confirm in the second window
    assert app._done
    result = app.result
    assert isinstance(result, ObjectSelection)
    assert result.mask.shape == image.shape[:2]
    assert result.positive_points and result.negative_points


def test_select_object_requires_positive_stroke(monkeypatch):
    monkeypatch.setattr(WindowApp, "flush", lambda self: None)
    app = SelectObjectApp(make_image(), lambda p, n: np.zeros((480, 640), dtype=bool))
    drag(app, cv2.EVENT_RBUTTONDOWN, [(10, 10), (20, 20)])
    app.on_key(13)
    assert app.mode == "annotate"
    assert "positive" in app.message


def test_select_object_reset(monkeypatch):
    monkeypatch.setattr(WindowApp, "flush", lambda self: None)
    image = make_image()
    app = SelectObjectApp(image, lambda p, n: np.ones(image.shape[:2], dtype=bool))
    drag(app, cv2.EVENT_LBUTTONDOWN, [(100, 240), (200, 240)])
    app.on_key(ord("r"))
    assert app.mode == "annotate" and app.strokes

    app.mode = "preview"
    app.mask = np.ones(image.shape[:2], dtype=bool)
    app.on_key(ord("r"))
    assert app._done and app._redraw_requested
    assert not app.strokes and app.mask is None


def test_select_object_segments_between_windows(monkeypatch):
    image = make_image()
    events = []
    selection = ObjectSelection(np.ones(image.shape[:2], dtype=bool), [[1, 1]], [])

    def collect_strokes(self):
        events.append("draw-window-closed")
        return ([[10, 10]], [])

    def review_mask(self, mask):
        events.append(("review", mask.shape))
        return selection

    def fake_segment(positive, negative):
        events.append(("segment", positive, negative))
        return np.ones(image.shape[:2], dtype=bool)

    monkeypatch.setattr(SelectObjectApp, "collect_strokes", collect_strokes)
    monkeypatch.setattr(SelectObjectApp, "review_mask", review_mask)
    result = select_object(image, fake_segment)

    assert result is selection
    assert events == ["draw-window-closed", ("segment", [[10, 10]], []), ("review", image.shape[:2])]


# -- step 2 UI logic ---------------------------------------------------------------
def test_reference_flow(monkeypatch):
    image = make_image()
    app = ReferenceLineApp(image, meters=1.8)

    drag(app, cv2.EVENT_LBUTTONDOWN, [(100, 100), (400, 100)])
    assert app.has_line
    assert app.pixel_length == pytest.approx(300.0)

    app.on_key(13)  # Enter -> confirm the line
    assert app._done
    result = app.result
    assert isinstance(result, ReferenceScale)
    assert result.start == [100, 100] and result.end == [400, 100]
    assert result.px_per_meter == pytest.approx(300.0 / 1.8)
    data = result.to_dict()
    assert data["meters_per_px"] == pytest.approx(1.8 / 300.0)
    json.dumps(data)  # must be JSON-serializable


def test_reference_rejects_short_line():
    app = ReferenceLineApp(make_image())
    drag(app, cv2.EVENT_LBUTTONDOWN, [(100, 100), (102, 101)])
    assert not app.has_line
    assert "too short" in app.message


def test_reference_rejects_invalid_input(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda prompt="": "abc")
    assert prompt_meters() is None


def test_reference_clear():
    app = ReferenceLineApp(make_image())
    drag(app, cv2.EVENT_LBUTTONDOWN, [(100, 100), (400, 100)])
    app.on_key(ord("u"))
    assert not app.has_line


# -- pipeline saving ------------------------------------------------------------------
def test_pipeline_save(tmp_path):
    settings = Settings(outputs_root=tmp_path)
    pipeline = FurniturePipeline(settings)
    image = make_image()
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[100:200, 100:200] = True
    selection = ObjectSelection(mask=mask, positive_points=[[320, 240]], negative_points=[[20, 20]])
    reference = ReferenceScale(start=[100, 100], end=[400, 100], pixel_length=300.0, meters=1.8)

    context = PipelineContext(
        image_path=tmp_path / "room.jpg",
        settings=settings,
        image_bgr=image,
        original_size=(480, 640),
        resize_scale=2.0,
        selection=selection,
        reference=reference,
    )
    out_dir = OutputWriter().save(context)
    assert out_dir.parent == tmp_path
    for name in ("image.png", "mask.png", "overlay.png", "selection.json", "reference.json", "meta.json"):
        assert (out_dir / name).is_file(), name

    saved_mask = load_image_bgr(out_dir / "mask.png")[:, :, 0]
    assert saved_mask.max() == 255 and (saved_mask > 0).sum() == 100 * 100
    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["resize_scale"] == 2.0
    reference_data = json.loads((out_dir / "reference.json").read_text())
    assert reference_data["px_per_meter"] == pytest.approx(300.0 / 1.8)


def test_pipeline_latency_stats(tmp_path, capsys):
    stats = {
        "step_0_resize": 0.1,
        "step_1_segment": 1.2,
        "step_2_reference": 0.3,
        "step_3_moge": None,
        "step_4_fit_3d": None,
        "save_outputs": 0.4,
        "total": 2.0,
    }
    OutputWriter.write_stats(tmp_path, stats)
    saved = json.loads((tmp_path / "stats.json").read_text())
    assert saved == {"latency_seconds": stats}

    print_latency_stats(stats)
    assert "step_3_moge: skipped" in capsys.readouterr().out


@pytest.mark.parametrize("debug", [False, True])
def test_wall_stage_visualizes_walls_only_in_debug(tmp_path, monkeypatch, debug):
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=bool)
    floor = FloorPlane(np.zeros(3), np.array([0.0, 1.0, 0.0]), 1.0, 100)
    box = Box3D(np.zeros(3), np.eye(3), np.ones(3))
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
        inlier_count=100,
        num_candidates=200,
        rmse=0.01,
        width=2.0,
        height=2.0,
        confidence=0.8,
    )
    context = PipelineContext(
        image_path=tmp_path / "room.png",
        settings=Settings(debug=debug),
        image_bgr=image,
        selection=ObjectSelection(mask, [], []),
        moge=MogeResult(
            np.zeros((10, 10, 3)),
            {
                "image_size": [10, 10],
                "intrinsics": [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]],
            },
            glb_bytes=b"source",
        ),
        point_map=np.zeros((10, 10, 3)),
        mask_pm=mask,
        scale_correction=1.0,
        floor=floor,
        box=box,
    )
    calls = {}
    monkeypatch.setattr(
        "dreamroom.pipeline.stages.walls.fit_wall_planes",
        lambda *args: [wall],
    )

    def fake_draw(*args):
        calls["draw_walls"] = args[-1]
        return image

    def fake_export(*args):
        calls["glb_walls"] = args[-1]
        return b"debug-glb"

    monkeypatch.setattr("dreamroom.pipeline.stages.walls.draw_debug_2d", fake_draw)
    monkeypatch.setattr("dreamroom.pipeline.stages.walls.export_debug_glb", fake_export)

    WallStage().run(context)

    assert context.walls[0] is wall
    assert calls["draw_walls"] is (context.walls if debug else None)
    if debug:
        assert calls["glb_walls"] is context.walls
    else:
        assert "glb_walls" not in calls


def test_pipeline_run_writes_latency_stats(tmp_path, monkeypatch):
    image_path = tmp_path / "room.png"
    cv2.imwrite(str(image_path), make_image())
    mask = np.zeros((480, 640), dtype=bool)
    mask[100:200, 100:200] = True
    selection = ObjectSelection(mask=mask, positive_points=[], negative_points=[])
    reference = ReferenceScale(start=[100, 100], end=[400, 100], pixel_length=300.0, meters=1.8)
    monkeypatch.setattr(
        "dreamroom.pipeline.stages.selection.select_object",
        lambda *args, **kwargs: selection,
    )
    monkeypatch.setattr(
        "dreamroom.pipeline.stages.reference.get_reference_scale",
        lambda *args, **kwargs: reference,
    )

    result = FurniturePipeline(Settings(outputs_root=tmp_path, moge_enabled=False)).run(image_path)

    assert result is not None
    stats = json.loads((result.output_dir / "stats.json").read_text())
    assert stats["latency_seconds"]["step_0_resize"] >= 0
    assert stats["latency_seconds"]["step_1_segment"] >= 0
    assert stats["latency_seconds"]["step_2_reference"] >= 0
    assert stats["latency_seconds"]["step_3_moge"] is None
    assert stats["latency_seconds"]["step_4_fit_3d"] is None
    assert stats["latency_seconds"]["step_5_fit_walls"] is None
    assert stats["latency_seconds"]["step_6_target_box"] is None
    assert stats["latency_seconds"]["total"] >= stats["latency_seconds"]["step_0_resize"]
    assert result.latency_seconds == stats["latency_seconds"]


@pytest.mark.parametrize("debug", [False, True])
def test_placement_stage_writes_separate_debug_assets(tmp_path, monkeypatch, debug):
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=bool)
    orientation = PlacementOrientation(
        mode="ambiguous",
        primary_rear_face=None,
        secondary_anchor_face=None,
        primary_wall_index=None,
        secondary_wall_index=None,
        confidence=0.0,
        reason="test",
    )
    context = PipelineContext(
        image_path=tmp_path / "room.png",
        settings=Settings(debug=debug),
        image_bgr=image,
        selection=ObjectSelection(mask, [], []),
        moge=MogeResult(
            np.zeros((10, 10, 3)),
            {
                "image_size": [10, 10],
                "intrinsics": [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]],
            },
            glb_bytes=b"source",
        ),
        point_map=np.zeros((10, 10, 3)),
        mask_pm=mask,
        scale_correction=1.0,
        floor=FloorPlane(np.zeros(3), np.array([0.0, 1.0, 0.0]), 1.0, 100),
        box=Box3D(np.zeros(3), np.eye(3), np.ones(3)),
    )
    calls = []
    monkeypatch.setattr(
        "dreamroom.pipeline.stages.placement.infer_placement_orientation",
        lambda *args: orientation,
    )
    monkeypatch.setattr(
        "dreamroom.pipeline.stages.placement.draw_placement_debug_2d",
        lambda *args: calls.append("2d") or image,
    )
    monkeypatch.setattr(
        "dreamroom.pipeline.stages.placement.export_placement_debug_glb",
        lambda *args: calls.append("3d") or b"placement-glb",
    )

    PlacementStage().run(context)

    assert context.placement_orientation is orientation
    if debug:
        assert calls == ["2d", "3d"]
        assert context.debug_placement_2d is image
        assert context.debug_placement_3d == b"placement-glb"
    else:
        assert calls == []
        assert context.debug_placement_2d is None
        assert context.debug_placement_3d is None


def test_production_moge_outputs_only_save_2d_debug(tmp_path):
    context = PipelineContext(
        image_path=tmp_path / "room.jpg",
        settings=Settings(outputs_root=tmp_path, debug=False),
        moge=MogeResult(
            np.zeros((2, 2, 3), dtype=np.float32),
            {"image_size": [2, 2], "intrinsics": np.eye(3).tolist()},
            glb_bytes=b"glb",
            depth_png=b"depth",
            normal_png=b"normal",
        ),
        box=Box3D(np.zeros(3), np.eye(3), np.ones(3)),
        floor=FloorPlane(np.zeros(3), np.array([0.0, 1.0, 0.0]), 1.0, 4),
        scale_correction=1.0,
        placement_orientation=PlacementOrientation(
            mode="ambiguous",
            primary_rear_face=None,
            secondary_anchor_face=None,
            primary_wall_index=None,
            secondary_wall_index=None,
            confidence=0.0,
            reason="test",
        ),
        debug_2d=make_image(8, 8),
    )

    OutputWriter.save_moge_outputs(tmp_path, context)

    assert (tmp_path / "box3d.json").is_file()
    assert (tmp_path / "walls3d.json").is_file()
    assert json.loads((tmp_path / "walls3d.json").read_text())["walls"] == []
    assert json.loads((tmp_path / "placement.json").read_text())["mode"] == "ambiguous"
    assert not (tmp_path / "target_box3d.json").exists()
    assert (tmp_path / "debug_2d.png").is_file()
    for name in ("point_map.npy", "moge_metadata.json", "output.glb", "depth.png", "normal.png"):
        assert not (tmp_path / name).exists(), name


def test_target_box_and_separate_debug_outputs(tmp_path):
    box = Box3D(np.zeros(3), np.eye(3), np.ones(3))
    orientation = PlacementOrientation(
        mode="free_standing",
        primary_rear_face="axis0_negative",
        secondary_anchor_face=None,
        primary_wall_index=None,
        secondary_wall_index=None,
        confidence=0.6,
        reason="test",
    )
    target = TargetBoxPlacement(
        box=box,
        rear_anchor=np.zeros(3),
        rear_face_id="axis0_negative",
        primary_wall_index=None,
        secondary_wall_index=None,
        wall_aligned=False,
        tilt_degrees=None,
        primary_wall_distance=None,
    )
    context = PipelineContext(
        image_path=tmp_path / "room.jpg",
        settings=Settings(outputs_root=tmp_path, debug=True),
        moge=MogeResult(
            np.zeros((2, 2, 3), dtype=np.float32),
            {"image_size": [2, 2], "intrinsics": np.eye(3).tolist()},
        ),
        box=box,
        floor=FloorPlane(np.zeros(3), np.array([0.0, 1.0, 0.0]), 1.0, 4),
        scale_correction=1.0,
        placement_orientation=orientation,
        target_placement=target,
        debug_placement_2d=make_image(8, 8),
        debug_placement_3d=b"placement-glb",
    )

    OutputWriter.save_moge_outputs(tmp_path, context)

    target_data = json.loads((tmp_path / "target_box3d.json").read_text())
    assert target_data["rear_face_id"] == "axis0_negative"
    assert (tmp_path / "debug_placement_2d.png").is_file()
    assert (tmp_path / "debug_placement_3d.glb").read_bytes() == b"placement-glb"


def test_mask_to_uint8():
    mask = np.array([[True, False]])
    assert mask_to_uint8(mask).tolist() == [[255, 0]]
