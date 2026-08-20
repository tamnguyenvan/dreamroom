"""Tests for fal.ai SAM 3 surface segmentation and pipeline staging."""

from __future__ import annotations

import base64
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from dreamroom.config import Settings
from dreamroom.geometry3d import Box3D, FloorPlane
from dreamroom.moge_client import MogeResult
from dreamroom.pipeline.models import PipelineContext
from dreamroom.pipeline.outputs import OutputWriter
from dreamroom.pipeline.stages.base import StageStatus
from dreamroom.pipeline.stages.surfaces import SurfaceStage
from dreamroom.sam3_client import Sam3Client, Sam3Mask, SurfaceSegmentation


def _mask_data_uri(mask: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    assert ok
    payload = base64.b64encode(encoded.tobytes()).decode()
    return f"data:image/png;base64,{payload}"


def test_sam3_client_uploads_once_and_runs_surface_prompts(monkeypatch):
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    source_masks = {}
    for index, label in enumerate(("wall", "floor", "rug"), start=1):
        mask = np.zeros((6, 8), dtype=bool)
        mask[index : index + 2, 1:7] = True
        source_masks[label] = mask

    uploads = []
    calls = []

    def upload_file(path):
        uploaded = cv2.imread(str(path))
        uploads.append(uploaded.shape)
        return "https://fal.example/input.png"

    def subscribe(model, *, arguments, client_timeout):
        calls.append((model, arguments, client_timeout))
        label = arguments["prompt"]
        score = 0.1 if label == "rug" else 0.9
        return {
            "masks": [{"url": _mask_data_uri(source_masks[label])}],
            "metadata": [{"index": 0, "score": score, "box": [0.5, 0.5, 0.5, 0.5]}],
        }

    monkeypatch.setenv("FAL_KEY", "test-key")
    monkeypatch.setitem(
        sys.modules,
        "fal_client",
        SimpleNamespace(upload_file=upload_file, subscribe=subscribe),
    )

    result = Sam3Client(timeout=12.0, min_score=0.25).segment_surfaces(image)

    assert uploads == [(6, 8, 3)]
    assert {call[1]["prompt"] for call in calls} == {"wall", "floor", "rug"}
    assert all(call[0] == "fal-ai/sam-3/image" for call in calls)
    assert all(call[1]["apply_mask"] is False for call in calls)
    assert all(call[2] == 12.0 for call in calls)
    assert np.array_equal(result.combined_mask("wall"), source_masks["wall"])
    assert np.array_equal(result.combined_mask("floor"), source_masks["floor"])
    assert not result.combined_mask("rug").any()


def test_sam3_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("FAL_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FAL_KEY"):
        Sam3Client().segment_surfaces(np.zeros((2, 2, 3), dtype=np.uint8))


@pytest.mark.parametrize("debug", [False, True])
def test_surface_stage_resizes_masks_and_keeps_debug_separate(tmp_path, debug):
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    floor = np.zeros((8, 10), dtype=bool)
    floor[5:, :] = True
    rug = np.zeros_like(floor)
    rug[5:7, 3:7] = True
    wall = np.zeros_like(floor)
    wall[:5, 1:9] = True
    segmentation = SurfaceSegmentation(
        masks={
            "wall": [Sam3Mask("wall", wall, 0.9)],
            "floor": [Sam3Mask("floor", floor, 0.8)],
            "rug": [Sam3Mask("rug", rug, 0.7)],
        },
        image_shape=(8, 10),
        model="test-model",
    )

    class FakeClient:
        def segment_surfaces(self, value):
            assert value is image
            return segmentation

    context = PipelineContext(
        image_path=tmp_path / "room.png",
        settings=Settings(debug=debug),
        image_bgr=image,
        moge=MogeResult(
            point_map=np.zeros((4, 5, 3), dtype=np.float32),
            metadata={"image_size": [5, 4], "intrinsics": np.eye(3).tolist()},
        ),
    )

    status = SurfaceStage(lambda _: FakeClient()).run(context)

    assert status is StageStatus.COMPLETED
    assert context.floor_surface_mask_pm.shape == (4, 5)
    assert context.rug_surface_mask_pm.shape == (4, 5)
    assert len(context.wall_surface_masks_pm) == 1
    assert context.wall_surface_masks_pm[0].shape == (4, 5)
    if debug:
        assert context.debug_surfaces_2d.shape == image.shape
    else:
        assert context.debug_surfaces_2d is None


def test_surface_outputs_are_separate_debug_assets(tmp_path):
    mask = np.zeros((6, 8), dtype=bool)
    mask[2:5, 2:7] = True
    segmentation = SurfaceSegmentation(
        masks={
            "wall": [Sam3Mask("wall", mask, 0.9)],
            "floor": [Sam3Mask("floor", mask, 0.8)],
            "rug": [],
        },
        image_shape=(6, 8),
        model="test-model",
    )
    context = PipelineContext(
        image_path=tmp_path / "room.png",
        settings=Settings(debug=True),
        moge=MogeResult(
            point_map=np.zeros((6, 8, 3), dtype=np.float32),
            metadata={"image_size": [8, 6], "intrinsics": np.eye(3).tolist()},
        ),
        surface_segmentation=segmentation,
        box=Box3D(np.zeros(3), np.eye(3), np.ones(3)),
        floor=FloorPlane(np.zeros(3), np.array([0.0, 1.0, 0.0]), 1.0, 48),
        scale_correction=1.0,
        debug_surfaces_2d=np.zeros((6, 8, 3), dtype=np.uint8),
    )

    OutputWriter.save_moge_outputs(tmp_path, context)

    assert (tmp_path / "surfaces.json").is_file()
    assert (tmp_path / "sam3_floor_mask.png").is_file()
    assert (tmp_path / "sam3_rug_mask.png").is_file()
    assert (tmp_path / "sam3_wall_mask.png").is_file()
    assert (tmp_path / "debug_surfaces_2d.png").is_file()
