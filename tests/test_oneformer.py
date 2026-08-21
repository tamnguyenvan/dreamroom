"""Tests for the remote OneFormer surface client and fallback order."""

from __future__ import annotations

import base64
from types import SimpleNamespace

import cv2
import numpy as np

from dreamroom.config import Settings
from dreamroom.moge_client import MogeResult
from dreamroom.oneformer_client import OneFormerClient
from dreamroom.pipeline.models import PipelineContext
from dreamroom.pipeline.stages.base import StageStatus
from dreamroom.pipeline.stages.surfaces import SurfaceStage
from dreamroom.sam3_client import Sam3Mask, SurfaceSegmentation


def _mask_data_uri(mask: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    assert ok
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode()


def _segmentation(provider: str, *, include_wall: bool = True) -> SurfaceSegmentation:
    wall = np.zeros((6, 8), dtype=bool)
    wall[:3, 1:7] = True
    floor = np.zeros_like(wall)
    floor[3:, :] = True
    masks = {
        "wall": [Sam3Mask("wall", wall)] if include_wall else [],
        "floor": [Sam3Mask("floor", floor)],
        "rug": [],
    }
    return SurfaceSegmentation(masks, wall.shape, f"{provider}-model", provider)


def test_oneformer_client_decodes_provider_masks(monkeypatch):
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    calls = []
    response_data = {
        "provider": "oneformer",
        "model": "test-model",
        "image_size_hw": [6, 8],
        "masks": {
            "wall": [{"mask": _mask_data_uri(np.ones((6, 8), dtype=bool))}],
            "floor": [],
            "rug": [],
        },
    }

    def post(url, *, json, timeout):
        calls.append((url, json, timeout))
        return SimpleNamespace(status_code=200, text="", json=lambda: response_data)

    monkeypatch.setattr("dreamroom.oneformer_client.requests.post", post)
    result = OneFormerClient("https://oneformer.example/segment", timeout=12).segment_surfaces(
        image
    )

    assert calls[0][0] == "https://oneformer.example/segment"
    assert calls[0][2] == 12
    assert calls[0][1]["image"].startswith("data:image/png;base64,")
    assert result.provider == "oneformer"
    assert result.model == "test-model"
    assert result.instances("wall")[0].mask.all()


def test_surface_stage_prefers_oneformer_over_sam3(tmp_path):
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    context = PipelineContext(
        image_path=tmp_path / "room.png",
        settings=Settings(oneformer_endpoint="https://oneformer.example"),
        image_bgr=image,
        moge=MogeResult(
            np.zeros((6, 8, 3)),
            {"image_size": [8, 6], "intrinsics": np.eye(3).tolist()},
        ),
    )
    sam3_called = False

    def sam3_factory(_):
        nonlocal sam3_called
        sam3_called = True
        raise AssertionError("SAM3 should not run after a usable OneFormer result")

    status = SurfaceStage(
        oneformer_factory=lambda _: SimpleNamespace(
            segment_surfaces=lambda _: _segmentation("oneformer")
        ),
        sam3_factory=sam3_factory,
    ).run(context)

    assert status is StageStatus.COMPLETED
    assert context.surface_segmentation is not None
    assert context.surface_segmentation.provider == "oneformer"
    assert not sam3_called


def test_surface_stage_uses_sam3_when_oneformer_is_incomplete(tmp_path):
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    context = PipelineContext(
        image_path=tmp_path / "room.png",
        settings=Settings(oneformer_endpoint="https://oneformer.example"),
        image_bgr=image,
    )
    status = SurfaceStage(
        oneformer_factory=lambda _: SimpleNamespace(
            segment_surfaces=lambda _: _segmentation("oneformer", include_wall=False)
        ),
        sam3_factory=lambda _: SimpleNamespace(
            segment_surfaces=lambda _: _segmentation("sam3")
        ),
    ).run(context)

    assert status is StageStatus.COMPLETED
    assert context.surface_segmentation is not None
    assert context.surface_segmentation.provider == "sam3"
