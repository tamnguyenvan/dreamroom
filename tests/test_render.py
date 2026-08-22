"""Tests for target-box render preparation and Seedream invocation."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np

from dreamroom.config import Settings
from dreamroom.gemini_client import GeminiEditResult
from dreamroom.geometry3d import Box3D, FloorPlane
from dreamroom.moge_client import MogeResult
from dreamroom.pipeline.models import PipelineContext
from dreamroom.pipeline.outputs import OutputWriter
from dreamroom.pipeline.stages.base import StageStatus
from dreamroom.pipeline.stages.removal import RemovalStage
from dreamroom.pipeline.stages.render import FurnitureStage, RenderStage
from dreamroom.placement_geometry import TargetBoxPlacement
from dreamroom.render_viz import draw_target_box_2d
from dreamroom.seedream_client import SeedreamClient, SeedreamResult
from dreamroom.ui.strokes import ObjectSelection


def _target() -> TargetBoxPlacement:
    box = Box3D(
        center=np.array([0.0, -0.5, -3.0]),
        axes=np.eye(3),
        extents=np.array([1.0, 1.2, 1.0]),
    )
    return TargetBoxPlacement(
        box=box,
        rear_anchor=np.array([0.0, -1.0, -3.0]),
        rear_face_id="axis1_negative",
        primary_wall_index=None,
        secondary_wall_index=None,
        wall_aligned=False,
        tilt_degrees=None,
        primary_wall_distance=None,
    )


def _encoded_input(data_url: str) -> np.ndarray:
    encoded = base64.b64decode(data_url.split(",", 1)[1])
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    return image


def test_seedream_sends_two_images_and_fast_options(monkeypatch):
    requests = {}

    class Response:
        status_code = 200
        text = ""

        def json(self):
            return {
                "model": "dola-seedream-5-0-pro-260628",
                "data": [{"url": "https://example.test/render.jpg", "size": "1024x1024"}],
                "usage": {"generated_images": 1},
            }

    class Download:
        status_code = 200
        content = b"jpeg-bytes"
        text = ""

    def post(url, *, headers, json, timeout):
        requests["url"] = url
        requests["headers"] = headers
        requests["json"] = json
        requests["timeout"] = timeout
        return Response()

    def get(url, *, timeout):
        requests["download_url"] = url
        return Download()

    monkeypatch.setattr("dreamroom.seedream_client.requests.post", post)
    monkeypatch.setattr("dreamroom.seedream_client.requests.get", get)

    room = np.zeros((8, 12, 3), dtype=np.uint8)
    furniture = np.zeros((6, 5, 3), dtype=np.uint8)
    result = SeedreamClient(api_key="secret", timeout=17.0).generate(
        room, furniture, "replace Image 1 using Image 2"
    )

    payload = requests["json"]
    assert payload["image"] and len(payload["image"]) == 2
    assert _encoded_input(payload["image"][0]).shape[:2] == room.shape[:2]
    assert _encoded_input(payload["image"][1]).shape[:2] == furniture.shape[:2]
    assert payload["model"] == "dola-seedream-5-0-pro-260628"
    assert payload["size"] == "1K"
    assert payload["response_format"] == "url"
    assert payload["output_format"] == "jpeg"
    assert payload["watermark"] is False
    assert payload["optimize_prompt_options"] == {"mode": "fast"}
    assert requests["headers"]["Authorization"] == "Bearer secret"
    assert result.image_bytes == b"jpeg-bytes"
    assert result.image_url == "https://example.test/render.jpg"


def test_draw_target_box_is_red():
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    rendered = draw_target_box_2d(
        image,
        _target(),
        np.array([[600.0, 0.0, 400.0], [0.0, 600.0, 300.0], [0.0, 0.0, 1.0]]),
        (1.0, 1.0),
    )
    red_pixels = (rendered[:, :, 2] > 200) & (rendered[:, :, 1] < 50)
    assert int(red_pixels.sum()) > 0


def test_render_stage_resizes_furniture_and_sends_two_inputs(tmp_path):
    furniture_path = tmp_path / "furniture.jpg"
    cv2.imwrite(str(furniture_path), np.zeros((900, 700, 3), dtype=np.uint8))
    room = np.zeros((600, 800, 3), dtype=np.uint8)
    fake_result = SeedreamResult(
        image_bytes=b"rendered-jpeg",
        image_url="https://example.test/result.jpg",
        response={"data": [{"url": "https://example.test/result.jpg"}]},
        elapsed_seconds=1.25,
    )

    class FakeClient:
        def generate(self, room_image, furniture_image, prompt):
            assert room_image.shape[:2] == room.shape[:2]
            assert max(furniture_image.shape[:2]) == 512
            assert "Image 1" in prompt and "Image 2" in prompt
            assert "red wireframe target box" in prompt
            return fake_result

    class FakeGeminiClient:
        def remove_object(self, image, prompt):
            assert image.shape[:2] == (600, 600)
            assert prompt == (
                "Remove the selected object and keep everything else in the room unchanged."
            )
            assert np.any((image[:, :, 2] > 200) & (image[:, :, 1] < 50))
            encoded = cv2.imencode(".png", np.full((8, 8, 3), 7, dtype=np.uint8))[1]
            return GeminiEditResult(
                image_bytes=encoded.tobytes(),
                image_url="data:image/png;base64,removed",
                response={"images": [{"url": "data:image/png;base64,removed"}]},
                elapsed_seconds=0.25,
            )

    context = PipelineContext(
        image_path=tmp_path / "room.jpg",
        settings=Settings(furniture_path=furniture_path, debug=True),
        image_bgr=room,
        moge=MogeResult(
            np.zeros((300, 400, 3), dtype=np.float32),
            {
                "image_size": [400, 300],
                "intrinsics": [[0.75, 0.0, 0.5], [0.0, 0.75, 0.5], [0.0, 0.0, 1.0]],
            },
        ),
        box=_target().box,
        floor=FloorPlane(np.zeros(3), np.array([0.0, 1.0, 0.0]), 1.0, 100),
        scale_correction=1.0,
        target_placement=_target(),
        selection=ObjectSelection(
            mask=np.pad(
                np.ones((40, 40), dtype=bool),
                ((260, 300), (380, 380)),
            ),
            positive_points=[[400, 280]],
            negative_points=[],
        ),
    )

    prepare_status = FurnitureStage().run(context)
    removal_status = RemovalStage(lambda _: FakeGeminiClient()).run(context)
    status = RenderStage(lambda _: FakeClient()).run(context)

    assert prepare_status is StageStatus.COMPLETED
    assert removal_status is StageStatus.COMPLETED
    assert status is StageStatus.COMPLETED
    assert context.render_room is not None
    assert context.render_furniture is not None
    assert max(context.render_furniture.shape[:2]) == 512
    assert context.rendered_image == b"rendered-jpeg"
    assert context.render_metadata["input_mode"] == "two_images"
    assert context.render_metadata["object_removal"]["crop"]["size"] == 600
    assert json.loads(json.dumps(context.render_metadata))["furniture_input_size_hw"] == [512, 398]

    OutputWriter.save_moge_outputs(tmp_path, context)
    assert (tmp_path / "render_room_target_box.png").is_file()
    assert (tmp_path / "render_furniture_reference.png").is_file()
    assert (tmp_path / "render_object_removal_input.png").is_file()
    assert (tmp_path / "render_object_removed_patch.png").is_file()
    assert (tmp_path / "render_room_object_removed.png").is_file()
    assert (tmp_path / "rendered_furniture.jpg").read_bytes() == b"rendered-jpeg"
    assert (tmp_path / "render.json").is_file()
