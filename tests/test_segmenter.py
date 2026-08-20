"""Tests for the remote SimpleClick client."""

from __future__ import annotations

import base64

import cv2
import numpy as np

from dreamroom.config import Settings
from dreamroom.segmenter import SimpleClickSegmenter


def _mask_base64(mask: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    assert ok
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def test_remote_segmenter_sends_contract_and_resizes_mask(monkeypatch):
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    response_mask = np.zeros((3, 4), dtype=bool)
    response_mask[1:, 1:3] = True
    request = {}

    class Response:
        status_code = 200
        text = ""

        def json(self):
            return {"mask": _mask_base64(response_mask)}

    def post(url, *, json, timeout):
        request.update(url=url, json=json, timeout=timeout)
        return Response()

    monkeypatch.setattr("dreamroom.segmenter.requests.post", post)
    client = SimpleClickSegmenter(
        Settings(
            simpleclick_endpoint="https://example.test/segment/",
            simpleclick_timeout=12.0,
            max_points=2,
        )
    )

    mask = client.segment(
        image,
        [[1, 1], [2, 2], [3, 3]],
        [[0, 0]],
    )

    assert request["url"] == "https://example.test/segment"
    assert request["timeout"] == 12.0
    assert request["json"]["positive_points"] == [[1, 1], [3, 3]]
    assert request["json"]["negative_points"] == [[0, 0]]
    assert isinstance(request["json"]["image"], str)
    assert mask.shape == image.shape[:2]
    assert mask.dtype == bool
    assert mask[3, 2]
    assert not mask[0, 0]
