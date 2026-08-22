"""Tests for the fal.ai Gemini Nano Banana Lite client."""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from dreamroom.gemini_client import GeminiClient


def test_gemini_client_uploads_square_input_and_uses_edit_schema(monkeypatch):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    encoded = cv2.imencode(".png", image)[1].tobytes()
    output_url = "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")
    calls = {}

    def upload_file(path: Path):
        calls["uploaded_shape"] = cv2.imread(str(path)).shape[:2]
        return "https://fal.example/input.png"

    def subscribe(model, *, arguments, client_timeout):
        calls["model"] = model
        calls["arguments"] = arguments
        calls["timeout"] = client_timeout
        return {"images": [{"url": output_url}], "description": "removed"}

    monkeypatch.setenv("FAL_KEY", "test-key")
    monkeypatch.setitem(
        sys.modules,
        "fal_client",
        SimpleNamespace(upload_file=upload_file, subscribe=subscribe),
    )

    result = GeminiClient(timeout=17.0).remove_object(
        image,
        "remove the object",
        aspect_ratio="16:9",
    )

    assert calls["model"] == "google/nano-banana-lite/edit"
    assert calls["uploaded_shape"] == (8, 8)
    assert calls["arguments"] == {
        "prompt": "remove the object",
        "image_urls": ["https://fal.example/input.png"],
        "aspect_ratio": "16:9",
        "output_format": "png",
        "num_images": 1,
        "limit_generations": True,
    }
    assert calls["timeout"] == 17.0
    assert result.image_bytes == encoded
