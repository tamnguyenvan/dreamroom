"""Modal FastAPI endpoint for OneFormer ADE20K room-surface segmentation.

Deploy with::

    modal deploy third_party/OneFormer/oneformer_api.py

The endpoint accepts a JSON payload containing a base64 image data URL::

    {"image": "data:image/png;base64,..."}

It runs OneFormer in semantic mode and returns binary PNG masks for the ADE20K
``wall``, ``floor``, and ``rug`` classes.  The application intentionally keeps
the model-specific dependencies inside the Modal image; the local pipeline only
needs ``requests`` to call the endpoint.
"""

from __future__ import annotations

import base64
import binascii
import io
import os
from pathlib import Path

import modal


APP_NAME = "oneformer-semantic-segmentation"
MODEL_ID = os.getenv("ONEFORMER_MODEL_ID", "shi-labs/oneformer_ade20k_swin_large")
MODEL_DIR = "/opt/models/oneformer"
MAX_IMAGE_BYTES = 25 * 1024 * 1024
TARGET_ALIASES = {
    "wall": ("wall",),
    "floor": ("floor",),
    "rug": ("rug", "carpet"),
}


def _download_model() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_DIR)


def _decode_base64_image(value: object) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError("image must be a non-empty base64 string or data URL")
    encoded = value.split(",", 1)[1] if value.startswith("data:") else value
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image is not valid base64") from exc
    if not raw:
        raise ValueError("image is empty")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit"
        )
    return raw


def _mask_as_data_url(mask) -> str:
    from PIL import Image

    output = io.BytesIO()
    Image.fromarray((mask.astype("uint8") * 255)).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "transformers>=4.38,<5",
        "huggingface_hub>=0.20,<1",
        "Pillow>=9.5,<12",
        "numpy<2",
        "scipy>=1.10,<1.16",
        "fastapi[standard]>=0.115,<1",
    )
    .run_function(_download_model)
)

app = modal.App(APP_NAME)


@app.cls(
    image=image,
    gpu="A10G",
    scaledown_window=300,
    timeout=300,
)
@modal.concurrent(max_inputs=4)
class OneFormerSegmenter:
    """Keep the large OneFormer checkpoint warm between requests."""

    @modal.enter()
    def load_model(self) -> None:
        import torch
        from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = OneFormerProcessor.from_pretrained(MODEL_DIR)
        self.model = OneFormerForUniversalSegmentation.from_pretrained(MODEL_DIR)
        self.model.to(self.device).eval()
        self.id_to_label = {
            int(index): self._normalize_label(label)
            for index, label in self.model.config.id2label.items()
        }

        warmup = self.processor(
            images=self._blank_image(),
            task_inputs=["semantic"],
            return_tensors="pt",
        )
        warmup = self._to_device(warmup)
        with torch.inference_mode():
            self.model(**warmup)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    @modal.fastapi_endpoint(method="POST", docs=True)
    def segment(self, payload: dict) -> dict:
        from fastapi import HTTPException
        from PIL import Image

        try:
            image = Image.open(io.BytesIO(_decode_base64_image(payload.get("image")))).convert(
                "RGB"
            )
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        import torch

        inputs = self.processor(
            images=image,
            task_inputs=["semantic"],
            return_tensors="pt",
        )
        inputs = self._to_device(inputs)
        try:
            with torch.inference_mode():
                outputs = self.model(**inputs)
            semantic_map = self.processor.post_process_semantic_segmentation(
                outputs,
                target_sizes=[image.size[::-1]],
            )[0]
        except Exception as exc:
            raise HTTPException(status_code=500, detail="OneFormer inference failed") from exc

        predicted = semantic_map.detach().cpu().numpy()
        masks = {}
        label_ids = {}
        for target, aliases in TARGET_ALIASES.items():
            ids = [
                index
                for index, label in self.id_to_label.items()
                if label in aliases
            ]
            label_ids[target] = ids
            mask = (predicted[..., None] == ids).any(axis=-1) if ids else predicted == -1
            masks[target] = []
            if mask.any():
                masks[target].append(
                    {
                        "mask": _mask_as_data_url(mask),
                        "score": None,
                        "box": None,
                    }
                )

        return {
            "model": MODEL_ID,
            "provider": "oneformer",
            "image_size_hw": [image.height, image.width],
            "label_ids": label_ids,
            "masks": masks,
        }

    @staticmethod
    def _normalize_label(value: object) -> str:
        return " ".join(str(value).lower().replace("_", " ").split())

    def _to_device(self, inputs):
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    @staticmethod
    def _blank_image():
        from PIL import Image

        return Image.new("RGB", (64, 64), (0, 0, 0))


if __name__ == "__main__":
    print(f"Deploy with: modal deploy {Path(__file__).name}")
