"""Modal FastAPI endpoint for SimpleClick interactive segmentation."""

from __future__ import annotations

import base64
import binascii
from pathlib import Path

import modal


APP_NAME = "simpleclick-interactive-segmentation"
SIMPLECLICK_ROOT = "/opt/SimpleClick"
CHECKPOINT_PATH = f"{SIMPLECLICK_ROOT}/weights/simpleclick_models/cocolvis_vit_huge.pth"
CHECKPOINT_ID = "1GXk6q5fwKo2twkY5ZZGjVKCgJv7XeLAW"
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_POINTS = 24
MAX_LONGEST_SIZE = 800
MODEL_INPUT_SIZE = 448
DEFAULT_THRESHOLD = 0.49


def _sample_points(points: list[list[float]], limit: int = MAX_POINTS) -> list[list[int]]:
    """Keep points distributed over the complete open stroke."""

    if len(points) <= limit:
        return [[int(round(point[0])), int(round(point[1]))] for point in points]

    step = (len(points) - 1) / (limit - 1)
    indices = [round(index * step) for index in range(limit)]
    return [[int(round(points[index][0])), int(round(points[index][1]))] for index in indices]


def _decode_base64_image(value: object) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError("image must be a non-empty base64 string")

    if value.startswith("data:"):
        if "," not in value:
            raise ValueError("image data URL is missing its payload")
        encoded = value.split(",", 1)[1]
    else:
        encoded = value
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image is not valid base64") from exc

    if not image_bytes:
        raise ValueError("image is empty")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError(f"image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit")
    return image_bytes


def _validate_points(
    value: object,
    width: int,
    height: int,
    field_name: str = "positive_points",
    *,
    required: bool = True,
) -> list[list[int]]:
    if value is None and not required:
        return []
    if not isinstance(value, list) or (required and not value):
        if required:
            raise ValueError(f"{field_name} must contain at least one [x, y] point")
        raise ValueError(f"{field_name} must be a list of [x, y] points")

    points: list[list[float]] = []
    for index, point in enumerate(value):
        if (
            not isinstance(point, (list, tuple))
            or len(point) != 2
            or not all(isinstance(coordinate, (int, float)) for coordinate in point)
        ):
            raise ValueError(f"{field_name}[{index}] must be [x, y]")

        x, y = float(point[0]), float(point[1])
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"{field_name}[{index}] is outside the image bounds")
        points.append([x, y])

    return _sample_points(points)


def _image_from_bytes(image_bytes: bytes):
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("image could not be decoded; use a PNG or JPEG image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _mask_as_png_base64(mask) -> str:
    import cv2
    import numpy as np

    mask_image = np.where(mask, 255, 0).astype(np.uint8)
    encoded, buffer = cv2.imencode(".png", mask_image)
    if not encoded:
        raise RuntimeError("failed to encode segmentation mask")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("build-essential", "git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.2.2",
        "torchvision==0.17.2",
    )
    .pip_install(
        "fastapi[standard]>=0.115,<1",
        "gdown>=5.2,<6",
        "numpy==1.23.5",
        "opencv-python-headless>=4.10,<5",
        "Pillow>=9.5,<12",
        "PyYAML>=6,<7",
        "protobuf==3.20.3",
        "tensorboard==2.8.0",
        "albumentations==0.5.2",
        "Cython==0.29.32",
        "easydict>=1.9,<2",
        "mmcv==1.6.2",
        "scipy>=1.10,<2",
        "timm==0.6.11",
        env={"MMCV_WITH_OPS": "0"},
    )
    .run_commands(
        f"git clone --depth 1 --branch v1.0 https://github.com/uncbiag/SimpleClick {SIMPLECLICK_ROOT}",
        f"mkdir -p {Path(CHECKPOINT_PATH).parent}",
        f"gdown {CHECKPOINT_ID} -O {CHECKPOINT_PATH}",
    )
)

app = modal.App(APP_NAME)


@app.cls(
    image=image,
    gpu="A10G",
    scaledown_window=300,
    timeout=300,
)
@modal.concurrent(max_inputs=10)
class SimpleClickSegmenter:
    """Keep the 659M-parameter model warm for repeated requests."""

    @modal.enter()
    def load_model(self) -> None:
        import sys

        import torch

        sys.path.insert(0, SIMPLECLICK_ROOT)
        from isegm.inference import utils
        from isegm.inference.predictors import get_predictor

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = utils.load_is_model(
            CHECKPOINT_PATH,
            self.device,
            eval_ritm=False,
            cpu_dist_maps=True,
        )
        self.predictor = get_predictor(
            self.model,
            "NoBRS",
            self.device,
            prob_thresh=DEFAULT_THRESHOLD,
            with_flip=True,
            zoom_in_params={
                "skip_clicks": -1,
                "target_size": (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                "expansion_ratio": 1.4,
            },
            predictor_params={"max_size": MAX_LONGEST_SIZE},
        )

    @modal.fastapi_endpoint(method="POST", docs=True)
    def segment(self, payload: dict) -> dict:
        """Segment one image using positive and optional negative ``[x, y]`` points."""

        from fastapi import HTTPException

        import torch

        try:
            image_rgb = _image_from_bytes(_decode_base64_image(payload.get("image")))
            height, width = image_rgb.shape[:2]
            positive_points = _validate_points(
                payload.get("positive_points"),
                width,
                height,
                "positive_points",
                required=True,
            )
            negative_points = _validate_points(
                payload.get("negative_points"),
                width,
                height,
                "negative_points",
                required=False,
            )
            threshold = float(payload.get("threshold", DEFAULT_THRESHOLD))
            if not 0 < threshold < 1:
                raise ValueError("threshold must be between 0 and 1")
        except (AttributeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        from isegm.inference.clicker import Click, Clicker

        clicker = Clicker(
            init_clicks=[
                *[
                    Click(is_positive=True, coords=(point[1], point[0]))
                    for point in positive_points
                ],
                *[
                    Click(is_positive=False, coords=(point[1], point[0]))
                    for point in negative_points
                ],
            ]
        )

        try:
            with torch.inference_mode():
                self.predictor.set_input_image(image_rgb)
                probabilities = self.predictor.get_prediction(clicker)
        except Exception as exc:
            raise HTTPException(status_code=500, detail="SimpleClick inference failed") from exc

        mask = probabilities > threshold
        return {
            "mask": _mask_as_png_base64(mask),
            "mask_format": "png",
            "mask_shape": [height, width],
            "positive_points_used": positive_points,
            "negative_points_used": negative_points,
            "threshold": threshold,
        }


if __name__ == "__main__":
    print(f"Deploy with: modal deploy {Path(__file__).name}")
