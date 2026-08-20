"""Client for the deployed MoGe-2 API (production Modal endpoint).

POSTs one image to ``/predict`` and parses the returned ZIP:

- ``output.glb``      textured mesh (glb camera coords: +X right, +Y up, -Z forward)
- ``point_map.npy``   HxWx3 float32 metric points, NaN where invalid
- ``depth.png`` / ``normal.png``  debug visualizations
- ``metadata.json``   image size and normalized intrinsics (pp = 0.5, 0.5)

Note: the API downsizes the image to max-side 800, so the point map
resolution differs from the pipeline working image (max-side 1280).
"""

from __future__ import annotations

import io
import json
import logging
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import requests

from .config import DEFAULT_MOGE_ENDPOINT

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = f"{DEFAULT_MOGE_ENDPOINT.rstrip('/')}/predict"


@dataclass
class MogeResult:
    """Parsed MoGe-2 response."""

    point_map: np.ndarray  # HxWx3 float32, NaN where invalid
    metadata: dict
    glb_bytes: bytes | None = None
    depth_png: bytes | None = None
    normal_png: bytes | None = None

    @property
    def image_size(self) -> tuple[int, int]:
        """(width, height) of the point map."""

        width, height = self.metadata["image_size"]
        return int(width), int(height)

    @property
    def intrinsics(self) -> np.ndarray:
        """3x3 normalized intrinsics (fx,fy divided by width/height)."""

        return np.asarray(self.metadata["intrinsics"], dtype=np.float64)

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "point_map.npy", self.point_map)
        (out_dir / "moge_metadata.json").write_text(json.dumps(self.metadata, indent=2))
        if self.glb_bytes is not None:
            (out_dir / "output.glb").write_bytes(self.glb_bytes)
        if self.depth_png is not None:
            (out_dir / "depth.png").write_bytes(self.depth_png)
        if self.normal_png is not None:
            (out_dir / "normal.png").write_bytes(self.normal_png)


class MogeClient:
    """Thin client around the MoGe-2 ``/predict`` endpoint."""

    def __init__(self, endpoint: str | None = None, timeout: float = 300.0) -> None:
        endpoint = (endpoint or DEFAULT_ENDPOINT).rstrip("/")
        if not endpoint.endswith("/predict"):
            endpoint += "/predict"
        self.endpoint = endpoint
        self.timeout = timeout

    def predict(
        self,
        image_bgr: np.ndarray,
        *,
        include_mesh: bool = False,
        include_debug: bool = False,
    ) -> MogeResult:
        """Send one BGR image with optional mesh/debug response assets."""

        ok, png = cv2.imencode(".png", image_bgr)
        if not ok:
            raise ValueError("failed to PNG-encode the input image")
        logger.info("calling MoGe-2 at %s", self.endpoint)
        tic = time.time()
        response = requests.post(
            self.endpoint,
            files={"file": ("image.png", png.tobytes(), "image/png")},
            data={
                "include_mesh": "true" if include_mesh else "false",
                "include_debug": "true" if include_debug else "false",
            },
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"MoGe-2 HTTP {response.status_code}: {response.text[:300]}"
            )
        result = self.parse_zip(response.content)
        logger.info(
            "MoGe-2 responded in %.1fs: point map %s",
            time.time() - tic,
            result.point_map.shape,
        )
        return result

    @staticmethod
    def parse_zip(content: bytes) -> MogeResult:
        """Parse the response ZIP (separated from the network for testing)."""

        fields: dict = {"metadata": None, "point_map": None}
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            for name in archive.namelist():
                data = archive.read(name)
                if name == "point_map.npy":
                    fields["point_map"] = np.load(io.BytesIO(data), allow_pickle=False)
                elif name == "metadata.json":
                    fields["metadata"] = json.loads(data.decode())
                elif name == "output.glb":
                    fields["glb_bytes"] = data
                elif name == "depth.png":
                    fields["depth_png"] = data
                elif name == "normal.png":
                    fields["normal_png"] = data
        if fields["point_map"] is None or fields["metadata"] is None:
            raise RuntimeError("MoGe-2 response is missing point_map.npy or metadata.json")
        return MogeResult(**fields)
