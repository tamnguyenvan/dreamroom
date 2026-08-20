"""Central configuration for the dreamroom pipeline.

Environment variables are read once at import time:

- ``DREAMROOM_SIMPLECLICK_ROOT``: path of the local SimpleClick clone.
- ``DREAMROOM_CHECKPOINT``: path of the SimpleClick checkpoint file.
- ``DREAMROOM_MOGE_ENDPOINT``: optional MoGe-2 endpoint override.
- ``DREAMROOM_SAM3_MODEL``: optional fal.ai SAM 3 model ID override.
- ``FAL_KEY``: fal.ai API credential consumed by ``fal-client``.
- ``ARK_API_KEY``: BytePlus ModelArk API credential for Seedream rendering.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the pipeline steps."""

    # Step 0: resize
    max_side: int = 1280

    # Step 1: SimpleClick segmentation (ported from the Modal app)
    simpleclick_root: Path = _env_path(
        "DREAMROOM_SIMPLECLICK_ROOT", PROJECT_ROOT / "third_party" / "SimpleClick"
    )
    checkpoint_path: Path = _env_path(
        "DREAMROOM_CHECKPOINT", PROJECT_ROOT / "weights" / "cocolvis_vit_huge.pth"
    )
    threshold: float = 0.49
    max_points: int = 24
    model_input_size: int = 448
    max_longest_size: int = 800
    zoom_in_expansion: float = 1.4
    with_flip: bool = True  # disable on CPU for ~2x faster clicks

    # Step 3: MoGe-2 API (production Modal endpoint is the default)
    moge_enabled: bool = True
    debug: bool = False  # request and persist the slower MoGe debug assets
    moge_endpoint: str | None = os.getenv("DREAMROOM_MOGE_ENDPOINT") or None
    moge_timeout: float = 300.0

    # Step 4: fal.ai SAM 3 room-surface segmentation
    sam3_model: str = os.getenv("DREAMROOM_SAM3_MODEL", "fal-ai/sam-3/image")
    sam3_timeout: float = 300.0
    sam3_min_score: float = 0.25

    # Step 7: optional replacement dimensions in meters
    target_width_m: float | None = None
    target_depth_m: float | None = None
    target_height_m: float | None = None

    # Step 8: optional Seedream furniture render
    furniture_path: Path | None = None
    seedream_endpoint: str = os.getenv(
        "DREAMROOM_SEEDREAM_ENDPOINT",
        "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations",
    )
    seedream_model: str = os.getenv(
        "DREAMROOM_SEEDREAM_MODEL", "dola-seedream-5-0-pro-260628"
    )
    seedream_timeout: float = 300.0

    # UI
    max_display_width: int = 1200

    # Outputs
    outputs_root: Path = PROJECT_ROOT / "outputs"
