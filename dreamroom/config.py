"""Central configuration for the dreamroom pipeline.

Environment variables are read once at import time:

- ``DREAMROOM_SIMPLECLICK_ENDPOINT``: optional remote SimpleClick endpoint.
- ``DREAMROOM_MOGE_ENDPOINT``: optional MoGe-2 endpoint override.
- ``DREAMROOM_ONEFORMER_ENDPOINT``: optional OneFormer endpoint override.
- ``DREAMROOM_SAM3_MODEL``: optional fal.ai SAM 3 model ID override.
- ``FAL_KEY``: fal.ai API credential consumed by ``fal-client``.
- ``ARK_API_KEY``: BytePlus ModelArk API credential for Seedream rendering.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SIMPLECLICK_ENDPOINT = (
    "https://blakestieper--simpleclick-interactive-segmentation-simpl-771f03.modal.run"
)
DEFAULT_MOGE_ENDPOINT = "https://blakestieper--moge-2-api-web.modal.run"
DEFAULT_ONEFORMER_ENDPOINT = "https://blakestieper--oneformer-semantic-segmentation-oneformers-e54847.modal.run"


@dataclass(frozen=True)
class Settings:
    """Runtime settings for pipeline tasks."""

    # Image preparation
    max_side: int = 1280

    # Remote SimpleClick segmentation
    simpleclick_endpoint: str = os.getenv(
        "DREAMROOM_SIMPLECLICK_ENDPOINT", DEFAULT_SIMPLECLICK_ENDPOINT
    )
    simpleclick_timeout: float = 300.0
    threshold: float = 0.49
    max_points: int = 24

    # MoGe-2 API (production Modal endpoint is the default)
    moge_enabled: bool = True
    debug: bool = False  # request and persist the slower MoGe debug assets
    moge_endpoint: str | None = (
        os.getenv("DREAMROOM_MOGE_ENDPOINT") or DEFAULT_MOGE_ENDPOINT
    )
    moge_timeout: float = 300.0

    # Remote OneFormer semantic room-surface segmentation
    oneformer_endpoint: str | None = DEFAULT_ONEFORMER_ENDPOINT
    oneformer_timeout: float = 300.0

    # fal.ai SAM 3 room-surface segmentation
    sam3_model: str = os.getenv("DREAMROOM_SAM3_MODEL", "fal-ai/sam-3/image")
    sam3_timeout: float = 300.0
    sam3_min_score: float = 0.25

    # fal.ai Gemini object removal
    gemini_model: str = os.getenv(
        "DREAMROOM_GEMINI_MODEL", "google/nano-banana-lite/edit"
    )
    gemini_timeout: float = 300.0

    # Optional replacement dimensions in meters
    target_width_m: float | None = None
    target_depth_m: float | None = None
    target_height_m: float | None = None
    wall_snap_distance_m: float = 0.4

    # Optional Seedream furniture render
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
