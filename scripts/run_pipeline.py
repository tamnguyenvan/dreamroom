#!/usr/bin/env python
"""Run the concurrent furniture replacement pipeline on one image.

Example:
    python scripts/run_pipeline.py --image path/to/room.jpg
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dreamroom.config import Settings  # noqa: E402
from dreamroom.pipeline import FurniturePipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="input image path")
    parser.add_argument("--output-dir", type=Path, default=None, help="defaults to outputs/<name>-<timestamp>")
    parser.add_argument("--max-side", type=int, default=None, help="resize target (default 1280)")
    parser.add_argument("--threshold", type=float, default=None, help="mask threshold (default 0.49)")
    parser.add_argument("--max-display-width", type=int, default=None, help="UI window width limit")
    parser.add_argument(
        "--moge-endpoint",
        default=None,
        help="MoGe-2 API URL (default: production deployment / DREAMROOM_MOGE_ENDPOINT)",
    )
    parser.add_argument("--moge-timeout", type=float, default=None, help="MoGe-2 request timeout (s)")
    parser.add_argument(
        "--sam3-model",
        default=None,
        help="fal.ai SAM 3 model ID (default: fal-ai/sam-3/image)",
    )
    parser.add_argument(
        "--sam3-timeout", type=float, default=None, help="SAM 3 request timeout (s)"
    )
    parser.add_argument(
        "--sam3-min-score",
        type=float,
        default=None,
        help="minimum SAM 3 mask confidence (default 0.25)",
    )
    parser.add_argument("--gemini-model", default=None, help="Gemini fal.ai model ID")
    parser.add_argument(
        "--gemini-timeout", type=float, default=None, help="Gemini fal.ai request timeout (s)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="request and save MoGe mesh/debug assets (slower)",
    )
    parser.add_argument(
        "--new-dimensions",
        type=float,
        nargs=3,
        metavar=("WIDTH", "DEPTH", "HEIGHT"),
        default=None,
        help="target width, depth, and height in meters",
    )
    parser.add_argument(
        "--wall-snap-distance",
        type=float,
        default=None,
        help="maximum rear-face distance to snap to its wall in meters (default 0.4)",
    )
    parser.add_argument(
        "--furniture",
        type=Path,
        default=None,
        help="furniture reference image; requires --new-dimensions",
    )
    parser.add_argument("--seedream-endpoint", default=None, help="Seedream API URL")
    parser.add_argument("--seedream-model", default=None, help="Seedream model ID")
    parser.add_argument(
        "--seedream-timeout", type=float, default=None, help="Seedream request timeout (s)"
    )
    parser.add_argument(
        "--skip-moge",
        action="store_true",
        help="skip MoGe, SAM 3, geometry, placement, and rendering",
    )
    args = parser.parse_args()
    dimensions = tuple(args.new_dimensions) if args.new_dimensions is not None else None
    if dimensions is not None:
        if any(value <= 0 for value in dimensions):
            parser.error("target dimensions must be positive")
    if args.sam3_min_score is not None and not 0.0 <= args.sam3_min_score <= 1.0:
        parser.error("--sam3-min-score must be between 0 and 1")
    if args.wall_snap_distance is not None and args.wall_snap_distance < 0.0:
        parser.error("--wall-snap-distance must be non-negative")
    if args.furniture is not None and dimensions is None:
        parser.error("--furniture requires --new-dimensions WIDTH DEPTH HEIGHT")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy_logger in ("httpx", "httpcore", "fal", "fal_client"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    settings = Settings()
    overrides = {}
    if args.max_side is not None:
        overrides["max_side"] = args.max_side
    if args.threshold is not None:
        overrides["threshold"] = args.threshold
    if args.max_display_width is not None:
        overrides["max_display_width"] = args.max_display_width
    if args.moge_endpoint is not None:
        overrides["moge_endpoint"] = args.moge_endpoint
    if args.moge_timeout is not None:
        overrides["moge_timeout"] = args.moge_timeout
    if args.sam3_model is not None:
        overrides["sam3_model"] = args.sam3_model
    if args.sam3_timeout is not None:
        overrides["sam3_timeout"] = args.sam3_timeout
    if args.sam3_min_score is not None:
        overrides["sam3_min_score"] = args.sam3_min_score
    if args.gemini_model is not None:
        overrides["gemini_model"] = args.gemini_model
    if args.gemini_timeout is not None:
        overrides["gemini_timeout"] = args.gemini_timeout
    if args.new_dimensions is not None:
        overrides["target_width_m"] = args.new_dimensions[0]
        overrides["target_depth_m"] = args.new_dimensions[1]
        overrides["target_height_m"] = args.new_dimensions[2]
    if args.wall_snap_distance is not None:
        overrides["wall_snap_distance_m"] = args.wall_snap_distance
    if args.furniture is not None:
        overrides["furniture_path"] = args.furniture
    if args.seedream_endpoint is not None:
        overrides["seedream_endpoint"] = args.seedream_endpoint
    if args.seedream_model is not None:
        overrides["seedream_model"] = args.seedream_model
    if args.seedream_timeout is not None:
        overrides["seedream_timeout"] = args.seedream_timeout
    if args.debug:
        overrides["debug"] = True
    if args.skip_moge:
        overrides["moge_enabled"] = False
    if overrides:
        settings = replace(settings, **overrides)

    result = FurniturePipeline(settings).run(args.image, args.output_dir)
    if result is None:
        raise SystemExit("pipeline aborted")
    print(f"scale: 1 px = {result.reference.meters_per_px * 100:.2f} cm")


if __name__ == "__main__":
    main()
