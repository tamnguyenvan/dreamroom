#!/usr/bin/env python
"""Run the furniture replacement pipeline (steps 0-6) on one image.

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
        "--no-flip",
        action="store_true",
        help="disable test-time flip (about 2x faster on CPU, slightly lower quality)",
    )
    parser.add_argument(
        "--moge-endpoint",
        default=None,
        help="MoGe-2 API URL (default: production deployment / DREAMROOM_MOGE_ENDPOINT)",
    )
    parser.add_argument("--moge-timeout", type=float, default=None, help="MoGe-2 request timeout (s)")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="request and save MoGe mesh/debug assets (slower)",
    )
    parser.add_argument("--new-width", type=float, default=None, help="target width in meters")
    parser.add_argument("--new-depth", type=float, default=None, help="target depth in meters")
    parser.add_argument("--new-height", type=float, default=None, help="target height in meters")
    parser.add_argument("--skip-moge", action="store_true", help="run only steps 0-2")
    args = parser.parse_args()
    dimensions = (args.new_width, args.new_depth, args.new_height)
    if any(value is not None for value in dimensions):
        if not all(value is not None for value in dimensions):
            parser.error("--new-width, --new-depth, and --new-height are required together")
        if any(value <= 0 for value in dimensions):
            parser.error("target dimensions must be positive")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    settings = Settings()
    overrides = {}
    if args.max_side is not None:
        overrides["max_side"] = args.max_side
    if args.threshold is not None:
        overrides["threshold"] = args.threshold
    if args.max_display_width is not None:
        overrides["max_display_width"] = args.max_display_width
    if args.no_flip:
        overrides["with_flip"] = False
    if args.moge_endpoint is not None:
        overrides["moge_endpoint"] = args.moge_endpoint
    if args.moge_timeout is not None:
        overrides["moge_timeout"] = args.moge_timeout
    if args.new_width is not None:
        overrides["target_width_m"] = args.new_width
        overrides["target_depth_m"] = args.new_depth
        overrides["target_height_m"] = args.new_height
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
