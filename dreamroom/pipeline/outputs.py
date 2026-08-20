"""Persistence for completed pipeline artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from ..image_ops import mask_to_uint8, save_image
from ..ui import overlay_mask
from .models import PipelineContext


class OutputWriter:
    """Write the stable output contract for a pipeline run."""

    def save(self, context: PipelineContext, output_dir: str | Path | None = None) -> Path:
        if context.image_bgr is None or context.selection is None or context.reference is None:
            raise RuntimeError("cannot save an incomplete pipeline context")
        out_dir = self._resolve_output_dir(context, output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if not context.settings.debug and context.rendered_image is not None:
            (out_dir / "rendered_furniture.jpg").write_bytes(context.rendered_image)
            return out_dir

        save_image(out_dir / "image.png", context.image_bgr)
        save_image(out_dir / "mask.png", mask_to_uint8(context.selection.mask))
        save_image(out_dir / "overlay.png", overlay_mask(context.image_bgr, context.selection.mask))
        (out_dir / "selection.json").write_text(
            json.dumps(
                {
                    "positive_points": context.selection.positive_points,
                    "negative_points": context.selection.negative_points,
                    "mask_area_px": int(context.selection.mask.sum()),
                },
                indent=2,
            )
        )
        (out_dir / "reference.json").write_text(
            json.dumps(context.reference.to_dict(), indent=2)
        )
        assert context.original_size is not None
        (out_dir / "meta.json").write_text(
            json.dumps(
                {
                    "source_image": str(context.image_path.resolve()),
                    "original_size_hw": list(context.original_size),
                    "resized_size_hw": list(context.image_bgr.shape[:2]),
                    "resize_scale": context.resize_scale,
                    "note": "original_coord = resized_coord * resize_scale",
                    "max_side": context.settings.max_side,
                    "threshold": context.settings.threshold,
                    "simpleclick_endpoint": context.settings.simpleclick_endpoint,
                    "created": datetime.now().isoformat(timespec="seconds"),
                },
                indent=2,
            )
        )
        if context.moge is not None:
            self.save_moge_outputs(out_dir, context)
        return out_dir

    @staticmethod
    def _resolve_output_dir(context: PipelineContext, output_dir: str | Path | None) -> Path:
        if output_dir is not None:
            return Path(output_dir)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return context.settings.outputs_root / f"{context.image_path.stem}-{stamp}"

    @staticmethod
    def save_moge_outputs(out_dir: Path, context: PipelineContext) -> None:
        assert context.moge is not None
        if context.settings.debug:
            context.moge.save(out_dir)
        if context.box is None or context.floor is None or context.scale_correction is None:
            raise RuntimeError("MoGe output is missing fitted geometry")
        (out_dir / "box3d.json").write_text(
            json.dumps(
                {
                    "box": context.box.to_dict(),
                    "floor_plane": context.floor.to_dict(),
                    "floor_fit_method": context.floor_fit_method,
                    "scale_correction": context.scale_correction,
                    "calibration": context.calibration,
                    "coordinate_frame": "MoGe glb camera (+X right, +Y up, -Z forward), meters",
                },
                indent=2,
            )
        )
        (out_dir / "walls3d.json").write_text(
            json.dumps(
                {
                    "walls": [wall.to_dict() for wall in context.walls],
                    "floor_plane": context.floor.to_dict(),
                    "floor_fit_method": context.floor_fit_method,
                    "wall_fit_method": context.wall_fit_method,
                    "scale_correction": context.scale_correction,
                    "coordinate_frame": "MoGe glb camera (+X right, +Y up, -Z forward), meters",
                },
                indent=2,
            )
        )
        if context.surface_segmentation is not None:
            (out_dir / "surfaces.json").write_text(
                json.dumps(context.surface_segmentation.to_dict(), indent=2)
            )
            if context.settings.debug:
                save_image(
                    out_dir / "sam3_floor_mask.png",
                    mask_to_uint8(
                        context.surface_segmentation.combined_mask("floor")
                    ),
                )
                save_image(
                    out_dir / "sam3_rug_mask.png",
                    mask_to_uint8(context.surface_segmentation.combined_mask("rug")),
                )
                save_image(
                    out_dir / "sam3_wall_mask.png",
                    mask_to_uint8(context.surface_segmentation.combined_mask("wall")),
                )
        if context.debug_surfaces_2d is not None:
            save_image(out_dir / "debug_surfaces_2d.png", context.debug_surfaces_2d)
        if context.debug_2d is not None:
            save_image(out_dir / "debug_2d.png", context.debug_2d)
        if context.debug_3d is not None:
            (out_dir / "debug_3d.glb").write_bytes(context.debug_3d)
        if context.placement_orientation is not None:
            (out_dir / "placement.json").write_text(
                json.dumps(context.placement_orientation.to_dict(), indent=2)
            )
        if context.target_placement is not None:
            (out_dir / "target_box3d.json").write_text(
                json.dumps(context.target_placement.to_dict(), indent=2)
            )
        if context.debug_placement_2d is not None:
            save_image(
                out_dir / "debug_placement_2d.png",
                context.debug_placement_2d,
            )
        if context.debug_placement_3d is not None:
            (out_dir / "debug_placement_3d.glb").write_bytes(
                context.debug_placement_3d
            )
        if context.render_room is not None:
            save_image(out_dir / "render_room_target_box.png", context.render_room)
        if context.render_furniture is not None:
            save_image(out_dir / "render_furniture_reference.png", context.render_furniture)
        if context.rendered_image is not None:
            (out_dir / "rendered_furniture.jpg").write_bytes(context.rendered_image)
        if context.render_metadata is not None:
            (out_dir / "render.json").write_text(
                json.dumps(context.render_metadata, indent=2)
            )

    @staticmethod
    def write_stats(out_dir: Path, report: dict) -> None:
        (out_dir / "stats.json").write_text(json.dumps(report, indent=2))
