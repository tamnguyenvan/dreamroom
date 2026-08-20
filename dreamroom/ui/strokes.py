"""Select an object by drawing polylines and remote SimpleClick segmentation.

Controls (annotate window):
    left-drag   positive polyline (red) - on the object
    right-drag  negative polyline (blue) - on the background
    u           undo last stroke
    c           clear all strokes
    Enter/Space  finish drawing; the app closes the window and runs segmentation
Preview window (after segmentation):
    y / Enter   confirm the mask
    r / n       redraw the strokes
    Esc / q     abort the pipeline
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from ..segmenter import sample_points
from .window import ENTER_KEYS, WindowApp, draw_banner, overlay_mask

POSITIVE_COLOR = (50, 50, 230)  # red (BGR)
NEGATIVE_COLOR = (230, 130, 40)  # blue (BGR)
MASK_COLOR = (0, 200, 0)  # green overlay
MIN_DRAG_DISTANCE_PX = 3  # minimum spacing between stroke points (display px)

SegmentFn = Callable[[list[list[int]], list[list[int]]], np.ndarray]


@dataclass
class ObjectSelection:
    """Confirmed object selection in resized-image coordinates."""

    mask: np.ndarray
    positive_points: list[list[int]]
    negative_points: list[list[int]]


class SelectObjectApp(WindowApp):
    """Stroke annotation + remote SimpleClick segmentation with confirmation."""

    def __init__(
        self,
        image_bgr: np.ndarray,
        segment_fn: SegmentFn,
        max_points: int = 24,
        max_display_width: int = 1200,
    ) -> None:
        super().__init__(image_bgr, "dreamroom | select object", max_display_width)
        self._segment_fn = segment_fn
        self.max_points = max_points
        self.strokes: list[tuple[bool, list[list[int]]]] = []
        self.current_stroke: list[list[int]] = []
        self.current_is_positive = True
        self.dragging = False
        self.mask: np.ndarray | None = None
        self.mode = "annotate"  # annotate | preview
        self.message = ""
        self._redraw_requested = False

    # -- point collection -----------------------------------------------------
    def points(self, is_positive: bool) -> list[list[int]]:
        points = [
            point
            for stroke_is_positive, stroke in self.strokes
            if stroke_is_positive == is_positive
            for point in stroke
        ]
        return sample_points(points, self.max_points)

    # -- mouse ------------------------------------------------------------------
    def on_mouse(self, event: int, x: int, y: int, _flags: int) -> None:
        if self.mode != "annotate":
            return
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            self.dragging = True
            self.current_is_positive = event == cv2.EVENT_LBUTTONDOWN
            self.current_stroke = []
            self._add_drag_point(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self._add_drag_point(x, y)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP) and self.dragging:
            self._add_drag_point(x, y)
            self.dragging = False
            if self.current_stroke:
                self.strokes.append((self.current_is_positive, self.current_stroke))
            self.current_stroke = []

    def _add_drag_point(self, x: int, y: int) -> None:
        if self.current_stroke:
            previous = self.to_display_point(self.current_stroke[-1])
            if abs(x - previous[0]) + abs(y - previous[1]) < MIN_DRAG_DISTANCE_PX:
                return
        self.current_stroke.append(self.to_image_point(x, y))

    # -- keys -------------------------------------------------------------------
    def on_key(self, key: int) -> None:
        if self.mode == "busy":
            return
        if self.mode == "annotate":
            self._on_annotate_key(key)
        else:
            self._on_preview_key(key)

    def _on_annotate_key(self, key: int) -> None:
        if key == ord("u"):
            if self.strokes:
                self.strokes.pop()
        elif key == ord("c"):
            self.strokes.clear()
            self.current_stroke = []
        elif key in ENTER_KEYS or key == ord(" "):
            if self.points(is_positive=True):
                self._done = True
            else:
                self.message = "draw at least one positive stroke first"

    def _on_preview_key(self, key: int) -> None:
        if key == ord("y") or key in ENTER_KEYS:
            self.result = ObjectSelection(
                mask=self.mask,
                positive_points=self.points(is_positive=True),
                negative_points=self.points(is_positive=False),
            )
            self._done = True
        elif key in (ord("n"), ord("r")):
            self.strokes.clear()
            self.current_stroke = []
            self.mask = None
            self._redraw_requested = True
            self._done = True

    # -- two-phase flow --------------------------------------------------------
    def collect_strokes(self) -> tuple[list[list[int]], list[list[int]]] | None:
        """Collect strokes until Enter closes the annotation window."""

        previous_message = self.message
        self.mode = "annotate"
        self._done = False
        self._aborted = False
        self.result = None
        self.message = previous_message
        self.run()
        if self._aborted:
            return None
        positive = self.points(is_positive=True)
        if not positive:
            return None
        return positive, self.points(is_positive=False)

    def review_mask(self, mask: np.ndarray) -> ObjectSelection | str | None:
        """Show a segmented mask and return confirm, redraw, or exit."""

        self.mask = mask
        self.mode = "preview"
        self._done = False
        self._aborted = False
        self.result = None
        self._redraw_requested = False
        self.run()
        if self._aborted:
            return None
        if self._redraw_requested:
            return "redraw"
        return self.result

    # -- rendering --------------------------------------------------------------
    def render(self) -> np.ndarray:
        frame = self.base_frame()
        if self.mask is not None:
            frame = overlay_mask(frame, self.mask, color=MASK_COLOR)
        for is_positive, stroke in self.strokes:
            self._draw_stroke(frame, stroke, is_positive)
        if self.current_stroke:
            self._draw_stroke(frame, self.current_stroke, self.current_is_positive)
        return draw_banner(frame, self._banner_lines())

    def _draw_stroke(self, frame: np.ndarray, stroke: list[list[int]], is_positive: bool) -> None:
        color = POSITIVE_COLOR if is_positive else NEGATIVE_COLOR
        display_points = np.array([self.to_display_point(p) for p in stroke], dtype=np.int32)
        if len(display_points) > 1:
            cv2.polylines(frame, [display_points], False, color, 2, cv2.LINE_AA)
        for point in display_points:
            cv2.circle(frame, tuple(point), 2, color, -1, cv2.LINE_AA)

    def _banner_lines(self) -> list[str]:
        if self.mode == "preview":
            return ["confirm mask? [y/Enter] accept   [r/n] redraw   [Esc] exit"]
        lines = [
            "left-drag: positive stroke   right-drag: negative stroke",
            "[Enter] finish and segment   [u] undo   [c] clear   [Esc] abort",
        ]
        if self.message:
            lines.append(self.message)
        return lines


def select_object(
    image_bgr: np.ndarray,
    segment_fn: SegmentFn,
    max_points: int = 24,
    max_display_width: int = 1200,
) -> ObjectSelection | None:
    """Collect strokes, close the window, segment, then confirm the mask."""

    app = SelectObjectApp(image_bgr, segment_fn, max_points, max_display_width)
    while True:
        strokes = app.collect_strokes()
        if strokes is None:
            return None
        positive, negative = strokes
        try:
            print("[object] annotation complete; calling remote SimpleClick...")
            mask = segment_fn(positive, negative)
        except Exception as exc:  # noqa: BLE001 - report and let the user retry
            app.message = f"segmentation failed: {exc}"
            continue
        reviewed = app.review_mask(mask)
        if reviewed == "redraw":
            continue
        return reviewed if isinstance(reviewed, ObjectSelection) else None
