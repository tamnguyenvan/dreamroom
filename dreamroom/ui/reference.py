"""Draw a reference line and enter its real-world length in meters.

Controls:
    console     enter the known line length before opening the window
    left-drag   draw the reference line (yellow)
    u / c       clear the line and redraw
    Enter       confirm the drawn line
    Esc / q     abort the pipeline
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .window import ENTER_KEYS, WindowApp, draw_banner, overlay_mask

LINE_COLOR = (0, 220, 230)  # yellow (BGR)
MIN_LINE_LENGTH_PX = 5.0


@dataclass
class ReferenceScale:
    """Confirmed reference scale mapping pixels to real-world meters."""

    start: list[int]
    end: list[int]
    pixel_length: float
    meters: float

    @property
    def px_per_meter(self) -> float:
        return self.pixel_length / self.meters

    @property
    def meters_per_px(self) -> float:
        return self.meters / self.pixel_length

    def to_dict(self) -> dict:
        data = asdict(self)
        data["px_per_meter"] = self.px_per_meter
        data["meters_per_px"] = self.meters_per_px
        return data


def prompt_meters() -> float | None:
    """Ask for a positive length in meters on the terminal."""

    print("\n\n\n==================================")
    print("enter the reference line length in meters (e.g. 1.8):")
    print("==================================")
    try:
        raw = input("  > ").strip()
    except EOFError:
        return None
    try:
        value = float(raw)
    except ValueError:
        print("  not a number - restart the reference step to try again")
        return None
    if value <= 0:
        print("  length must be positive - restart the reference step to try again")
        return None
    return value


class ReferenceLineApp(WindowApp):
    """Single straight reference line + terminal input for its length."""

    def __init__(
        self,
        image_bgr: np.ndarray,
        mask: np.ndarray | None = None,
        max_display_width: int = 1200,
        meters: float | None = None,
    ) -> None:
        super().__init__(image_bgr, "dreamroom | reference scale", max_display_width)
        self.mask = mask
        self.start: list[int] | None = None
        self.end: list[int] | None = None
        self.dragging = False
        self.meters = meters
        self.message = ""

    # -- helpers ---------------------------------------------------------------
    @property
    def has_line(self) -> bool:
        return self.start is not None and self.end is not None

    @property
    def pixel_length(self) -> float:
        if not self.has_line:
            return 0.0
        return math.dist(self.start, self.end)

    def _clear(self) -> None:
        self.start = None
        self.end = None
        self.dragging = False

    # -- mouse -------------------------------------------------------------------
    def on_mouse(self, event: int, x: int, y: int, _flags: int) -> None:
        if self._done:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start = self.to_image_point(x, y)
            self.end = self.start
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end = self.to_image_point(x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.end = self.to_image_point(x, y)
            self.dragging = False
            if self.pixel_length < MIN_LINE_LENGTH_PX:
                self._clear()
                self.message = "line too short - drag a longer line"


    # -- keys ---------------------------------------------------------------------
    def on_key(self, key: int) -> None:
        if key in (ord("u"), ord("c")):
            self._clear()
            self.message = ""
        elif key in ENTER_KEYS and self.has_line and self.meters is not None:
            self.result = ReferenceScale(
                start=list(self.start),
                end=list(self.end),
                pixel_length=round(self.pixel_length, 2),
                meters=self.meters,
            )
            self._done = True

    # -- rendering ------------------------------------------------------------------
    def render(self) -> np.ndarray:
        frame = self.base_frame()
        if self.mask is not None:
            frame = overlay_mask(frame, self.mask, alpha=0.25)
        if self.has_line:
            start = self.to_display_point(self.start)
            end = self.to_display_point(self.end)
            cv2.line(frame, start, end, LINE_COLOR, 2, cv2.LINE_AA)
            cv2.circle(frame, start, 4, LINE_COLOR, -1, cv2.LINE_AA)
            cv2.circle(frame, end, 4, LINE_COLOR, -1, cv2.LINE_AA)
            midpoint = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2 - 8)
            cv2.putText(
                frame,
                self._line_label(),
                midpoint,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                LINE_COLOR,
                1,
                cv2.LINE_AA,
            )
        return draw_banner(frame, self._banner_lines())

    def _line_label(self) -> str:
        label = f"{self.pixel_length:.0f} px"
        if self.meters is not None:
            label += f" = {self.meters:g} m  ({self.pixel_length / self.meters:.1f} px/m)"
        return label

    def _banner_lines(self) -> list[str]:
        lines = ["left-drag: draw a reference line on an object of known length"]
        if self.has_line and not self.dragging and self.meters is not None:
            lines.append(f"known length = {self.meters:g} m   [Enter] confirm   [u/c] redraw")
        else:
            lines.append("[u/c] redraw   [Esc] abort")
        if self.message:
            lines.append(self.message)
        return lines


def get_reference_scale(
    image_bgr: np.ndarray,
    mask: np.ndarray | None = None,
    max_display_width: int = 1200,
) -> ReferenceScale | None:
    """Read the known length, then draw and confirm its image-space line."""

    meters = prompt_meters()
    if meters is None:
        return None
    return ReferenceLineApp(image_bgr, mask, max_display_width, meters).run()
