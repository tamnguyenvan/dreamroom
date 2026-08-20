"""Shared OpenCV window machinery for the interactive pipeline steps.

Frames are rendered at display resolution and shown in an AUTOSIZE window,
so mouse callback coordinates always map 1:1 to display pixels and are
converted to image coordinates by dividing by ``display_scale``.
"""

from __future__ import annotations

import cv2
import numpy as np

ESC = 27
ENTER_KEYS = frozenset({10, 13})
NO_KEY = 255


def overlay_mask(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 220, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend ``color`` over ``image_bgr`` where ``mask`` is true."""

    if mask.shape[:2] != image_bgr.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (image_bgr.shape[1], image_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    colored = np.zeros_like(image_bgr)
    colored[:, :] = color
    blended = cv2.addWeighted(image_bgr, 1.0 - alpha, colored, alpha, 0)
    return np.where(mask[..., None], blended, image_bgr)


def draw_banner(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw a translucent instruction banner at the top of ``frame``."""

    if not lines:
        return frame
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    line_height = 20
    box_height = line_height * len(lines) + 10
    banner = frame.copy()
    cv2.rectangle(banner, (0, 0), (frame.shape[1], box_height), (30, 30, 30), -1)
    frame = cv2.addWeighted(banner, 0.75, frame, 0.25, 0)
    for index, line in enumerate(lines):
        cv2.putText(
            frame, line, (8, 18 + index * line_height), font, scale, (240, 240, 240), thickness, cv2.LINE_AA
        )
    return frame


class WindowApp:
    """Base class: display scaling, mouse callback and the main event loop.

    Subclasses override :meth:`render`, :meth:`on_mouse` and :meth:`on_key`,
    set ``self.result`` and ``self._done = True`` to finish, and read the
    value returned by :meth:`run` (``None`` when the user aborts).
    """

    def __init__(self, image_bgr: np.ndarray, title: str, max_display_width: int = 1200) -> None:
        self.image = image_bgr
        self.title = title
        self.height, self.width = image_bgr.shape[:2]
        self.display_scale = min(1.0, max_display_width / self.width)
        self.display_width = max(1, round(self.width * self.display_scale))
        self.display_height = max(1, round(self.height * self.display_scale))
        self.result = None
        self._done = False
        self._aborted = False

    # -- coordinate helpers -------------------------------------------------
    def to_image_point(self, x: int, y: int) -> list[int]:
        image_x = min(self.width - 1, max(0, round(x / self.display_scale)))
        image_y = min(self.height - 1, max(0, round(y / self.display_scale)))
        return [image_x, image_y]

    def to_display_point(self, point: list[int] | tuple[int, int]) -> tuple[int, int]:
        return (
            round(point[0] * self.display_scale),
            round(point[1] * self.display_scale),
        )

    def base_frame(self) -> np.ndarray:
        return cv2.resize(self.image, (self.display_width, self.display_height))

    # -- hooks to override ----------------------------------------------------
    def render(self) -> np.ndarray:
        return self.base_frame()

    def on_mouse(self, event: int, x: int, y: int, flags: int) -> None:
        pass

    def on_key(self, key: int) -> None:
        pass

    # -- main loop ------------------------------------------------------------
    def run(self, *, close_is_abort: bool = True):
        cv2.namedWindow(self.title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.title, self._mouse_callback)
        try:
            while not (self._done or self._aborted):
                cv2.imshow(self.title, self.render())
                key = cv2.waitKey(30) & 0xFF
                if key == ESC:
                    self._aborted = True
                    break
                if not self._window_visible():
                    if close_is_abort:
                        self._aborted = True
                    break
                if key != NO_KEY:
                    self.on_key(key)
        finally:
            cv2.destroyWindow(self.title)
            cv2.waitKey(1)
        return None if self._aborted else self.result

    def flush(self) -> None:
        """Redraw immediately (used before a long blocking operation)."""

        cv2.imshow(self.title, self.render())
        cv2.waitKey(1)

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, _param: object) -> None:
        self.on_mouse(event, x, y, flags)

    def _window_visible(self) -> bool:
        try:
            return cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False
