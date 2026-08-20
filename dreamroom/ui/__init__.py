"""OpenCV-based interactive UI steps for the dreamroom pipeline."""

from .window import overlay_mask
from .strokes import ObjectSelection, SelectObjectApp, select_object
from .reference import ReferenceScale, ReferenceLineApp, get_reference_scale

__all__ = [
    "ObjectSelection",
    "SelectObjectApp",
    "select_object",
    "ReferenceScale",
    "ReferenceLineApp",
    "get_reference_scale",
]
