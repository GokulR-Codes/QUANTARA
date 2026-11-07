"""Bounding-box validation and target-crop helpers (pure NumPy/OpenCV, no I/O)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


class BBoxError(ValueError):
    """Raised when a bounding box is invalid for the given frame."""


@dataclass(frozen=True)
class BBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def diagonal(self) -> float:
        return float(np.hypot(self.w, self.h))

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


def validate_bbox(
    x: int, y: int, w: int, h: int,
    frame_width: int, frame_height: int,
    min_side: int = 8,
) -> BBox:
    """Strictly validate an operator-selected box against the *actual* frame size.

    Raises :class:`BBoxError` with a specific message for every failure mode; the box is
    never clamped or "repaired" because silently altering the operator's selection could
    change which person is tracked.
    """
    for name, v in (("x", x), ("y", y), ("w", w), ("h", h)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise BBoxError(f"bbox {name} must be an integer, got {type(v).__name__}")
    if frame_width <= 0 or frame_height <= 0:
        raise BBoxError(f"invalid frame size {frame_width}x{frame_height}")
    if x < 0 or y < 0:
        raise BBoxError(f"bbox origin ({x},{y}) is outside the frame (negative coordinate)")
    if w < min_side or h < min_side:
        raise BBoxError(
            f"bbox {w}x{h} is too small; each side must be at least {min_side} pixels"
        )
    if x >= frame_width or y >= frame_height:
        raise BBoxError(f"bbox origin ({x},{y}) is outside the {frame_width}x{frame_height} frame")
    if x + w > frame_width:
        raise BBoxError(
            f"bbox exceeds frame width: x+w={x + w} > {frame_width}"
        )
    if y + h > frame_height:
        raise BBoxError(
            f"bbox exceeds frame height: y+h={y + h} > {frame_height}"
        )
    return BBox(int(x), int(y), int(w), int(h))


def visible_fraction(bbox: BBox, frame_width: int, frame_height: int) -> float:
    """Fraction of the box area that lies inside the frame (0..1)."""
    if bbox.area <= 0:
        return 0.0
    ix = max(0, min(bbox.x2, frame_width) - max(bbox.x, 0))
    iy = max(0, min(bbox.y2, frame_height) - max(bbox.y, 0))
    return float(ix * iy) / float(bbox.area)


def iou(a: BBox, b: BBox) -> float:
    ix = max(0, min(a.x2, b.x2) - max(a.x, b.x))
    iy = max(0, min(a.y2, b.y2) - max(a.y, b.y))
    inter = ix * iy
    union = a.area + b.area - inter
    return float(inter) / float(union) if union > 0 else 0.0


def crop_target(
    frame_bgr: np.ndarray,
    bbox: BBox,
    out_size: tuple[int, int] = (112, 112),
    context_pad: float = 0.10,
) -> np.ndarray:
    """Crop the target box (+ small context pad), letterboxed to ``out_size`` (H, W).

    Aspect ratio is preserved; only pixels inside the padded box are used, so a crop never
    contains regions chosen for any other reason than being near the tracked target.
    """
    H, W = frame_bgr.shape[:2]
    out_h, out_w = out_size
    px = int(round(bbox.w * context_pad))
    py = int(round(bbox.h * context_pad))
    x0, y0 = max(0, bbox.x - px), max(0, bbox.y - py)
    x1, y1 = min(W, bbox.x2 + px), min(H, bbox.y2 + py)
    region = frame_bgr[y0:y1, x0:x1]
    if region.size == 0:
        raise BBoxError("bbox does not intersect the frame; cannot crop")
    rh, rw = region.shape[:2]
    scale = min(out_w / rw, out_h / rh)
    nw, nh = max(1, int(round(rw * scale))), max(1, int(round(rh * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(region, (nw, nh), interpolation=interp)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    oy, ox = (out_h - nh) // 2, (out_w - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def clamp_bbox(bbox: BBox, frame_width: int, frame_height: int) -> Optional[BBox]:
    """Clamp to the frame (for cropping only). Returns None if nothing remains."""
    x0, y0 = max(0, bbox.x), max(0, bbox.y)
    x1, y1 = min(frame_width, bbox.x2), min(frame_height, bbox.y2)
    if x1 <= x0 or y1 <= y0:
        return None
    return BBox(x0, y0, x1 - x0, y1 - y0)
