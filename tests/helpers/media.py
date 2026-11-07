"""Synthetic ground-truth videos for tests (never presented as real recordings)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# BGR quadrant colours of the synthetic "participant" (a high-contrast square)
_TR, _BL, _BR = (0, 200, 255), (0, 200, 0), (200, 60, 0)
_SWAP = (255, 0, 255)   # a *different-looking* object that replaces the target


def make_target_video(
    path: Path, *, n_frames: int = 40, fps: float = 10.0, size: tuple[int, int] = (160, 120),
    box: int = 36, x0: int = 10, y0: int = 40, x_step: int = 3,
    vanish_at: Optional[int] = None, swap_at: Optional[int] = None, seed: int = 0,
    fourcc: str = "mp4v",
) -> dict:
    """Write a video of a square moving right over a static noise texture.

    * ``vanish_at``: from this frame on the target is not drawn at all (target loss).
    * ``swap_at``: from this frame on a differently-coloured square occupies the target's
      place (tests that the tracker never silently continues on a different object).
    Returns ground truth: per-frame true box, fps and size.
    """
    w, h = size
    rng = np.random.RandomState(seed)
    bg = rng.randint(0, 60, (h, w, 3)).astype(np.uint8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    assert writer.isOpened(), "cv2.VideoWriter failed to open"
    truth = []
    half = box // 2
    for i in range(n_frames):
        f = bg.copy()
        x = x0 + x_step * i
        y = y0 + int(4 * np.sin(i / 5.0))
        drawn = True
        if vanish_at is not None and i >= vanish_at:
            drawn = False
        elif swap_at is not None and i >= swap_at:
            f[y:y + box, x:x + box] = _SWAP
        else:
            gray = (20 + 5 * i) % 256   # marker wraps past frame 47 (uint8)
            f[y:y + half, x:x + half] = (gray, gray, gray)
            f[y:y + half, x + half:x + box] = _TR
            f[y + half:y + box, x:x + half] = _BL
            f[y + half:y + box, x + half:x + box] = _BR
        writer.write(f)
        truth.append({"i": i, "t_ms": i * 1000.0 / fps, "x": x, "y": y, "w": box, "h": box, "drawn": drawn})
    writer.release()
    return {"truth": truth, "fps": fps, "size": size, "box": box, "path": str(path)}


def initial_bbox(gt: dict) -> dict:
    t = gt["truth"][0]
    return {"x": t["x"], "y": t["y"], "w": t["w"], "h": t["h"]}
