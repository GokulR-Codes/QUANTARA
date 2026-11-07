"""Single-target tracker with explicit loss / ambiguity reporting.

IMPORTANT LIMITS (also stated in docs/tracking.md)
-------------------------------------------------
* The tracker follows the region the **operator selected**.  It does not detect people,
  recognise faces, estimate age, or know who anyone is.
* OpenCV trackers are appearance-based single-object trackers.  They can drift onto a
  similar-looking person *without reporting failure*.  This module adds heuristic guards
  (appearance similarity to the selected template, motion plausibility, scale plausibility)
  that reduce — but cannot eliminate — that risk.  Frames that fail a guard are reported
  as ``ambiguous`` and are never used for crops.
* After the tracker reports failure (or the box leaves the frame) the target is
  ``lost``.  By default it is **not** re-acquired automatically: all remaining frames are
  invalid, because "finding something again" cannot prove it is the same person.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import cv2
import numpy as np

from backend.app.core.logging_config import get_logger
from src.video.geometry import BBox, BBoxError, clamp_bbox, visible_fraction

logger = get_logger(__name__)


class TrackStatus(str, Enum):
    TRACKED = "tracked"            # trusted: a crop may be generated
    AMBIGUOUS = "ambiguous"        # tracker reports success but guards say "don't trust"
    LOST = "lost"                  # tracker reported failure (terminal by default)
    OUT_OF_FRAME = "out_of_frame"  # box (mostly) left the frame (treated like lost)


class TrackerInitError(RuntimeError):
    pass


@dataclass
class TrackerParams:
    min_similarity: float = 0.30
    resume_similarity: float = 0.45
    max_jump_ratio: float = 1.0
    max_area_ratio: float = 3.0
    min_visible_fraction: float = 0.5
    min_box_px: int = 8
    confirm_frames: int = 3
    allow_reacquisition: bool = False


@dataclass
class StepResult:
    status: TrackStatus
    bbox: Optional[BBox] = None          # tracker's raw output (may extend past the frame)
    similarity: Optional[float] = None   # HSV-histogram correlation vs selected template
    reason: str = ""
    resumed: bool = False                # True on the first trusted frame after ambiguity


def create_cv_tracker() -> tuple[Any, str]:
    """Create the best available OpenCV tracker and report which one it really is."""
    attempts: list[tuple[str, Callable[[], Any]]] = []
    for label, owner, fname in (
        ("CSRT", cv2, "TrackerCSRT_create"),
        ("CSRT(legacy)", getattr(cv2, "legacy", None), "TrackerCSRT_create"),
        ("KCF", cv2, "TrackerKCF_create"),
        ("KCF(legacy)", getattr(cv2, "legacy", None), "TrackerKCF_create"),
        ("MIL", cv2, "TrackerMIL_create"),
    ):
        fn = getattr(owner, fname, None) if owner is not None else None
        if fn is not None:
            attempts.append((label, fn))
    if not attempts:
        raise RuntimeError(
            "No OpenCV single-object tracker is available in this OpenCV build "
            f"({cv2.__version__}). Install opencv-contrib-python-headless."
        )
    label, fn = attempts[0]
    return fn(), label


def hsv_histogram(frame_bgr: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
    """Normalised H-S histogram of the (clamped) box region, or None if empty."""
    H, W = frame_bgr.shape[:2]
    cb = clamp_bbox(bbox, W, H)
    if cb is None or cb.w < 2 or cb.h < 2:
        return None
    region = frame_bgr[cb.y:cb.y2, cb.x:cb.x2]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    return hist


def histogram_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_CORREL))


class TargetTracker:
    """Wraps an OpenCV tracker and classifies every step."""

    def __init__(
        self,
        params: Optional[TrackerParams] = None,
        tracker_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.params = params or TrackerParams()
        self._factory = tracker_factory
        self._tracker: Any = None
        self.backend_name: str = "uninitialised"
        self._template: Optional[np.ndarray] = None
        self._last_good: Optional[BBox] = None
        self._steps_since_good = 0
        self._state = "tracking"   # tracking | suspect | lost
        self._confirm = 0
        self._frame_size: tuple[int, int] = (0, 0)
        self.initial_bbox: Optional[BBox] = None

    # -- lifecycle -------------------------------------------------------
    def init(self, frame_bgr: np.ndarray, bbox: BBox) -> None:
        H, W = frame_bgr.shape[:2]
        self._frame_size = (W, H)
        template = hsv_histogram(frame_bgr, bbox)
        if template is None:
            raise TrackerInitError("Selected box contains no usable pixels")
        self._template = template
        if self._factory is not None:
            self._tracker, self.backend_name = self._factory(), "injected"
        else:
            self._tracker, self.backend_name = create_cv_tracker()
        # OpenCV >= 4.5.1 returns None from init(); only an explicit False means failure.
        res = self._tracker.init(frame_bgr, bbox.as_tuple())
        if res is False:
            raise TrackerInitError("Tracker initialisation failed for the selected box")
        self.initial_bbox = bbox
        self._last_good = bbox
        self._steps_since_good = 0
        self._state = "tracking"
        self._confirm = 0
        logger.info("Tracker initialised (%s) with box %s", self.backend_name, bbox.as_dict())

    @property
    def is_lost(self) -> bool:
        return self._state == "lost"

    # -- per-frame update ---------------------------------------------------
    def update(self, frame_bgr: np.ndarray) -> StepResult:
        if self._tracker is None or self._template is None or self._last_good is None:
            raise RuntimeError("TargetTracker.update() called before init()")
        p = self.params
        W, H = self._frame_size

        if self._state == "lost" and not p.allow_reacquisition:
            return StepResult(
                TrackStatus.LOST, None, None,
                "target lost earlier; automatic re-acquisition is disabled",
            )

        ok, raw = self._tracker.update(frame_bgr)
        self._steps_since_good += 1

        if not ok or raw is None:
            self._state = "lost"
            return StepResult(TrackStatus.LOST, None, None, "tracker reported failure")

        try:
            box = BBox(int(round(raw[0])), int(round(raw[1])), int(round(raw[2])), int(round(raw[3])))
        except (TypeError, ValueError, IndexError):
            self._state = "lost"
            return StepResult(TrackStatus.LOST, None, None, "tracker returned a malformed box")

        vis = visible_fraction(box, W, H)
        if vis < p.min_visible_fraction:
            self._state = "lost"
            return StepResult(
                TrackStatus.OUT_OF_FRAME, box, None,
                f"only {vis:.0%} of the box is inside the frame",
            )
        if self._state == "lost":
            # Re-acquisition explicitly enabled: treat as a suspect candidate to confirm.
            self._state, self._confirm = "suspect", 0

        reasons: list[str] = []
        if box.w < p.min_box_px or box.h < p.min_box_px:
            reasons.append(f"box collapsed to {box.w}x{box.h}px")

        hist = hsv_histogram(frame_bgr, box)
        sim = histogram_similarity(self._template, hist) if hist is not None else None
        threshold = p.min_similarity if self._state == "tracking" else p.resume_similarity
        if sim is None:
            reasons.append("no pixels under box")
        elif sim < threshold:
            reasons.append(f"appearance similarity {sim:.2f} < {threshold:.2f}")

        ref = self._last_good
        allowance = p.max_jump_ratio * (1 + max(0, self._steps_since_good - 1))
        cx, cy = box.center
        rx, ry = ref.center
        jump = float(np.hypot(cx - rx, cy - ry)) / max(ref.diagonal, 1.0)
        if jump > allowance:
            reasons.append(f"implausible jump {jump:.2f} box-diagonals (limit {allowance:.2f})")
        if box.area > 0 and ref.area > 0:
            ratio = max(box.area / ref.area, ref.area / box.area)
            if ratio > p.max_area_ratio:
                reasons.append(f"implausible scale change x{ratio:.1f}")

        if reasons:
            self._confirm = 0
            if self._state == "tracking":
                self._state = "suspect"
            return StepResult(TrackStatus.AMBIGUOUS, box, sim, "; ".join(reasons))

        # Guards passed
        if self._state == "tracking":
            self._last_good, self._steps_since_good = box, 0
            return StepResult(TrackStatus.TRACKED, box, sim)

        # state == suspect: require consecutive confirmations before trusting again
        self._confirm += 1
        if self._confirm >= p.confirm_frames:
            self._state, self._confirm = "tracking", 0
            self._last_good, self._steps_since_good = box, 0
            return StepResult(TrackStatus.TRACKED, box, sim, "resumed after ambiguity", resumed=True)
        return StepResult(
            TrackStatus.AMBIGUOUS, box, sim,
            f"confirming candidate ({self._confirm}/{p.confirm_frames})",
        )
