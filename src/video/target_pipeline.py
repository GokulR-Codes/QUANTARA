"""Target-specific frame sampling: tracking output aligned to sampled-frame timestamps.

For every sampled frame (one per ``1/sample_fps`` slot, starting at the operator's
selection frame) this module records *exactly one* :class:`SampleRecord` that states
whether the target was trusted on that frame, and — only if it was — writes a crop of the
tracked box.  Frames where the target is lost, ambiguous or out of frame produce **no
crop**; they are never replaced by another region or another person.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import cv2

from backend.app.core.errors import JobCancelled, VideoProcessingError
from backend.app.core.logging_config import get_logger
from src.video.extractor import FrameReader, FrameSampler
from src.video.geometry import BBox, crop_target
from src.video.tracker import StepResult, TargetTracker, TrackerParams, TrackStatus

logger = get_logger(__name__)

TRACKING_FILE = "tracking.json"
CROPS_DIR = "crops"
SCHEMA_VERSION = 1


@dataclass
class SampleRecord:
    sample_index: int
    frame_index: int
    timestamp_ms: float
    status: str                      # TrackStatus value
    bbox: Optional[dict] = None      # tracker box in source-pixel coordinates
    similarity: Optional[float] = None
    reason: str = ""
    resumed: bool = False
    crop_file: Optional[str] = None  # relative to the job directory; only when status == tracked


@dataclass
class TargetExtractionResult:
    samples: list[SampleRecord] = field(default_factory=list)
    intervals: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def tracked(self) -> list[SampleRecord]:
        return [s for s in self.samples if s.status == TrackStatus.TRACKED.value]


def build_intervals(samples: list[SampleRecord], sample_interval_ms: float) -> list[dict]:
    """Group consecutive non-tracked samples into invalid intervals (per status)."""
    intervals: list[dict] = []
    cur: Optional[dict] = None
    for s in samples:
        if s.status == TrackStatus.TRACKED.value:
            cur = None
            continue
        if cur is not None and cur["status"] == s.status and cur["end_sample_index"] == s.sample_index - 1:
            cur["end_sample_index"] = s.sample_index
            cur["end_ms"] = s.timestamp_ms
            cur["n_samples"] += 1
            if s.reason and s.reason not in cur["reasons"] and len(cur["reasons"]) < 4:
                cur["reasons"].append(s.reason)
        else:
            cur = {
                "status": s.status,
                "start_sample_index": s.sample_index,
                "end_sample_index": s.sample_index,
                "start_ms": s.timestamp_ms,
                "end_ms": s.timestamp_ms,
                "n_samples": 1,
                "reasons": [s.reason] if s.reason else [],
            }
            intervals.append(cur)
    for iv in intervals:
        # Each sample represents one sampling slot, so the interval covers n slots.
        iv["duration_ms"] = round(iv["n_samples"] * sample_interval_ms, 3)
    return intervals


def extract_target_crops(
    video_path: Path,
    job_dir: Path,
    bbox: BBox,
    selection_frame_index: int,
    *,
    sample_fps: float = 2.0,
    tracking_fps: float = 10.0,
    max_samples: int = 150,
    crop_size: tuple[int, int] = (112, 112),
    context_pad: float = 0.10,
    params: Optional[TrackerParams] = None,
    tracker_factory: Optional[Callable[[], Any]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    video_duration_s: Optional[float] = None,
) -> TargetExtractionResult:
    """Track the operator-selected target and write timestamp-aligned crops.

    Raises
    ------
    VideoProcessingError  if the selection frame cannot be found / tracker cannot start.
    JobCancelled          if ``should_cancel()`` returns True (partial output is left for
                          the caller to clean up).
    """
    job_dir = Path(job_dir)
    crops_dir = job_dir / CROPS_DIR
    crops_dir.mkdir(parents=True, exist_ok=True)
    params = params or TrackerParams()

    sample_interval_ms = 1000.0 / sample_fps
    track_interval_ms = 1000.0 / max(tracking_fps, 0.1)
    eps = 1e-3

    tracker = TargetTracker(params, tracker_factory)
    samples: list[SampleRecord] = []
    frames_decoded = skipped_pre = track_updates = 0
    initialised = False
    sampler: Optional[FrameSampler] = None
    next_track_ms = 0.0
    truncated = False
    last_sample_ms = 0.0
    selection_time_ms: Optional[float] = None

    def _write_sample(df_index: int, t_ms: float, frame, step: StepResult) -> None:
        crop_rel: Optional[str] = None
        if step.status == TrackStatus.TRACKED and step.bbox is not None:
            crop = crop_target(frame, step.bbox, crop_size, context_pad)
            crop_rel = f"{CROPS_DIR}/crop_{len(samples):05d}.png"
            if not cv2.imwrite(str(job_dir / crop_rel), crop):
                raise VideoProcessingError(f"Could not write crop {crop_rel}")
        samples.append(SampleRecord(
            sample_index=len(samples),
            frame_index=df_index,
            timestamp_ms=round(t_ms, 3),
            status=step.status.value,
            bbox=step.bbox.as_dict() if step.bbox is not None else None,
            similarity=round(step.similarity, 4) if step.similarity is not None else None,
            reason=step.reason,
            resumed=step.resumed,
            crop_file=crop_rel,
        ))

    with FrameReader(video_path) as reader:
        for df in reader:
            frames_decoded += 1
            if df.index < selection_frame_index:
                skipped_pre += 1
                continue

            if not initialised:
                if df.index != selection_frame_index:
                    raise VideoProcessingError(
                        f"Selection frame {selection_frame_index} was not found in the video"
                    )
                try:
                    tracker.init(df.frame, bbox)
                except Exception as exc:  # TrackerInitError or OpenCV error
                    raise VideoProcessingError(f"Tracker could not start: {exc}") from exc
                initialised = True
                selection_time_ms = df.t_ms
                sampler = FrameSampler(sample_fps, start_ms=df.t_ms)  # first sample *is* the selection frame
                sampler.should_sample(df.t_ms)
                next_track_ms = (math.floor((df.t_ms + eps) / track_interval_ms) + 1) * track_interval_ms
                _write_sample(df.index, df.t_ms, df.frame, StepResult(
                    TrackStatus.TRACKED, bbox, 1.0, "operator-selected frame",
                ))
                last_sample_ms = df.t_ms
                continue

            assert sampler is not None
            sampled = sampler.should_sample(df.t_ms)
            if sampled and len(samples) >= max_samples:
                truncated = True
                break
            if not sampled and df.t_ms + eps < next_track_ms:
                continue

            step = tracker.update(df.frame)
            track_updates += 1
            next_track_ms = (math.floor((df.t_ms + eps) / track_interval_ms) + 1) * track_interval_ms
            if track_updates % 25 == 0 and should_cancel is not None and should_cancel():
                raise JobCancelled()
            if sampled:
                _write_sample(df.index, df.t_ms, df.frame, step)
                last_sample_ms = df.t_ms
        timestamp_source = reader.timestamp_source

    if not initialised:
        raise VideoProcessingError(
            f"Selection frame {selection_frame_index} was not found in the video "
            f"({frames_decoded} frames decoded)"
        )
    if should_cancel is not None and should_cancel():
        raise JobCancelled()

    intervals = build_intervals(samples, sample_interval_ms)
    counts = {s.value: 0 for s in TrackStatus}
    for s in samples:
        counts[s.status] += 1
    total = len(samples)
    tracked = counts[TrackStatus.TRACKED.value]
    lost_like = [iv for iv in intervals if iv["status"] in (TrackStatus.LOST.value, TrackStatus.OUT_OF_FRAME.value)]
    summary = {
        "tracking_enabled": True,
        "tracker_backend": tracker.backend_name,
        "timestamp_source": timestamp_source,
        "selection": {
            "frame_index": selection_frame_index,
            "time_ms": selection_time_ms,
            "bbox": bbox.as_dict(),
        },
        "sample_fps": sample_fps,
        "tracking_fps": tracking_fps,
        "frames_decoded": frames_decoded,
        "frames_before_selection_skipped": skipped_pre,
        "tracker_updates": track_updates,
        "samples_total": total,
        "frames_tracked": tracked,
        "frames_ambiguous": counts[TrackStatus.AMBIGUOUS.value],
        "frames_lost": counts[TrackStatus.LOST.value] + counts[TrackStatus.OUT_OF_FRAME.value],
        "frames_out_of_frame": counts[TrackStatus.OUT_OF_FRAME.value],
        "frames_invalid": total - tracked,
        "tracking_coverage_pct": round(100.0 * tracked / total, 2) if total else 0.0,
        "tracking_failures": len(lost_like),
        "ambiguity_events": sum(1 for iv in intervals if iv["status"] == TrackStatus.AMBIGUOUS.value),
        "resumed_after_ambiguity": sum(1 for s in samples if s.resumed),
        "first_loss_ms": lost_like[0]["start_ms"] if lost_like else None,
        "crops_generated": tracked,
        "intervals": intervals,
        "truncated": truncated,
        "last_sample_ms": last_sample_ms,
        "reacquisition_enabled": params.allow_reacquisition,
    }
    if video_duration_s is not None:
        summary["video_duration_s"] = round(video_duration_s, 3)

    result = TargetExtractionResult(samples=samples, intervals=intervals, summary=summary)
    doc = {
        "schema_version": SCHEMA_VERSION,
        "summary": summary,
        "tracker_params": asdict(params),
        "crop": {"size_hw": list(crop_size), "context_pad": context_pad, "letterboxed": True},
        "samples": [asdict(s) for s in samples],
    }
    (job_dir / TRACKING_FILE).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    logger.info(
        "Target crops: %d samples, tracked=%d ambiguous=%d lost=%d (backend=%s)",
        total, tracked, summary["frames_ambiguous"], summary["frames_lost"], tracker.backend_name,
    )
    return result
