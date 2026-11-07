"""Video decoding, probing, and timestamp-based frame sampling.

Design notes
------------
* Frames are sampled by **timestamp** (every ``1 / sample_fps`` seconds), not by a
  frame-index stride derived from the reported FPS.  Browser ``MediaRecorder`` WebM
  files frequently report a meaningless FPS (e.g. 1000) and no frame count, which
  breaks index-stride sampling.
* Every decoded frame carries a timestamp (``t_ms``).  The container timestamp is used
  when it is monotonic; otherwise a timestamp is synthesised from the frame index and
  the FPS, and :attr:`FrameReader.timestamp_source` records that this happened.
* Padding is **mask-only**: no synthetic zero-frame files are written to disk and
  padded positions are never presented as real frames.
"""
from __future__ import annotations

import datetime
import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from backend.app.core.errors import VideoProcessingError, VideoValidationError
from backend.app.core.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_FALLBACK_FPS = 25.0
_MIN_VALID_FPS, _MAX_VALID_FPS = 1.0, 240.0


def _fps_is_valid(fps: float) -> bool:
    return bool(fps) and math.isfinite(fps) and _MIN_VALID_FPS <= fps <= _MAX_VALID_FPS


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------
@dataclass
class DecodedFrame:
    index: int          # 0-based index of the decoded frame in the file
    t_ms: float         # presentation timestamp in milliseconds
    frame: np.ndarray   # BGR uint8, original resolution


class FrameReader:
    """Iterate over every decoded frame of a video with robust timestamps."""

    def __init__(self, video_path: Path) -> None:
        self.video_path = Path(video_path)
        self._cap: Optional[cv2.VideoCapture] = None
        self.fps_reported: float = 0.0
        self.fps_used: float = DEFAULT_FALLBACK_FPS
        self._n_container = 0
        self._n_synth = 0

    def __enter__(self) -> "FrameReader":
        if not self.video_path.exists():
            raise VideoValidationError(f"Video file not found: {self.video_path}")
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            raise VideoValidationError(
                "Cannot open video file — format may be unsupported or the file is corrupt"
            )
        self.fps_reported = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps_used = self.fps_reported if _fps_is_valid(self.fps_reported) else DEFAULT_FALLBACK_FPS
        return self

    def __exit__(self, *exc) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __iter__(self) -> Iterator[DecodedFrame]:
        assert self._cap is not None, "FrameReader must be used as a context manager"
        idx = 0
        prev_t = -1.0
        step = 1000.0 / self.fps_used
        while True:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                return
            raw = float(self._cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            usable = math.isfinite(raw) and raw > prev_t and not (idx > 0 and raw <= 0.0)
            if idx == 0 and math.isfinite(raw) and raw >= 0.0:
                usable = True
            if usable:
                t_ms = raw
                self._n_container += 1
            else:
                t_ms = (prev_t + step) if idx > 0 else 0.0
                self._n_synth += 1
            prev_t = t_ms
            yield DecodedFrame(index=idx, t_ms=t_ms, frame=frame)
            idx += 1

    @property
    def timestamp_source(self) -> str:
        if self._n_container and not self._n_synth:
            return "container_pts"
        if self._n_synth and not self._n_container:
            return "synthesised_from_fps"
        if self._n_container and self._n_synth:
            return "mixed"
        return "none"


class FrameSampler:
    """Choose one frame per ``1/sample_fps`` slot, based on timestamps."""

    def __init__(self, sample_fps: float, start_ms: float = 0.0) -> None:
        if sample_fps <= 0:
            raise ValueError("sample_fps must be > 0")
        self.interval_ms = 1000.0 / sample_fps
        self._next_slot_ms = float(start_ms)   # first slot: a frame at/after start_ms
        self._eps = 1e-3

    def should_sample(self, t_ms: float) -> bool:
        if t_ms + self._eps < self._next_slot_ms:
            return False
        self._next_slot_ms = (math.floor((t_ms + self._eps) / self.interval_ms) + 1) * self.interval_ms
        return True


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------
def probe_video(video_path: Path) -> dict:
    """Validate a video and return trustworthy metadata.

    Dimensions always come from an actually decoded frame.  If the container's FPS or
    frame count is unusable (typical for browser-recorded WebM) the file is scanned once
    to obtain the real frame count and duration.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise VideoValidationError(f"Video file not found: {video_path}")
    if video_path.stat().st_size == 0:
        raise VideoValidationError("Video file is empty")

    warnings: list[str] = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise VideoValidationError(
            "Cannot open video file — format may be unsupported or the file is corrupt"
        )
    fps_reported = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    count_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    codec = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")
    cap.release()

    needs_scan = (not _fps_is_valid(fps_reported)) or count_reported <= 0
    width = height = 0
    decoded = 0
    first_t = last_t = 0.0
    source = "none"
    with FrameReader(video_path) as reader:
        for df in reader:
            if decoded == 0:
                height, width = df.frame.shape[:2]
                first_t = df.t_ms
                if not needs_scan:
                    decoded = 1
                    last_t = df.t_ms
                    break
            decoded += 1
            last_t = df.t_ms
        source = reader.timestamp_source

    if width <= 0 or height <= 0 or decoded == 0:
        raise VideoValidationError("Video contains no decodable frames")

    if needs_scan:
        frame_count = decoded
        if decoded > 1 and last_t > first_t:
            span_s = (last_t - first_t) / 1000.0
            fps_used = (decoded - 1) / span_s
            fps_used = fps_used if _fps_is_valid(fps_used) else DEFAULT_FALLBACK_FPS
            duration_s = span_s + 1.0 / fps_used
        else:
            fps_used = DEFAULT_FALLBACK_FPS
            duration_s = decoded / fps_used
        warnings.append(
            f"Container reported fps={fps_reported:g} / frame_count={count_reported}; "
            f"values were derived by decoding the file (fps≈{fps_used:.2f}, frames={decoded})."
        )
    else:
        frame_count = count_reported
        fps_used = fps_reported
        duration_s = frame_count / fps_used

    return {
        "fps": float(fps_used),
        "fps_reported": float(fps_reported),
        "frame_count": int(frame_count),
        "width": int(width),
        "height": int(height),
        "codec": codec,
        "duration_s": float(duration_s),
        "timestamp_source": source,
        "scanned": bool(needs_scan),
        "warnings": warnings,
    }


def read_frame_at(video_path: Path, t_ms: float = 0.0, frame_index: Optional[int] = None) -> DecodedFrame:
    """Return the first decoded frame at/after ``t_ms`` (or exactly ``frame_index``)."""
    with FrameReader(video_path) as reader:
        last: Optional[DecodedFrame] = None
        for df in reader:
            last = df
            if frame_index is not None:
                if df.index == frame_index:
                    return df
                if df.index > frame_index:
                    break
            elif df.t_ms + 1e-3 >= t_ms:
                return df
        if frame_index is not None:
            raise VideoValidationError(f"Video has no frame with index {frame_index}")
        if last is None:
            raise VideoValidationError("Video contains no decodable frames")
        return last  # t_ms beyond the end: return the final frame


# ---------------------------------------------------------------------------
# Legacy whole-frame extractor (kept for API compatibility)
# ---------------------------------------------------------------------------
@dataclass
class FrameMetadata:
    frame_index: int
    timestamp_ms: float
    source_video: str
    height: int
    width: int
    is_padding: bool = False


@dataclass
class ExtractionManifest:
    video_path: str
    job_id: str
    source_duration_s: float
    source_fps: float
    source_width: int
    source_height: int
    source_codec: str
    num_frames_extracted: int
    extraction_fps: float
    max_frames: int
    frame_height: int
    frame_width: int
    frames: list[FrameMetadata] = field(default_factory=list)
    extraction_duration_s: float = 0.0
    padding_applied: int = 0
    truncation_applied: bool = False
    created_at: str = ""
    timestamp_source: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class VideoFrameExtractor:
    """Extracts whole (non-target-specific) frames with timestamps.

    The processing worker uses :mod:`src.video.target_pipeline` for target-specific
    crops; this class remains useful for full-frame sampling and for validation.
    """

    def __init__(
        self,
        sample_fps: float = 2.0,
        max_frames: int = 150,
        frame_height: int = 112,
        frame_width: int = 112,
        sequence_length: int = 30,
    ) -> None:
        self.sample_fps = sample_fps
        self.max_frames = max_frames
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.sequence_length = sequence_length

    def validate_video(self, video_path: Path) -> dict:
        """Validate a video file and return metadata. Raises VideoValidationError."""
        return probe_video(video_path)

    def extract_frames(
        self,
        video_path: Path,
        output_dir: Path,
        job_id: Optional[str] = None,
    ) -> ExtractionManifest:
        """Extract timestamp-sampled frames and write ``manifest.json``.

        Only real frames are written to disk.  If fewer than ``sequence_length`` frames
        exist, the manifest lists the shortfall as ``is_padding`` entries (mask-only);
        if more exist, the surplus is **not** written, so disk contents and manifest agree.
        """
        job_id = job_id or str(uuid.uuid4())
        t0 = time.monotonic()
        meta = probe_video(video_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        sampler = FrameSampler(self.sample_fps)
        limit = min(self.max_frames, self.sequence_length)
        frames_meta: list[FrameMetadata] = []
        truncated = False

        with FrameReader(video_path) as reader:
            for df in reader:
                if not sampler.should_sample(df.t_ms):
                    continue
                if len(frames_meta) >= limit:
                    truncated = True
                    break
                resized = cv2.resize(df.frame, (self.frame_width, self.frame_height))
                np.save(str(output_dir / f"frame_{len(frames_meta):05d}.npy"), resized)
                frames_meta.append(FrameMetadata(
                    frame_index=df.index,
                    timestamp_ms=df.t_ms,
                    source_video=str(video_path),
                    height=self.frame_height,
                    width=self.frame_width,
                ))
            ts_source = reader.timestamp_source

        padding = max(0, self.sequence_length - len(frames_meta))
        for _ in range(padding):
            frames_meta.append(FrameMetadata(
                frame_index=-1, timestamp_ms=-1.0, source_video=str(video_path),
                height=self.frame_height, width=self.frame_width, is_padding=True,
            ))
        if padding:
            logger.info("Sequence short by %d frames; marked as mask-only padding", padding)

        elapsed = time.monotonic() - t0
        manifest = ExtractionManifest(
            video_path=str(video_path),
            job_id=job_id,
            source_duration_s=meta["duration_s"],
            source_fps=meta["fps"],
            source_width=meta["width"],
            source_height=meta["height"],
            source_codec=meta["codec"],
            num_frames_extracted=len(frames_meta) - padding,
            extraction_fps=self.sample_fps,
            max_frames=self.max_frames,
            frame_height=self.frame_height,
            frame_width=self.frame_width,
            frames=frames_meta,
            extraction_duration_s=elapsed,
            padding_applied=padding,
            truncation_applied=truncated,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            timestamp_source=ts_source,
        )
        (output_dir / "manifest.json").write_text(manifest.to_json(), encoding="utf-8")
        logger.info(
            "Extracted %d frames in %.2fs (padding=%d, truncated=%s, timestamps=%s)",
            manifest.num_frames_extracted, elapsed, padding, truncated, ts_source,
        )
        return manifest

    def load_frame_sequence(self, frame_dir: Path, include_padding: bool = False) -> np.ndarray:
        """Load extracted frames from disk. Returns (T, H, W, C) uint8.

        Only real frames are returned unless ``include_padding`` is set, in which case
        zero frames are appended for the manifest's padding entries.
        """
        manifest_path = frame_dir / "manifest.json"
        if not manifest_path.exists():
            raise VideoProcessingError(f"Manifest not found in {frame_dir}")
        files = sorted(frame_dir.glob("frame_*.npy"))
        if not files:
            raise VideoProcessingError(f"No frame files found in {frame_dir}")
        frames = [np.load(str(f)) for f in files]
        if include_padding:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            n_pad = int(manifest.get("padding_applied", 0))
            frames.extend(np.zeros_like(frames[0]) for _ in range(n_pad))
        return np.stack(frames, axis=0)
