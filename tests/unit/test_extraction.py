"""Timestamp-based frame sampling, metadata probing, and WebM robustness."""
import json
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from backend.app.core.errors import VideoValidationError
from src.video.extractor import (
    FrameReader, FrameSampler, VideoFrameExtractor, probe_video, read_frame_at,
)
from tests.helpers.media import make_target_video

def _ffmpeg_has_libvpx() -> bool:
    try:
        return shutil.which("ffmpeg") is not None and "libvpx" in subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    except Exception:
        return False


def _ffmpeg_version_tuple() -> tuple[int, ...]:
    """Return (major, minor) of the ffmpeg binary, or (0, 0) on failure."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True).stdout
        for part in out.split():
            if part.startswith("version"):
                continue
            nums = part.split(".")
            if nums[0].isdigit():
                return tuple(int(x) for x in nums[:2] if x.isdigit())
    except Exception:
        pass
    return (0, 0)


HAS_FFMPEG = _ffmpeg_has_libvpx()
_FFMPEG_VER = _ffmpeg_version_tuple()
# -vsync was removed in ffmpeg 5.1; use -fps_mode on newer builds
_VSYNC_FLAG = ["-fps_mode", "vfr"] if _FFMPEG_VER >= (5, 1) else ["-vsync", "vfr"]


class TestFrameSampler:
    def test_picks_first_frame_of_each_slot(self):
        s = FrameSampler(2.0)                                  # slots every 500 ms
        times = [0, 33, 66, 480, 499, 501, 530, 990, 1010, 1499, 1500, 1520]
        picked = [t for t in times if s.should_sample(t)]
        assert picked == [0, 501, 1010, 1500]                  # 990 is in slot 1 already taken

    def test_gap_in_timestamps_does_not_cause_a_burst(self):
        s = FrameSampler(2.0)
        picked = [t for t in [0, 3000, 3033, 3066, 3500] if s.should_sample(t)]
        assert picked == [0, 3000, 3500]

    def test_start_ms_offsets_first_slot(self):
        s = FrameSampler(2.0, start_ms=1000)
        assert [t for t in [0, 500, 999, 1000, 1100, 1500] if s.should_sample(t)] == [1000, 1500]

    def test_invalid_fps(self):
        with pytest.raises(ValueError):
            FrameSampler(0)


class TestProbe:
    def test_mp4_metadata_matches_ground_truth(self, tmp_path):
        gt = make_target_video(tmp_path / "a.mp4", n_frames=30, fps=10.0)
        m = probe_video(tmp_path / "a.mp4")
        assert (m["width"], m["height"]) == gt["size"]
        assert m["fps"] == pytest.approx(10.0) and m["frame_count"] == 30
        assert m["duration_s"] == pytest.approx(3.0) and not m["scanned"]

    @pytest.mark.parametrize("content", [b"", b"not a video at all", b"\x00" * 4096])
    def test_garbage_is_rejected(self, tmp_path, content):
        p = tmp_path / "bad.mp4"
        p.write_bytes(content)
        with pytest.raises(VideoValidationError):
            probe_video(p)

    def test_missing_file(self, tmp_path):
        with pytest.raises(VideoValidationError, match="not found"):
            probe_video(tmp_path / "nope.mp4")

    @pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg with libvpx not available")
    def test_live_mode_webm_gets_true_frame_count_and_duration(self, tmp_path):
        """Emulates a browser MediaRecorder file (live muxing: no duration/cues, ms timebase).

        NOTE: produced by ffmpeg, not by a real browser; see docs/verification.md.
        """
        p = tmp_path / "live.webm"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=15",
             "-t", "3", "-vf", "setpts='PTS+0.004*sin(N)/TB'", "-c:v", "libvpx", "-b:v", "500k",
             *_VSYNC_FLAG,
             "-video_track_timescale", "1000", "-f", "webm", "-live", "1", str(p)], check=True)
        # independent ground truth from ffprobe
        out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                              "-show_entries", "stream=nb_read_frames", "-of", "json", str(p)],
                             capture_output=True, text=True, check=True).stdout
        true_frames = int(json.loads(out)["streams"][0]["nb_read_frames"])
        m = probe_video(p)
        assert m["scanned"] and m["frame_count"] == true_frames > 0
        assert m["duration_s"] == pytest.approx(3.0, abs=0.15)
        assert m["timestamp_source"] == "container_pts"
        assert any("derived by decoding" in w for w in m["warnings"])
        # sampling is by timestamp, so ~2 samples/s even though container metadata is unreliable
        s = FrameSampler(2.0)
        with FrameReader(p) as r:
            picked = [df.t_ms for df in r if s.should_sample(df.t_ms)]
        assert 6 <= len(picked) <= 7
        assert all(b - a >= 450 for a, b in zip(picked, picked[1:]))


class TestFrameReaderTimestamps:
    def test_timestamps_follow_frame_times_and_are_monotonic(self, tmp_path):
        make_target_video(tmp_path / "a.mp4", n_frames=20, fps=10.0)
        with FrameReader(tmp_path / "a.mp4") as r:
            ts = [(df.index, df.t_ms) for df in r]
            assert r.timestamp_source == "container_pts"
        assert [i for i, _ in ts] == list(range(20))
        assert [t for _, t in ts] == pytest.approx([i * 100.0 for i in range(20)], abs=1.0)

    def test_read_frame_at_time_and_index_agree(self, tmp_path):
        make_target_video(tmp_path / "a.mp4", n_frames=20, fps=10.0)
        by_time = read_frame_at(tmp_path / "a.mp4", t_ms=1000)
        by_index = read_frame_at(tmp_path / "a.mp4", frame_index=by_time.index)
        assert by_time.index == 10 and by_time.t_ms == pytest.approx(1000, abs=1)
        assert np.array_equal(by_time.frame, by_index.frame)

    def test_read_frame_at_missing_index(self, tmp_path):
        make_target_video(tmp_path / "a.mp4", n_frames=5, fps=10.0)
        with pytest.raises(VideoValidationError, match="no frame with index"):
            read_frame_at(tmp_path / "a.mp4", frame_index=99)


class TestLegacyExtractor:
    def test_samples_by_timestamp_with_mask_only_padding(self, tmp_path):
        make_target_video(tmp_path / "a.mp4", n_frames=30, fps=10.0)
        out = tmp_path / "frames"
        m = VideoFrameExtractor(sample_fps=2.0, sequence_length=10).extract_frames(tmp_path / "a.mp4", out, "j")
        real = [f for f in m.frames if not f.is_padding]
        assert [round(f.timestamp_ms) for f in real] == [0, 500, 1000, 1500, 2000, 2500]
        assert [f.frame_index for f in real] == [0, 5, 10, 15, 20, 25]
        assert m.padding_applied == 4 and m.num_frames_extracted == 6 and not m.truncation_applied
        assert len(list(out.glob("frame_*.npy"))) == 6, "no fabricated zero-frame files for padding"

    def test_truncation_keeps_disk_and_manifest_consistent(self, tmp_path):
        make_target_video(tmp_path / "a.mp4", n_frames=30, fps=10.0)
        out = tmp_path / "frames"
        ex = VideoFrameExtractor(sample_fps=2.0, sequence_length=3)
        m = ex.extract_frames(tmp_path / "a.mp4", out, "j")
        assert m.truncation_applied and m.num_frames_extracted == 3 and m.padding_applied == 0
        assert len(list(out.glob("frame_*.npy"))) == 3
        assert ex.load_frame_sequence(out).shape[0] == 3
        assert ex.load_frame_sequence(out, include_padding=False).shape[0] == 3
