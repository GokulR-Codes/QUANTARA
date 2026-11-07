"""Tracker state machine (scripted backend) and real-OpenCV behaviour."""
import numpy as np
import pytest

from src.video.geometry import BBox, iou
from src.video.tracker import (
    TargetTracker, TrackerInitError, TrackerParams, TrackStatus, create_cv_tracker,
)

W, H = 200, 150
ORANGE, MAGENTA = (0, 140, 255), (255, 0, 255)


def scene(x, y, color=ORANGE, size=30, seed=1):
    f = np.random.RandomState(seed).randint(0, 50, (H, W, 3)).astype(np.uint8)
    f[y:y + size, x:x + size] = color
    return f


class Scripted:
    """Stand-in for an OpenCV tracker returning pre-scripted (ok, box) results."""
    def __init__(self, script, init_result=None):
        self.script, self.calls, self._init = list(script), 0, init_result

    def init(self, frame, bbox):
        return self._init

    def update(self, frame):
        r = self.script[self.calls]
        self.calls += 1
        return r


def make(script, **params):
    sc = Scripted(script)
    tr = TargetTracker(TrackerParams(**params), tracker_factory=lambda: sc)
    tr.init(scene(50, 50), BBox(50, 50, 30, 30))
    return tr, sc


class TestStateMachine:
    def test_trusted_when_all_guards_pass(self):
        tr, _ = make([(True, (53, 50, 30, 30))])
        r = tr.update(scene(53, 50))
        assert r.status == TrackStatus.TRACKED and r.similarity > 0.9 and r.bbox == BBox(53, 50, 30, 30)

    def test_init_returning_none_is_success_but_false_is_failure(self):
        TargetTracker(tracker_factory=lambda: Scripted([], init_result=None)).init(scene(50, 50), BBox(50, 50, 30, 30))
        with pytest.raises(TrackerInitError):
            TargetTracker(tracker_factory=lambda: Scripted([], init_result=False)).init(scene(50, 50), BBox(50, 50, 30, 30))

    def test_tracker_failure_is_lost_and_terminal_by_default(self):
        tr, sc = make([(False, None), (True, (55, 50, 30, 30))])
        assert tr.update(scene(55, 50)).status == TrackStatus.LOST
        r2 = tr.update(scene(55, 50))            # tracker "recovers" -> must NOT be trusted
        assert r2.status == TrackStatus.LOST and "re-acquisition is disabled" in r2.reason
        assert sc.calls == 1, "backend must not be consulted again after a loss"

    def test_different_looking_object_is_ambiguous_not_tracked(self):
        tr, _ = make([(True, (52, 50, 30, 30))])
        r = tr.update(scene(52, 50, color=MAGENTA))
        assert r.status == TrackStatus.AMBIGUOUS and "similarity" in r.reason

    def test_implausible_jump_is_ambiguous(self):
        tr, _ = make([(True, (150, 100, 30, 30))])
        r = tr.update(scene(150, 100))           # looks identical but teleported ~100px
        assert r.status == TrackStatus.AMBIGUOUS and "jump" in r.reason

    def test_implausible_scale_change_is_ambiguous(self):
        tr, _ = make([(True, (50, 50, 80, 80))])
        r = tr.update(scene(50, 50, size=80))
        assert r.status == TrackStatus.AMBIGUOUS and "scale" in r.reason

    def test_box_leaving_frame_is_out_of_frame_and_terminal(self):
        tr, _ = make([(True, (W - 5, 50, 30, 30)), (True, (60, 50, 30, 30))])
        assert tr.update(scene(W - 5, 50)).status == TrackStatus.OUT_OF_FRAME
        assert tr.update(scene(60, 50)).status == TrackStatus.LOST

    def test_malformed_box_is_lost(self):
        tr, _ = make([(True, None)])
        assert tr.update(scene(50, 50)).status == TrackStatus.LOST

    def test_resume_after_ambiguity_requires_consecutive_confirmations(self):
        good = (True, (52, 50, 30, 30))
        tr, _ = make([good, good, good, good])
        bad_frame, ok_frame = scene(52, 50, color=MAGENTA), scene(52, 50)
        statuses = [tr.update(f) for f in (bad_frame, ok_frame, ok_frame, ok_frame)]
        assert [s.status for s in statuses] == [TrackStatus.AMBIGUOUS] * 3 + [TrackStatus.TRACKED]
        assert statuses[3].resumed and "resumed" in statuses[3].reason
        assert "confirming" in statuses[1].reason

    def test_confirmation_resets_if_a_bad_frame_interrupts(self):
        good = (True, (52, 50, 30, 30))
        tr, _ = make([good] * 6)
        ok, bad = scene(52, 50), scene(52, 50, color=MAGENTA)
        seq = [bad, ok, ok, bad, ok, ok]          # never 3 good in a row
        assert all(tr.update(f).status == TrackStatus.AMBIGUOUS for f in seq)

    def test_reacquisition_enabled_still_needs_confirmation(self):
        good = (True, (52, 50, 30, 30))
        tr, sc = make([(False, None), good, good, good], allow_reacquisition=True)
        ok = scene(52, 50)
        res = [tr.update(ok) for _ in range(4)]
        # lost -> candidate seen (1/3) -> (2/3) -> third consecutive confirmation => trusted again, flagged
        assert [r.status for r in res] == [TrackStatus.LOST, TrackStatus.AMBIGUOUS,
                                           TrackStatus.AMBIGUOUS, TrackStatus.TRACKED]
        assert res[3].resumed
        assert sc.calls == 4

    def test_update_before_init_is_a_programming_error(self):
        with pytest.raises(RuntimeError):
            TargetTracker().update(scene(0, 0))


class TestRealOpenCVTracker:
    """Runs the actual OpenCV tracker on synthetic frames (no video I/O)."""

    def _run(self, frames, bbox):
        tr = TargetTracker(TrackerParams())
        tr.init(frames[0], bbox)
        return tr, [tr.update(f) for f in frames[1:]]

    def test_backend_is_csrt_not_silently_downgraded(self):
        _, name = create_cv_tracker()
        assert name in ("CSRT", "CSRT(legacy)"), f"expected CSRT, got {name} — install opencv-contrib-python-headless"

    def test_follows_a_moving_target(self):
        xs = [20 + 3 * i for i in range(30)]
        frames = [scene(x, 60) for x in xs]
        tr, res = self._run(frames, BBox(xs[0], 60, 30, 30))
        tracked = [r for r in res if r.status == TrackStatus.TRACKED]
        assert tr.backend_name.startswith("CSRT")
        assert len(tracked) >= 0.9 * len(res)
        # Box ACCURACY is modest (CSRT's box shrinks on a flat-coloured square: IoU decays to
        # ~0.35), so assert what matters for "right target": the box stays on the object.
        for r, x in zip(res, xs[1:]):
            if r.status != TrackStatus.TRACKED:
                continue
            true_box = BBox(x, 60, 30, 30)
            cx, cy = r.bbox.center
            assert true_box.x <= cx <= true_box.x2 and true_box.y <= cy <= true_box.y2, "box centre left the target"
            assert iou(r.bbox, true_box) > 0.25

    def test_vanished_target_is_never_reported_as_tracked(self):
        frames = [scene(20 + 3 * i, 60) for i in range(15)]
        frames += [scene(0, 0, size=1) for _ in range(10)]        # target gone (background only)
        tr, res = self._run(frames, BBox(20, 60, 30, 30))
        after = res[14:]                                           # frames 15..24: absent
        assert after and all(r.status != TrackStatus.TRACKED for r in after), [r.status for r in after]

    def test_target_replaced_by_different_object_is_never_reported_as_tracked(self):
        frames = [scene(20 + 3 * i, 60) for i in range(15)]
        frames += [scene(20 + 3 * 14, 60, color=MAGENTA) for _ in range(10)]
        tr, res = self._run(frames, BBox(20, 60, 30, 30))
        assert all(r.status != TrackStatus.TRACKED for r in res[14:]), [(r.status.value, r.reason) for r in res[14:]]
