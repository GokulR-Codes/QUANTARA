"""End-to-end verification through the real HTTP API (TestClient runs background tasks inline).

Acceptance workflow under test:
 upload -> explicit participant selection -> target tracking -> timestamp-aligned crops
 -> CNN -> LSTM -> fusion -> QAOA-selected mapping -> VQC -> persisted, independently retrievable result.

The synthetic bundle used for the VQC stage is a plumbing fixture (see tests/helpers/bundles.py),
NOT a trained research model.
"""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.db import SessionLocal
from backend.app.models import Job
from backend.app.services.job_service import cancel_job, get_job, job_dir_for
from src.pipeline.bundle import FILE_NAMES, MANIFEST_NAME, load_bundle, predict_with_bundle, sha256_file
from src.pipeline.features import build_untrained_upstream, load_crop_sequence, run_cnn_lstm_fusion
from src.video.extractor import FrameReader
from src.video.geometry import BBox, crop_target, iou
from tests.helpers.api import run_job, select, upload
from tests.integration.conftest import make_variant

V = "/api/v1/videos"


def tracking_doc(client, job_id):
    r = client.get(f"{V}/{job_id}/tracking")
    assert r.status_code == 200, r.text
    return r.json()


# ───────────────────────────── 1. upload & frame extraction ─────────────────────────────
class TestUploadAndFrames:
    def test_upload_records_true_metadata_from_decoded_video(self, client, moving_video):
        path, gt = moving_video
        r = upload(client, path)
        assert r.status_code == 202
        job_id = r.json()["job_id"]
        st = client.get(f"{V}/{job_id}/status").json()
        assert st["status"] == "uploaded"
        md = st["video_metadata"]
        assert (md["width"], md["height"]) == gt["size"]
        assert md["fps"] == pytest.approx(10.0) and md["duration_s"] == pytest.approx(4.0)

    def test_frame_info_and_preview_return_the_exact_frame(self, client, moving_video):
        path, gt = moving_video
        job_id = upload(client, path).json()["job_id"]
        info = client.get(f"{V}/{job_id}/frame-info", params={"t_ms": 1000}).json()
        assert info["frame_index"] == 10 and info["time_ms"] == pytest.approx(1000, abs=1)
        assert (info["width"], info["height"]) == gt["size"]
        img = client.get(f"{V}/{job_id}/preview-frame", params={"frame_index": info["frame_index"]})
        assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
        assert img.headers["x-frame-index"] == "10"
        dec = cv2.imdecode(np.frombuffer(img.content, np.uint8), cv2.IMREAD_COLOR)
        assert dec.shape[:2] == (gt["size"][1], gt["size"][0])
        # the previewed frame is frame 10 of the source video (compare with an independent decode)
        with FrameReader(path) as r:
            src = next(df.frame for df in r if df.index == 10)
        assert np.abs(dec.astype(int) - src.astype(int)).mean() < 6      # JPEG tolerance

    @pytest.mark.parametrize("name,content,mime", [
        ("empty.mp4", b"", "video/mp4"),
        ("notes.txt", b"hello world, this is text" * 4, "text/plain"),
        ("fake.mp4", b"MZ\x90\x00" + b"\x00" * 200, "video/mp4"),                  # not a video container
        ("mismatch.mp4", b"\x1a\x45\xdf\xa3" + b"\x00" * 200, "video/mp4"),        # EBML content, .mp4 name
        ("corrupt.mp4", b"\x00\x00\x00\x18ftypmp42" + b"\xff" * 5000, "video/mp4"),  # valid header, undecodable
    ])
    def test_invalid_uploads_are_rejected_with_422_and_leave_nothing_behind(self, client, settings, name, content, mime):
        before = set(Path(settings.upload_dir).glob("*"))
        r = client.post(f"{V}/upload", files={"file": (name, content, mime)})
        assert r.status_code == 422, (r.status_code, r.text)
        assert set(Path(settings.upload_dir).glob("*")) == before, "rejected upload left a file on disk"

    def test_oversized_upload_rejected_and_removed(self, client, settings, moving_video, set_setting):
        path, _ = moving_video
        set_setting(max_upload_size_mb=0)       # property max_upload_size_bytes derives from this
        before = set(Path(settings.upload_dir).glob("*"))
        r = upload(client, path)
        assert r.status_code == 422 and "limit" in r.text
        assert set(Path(settings.upload_dir).glob("*")) == before

    def test_too_long_video_rejected(self, client, moving_video, set_setting):
        path, _ = moving_video
        set_setting(max_video_duration_seconds=2)
        r = upload(client, path)
        assert r.status_code == 422 and "limit" in r.text


# ───────────────────────── 2/3. operator-selected target + bbox validation ─────────────────────────
class TestTargetSelection:
    def test_processing_requires_an_explicit_selection(self, client, moving_video):
        path, _ = moving_video
        job_id = upload(client, path).json()["job_id"]
        r = client.post(f"{V}/{job_id}/process")
        assert r.status_code == 409 and "No participant selected" in r.json()["detail"]
        assert client.get(f"{V}/{job_id}/status").json()["status"] == "uploaded"   # nothing was started

    def test_valid_selection_is_stored_and_reported(self, client, moving_video):
        path, gt = moving_video
        job_id = upload(client, path).json()["job_id"]
        bbox = {"x": 10, "y": 40, "w": 36, "h": 36}
        r = select(client, job_id, bbox, gt["size"], frame_index=0)
        assert r.status_code == 200
        body = r.json()
        assert body["bbox"] == bbox and body["frame_index"] == 0 and body["frame_time_ms"] == pytest.approx(0, abs=1)
        ts = client.get(f"{V}/{job_id}/status").json()["target_selection"]
        assert ts["bbox"] == bbox and ts["frame_index"] == 0 and (ts["frame_width"], ts["frame_height"]) == gt["size"]

    def test_selection_on_a_later_frame_records_that_frames_timestamp(self, client, moving_video):
        path, gt = moving_video
        job_id = upload(client, path).json()["job_id"]
        t = gt["truth"][10]
        r = select(client, job_id, {k: t[k] for k in "xywh"}, gt["size"], frame_index=10)
        assert r.status_code == 200 and r.json()["frame_index"] == 10
        assert r.json()["frame_time_ms"] == pytest.approx(1000, abs=1)

    @pytest.mark.parametrize("bbox,size,frame_index,needle", [
        ({"x": -1, "y": 10, "w": 30, "h": 30}, (160, 120), 0, "negative"),
        ({"x": 10, "y": 10, "w": 0, "h": 30}, (160, 120), 0, "too small"),
        ({"x": 10, "y": 10, "w": 30, "h": 5}, (160, 120), 0, "too small"),
        ({"x": 150, "y": 10, "w": 30, "h": 30}, (160, 120), 0, "exceeds frame width"),
        ({"x": 10, "y": 100, "w": 30, "h": 30}, (160, 120), 0, "exceeds frame height"),
        ({"x": 200, "y": 10, "w": 30, "h": 30}, (160, 120), 0, "outside"),
        ({"x": 10, "y": 10, "w": 30, "h": 30}, (80, 60), 0, "Frame size mismatch"),       # client-scaled coords
        ({"x": 10, "y": 10, "w": 30, "h": 30}, (1920, 1080), 0, "Frame size mismatch"),  # lying about size
        ({"x": 10, "y": 10, "w": 30, "h": 30}, (160, 120), 999, "no frame with index"),
    ])
    def test_invalid_or_out_of_bounds_selection_rejected_and_not_stored(self, client, moving_video, bbox, size, frame_index, needle):
        path, gt = moving_video
        job_id = upload(client, path).json()["job_id"]
        good = {"x": 10, "y": 40, "w": 36, "h": 36}
        assert select(client, job_id, good, gt["size"]).status_code == 200
        r = select(client, job_id, bbox, size, frame_index)
        assert r.status_code == 422 and needle in r.text, (r.status_code, r.text)
        # the earlier valid selection is untouched by the rejected one
        assert client.get(f"{V}/{job_id}/status").json()["target_selection"]["bbox"] == good

    def test_non_integer_and_missing_fields_are_422(self, client, moving_video):
        path, _ = moving_video
        job_id = upload(client, path).json()["job_id"]
        bad = {"bbox_x": "ten", "bbox_y": 1, "bbox_w": 30, "bbox_h": 30, "frame_width": 160, "frame_height": 120}
        assert client.post(f"{V}/{job_id}/select-target", json=bad).status_code == 422
        assert client.post(f"{V}/{job_id}/select-target", json={"bbox_x": 1}).status_code == 422

    def test_body_job_id_must_match_url(self, client, moving_video):
        path, gt = moving_video
        job_id = upload(client, path).json()["job_id"]
        body = {"job_id": "someone-else", "bbox_x": 10, "bbox_y": 40, "bbox_w": 36, "bbox_h": 36,
                "frame_width": gt["size"][0], "frame_height": gt["size"][1]}
        r = client.post(f"{V}/{job_id}/select-target", json=body)
        assert r.status_code == 422 and "does not match" in r.text

    def test_unknown_job_is_404(self, client):
        assert select(client, "nope", {"x": 1, "y": 1, "w": 20, "h": 20}, (160, 120)).status_code == 404


# ───────────────────────── 4. tracking-to-crop alignment ─────────────────────────
class TestTrackingCropAlignment:
    @pytest.fixture()
    def job(self, client, moving_video):
        path, gt = moving_video
        job_id, status = run_job(client, path, gt)
        return job_id, status, path, gt

    def test_every_sample_has_a_frame_timestamp_and_a_matching_crop(self, client, job):
        job_id, status, path, gt = job
        doc = tracking_doc(client, job_id)
        samples = doc["samples"]
        assert doc["summary"]["tracker_backend"].startswith("CSRT"), "real CSRT must have run, not a downgrade"
        assert [s["sample_index"] for s in samples] == list(range(len(samples))) and len(samples) == 8
        # samples at 0, 500, ... 3500 ms  == frame indices 0, 5, ... 35 at 10 fps
        assert [s["frame_index"] for s in samples] == [0, 5, 10, 15, 20, 25, 30, 35]
        assert [s["timestamp_ms"] for s in samples] == pytest.approx([i * 500.0 for i in range(8)], abs=1)
        frames = {}
        with FrameReader(path) as r:
            for df in r:
                if df.index in {s["frame_index"] for s in samples}:
                    frames[df.index] = (df.t_ms, df.frame)
        job_dir = job_dir_for(job_id)
        for s in samples:
            assert s["status"] == "tracked" and s["crop_file"]
            t_ms, frame = frames[s["frame_index"]]
            assert s["timestamp_ms"] == pytest.approx(t_ms, abs=0.01)          # timestamp of THAT frame
            # the crop on disk is exactly the stored box cropped out of THAT source frame
            expected = crop_target(frame, BBox(**s["bbox"]), (112, 112), 0.10)
            actual = cv2.imread(str(job_dir / s["crop_file"]), cv2.IMREAD_COLOR)
            assert np.array_equal(actual, expected), f"crop {s['sample_index']} does not match frame {s['frame_index']}"

    def test_tracked_boxes_follow_the_true_target_position_at_each_timestamp(self, client, job):
        job_id, _, _, gt = job
        for s in tracking_doc(client, job_id)["samples"]:
            truth = gt["truth"][s["frame_index"]]
            true_box = BBox(truth["x"], truth["y"], truth["w"], truth["h"])
            box = BBox(**s["bbox"])
            cx, cy = box.center
            assert true_box.x <= cx <= true_box.x2 and true_box.y <= cy <= true_box.y2, f"sample {s['sample_index']} off-target"
            assert iou(box, true_box) > 0.25

    def test_crops_contain_the_target_colours(self, client, job):
        job_id, _, _, _ = job
        job_dir = job_dir_for(job_id)
        for s in tracking_doc(client, job_id)["samples"]:
            crop = cv2.imread(str(job_dir / s["crop_file"]))
            orange = ((crop[..., 2] > 200) & (crop[..., 1] > 150) & (crop[..., 0] < 60)).mean()
            assert orange > 0.02, f"crop {s['sample_index']} has no target-coloured pixels"

    def test_summary_counts_are_consistent_with_files_and_records(self, client, job):
        job_id, status, _, _ = job
        tr, fp = status["target_tracking_summary"], status["frame_processing_summary"]
        crops = list((job_dir_for(job_id) / "crops").glob("*.png"))
        assert tr["samples_total"] == fp["frames_sampled"] == 8
        assert tr["frames_tracked"] == tr["crops_generated"] == fp["frames_accepted"] == len(crops) == 8
        assert tr["frames_lost"] == tr["frames_ambiguous"] == tr["frames_invalid"] == fp["frames_rejected"] == 0
        assert tr["tracking_coverage_pct"] == 100.0 and tr["intervals"] == []
        assert tr["selection"]["frame_index"] == 0 and tr["timestamp_source"] == "container_pts"

    def test_selection_on_a_later_frame_starts_tracking_there(self, client, moving_video):
        path, gt = moving_video
        _, status = run_job(client, path, gt, frame_index=10)
        tr = status["target_tracking_summary"]
        doc_samples = tracking_doc(client, status["job_id"])["samples"]
        assert doc_samples[0]["frame_index"] == 10 and doc_samples[0]["timestamp_ms"] == pytest.approx(1000, abs=1)
        assert tr["samples_total"] == 6 and tr["selection"]["frame_index"] == 10     # 1000..3500 ms
        assert doc_samples[0]["reason"] == "operator-selected frame"


# ───────────────────────── 5. target loss / ambiguity / never switching ─────────────────────────
class TestTargetLoss:
    def _assert_no_trust_after(self, doc, first_bad_ms):
        for s in doc["samples"]:
            if s["timestamp_ms"] >= first_bad_ms:
                assert s["status"] != "tracked", f"sample at {s['timestamp_ms']}ms trusted after the target vanished"
                assert s["crop_file"] is None
            else:
                assert s["status"] == "tracked" and s["crop_file"]

    def test_target_vanishing_late_is_recorded_and_crops_stop(self, client, tmp_path_factory):
        path, gt = make_variant(tmp_path_factory, "vanish_late.mp4", n_frames=40, fps=10.0, vanish_at=32)
        job_id, st = run_job(client, path, gt)
        doc = tracking_doc(client, job_id)
        self._assert_no_trust_after(doc, 3500.0)                          # frame 35 is the first sample >= frame 32
        tr = st["target_tracking_summary"]
        assert tr["frames_tracked"] == 7 and tr["frames_invalid"] == 1 and tr["crops_generated"] == 7
        assert len(tr["intervals"]) == 1
        iv = tr["intervals"][0]
        assert iv["status"] in ("lost", "ambiguous", "out_of_frame") and iv["start_ms"] == 3500.0 == iv["end_ms"]
        assert iv["n_samples"] == 1 and iv["reasons"]
        assert len(list((job_dir_for(job_id) / "crops").glob("*.png"))) == 7
        assert any("not trusted" in w for w in st["warnings"])
        assert st["feature_extraction_status"] == "completed_untrained_weights"   # 7 >= min_tracked_frames
        assert st["feature_extraction"]["n_embeddings"] == 7

    def test_target_vanishing_early_leaves_too_few_frames_and_no_features(self, client, tmp_path_factory):
        path, gt = make_variant(tmp_path_factory, "vanish_early.mp4", n_frames=40, fps=10.0, vanish_at=20)
        job_id, st = run_job(client, path, gt)
        self._assert_no_trust_after(tracking_doc(client, job_id), 2000.0)
        tr = st["target_tracking_summary"]
        assert tr["frames_tracked"] == 4 and tr["frames_invalid"] == 4 and tr["first_loss_ms"] in (None, 2000.0, 2500.0, 3000.0, 3500.0)
        # honesty: too little data -> explicit status, NO features, NO prediction
        assert st["status"] == "completed_without_prediction"
        assert st["feature_extraction_status"] == "insufficient_tracked_frames"
        assert st["model_status"] == "insufficient_tracked_frames" and st["prediction"] is None
        assert not (job_dir_for(job_id) / "features.npz").exists()

    def test_target_replaced_by_a_different_object_is_never_followed(self, client, tmp_path_factory):
        path, gt = make_variant(tmp_path_factory, "swap.mp4", n_frames=40, fps=10.0, swap_at=20)
        job_id, st = run_job(client, path, gt)
        doc = tracking_doc(client, job_id)
        self._assert_no_trust_after(doc, 2000.0)
        bad = [s for s in doc["samples"] if s["timestamp_ms"] >= 2000.0]
        assert bad and all(s["status"] in ("ambiguous", "lost", "out_of_frame") for s in bad)
        assert st["target_tracking_summary"]["intervals"], "invalid interval must be recorded"
        assert st["prediction"] is None

    def test_invalid_intervals_cover_exactly_the_untrusted_samples(self, client, tmp_path_factory):
        path, gt = make_variant(tmp_path_factory, "vanish_mid.mp4", n_frames=60, fps=10.0, x_step=2, vanish_at=22)
        job_id, st = run_job(client, path, gt)
        doc = tracking_doc(client, job_id)
        untrusted = [s["sample_index"] for s in doc["samples"] if s["status"] != "tracked"]
        covered = [i for iv in doc["summary"]["intervals"]
                   for i in range(iv["start_sample_index"], iv["end_sample_index"] + 1)]
        assert sorted(covered) == untrusted and untrusted
        for iv in doc["summary"]["intervals"]:
            assert iv["duration_ms"] == iv["n_samples"] * 500.0


# ───────────────────────── 6. CNN -> LSTM inference without a model bundle ─────────────────────────
class TestFeatureExtractionStage:
    def test_features_are_really_computed_persisted_and_reproducible(self, client, moving_video, settings):
        path, gt = moving_video
        job_id, st = run_job(client, path, gt)
        fe = st["feature_extraction"]
        assert st["feature_extraction_status"] == "completed_untrained_weights"
        assert fe["n_embeddings"] == 8 and fe["cnn_embedding_dim"] == settings.cnn_feature_dim
        assert fe["lstm_dim"] == settings.lstm_hidden_dim and fe["fused_dim"] == settings.fusion_output_dim
        assert fe["n_padded_masked"] == settings.sequence_length - 8
        assert fe["weights_provenance"]["kind"] == "untrained_random_init"
        assert any("untrained" in w for w in st["warnings"])
        z = np.load(job_dir_for(job_id) / "features.npz")
        assert z["cnn_embeddings"].shape == (8, settings.cnn_feature_dim) and z["fused"].shape == (settings.fusion_output_dim,)
        assert z["timestamps_ms"].tolist() == pytest.approx([i * 500.0 for i in range(8)], abs=1)
        # independent recomputation from the stored crops with identically-seeded modules
        out = run_cnn_lstm_fusion(load_crop_sequence(job_dir_for(job_id)), build_untrained_upstream(settings), settings.sequence_length)
        assert np.allclose(z["fused"], out.fused, atol=1e-6) and np.allclose(z["cnn_embeddings"], out.cnn_embeddings, atol=1e-6)
        assert fe["fused_l2_norm"] == pytest.approx(float(np.linalg.norm(z["fused"])), rel=1e-4)

    def test_without_a_bundle_no_prediction_is_claimed(self, client, moving_video):
        path, gt = moving_video
        job_id, st = run_job(client, path, gt)
        assert st["status"] == "completed_without_prediction"
        assert st["model_status"] == "model_not_ready" and "No trained pipeline bundle" in st["model_status_detail"]
        assert st["prediction"] is None and st["active_model_id"] is None
        assert st["selected_feature_names"] is None and st["quantum_configuration"] is None
        res = client.get(f"{V}/{job_id}/result").json()
        assert res["prediction"] is None and res["model_status"] == "model_not_ready"


# ───────────────────────── 7. QAOA -> VQC with a compatible checkpoint ─────────────────────────
class TestQuantumInference:
    def test_compatible_bundle_produces_a_labelled_prediction(self, client, moving_video, register_bundle, set_setting, synthetic_bundle_dir, settings):
        set_setting(allow_synthetic_model_inference=True)
        model_id = register_bundle()
        path, gt = moving_video
        job_id, st = run_job(client, path, gt)
        assert st["status"] == "completed" and st["model_status"] == "inference_completed"
        assert st["feature_extraction_status"] == "completed"                 # trained-bundle upstream weights
        assert st["feature_extraction"]["weights_provenance"]["kind"] == "trained_bundle"
        p = st["prediction"]
        assert p["synthetic_model"] is True and "SYNTHETIC" in p["disclaimer"] and "NOT clinical" in p["disclaimer"]
        assert sum(p["class_probabilities"]) == pytest.approx(1.0, abs=1e-5) and len(p["class_probabilities"]) == settings.num_classes
        assert p["predicted_label"] == settings.class_names[p["predicted_class"]]
        mapping = json.loads((synthetic_bundle_dir / FILE_NAMES["mapping"]).read_text())
        assert st["active_model_id"] == model_id and st["selected_feature_names"] == mapping["selected_feature_names"]
        qc = st["quantum_configuration"]
        assert qc["selected_indices"] == mapping["selected_indices"] and qc["fused_dim"] == settings.fusion_output_dim
        assert qc["vqc"]["input_dim"] == len(mapping["selected_indices"]) and qc["qaoa"]["target_k"] == 2
        assert st["model_provenance"]["training_data"]["kind"] == "synthetic_test_fixture" and st["model_provenance"]["synthetic"]
        assert st["model_provenance"]["file_checksums_sha256"]["vqc"] == sha256_file(synthetic_bundle_dir / FILE_NAMES["vqc"])
        # independent recomputation: load the bundle fresh and run the quantum head on the stored fused vector
        bundle = load_bundle(synthetic_bundle_dir, settings, allow_synthetic=True)
        fused = np.load(job_dir_for(job_id) / "features.npz")["fused"]
        again = predict_with_bundle(fused, bundle)
        assert again["predicted_class"] == p["predicted_class"]
        assert again["class_probabilities"] == pytest.approx(p["class_probabilities"], abs=1e-6)
        res = client.get(f"{V}/{job_id}/result").json()                       # persisted + retrievable
        assert res["prediction"]["class_probabilities"] == p["class_probabilities"]
        assert get_stored_result(job_id)["model_input_features"] == pytest.approx(again["mapped_input"], abs=1e-6)

    def test_synthetic_bundle_is_refused_by_default_and_no_prediction_is_made(self, client, moving_video, register_bundle):
        register_bundle()
        path, gt = moving_video
        job_id, st = run_job(client, path, gt)
        assert st["status"] == "completed_without_prediction" and st["model_status"] == "synthetic_checkpoint_refused"
        assert st["prediction"] is None and "SYNTHETIC" not in json.dumps(st["prediction"])
        assert st["feature_extraction_status"] == "completed_untrained_weights"   # not the refused bundle's weights

    def test_dimension_incompatible_bundle_is_reported(self, client, moving_video, register_bundle, synthetic_bundle_dir, set_setting, tmp_path):
        set_setting(allow_synthetic_model_inference=True)
        bad = tmp_path / "bad_dims"
        shutil.copytree(synthetic_bundle_dir, bad)
        mp = bad / FILE_NAMES["mapping"]
        m = json.loads(mp.read_text())
        m["fused_dim"] -= 1
        mp.write_text(json.dumps(m))
        man = json.loads((bad / MANIFEST_NAME).read_text())
        man["files"]["mapping"]["sha256"] = sha256_file(mp)
        (bad / MANIFEST_NAME).write_text(json.dumps(man))
        register_bundle(bad)
        path, gt = moving_video
        _, st = run_job(client, path, gt)
        assert st["model_status"] == "bundle_incompatible" and "fusion output" in st["model_status_detail"]
        assert st["prediction"] is None and st["status"] == "completed_without_prediction"

    def test_tampered_checkpoint_is_reported_corrupt(self, client, moving_video, register_bundle, synthetic_bundle_dir, set_setting, tmp_path):
        set_setting(allow_synthetic_model_inference=True)
        bad = tmp_path / "tampered"
        shutil.copytree(synthetic_bundle_dir, bad)
        register_bundle(bad)                                   # registered while intact ...
        with open(bad / FILE_NAMES["vqc"], "ab") as f:         # ... then modified on disk
            f.write(b"\x00")
        path, gt = moving_video
        _, st = run_job(client, path, gt)
        assert st["model_status"] == "bundle_corrupt" and st["prediction"] is None

    def test_registered_bundle_deleted_from_disk_is_model_not_ready(self, client, moving_video, register_bundle, synthetic_bundle_dir, set_setting, tmp_path):
        set_setting(allow_synthetic_model_inference=True)
        gone = tmp_path / "gone"
        shutil.copytree(synthetic_bundle_dir, gone)
        register_bundle(gone)
        shutil.rmtree(gone)
        path, gt = moving_video
        _, st = run_job(client, path, gt)
        assert st["model_status"] == "model_not_ready" and st["prediction"] is None

    def test_pennylane_unavailable_is_surfaced_not_hidden(self, client, moving_video, register_bundle, set_setting, monkeypatch):
        import sys
        set_setting(allow_synthetic_model_inference=True)
        register_bundle()
        monkeypatch.setitem(sys.modules, "src.models.quantum.vqc", None)
        path, gt = moving_video
        _, st = run_job(client, path, gt)
        assert st["model_status"] == "quantum_unavailable" and st["prediction"] is None
        assert any("PennyLane" in w for w in st["warnings"])

    def test_experiment_checkpoint_is_not_usable_for_video_inference(self, client, moving_video):
        from backend.app.services.model_service import register_model
        with SessionLocal() as s:
            register_model(s, "vqc", "exp-1", "/nonexistent/vqc_best.pt",
                           config={"training_data": {"kind": "synthetic_random"}}, make_active=True)
        listing = client.get("/api/v1/models").json()
        assert len(listing) == 1 and listing[0]["usable_for_video_inference"] is False
        assert listing[0]["synthetic"] is True and "cannot be used for video inference" in listing[0]["status_detail"]
        path, gt = moving_video
        _, st = run_job(client, path, gt)
        assert st["prediction"] is None and st["model_status"] == "model_not_ready"

    def test_models_listing_flags_usable_bundle_and_synthetic_provenance(self, client, register_bundle, set_setting):
        set_setting(allow_synthetic_model_inference=True)
        register_bundle()
        m = client.get("/api/v1/models").json()[0]
        assert m["model_type"] == "qaoa_vqc_bundle" and m["usable_for_video_inference"] is True and m["synthetic"] is True


def get_stored_result(job_id):
    with SessionLocal() as s:
        return dict(get_job(s, job_id).result)


# ───────────────────────── 8. job status & independent result retrieval ─────────────────────────
class TestStatusAndResultRetrieval:
    def test_result_is_202_until_finished(self, client, moving_video):
        path, _ = moving_video
        job_id = upload(client, path).json()["job_id"]
        r = client.get(f"{V}/{job_id}/result")
        assert r.status_code == 202 and r.json()["status"] == "uploaded"

    def test_result_matches_database_and_is_retrievable_by_an_independent_client(self, client, moving_video, register_bundle, set_setting):
        from backend.app.main import app
        set_setting(allow_synthetic_model_inference=True)
        register_bundle()
        path, gt = moving_video
        job_id, st = run_job(client, path, gt)
        with TestClient(app) as other:                          # a brand-new HTTP client / connection
            res = other.get(f"{V}/{job_id}/result")
            assert res.status_code == 200
            body = res.json()
        assert body == client.get(f"{V}/{job_id}/status").json()
        with SessionLocal() as s:                               # and the DB row holds the same data
            job = get_job(s, job_id)
            assert job.status == "completed" and job.result["prediction"] == body["prediction"]
            assert job.stage_progress["tracking"]["frames_tracked"] == body["target_tracking_summary"]["frames_tracked"]
        for key in ("job_id", "status", "video_metadata", "frame_processing_summary", "target_tracking_summary",
                    "target_selection", "feature_extraction", "model_status", "prediction", "artifact_references",
                    "processing_duration_s", "completed_at"):
            assert body[key] is not None, key

    def test_result_survives_deletion_of_raw_video_and_status_is_stable(self, client, moving_video):
        path, gt = moving_video
        job_id, st = run_job(client, path, gt)
        a = client.get(f"{V}/{job_id}/result").json()
        b = client.get(f"{V}/{job_id}/result").json()
        assert a == b and a["status"] == "completed_without_prediction"

    def test_unknown_job_404_everywhere(self, client):
        for path in ("status", "result", "tracking", "preview-frame", "frame-info"):
            assert client.get(f"{V}/nope/{path}").status_code == 404, path
        assert client.post(f"{V}/nope/process").status_code == 404
        assert client.post(f"{V}/nope/cancel").status_code == 404
        assert client.delete(f"{V}/nope").status_code == 404

    def test_predict_reruns_the_model_stage_from_stored_crops_after_the_raw_video_is_gone(self, client, moving_video, register_bundle, set_setting, settings):
        path, gt = moving_video
        job_id, st = run_job(client, path, gt)                  # no bundle yet
        assert st["prediction"] is None and st["artifact_references"]["raw_video"] == "deleted_per_retention_policy"
        set_setting(allow_synthetic_model_inference=True)
        model_id = register_bundle()
        r = client.post("/api/v1/predict", json={"job_id": job_id})
        assert r.status_code == 200, r.text
        st2 = client.get(f"{V}/{job_id}/result").json()
        assert st2["status"] == "completed" and st2["prediction"] is not None and st2["active_model_id"] == model_id
        assert st2["feature_extraction"]["weights_provenance"]["kind"] == "trained_bundle"
        # un-registering the model must clear the stale prediction on the next run
        with SessionLocal() as s:
            from backend.app.models import ModelRegistry
            s.query(ModelRegistry).delete()
            s.commit()
        assert client.post("/api/v1/predict", json={"job_id": job_id}).status_code == 200
        st3 = client.get(f"{V}/{job_id}/result").json()
        assert st3["status"] == "completed_without_prediction" and st3["prediction"] is None and st3["active_model_id"] is None
        assert st3["model_status"] == "model_not_ready"

    def test_predict_unknown_model_id_is_404_and_job_not_left_in_inference(self, client, moving_video):
        path, gt = moving_video
        job_id, _ = run_job(client, path, gt)
        r = client.post("/api/v1/predict", json={"job_id": job_id, "model_id": "does-not-exist"})
        assert r.status_code == 404
        assert client.get(f"{V}/{job_id}/status").json()["status"] == "completed_without_prediction"

    def test_predict_on_unprocessed_job_is_rejected(self, client, moving_video):
        path, _ = moving_video
        job_id = upload(client, path).json()["job_id"]
        assert client.post("/api/v1/predict", json={"job_id": job_id}).status_code == 409


# ───────────────────────── 9. cancellation, cleanup, retention ─────────────────────────
class _CancelAfter:
    """Tracker stub that cancels the job (via the DB) after N updates, then keeps 'tracking'."""
    def __init__(self, job_id_box, after):
        self.box, self.after, self.n = job_id_box, after, 0

    def init(self, frame, bbox):
        self.bbox = bbox
        return None

    def update(self, frame):
        self.n += 1
        if self.n == self.after:
            with SessionLocal() as s:
                cancel_job(s, get_job(s, self.box[0]))
        return True, tuple(float(v) for v in self.bbox)


class TestCancellationCleanupRetention:
    def test_cancel_during_tracking_stops_cleans_derived_files_and_keeps_raw_video(self, client, moving_video, monkeypatch):
        from backend.app.workers import processing_worker
        path, gt = moving_video
        up = upload(client, path).json()["job_id"]
        select(client, up, {k: gt["truth"][0][k] for k in "xywh"}, gt["size"])
        box = [up]
        monkeypatch.setattr(processing_worker, "TRACKER_FACTORY", lambda: _CancelAfter(box, after=3))
        assert client.post(f"{V}/{up}/process").status_code == 202       # runs inline; cancelled mid-tracking
        st = client.get(f"{V}/{up}/status").json()
        assert st["status"] == "cancelled"                               # never overwritten by a later stage
        assert st["prediction"] is None and st["feature_extraction_status"] is None
        assert not job_dir_for(up).exists(), "partial crops/tracking files must be removed"
        with SessionLocal() as s:
            job = get_job(s, up)
            assert job.stored_path and Path(job.stored_path).exists(), "raw video kept so the operator can retry"
        assert client.post("/api/v1/predict", json={"job_id": up}).status_code == 409
        # result endpoint treats cancelled as terminal (original code returned 202 forever)
        assert client.get(f"{V}/{up}/result").status_code == 200
        # a cancelled job can be reprocessed from scratch
        monkeypatch.setattr(processing_worker, "TRACKER_FACTORY", None)
        assert client.post(f"{V}/{up}/process").status_code == 202
        assert client.get(f"{V}/{up}/status").json()["status"] == "completed_without_prediction"

    @pytest.mark.parametrize("when", ["before_model_stage", "after_model_stage"])
    def test_cancel_between_stages_is_never_overwritten_by_a_later_stage(self, client, moving_video, monkeypatch, when):
        """No tracking-loop check can fire here: only the guard on status writes can stop the worker."""
        from backend.app.workers import processing_worker as pw
        path, gt = moving_video
        job_id = upload(client, path).json()["job_id"]
        select(client, job_id, {k: gt["truth"][0][k] for k in "xywh"}, gt["size"])
        real = pw.run_model_stage

        def cancel():
            with SessionLocal() as s:
                cancel_job(s, get_job(s, job_id))

        def wrapper(*a, **k):
            if when == "before_model_stage":
                cancel()
            out = real(*a, **k)
            if when == "after_model_stage":
                cancel()
            return out
        monkeypatch.setattr(pw, "run_model_stage", wrapper)
        assert client.post(f"{V}/{job_id}/process").status_code == 202
        st = client.get(f"{V}/{job_id}/status").json()
        assert st["status"] == "cancelled"
        assert st["prediction"] is None and st["model_status"] is None and st["feature_extraction_status"] is None
        assert not job_dir_for(job_id).exists()
        with SessionLocal() as s:
            job = get_job(s, job_id)
            assert job.result is None and job.stored_path and Path(job.stored_path).exists()   # raw kept, no result stored

    def test_cancel_before_start_is_respected_by_the_worker(self, client, moving_video):
        from backend.app.workers.processing_worker import run_processing_job
        path, gt = moving_video
        job_id = upload(client, path).json()["job_id"]
        select(client, job_id, {k: gt["truth"][0][k] for k in "xywh"}, gt["size"])
        assert client.post(f"{V}/{job_id}/cancel").json() == {"job_id": job_id, "cancelled": True, "status": "cancelled"}
        run_processing_job(job_id)                                       # as if dequeued late
        assert client.get(f"{V}/{job_id}/status").json()["status"] == "cancelled"
        assert not job_dir_for(job_id).exists()

    def test_cancel_of_finished_job_is_a_noop(self, client, moving_video):
        path, gt = moving_video
        job_id, _ = run_job(client, path, gt)
        r = client.post(f"{V}/{job_id}/cancel").json()
        assert r["cancelled"] is False and r["status"] == "completed_without_prediction"

    def test_default_retention_deletes_raw_video_only_after_outputs_are_persisted(self, client, moving_video, monkeypatch):
        from backend.app.services import job_service
        from backend.app.workers import processing_worker
        path, gt = moving_video
        job_id = upload(client, path).json()["job_id"]
        select(client, job_id, {k: gt["truth"][0][k] for k in "xywh"}, gt["size"])
        with SessionLocal() as s:
            raw = Path(get_job(s, job_id).stored_path)
        seen = {}
        real = job_service.update_job_status

        def spy(db, job, status, *a, **k):
            if status in ("completed", "completed_without_prediction"):
                d = job_dir_for(job_id)
                seen["raw_exists_at_final_commit"] = raw.exists()
                seen["tracking"] = (d / "tracking.json").exists()
                seen["crops"] = len(list((d / "crops").glob("*.png")))
                seen["features"] = (d / "features.npz").exists()
            return real(db, job, status, *a, **k)
        monkeypatch.setattr(processing_worker, "update_job_status", spy)
        assert client.post(f"{V}/{job_id}/process").status_code == 202
        assert seen == {"raw_exists_at_final_commit": True, "tracking": True, "crops": 8, "features": True}
        assert not raw.exists(), "raw video must be gone after completion (RETAIN_RAW_VIDEO=false)"
        st = client.get(f"{V}/{job_id}/status").json()
        assert st["artifact_references"]["raw_video"] == "deleted_per_retention_policy"
        with SessionLocal() as s:
            assert get_job(s, job_id).stored_path is None
        d = job_dir_for(job_id)                                           # derived artifacts survive
        assert (d / "tracking.json").exists() and (d / "features.npz").exists() and len(list((d / "crops").glob("*.png"))) == 8

    def test_after_raw_video_deletion_dependent_endpoints_fail_clearly_and_results_survive(self, client, moving_video):
        path, gt = moving_video
        job_id, st = run_job(client, path, gt)
        assert client.get(f"{V}/{job_id}/preview-frame").status_code == 410
        assert client.get(f"{V}/{job_id}/frame-info").status_code == 410
        r = client.post(f"{V}/{job_id}/process")
        assert r.status_code == 409 and "no longer available" in r.json()["detail"]
        assert client.get(f"{V}/{job_id}/status").json()["status"] == "completed_without_prediction"   # result not clobbered
        assert client.get(f"{V}/{job_id}/status").json()["target_tracking_summary"]["frames_tracked"] == 8

    def test_retain_raw_video_true_keeps_the_file_and_allows_reprocessing(self, client, moving_video, set_setting):
        set_setting(retain_raw_video=True)
        path, gt = moving_video
        job_id, st = run_job(client, path, gt)
        with SessionLocal() as s:
            raw = Path(get_job(s, job_id).stored_path)
        assert raw.exists() and st["artifact_references"]["raw_video"] == "retained"
        assert client.get(f"{V}/{job_id}/preview-frame").status_code == 200
        assert client.post(f"{V}/{job_id}/process").status_code == 202           # allowed: not 'completed'
        again = client.get(f"{V}/{job_id}/status").json()
        assert again["status"] == "completed_without_prediction" and again["target_tracking_summary"]["frames_tracked"] == 8

    def test_failure_keeps_raw_video_cleans_partial_outputs_and_can_be_retried(self, client, moving_video, monkeypatch):
        from backend.app.workers import processing_worker

        class Exploding:
            def init(self, frame, bbox):
                return False                                              # OpenCV-style init failure
            def update(self, frame):
                raise AssertionError("must not be reached")
        path, gt = moving_video
        job_id = upload(client, path).json()["job_id"]
        select(client, job_id, {k: gt["truth"][0][k] for k in "xywh"}, gt["size"])
        monkeypatch.setattr(processing_worker, "TRACKER_FACTORY", lambda: Exploding())
        assert client.post(f"{V}/{job_id}/process").status_code == 202
        st = client.get(f"{V}/{job_id}/status").json()
        assert st["status"] == "failed" and "Tracker could not start" in st["errors"]
        assert st["prediction"] is None and st["feature_extraction_status"] is None
        assert not job_dir_for(job_id).exists()
        with SessionLocal() as s:
            assert Path(get_job(s, job_id).stored_path).exists()
        monkeypatch.setattr(processing_worker, "TRACKER_FACTORY", None)
        assert client.post(f"{V}/{job_id}/process").status_code == 202           # retry
        assert client.get(f"{V}/{job_id}/status").json()["status"] == "completed_without_prediction"

    def test_delete_removes_raw_video_all_artifacts_and_db_row(self, client, moving_video, set_setting):
        set_setting(retain_raw_video=True)
        path, gt = moving_video
        job_id, _ = run_job(client, path, gt)
        with SessionLocal() as s:
            raw = Path(get_job(s, job_id).stored_path)
        assert raw.exists() and job_dir_for(job_id).exists()
        r = client.delete(f"{V}/{job_id}")
        assert r.status_code == 200 and sorted(r.json()["artifacts_removed"]) == ["job_artifacts", "raw_video"]
        assert not raw.exists() and not job_dir_for(job_id).exists()
        assert client.get(f"{V}/{job_id}/status").status_code == 404

    def test_delete_of_active_job_is_refused(self, client, moving_video):
        path, _ = moving_video
        job_id = upload(client, path).json()["job_id"]
        with SessionLocal() as s:
            j = get_job(s, job_id)
            j.status = "tracking"
            s.commit()
        assert client.delete(f"{V}/{job_id}").status_code == 409
        with SessionLocal() as s:
            j = get_job(s, job_id)
            j.status = "uploaded"
            s.commit()
        assert client.delete(f"{V}/{job_id}").status_code == 200

    def test_cannot_process_twice_concurrently_or_after_completion_with_prediction(self, client, moving_video, register_bundle, set_setting):
        set_setting(allow_synthetic_model_inference=True)
        register_bundle()
        path, gt = moving_video
        job_id, st = run_job(client, path, gt)
        assert st["status"] == "completed"
        assert client.post(f"{V}/{job_id}/process").status_code == 409
        with SessionLocal() as s:
            j = get_job(s, job_id)
            j.status = "tracking"
            s.commit()
        assert client.post(f"{V}/{job_id}/process").status_code == 409

    def test_selection_cannot_change_after_processing_started(self, client, moving_video):
        path, gt = moving_video
        job_id, _ = run_job(client, path, gt)
        r = select(client, job_id, {"x": 20, "y": 20, "w": 30, "h": 30}, gt["size"])
        assert r.status_code == 409
