"""Pipeline bundle: save/load, checksums, dimension/compatibility checks, quantum head."""
import json
import shutil
import sys

import numpy as np
import pytest

from src.pipeline.bundle import (
    FILE_NAMES, MANIFEST_NAME, BundleError, load_bundle, map_features, predict_with_bundle, sha256_file,
)


@pytest.fixture()
def bundle_copy(synthetic_bundle_dir, tmp_path):
    dst = tmp_path / "b"
    shutil.copytree(synthetic_bundle_dir, dst)
    return dst


def _rewrite(bundle, file_key, mutate):
    """Edit a JSON file of the bundle and refresh its manifest checksum (so ONLY the
    compatibility checks, not the checksum check, can catch the tampering)."""
    path = bundle / FILE_NAMES[file_key]
    doc = json.loads(path.read_text())
    mutate(doc)
    path.write_text(json.dumps(doc))
    man = json.loads((bundle / MANIFEST_NAME).read_text())
    man["files"][file_key]["sha256"] = sha256_file(path)
    (bundle / MANIFEST_NAME).write_text(json.dumps(man))


class TestLoad:
    def test_valid_bundle_loads_with_consistent_dimensions(self, synthetic_bundle_dir, settings):
        b = load_bundle(synthetic_bundle_dir, settings, allow_synthetic=True)
        d = b.manifest["dims"]
        k = len(b.mapping["selected_indices"])
        assert b.vqc.input_dim == k == d["vqc_input_dim"]
        assert b.upstream.cnn.feature_dim == b.upstream.lstm.input_dim == d["cnn_feature_dim"]
        assert b.upstream.lstm.hidden_dim == b.upstream.fusion.visual_dim
        assert b.upstream.fusion.output_dim == b.mapping["fused_dim"]
        assert set(b.mapping["selected_indices"]) <= set(b.mapping["candidate_indices"])
        assert b.synthetic and b.upstream.provenance["kind"] == "trained_bundle"

    def test_synthetic_bundle_refused_by_default(self, synthetic_bundle_dir, settings):
        assert settings.allow_synthetic_model_inference is False
        with pytest.raises(BundleError) as e:
            load_bundle(synthetic_bundle_dir, settings)
        assert e.value.code == "synthetic_checkpoint_refused"

    def test_modified_checkpoint_fails_checksum(self, bundle_copy, settings):
        with open(bundle_copy / FILE_NAMES["vqc"], "ab") as f:
            f.write(b"\x00")
        with pytest.raises(BundleError) as e:
            load_bundle(bundle_copy, settings, allow_synthetic=True)
        assert e.value.code == "checksum_mismatch"

    @pytest.mark.parametrize("key", ["vqc", "mapping", "upstream"])
    def test_missing_file_reported(self, bundle_copy, settings, key):
        (bundle_copy / FILE_NAMES[key]).unlink()
        with pytest.raises(BundleError) as e:
            load_bundle(bundle_copy, settings, allow_synthetic=True)
        assert e.value.code == "missing_files"

    def test_missing_directory_reported(self, tmp_path, settings):
        with pytest.raises(BundleError) as e:
            load_bundle(tmp_path / "nope", settings, allow_synthetic=True)
        assert e.value.code == "missing_files"

    def test_mapping_selecting_more_features_than_vqc_accepts(self, bundle_copy, settings):
        def add_feature(m):
            extra = next(i for i in m["candidate_indices"] if i not in m["selected_indices"])
            m["selected_indices"].append(extra)
            m["mean"].append(0.0)
            m["std"].append(1.0)
        _rewrite(bundle_copy, "mapping", add_feature)
        with pytest.raises(BundleError) as e:
            load_bundle(bundle_copy, settings, allow_synthetic=True)
        assert e.value.code == "incompatible_dimensions"

    def test_fused_dim_mismatch(self, bundle_copy, settings):
        _rewrite(bundle_copy, "mapping", lambda m: m.update(fused_dim=m["fused_dim"] - 1))
        with pytest.raises(BundleError, match="fusion output") as e:
            load_bundle(bundle_copy, settings, allow_synthetic=True)
        assert e.value.code == "incompatible_dimensions"

    def test_selected_index_out_of_range(self, bundle_copy, settings):
        def bad(m):
            m["selected_indices"][0] = m["fused_dim"] + 5
        _rewrite(bundle_copy, "mapping", bad)
        with pytest.raises(BundleError) as e:
            load_bundle(bundle_copy, settings, allow_synthetic=True)
        assert e.value.code == "incompatible_dimensions"

    def test_crop_size_mismatch_with_configured_pipeline(self, bundle_copy, settings):
        man = json.loads((bundle_copy / MANIFEST_NAME).read_text())
        man["dims"]["frame_hw"] = [64, 64]
        (bundle_copy / MANIFEST_NAME).write_text(json.dumps(man))
        with pytest.raises(BundleError, match="crops") as e:
            load_bundle(bundle_copy, settings, allow_synthetic=True)
        assert e.value.code == "incompatible_dimensions"

    def test_structured_feature_bundle_not_usable_for_video_only(self, bundle_copy, settings):
        man = json.loads((bundle_copy / MANIFEST_NAME).read_text())
        man["uses_structured_features"] = True
        (bundle_copy / MANIFEST_NAME).write_text(json.dumps(man))
        with pytest.raises(BundleError) as e:
            load_bundle(bundle_copy, settings, allow_synthetic=True)
        assert e.value.code == "structured_features_required"

    def test_pennylane_missing_is_reported_not_hidden(self, synthetic_bundle_dir, settings, monkeypatch):
        monkeypatch.setitem(sys.modules, "src.models.quantum.vqc", None)    # makes the import raise
        with pytest.raises(BundleError) as e:
            load_bundle(synthetic_bundle_dir, settings, allow_synthetic=True)
        assert e.value.code == "quantum_unavailable"


class TestQuantumHead:
    def test_mapping_selects_and_standardises_only_chosen_dims(self):
        mapping = {"fused_dim": 6, "selected_indices": [4, 1], "mean": [1.0, 2.0], "std": [2.0, 4.0]}
        fused = np.array([9, 6, 9, 9, 5, 9], np.float32)
        assert map_features(fused, mapping) == pytest.approx([(5 - 1) / 2, (6 - 2) / 4], abs=1e-6)

    def test_wrong_fused_length_rejected(self):
        with pytest.raises(BundleError, match="fused vector"):
            map_features(np.zeros(5, np.float32), {"fused_dim": 6, "selected_indices": [0], "mean": [0], "std": [1]})

    def test_prediction_is_a_valid_deterministic_distribution(self, synthetic_bundle_dir, settings):
        b = load_bundle(synthetic_bundle_dir, settings, allow_synthetic=True)
        fused = np.random.RandomState(0).normal(0, 1, b.mapping["fused_dim"]).astype(np.float32)
        p1, p2 = predict_with_bundle(fused, b), predict_with_bundle(fused, b)
        assert p1 == p2
        probs = p1["class_probabilities"]
        assert len(probs) == settings.num_classes and sum(probs) == pytest.approx(1.0, abs=1e-5)
        assert p1["predicted_class"] == int(np.argmax(probs))
        assert p1["predicted_label"] == settings.class_names[p1["predicted_class"]]
        assert len(p1["mapped_input"]) == len(b.mapping["selected_indices"])

    def test_prediction_depends_on_the_selected_dims_only(self, synthetic_bundle_dir, settings):
        b = load_bundle(synthetic_bundle_dir, settings, allow_synthetic=True)
        rng = np.random.RandomState(1)
        fused = rng.normal(0, 1, b.mapping["fused_dim"]).astype(np.float32)
        other = fused.copy()
        unused = [i for i in range(len(fused)) if i not in b.mapping["selected_indices"]]
        other[unused] += 100.0
        assert predict_with_bundle(fused, b)["class_probabilities"] == predict_with_bundle(other, b)["class_probabilities"]
        changed = fused.copy()
        changed[b.mapping["selected_indices"][0]] += 5.0
        assert predict_with_bundle(fused, b)["class_probabilities"] != predict_with_bundle(changed, b)["class_probabilities"]
