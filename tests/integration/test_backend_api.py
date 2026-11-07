"""Backend API behaviour outside the video workflow: health/readiness, registry, datasets, experiments."""
import sys

import pytest

from backend.app.db import SessionLocal


def component(data, name):
    return next(c for c in data["components"] if c["name"] == name)


class TestHealthEndpoints:
    def test_health(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok" and data["version"] == "1.0.0" and data["timestamp"]

    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200 and "service" in r.json()


class TestReadiness:
    def test_ready_reports_real_component_state_without_a_model(self, client):
        data = client.get("/api/v1/ready").json()
        assert data["ready"] is True and data["database_operational"] is True
        assert data["quantum_available"] is True and data["model_available"] is False
        assert component(data, "database")["status"] == "ok"
        assert component(data, "pennylane")["status"] == "ok" and "0.38.0" in component(data, "pennylane")["detail"]
        assert component(data, "pytorch")["status"] == "ok"
        tr = component(data, "opencv_tracker")
        assert tr["status"] == "ok" and "CSRT" in tr["detail"]
        am = component(data, "active_model")
        assert am["status"] == "unavailable" and "No pipeline bundle" in am["detail"]
        assert "without a prediction" in data["message"]

    def test_synthetic_bundle_is_not_reported_available_unless_explicitly_allowed(self, client, register_bundle, set_setting):
        register_bundle()
        data = client.get("/api/v1/ready").json()
        assert data["model_available"] is False
        assert "synthetic_checkpoint_refused" in component(data, "active_model")["detail"]
        set_setting(allow_synthetic_model_inference=True)
        data = client.get("/api/v1/ready").json()
        assert data["model_available"] is True and "synthetic-trained" in component(data, "active_model")["detail"]
        assert "inference pipeline ready" in data["message"]

    def test_degraded_tracker_backend_is_visible(self, client, monkeypatch):
        import src.video.tracker as t
        monkeypatch.setattr(t, "create_cv_tracker", lambda: (object(), "MIL"))
        data = client.get("/api/v1/ready").json()
        tr = component(data, "opencv_tracker")
        assert tr["status"] == "degraded" and "install opencv-contrib" in tr["detail"] and data["ready"] is True

    def test_no_tracker_at_all_means_not_ready(self, client, monkeypatch):
        import src.video.tracker as t

        def boom():
            raise RuntimeError("No OpenCV single-object tracker is available")
        monkeypatch.setattr(t, "create_cv_tracker", boom)
        data = client.get("/api/v1/ready").json()
        assert data["ready"] is False and component(data, "opencv_tracker")["status"] == "unavailable"

    def test_missing_pennylane_is_reported_not_hidden_but_processing_stays_ready(self, client, monkeypatch):
        monkeypatch.setitem(sys.modules, "pennylane", None)
        data = client.get("/api/v1/ready").json()
        assert data["quantum_available"] is False
        assert component(data, "pennylane")["status"] == "unavailable"
        assert data["ready"] is True       # video processing up to feature extraction does not need PennyLane


class TestModelsAPI:
    def test_list_models_empty(self, client):
        r = client.get("/api/v1/models")
        assert r.status_code == 200 and r.json() == []


class TestDatasetsAPI:
    def test_dataset_status(self, client):
        data = client.get("/api/v1/datasets/status").json()
        assert isinstance(data["dataset_available"], bool) and data["message"]

    def test_dataset_validate(self, client):
        r = client.post("/api/v1/datasets/validate")
        assert r.status_code == 200 and isinstance(r.json()["valid"], bool)


class TestFederatedAPI:
    def test_federated_status(self, client):
        data = client.get("/api/v1/federated/status").json()
        assert "federated_experiments" in data and "note" in data


class TestPredictAPI:
    def test_predict_missing_job(self, client):
        assert client.post("/api/v1/predict", json={"job_id": "doesnotexist"}).status_code == 404


class TestExperimentsAPI:
    def test_list_experiments(self, client):
        r = client.get("/api/v1/experiments")
        assert r.status_code == 200 and isinstance(r.json(), list)

    def test_classical_experiment_runs_and_reports_status(self, client):
        r = client.post("/api/v1/experiments", json={
            "name": "test classical", "experiment_type": "classical", "config": {"epochs": 2, "n_samples": 30}})
        assert r.status_code in (200, 202), r.text
        exp_id = r.json()["experiment_id"]
        st = client.get(f"/api/v1/experiments/{exp_id}/status")
        assert st.status_code == 200 and st.json()["experiment_id"] == exp_id
        assert st.json()["status"] in ("queued", "running", "completed", "failed")

    def test_missing_experiment_returns_404(self, client):
        assert client.get("/api/v1/experiments/nope/status").status_code == 404

    def test_qaoa_vqc_experiment_is_synthetic_and_never_usable_for_video(self, client):
        r = client.post("/api/v1/experiments", json={
            "name": "qaoa+vqc plumbing", "experiment_type": "qaoa_vqc",
            "config": {"n_features": 4, "n_qubits": 4, "n_samples": 12, "epochs": 1, "qaoa_depth": 1}})
        assert r.status_code in (200, 202), r.text
        exp = client.get(f"/api/v1/experiments/{r.json()['experiment_id']}/status").json()
        assert exp["status"] == "completed", exp
        res = client.get(f"/api/v1/experiments/{r.json()['experiment_id']}/results")
        assert res.status_code == 200
        models = client.get("/api/v1/models").json()
        assert models, "experiment should have registered its synthetic checkpoint"
        for m in models:
            assert m["usable_for_video_inference"] is False and m["synthetic"] is True
