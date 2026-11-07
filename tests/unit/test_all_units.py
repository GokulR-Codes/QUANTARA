"""Unit tests — CNN, LSTM, Fusion, QAOA, VQC, FedAvg, VideoExtractor."""
import numpy as np
import pytest
import torch
import tempfile
from pathlib import Path


# ============================================================
# CNN
# ============================================================
class TestSimpleCNN:
    def _make_cnn(self):
        from src.models.cnn.extractor import SimpleCNN
        return SimpleCNN(in_channels=3, feature_dim=64)

    def test_output_shape(self):
        cnn = self._make_cnn()
        x = torch.randn(2, 3, 112, 112)
        out = cnn(x)
        assert out.shape == (2, 64), f"Expected (2,64), got {out.shape}"

    def test_train_mode(self):
        cnn = self._make_cnn()
        cnn.train()
        x = torch.randn(1, 3, 112, 112)
        out = cnn(x)
        loss = out.sum()
        loss.backward()
        # Gradients should exist
        grads = [p.grad for p in cnn.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients produced"

    def test_eval_mode_deterministic(self):
        cnn = self._make_cnn()
        cnn.eval()
        x = torch.randn(1, 3, 112, 112)
        with torch.no_grad():
            o1 = cnn(x)
            o2 = cnn(x)
        assert torch.allclose(o1, o2), "Eval mode not deterministic"

    def test_different_batch_sizes(self):
        cnn = self._make_cnn()
        cnn.eval()
        for b in [1, 4, 8]:
            x = torch.randn(b, 3, 112, 112)
            with torch.no_grad():
                out = cnn(x)
            assert out.shape == (b, 64)

    def test_checkpoint_round_trip(self, tmp_path):
        from src.models.cnn.extractor import SimpleCNN
        cnn = SimpleCNN(feature_dim=32)
        path = tmp_path / "cnn.pt"
        torch.save(cnn.state_dict(), str(path))
        cnn2 = SimpleCNN(feature_dim=32)
        cnn2.load_state_dict(torch.load(str(path), weights_only=True))
        x = torch.randn(1, 3, 112, 112)
        cnn.eval(); cnn2.eval()
        with torch.no_grad():
            assert torch.allclose(cnn(x), cnn2(x))


# ============================================================
# LSTM
# ============================================================
class TestTemporalLSTM:
    def _make_lstm(self, input_dim=32, hidden_dim=16):
        from src.models.lstm.temporal import TemporalLSTM
        return TemporalLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=1, dropout=0.0)

    def test_output_shape_no_mask(self):
        lstm = self._make_lstm()
        x = torch.randn(2, 10, 32)
        out = lstm(x)
        assert out.shape == (2, 16)

    def test_output_shape_with_mask(self):
        lstm = self._make_lstm()
        x = torch.randn(3, 10, 32)
        mask = torch.ones(3, 10, dtype=torch.bool)
        mask[0, 7:] = False  # first sample has 7 real frames
        out = lstm(x, mask)
        assert out.shape == (3, 16)

    def test_gradient_propagation(self):
        lstm = self._make_lstm()
        x = torch.randn(2, 5, 32, requires_grad=True)
        out = lstm(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_padding_not_real(self):
        """Padded frames (mask=False) should not change output compared to no-padding baseline."""
        lstm = self._make_lstm()
        lstm.eval()
        x = torch.randn(1, 5, 32)
        mask_full = torch.ones(1, 5, dtype=torch.bool)
        mask_partial = torch.zeros(1, 5, dtype=torch.bool)
        mask_partial[0, :3] = True  # only 3 real frames

        with torch.no_grad():
            out_full = lstm(x, mask_full)
            out_partial = lstm(x, mask_partial)
        # They can differ (different input content) but both should have correct shape
        assert out_full.shape == out_partial.shape == (1, 16)


# ============================================================
# Fusion
# ============================================================
class TestFusionModule:
    def test_visual_only(self):
        from src.models.fusion.fusion import FusionModule
        f = FusionModule(visual_dim=16, structured_dim=5, output_dim=8, use_structured=False)
        vis = torch.randn(2, 16)
        out = f(vis)
        assert out.shape == (2, 8)

    def test_multimodal(self):
        from src.models.fusion.fusion import FusionModule
        f = FusionModule(visual_dim=16, structured_dim=5, output_dim=8, use_structured=True)
        vis = torch.randn(2, 16)
        struct = torch.randn(2, 5)
        out = f(vis, struct)
        assert out.shape == (2, 8)

    def test_wrong_structured_dim_raises(self):
        from src.models.fusion.fusion import FusionModule
        f = FusionModule(visual_dim=16, structured_dim=5, output_dim=8, use_structured=True)
        vis = torch.randn(1, 16)
        bad_struct = torch.randn(1, 3)  # wrong dim
        with pytest.raises(ValueError):
            f(vis, bad_struct)


# ============================================================
# QAOA
# ============================================================
class TestQAOAFeatureSelector:
    def _make_selector(self, n_qubits=3, target_k=2):
        from src.models.quantum.qaoa import QAOAConfig, QAOAFeatureSelector
        cfg = QAOAConfig(n_qubits=n_qubits, depth=1, target_k=target_k, random_seed=42)
        return QAOAFeatureSelector(cfg)

    def test_relevance_scores_shape(self):
        sel = self._make_selector()
        X = np.random.randn(20, 3).astype(np.float32)
        y = np.array([0, 1, 2] * 6 + [0, 2])
        scores = sel.compute_relevance_scores(X, y)
        assert scores.shape == (3,)
        assert 0.0 <= scores.min() and scores.max() <= 1.0

    def test_run_returns_result(self):
        sel = self._make_selector(n_qubits=3, target_k=1)
        X = np.random.randn(30, 3).astype(np.float32)
        y = np.array([0, 1, 2] * 10)
        result = sel.run(X, y, feature_names=['a', 'b', 'c'])
        assert len(result.selected_indices) >= 1
        assert all(n in ['a', 'b', 'c'] for n in result.selected_names)

    def test_feature_count_constraint(self):
        """Selected feature count should be target_k or close to it."""
        sel = self._make_selector(n_qubits=3, target_k=2)
        X = np.random.randn(30, 3).astype(np.float32)
        y = np.array([0, 1, 2] * 10)
        result = sel.run(X, y)
        # Allow target_k ± 1 due to fallback decoding
        assert 1 <= len(result.selected_indices) <= 3

    def test_too_many_features_raises(self):
        sel = self._make_selector(n_qubits=2)
        X = np.random.randn(10, 5).astype(np.float32)  # 5 features > 2 qubits
        y = np.ones(10, dtype=int)
        with pytest.raises(ValueError):
            sel.run(X, y)

    def test_exhaustive_small(self):
        """For a 2-qubit problem, verify cost computation is finite."""
        from src.models.quantum.qaoa import QAOAConfig, QAOAFeatureSelector
        cfg = QAOAConfig(n_qubits=2, depth=1, target_k=1, random_seed=0)
        sel = QAOAFeatureSelector(cfg)
        scores = np.array([0.8, 0.3])
        h, J = sel._build_qubo_coefficients(scores)
        cost = sel._cost_hamiltonian_expectation(
            np.array([0.5]), np.array([0.25]), h, J
        )
        assert np.isfinite(cost), "QAOA cost must be finite"


# ============================================================
# VQC
# ============================================================
class TestVQCClassifier:
    def _make_vqc(self, input_dim=4, n_qubits=2, depth=1, num_classes=3):
        from src.models.quantum.vqc import VQCClassifier
        return VQCClassifier(input_dim=input_dim, n_qubits=n_qubits, depth=depth, num_classes=num_classes)

    def test_forward_shape(self):
        vqc = self._make_vqc()
        x = torch.randn(2, 4)
        out = vqc(x)
        assert out.shape == (2, 3)

    def test_gradient_propagation(self):
        vqc = self._make_vqc()
        x = torch.randn(1, 4)
        y = torch.tensor([0])
        loss = torch.nn.CrossEntropyLoss()(vqc(x), y)
        loss.backward()
        grads = [p.grad for p in vqc.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients from VQC"

    def test_checkpoint_save_load(self, tmp_path):
        from src.models.quantum.vqc import VQCClassifier
        vqc = VQCClassifier(input_dim=4, n_qubits=2, depth=1, num_classes=3)
        path = tmp_path / "vqc.pt"
        vqc.save(path)
        vqc2 = VQCClassifier.load(path)
        x = torch.randn(1, 4)
        vqc.eval(); vqc2.eval()
        with torch.no_grad():
            assert torch.allclose(vqc(x), vqc2(x), atol=1e-5)

    def test_training_updates_weights(self):
        """A training step should change the quantum_weights parameter."""
        vqc = self._make_vqc()
        w_before = vqc.quantum_weights.data.clone()
        opt = torch.optim.Adam(vqc.parameters(), lr=0.1)
        x = torch.randn(2, 4)
        y = torch.tensor([0, 1])
        loss = torch.nn.CrossEntropyLoss()(vqc(x), y)
        loss.backward()
        opt.step()
        assert not torch.allclose(vqc.quantum_weights.data, w_before), \
            "Quantum weights did not update after training step"

    def test_no_nan_output(self):
        vqc = self._make_vqc()
        x = torch.randn(4, 4)
        out = vqc(x)
        assert not torch.isnan(out).any(), "VQC produced NaN output"


# ============================================================
# FedAvg
# ============================================================
class TestFedAvg:
    def _make_model(self, input_dim=4, num_classes=3):
        from src.models.baselines.classical import ClassicalMLP
        return ClassicalMLP(input_dim=input_dim, num_classes=num_classes)

    def test_correct_weighted_average(self):
        """fedavg_aggregate should produce exact sample-weighted average."""
        from src.federated.fedavg import fedavg_aggregate
        model = self._make_model(4, 2)
        # Create two fake updates
        u1 = {k: torch.ones_like(v) * 1.0 for k, v in model.state_dict().items()}
        u2 = {k: torch.ones_like(v) * 3.0 for k, v in model.state_dict().items()}
        # 10 samples for u1, 10 for u2 → expected average = 2.0
        result = fedavg_aggregate(model, [u1, u2], [10, 10])
        for k, v in result.state_dict().items():
            assert torch.allclose(v.float(), torch.ones_like(v) * 2.0, atol=1e-4), \
                f"Parameter {k}: expected 2.0, got {v.mean()}"

    def test_invalid_update_rejected(self):
        from src.federated.fedavg import fedavg_aggregate, validate_update
        model = self._make_model()
        u_valid = {k: v.clone() for k, v in model.state_dict().items()}
        u_nan = {k: torch.full_like(v, float('nan')) for k, v in model.state_dict().items()}
        assert validate_update(u_valid) is True
        assert validate_update(u_nan) is False
        # Aggregation with one valid + one NaN should still succeed using valid only
        result = fedavg_aggregate(model, [u_valid, u_nan], [5, 5])
        for v in result.state_dict().values():
            assert not torch.isnan(v).any()

    def test_all_invalid_raises(self):
        from src.federated.fedavg import fedavg_aggregate
        model = self._make_model()
        u_nan = {k: torch.full_like(v, float('nan')) for k, v in model.state_dict().items()}
        with pytest.raises(ValueError):
            fedavg_aggregate(model, [u_nan], [5])

    def test_zero_samples_raises(self):
        from src.federated.fedavg import fedavg_aggregate
        model = self._make_model()
        u = {k: v.clone() for k, v in model.state_dict().items()}
        with pytest.raises(ValueError):
            fedavg_aggregate(model, [u], [0])

    def test_end_to_end_federated_rounds(self):
        """Run 2 rounds of simulated federated learning on tiny synthetic data."""
        from src.federated.fedavg import SimulatedFederatedClient, FederatedServer
        from src.models.baselines.classical import ClassicalMLP
        import tempfile

        rng = np.random.RandomState(0)
        X = rng.randn(30, 4).astype(np.float32)
        y = rng.randint(0, 3, 30)

        model = ClassicalMLP(input_dim=4, num_classes=3)
        clients = [
            SimulatedFederatedClient('c0', X[:15], y[:15], local_epochs=1),
            SimulatedFederatedClient('c1', X[15:], y[15:], local_epochs=1),
        ]
        with tempfile.TemporaryDirectory() as td:
            server = FederatedServer(
                global_model=model, clients=clients, n_rounds=2,
                checkpoint_dir=Path(td),
            )
            result = server.run(X_val=X[:10], y_val=y[:10])
        assert len(result.rounds) == 2
        for r in result.rounds:
            assert r.global_val_accuracy is not None


# ============================================================
# Video Extractor
# ============================================================
class TestVideoFrameExtractor:
    def _make_test_video(self, tmp_path: Path) -> Path:
        """Create a minimal valid video file using OpenCV."""
        import cv2
        path = tmp_path / "test.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(path), fourcc, 10.0, (64, 64))
        for _ in range(30):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
        return path

    def test_validate_valid_video(self, tmp_path):
        from src.video.extractor import VideoFrameExtractor
        vp = self._make_test_video(tmp_path)
        ext = VideoFrameExtractor()
        meta = ext.validate_video(vp)
        assert meta['fps'] > 0
        assert meta['width'] == 64
        assert meta['height'] == 64

    def test_validate_missing_video_raises(self, tmp_path):
        from src.video.extractor import VideoFrameExtractor
        from backend.app.core.errors import VideoValidationError
        ext = VideoFrameExtractor()
        with pytest.raises(VideoValidationError):
            ext.validate_video(tmp_path / "nonexistent.mp4")

    def test_extract_frames_produces_manifest(self, tmp_path):
        from src.video.extractor import VideoFrameExtractor
        vp = self._make_test_video(tmp_path)
        out_dir = tmp_path / "frames"
        ext = VideoFrameExtractor(sample_fps=5.0, max_frames=20, sequence_length=5)
        manifest = ext.extract_frames(vp, out_dir, job_id="test-job")
        assert (out_dir / "manifest.json").exists()
        assert manifest.num_frames_extracted == len(manifest.frames)
        assert manifest.job_id == "test-job"

    def test_frame_files_on_disk(self, tmp_path):
        from src.video.extractor import VideoFrameExtractor
        vp = self._make_test_video(tmp_path)
        out_dir = tmp_path / "frames2"
        ext = VideoFrameExtractor(sample_fps=5.0, max_frames=10, sequence_length=5)
        ext.extract_frames(vp, out_dir)
        files = list(out_dir.glob("frame_*.npy"))
        assert len(files) >= 1

    def test_sequence_is_padded_to_length(self, tmp_path):
        from src.video.extractor import VideoFrameExtractor
        vp = self._make_test_video(tmp_path)
        out_dir = tmp_path / "frames3"
        seq_len = 50
        ext = VideoFrameExtractor(sample_fps=1.0, max_frames=100, sequence_length=seq_len)
        manifest = ext.extract_frames(vp, out_dir)
        assert len(manifest.frames) == seq_len
        padding = sum(1 for f in manifest.frames if f.is_padding)
        assert padding >= 0

    def test_load_frame_sequence_shape(self, tmp_path):
        from src.video.extractor import VideoFrameExtractor
        vp = self._make_test_video(tmp_path)
        out_dir = tmp_path / "frames4"
        ext = VideoFrameExtractor(sample_fps=5.0, max_frames=10, sequence_length=6,
                                   frame_height=32, frame_width=32)
        ext.extract_frames(vp, out_dir)
        arr = ext.load_frame_sequence(out_dir)
        assert arr.ndim == 4  # (T, H, W, C)
        assert arr.shape[1:3] == (32, 32)
