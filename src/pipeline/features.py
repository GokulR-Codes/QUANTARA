"""CNN -> LSTM -> fusion feature extraction over ordered target crops.

Honesty rules enforced here
---------------------------
* Embeddings are produced by *actually running* the modules; shapes and finiteness are
  checked and a non-finite output raises instead of being reported as success.
* Every result carries ``weights_provenance``.  Modules that were not loaded from a trained
  bundle are randomly initialised (seeded) and labelled ``untrained_random_init``; such
  embeddings demonstrate that the stage runs, not that the features are meaningful.
* Padding is mask-only: padded positions are zeros that the LSTM never sees (packed
  sequences), and they are counted separately from real frames.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from backend.app.core.errors import VideoProcessingError
from backend.app.core.logging_config import get_logger
from src.models.cnn.extractor import SimpleCNN
from src.models.fusion.fusion import FusionModule
from src.models.lstm.temporal import TemporalLSTM
from src.preprocessing.normalizer import IMAGENET_MEAN, IMAGENET_STD, FrameNormalizer
from src.video.target_pipeline import TRACKING_FILE

logger = get_logger(__name__)

FEATURES_FILE = "features.npz"
UNTRAINED = "untrained_random_init"


class FeatureExtractionError(RuntimeError):
    pass


@dataclass
class UpstreamWeights:
    cnn: SimpleCNN
    lstm: TemporalLSTM
    fusion: FusionModule
    provenance: dict = field(default_factory=dict)
    frame_mean: np.ndarray = field(default_factory=lambda: IMAGENET_MEAN.copy())
    frame_std: np.ndarray = field(default_factory=lambda: IMAGENET_STD.copy())

    @property
    def trained(self) -> bool:
        return self.provenance.get("kind") != UNTRAINED


def build_untrained_upstream(settings) -> UpstreamWeights:
    """Seeded randomly-initialised modules, labelled as untrained."""
    torch.manual_seed(settings.random_seed)
    cnn = SimpleCNN(feature_dim=settings.cnn_feature_dim)
    lstm = TemporalLSTM(
        input_dim=settings.cnn_feature_dim, hidden_dim=settings.lstm_hidden_dim,
        num_layers=settings.lstm_num_layers, dropout=settings.lstm_dropout,
    )
    fusion = FusionModule(
        visual_dim=settings.lstm_hidden_dim, output_dim=settings.fusion_output_dim,
        use_structured=False,
    )
    for m in (cnn, lstm, fusion):
        m.eval()
    return UpstreamWeights(
        cnn, lstm, fusion,
        provenance={"kind": UNTRAINED, "seed": settings.random_seed,
                    "note": "Randomly initialised; embeddings are not trained features."},
    )


# ---------------------------------------------------------------------------
@dataclass
class CropSequence:
    images: np.ndarray          # (N, H, W, 3) uint8 BGR, in temporal order
    timestamps_ms: list[float]
    sample_indices: list[int]
    frame_indices: list[int]


def load_crop_sequence(job_dir: Path) -> CropSequence:
    """Load the crops of *trusted* samples, in timestamp order, from tracking.json."""
    tpath = Path(job_dir) / TRACKING_FILE
    if not tpath.exists():
        raise VideoProcessingError(f"tracking.json not found in {job_dir}")
    doc = json.loads(tpath.read_text(encoding="utf-8"))
    rows = [s for s in doc["samples"] if s["status"] == "tracked" and s.get("crop_file")]
    rows.sort(key=lambda s: (s["timestamp_ms"], s["sample_index"]))
    imgs = []
    for s in rows:
        img = cv2.imread(str(Path(job_dir) / s["crop_file"]), cv2.IMREAD_COLOR)
        if img is None:
            raise VideoProcessingError(f"Crop file missing or unreadable: {s['crop_file']}")
        imgs.append(img)
    if not imgs:
        return CropSequence(np.zeros((0, 0, 0, 3), np.uint8), [], [], [])
    return CropSequence(
        images=np.stack(imgs), timestamps_ms=[s["timestamp_ms"] for s in rows],
        sample_indices=[s["sample_index"] for s in rows],
        frame_indices=[s["frame_index"] for s in rows],
    )


@dataclass
class ModelInput:
    frames: torch.Tensor        # (1, T, 3, H, W) float32, zeros at padded positions
    mask: torch.Tensor          # (1, T) bool, True = real frame
    kept_positions: list[int]   # positions into the CropSequence that were used
    n_real: int
    n_padded: int
    subsampled: bool
    n_available: int


def build_model_input(seq: CropSequence, sequence_length: int, upstream: UpstreamWeights) -> ModelInput:
    n = len(seq.images)
    if n == 0:
        raise FeatureExtractionError("No target crops available")
    subsampled = n > sequence_length
    if subsampled:
        kept = np.unique(np.linspace(0, n - 1, sequence_length).round().astype(int)).tolist()
    else:
        kept = list(range(n))
    normalizer = FrameNormalizer(mean=upstream.frame_mean, std=upstream.frame_std)
    real, _ = normalizer.build_sequence_tensor(seq.images[kept])        # (n_real, 3, H, W)
    n_real = real.shape[0]
    _, C, H, W = real.shape
    frames = torch.zeros(1, sequence_length, C, H, W, dtype=torch.float32)
    frames[0, :n_real] = real
    mask = torch.zeros(1, sequence_length, dtype=torch.bool)
    mask[0, :n_real] = True
    return ModelInput(frames, mask, kept, n_real, sequence_length - n_real, subsampled, n)


@dataclass
class FeatureOutput:
    cnn_embeddings: np.ndarray   # (n_real, D)
    lstm_repr: np.ndarray        # (hidden,)
    fused: np.ndarray            # (F,)
    n_real: int
    n_padded: int
    subsampled: bool
    n_available: int
    timestamps_ms: list[float]
    sample_indices: list[int]
    provenance: dict

    def summary(self) -> dict:
        return {
            "n_embeddings": int(self.n_real),
            "n_padded_masked": int(self.n_padded),
            "n_crops_available": int(self.n_available),
            "subsampled": bool(self.subsampled),
            "cnn_embedding_dim": int(self.cnn_embeddings.shape[1]),
            "lstm_dim": int(self.lstm_repr.shape[0]),
            "fused_dim": int(self.fused.shape[0]),
            "weights_provenance": self.provenance,
            "first_timestamp_ms": self.timestamps_ms[0],
            "last_timestamp_ms": self.timestamps_ms[-1],
            "fused_l2_norm": round(float(np.linalg.norm(self.fused)), 6),
        }


def _check_finite(name: str, arr: np.ndarray) -> None:
    if not np.isfinite(arr).all():
        raise FeatureExtractionError(f"{name} contains non-finite values")


@torch.no_grad()
def run_cnn_lstm_fusion(seq: CropSequence, upstream: UpstreamWeights, sequence_length: int) -> FeatureOutput:
    """Run the real modules over ordered crops: CNN per crop -> LSTM over time -> fusion."""
    mi = build_model_input(seq, sequence_length, upstream)
    for m in (upstream.cnn, upstream.lstm, upstream.fusion):
        m.eval()
    real_frames = mi.frames[0, : mi.n_real]                              # (n_real, 3, H, W)
    emb = upstream.cnn(real_frames)                                      # (n_real, D)
    if emb.ndim != 2 or emb.shape[0] != mi.n_real:
        raise FeatureExtractionError(f"Unexpected CNN output shape {tuple(emb.shape)}")
    D = emb.shape[1]
    padded = torch.zeros(1, sequence_length, D, dtype=emb.dtype)
    padded[0, : mi.n_real] = emb                                         # order preserved
    temporal = upstream.lstm(padded, mi.mask)                            # (1, hidden)
    fused = upstream.fusion(temporal)                                    # (1, F)
    out = FeatureOutput(
        cnn_embeddings=emb.numpy().astype(np.float32),
        lstm_repr=temporal[0].numpy().astype(np.float32),
        fused=fused[0].numpy().astype(np.float32),
        n_real=mi.n_real, n_padded=mi.n_padded, subsampled=mi.subsampled,
        n_available=mi.n_available,
        timestamps_ms=[seq.timestamps_ms[i] for i in mi.kept_positions],
        sample_indices=[seq.sample_indices[i] for i in mi.kept_positions],
        provenance=dict(upstream.provenance),
    )
    for name, arr in (("CNN embeddings", out.cnn_embeddings), ("LSTM output", out.lstm_repr), ("fused vector", out.fused)):
        _check_finite(name, arr)
    return out


def save_features(job_dir: Path, out: FeatureOutput) -> Path:
    path = Path(job_dir) / FEATURES_FILE
    np.savez_compressed(
        path,
        cnn_embeddings=out.cnn_embeddings, lstm_repr=out.lstm_repr, fused=out.fused,
        timestamps_ms=np.asarray(out.timestamps_ms, dtype=np.float64),
        sample_indices=np.asarray(out.sample_indices, dtype=np.int64),
        weights_provenance=json.dumps(out.provenance),
    )
    return path
