"""Pipeline bundle: upstream weights + QAOA feature mapping + VQC checkpoint.

A VQC checkpoint is only meaningful together with the *exact* upstream modules (CNN, LSTM,
fusion) whose fused output it was trained on, and with the QAOA-selected feature mapping
and standardisation applied to that output.  A bundle ties these together:

    <bundle_dir>/
        bundle_manifest.json   declared dims, provenance, class names, sha256 of every file
        upstream.pt            CNN + LSTM + fusion configs and state dicts
        qaoa_mapping.json      candidate/selected indices into the fused vector, mean/std
        vqc.pt                 VQCClassifier.save() checkpoint

``load_bundle`` verifies files, checksums, shapes and dimension chains and raises
:class:`BundleError` (with a machine-readable ``code``) for anything incompatible; callers
turn that into an explicit ``model_status`` and never fabricate a prediction.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from backend.app.core.logging_config import get_logger
from src.models.cnn.extractor import SimpleCNN
from src.models.fusion.fusion import FusionModule
from src.models.lstm.temporal import TemporalLSTM
from src.pipeline.features import UpstreamWeights
from src.preprocessing.normalizer import IMAGENET_MEAN, IMAGENET_STD

logger = get_logger(__name__)

SCHEMA_VERSION = "1"
MANIFEST_NAME = "bundle_manifest.json"
FILE_NAMES = {"upstream": "upstream.pt", "mapping": "qaoa_mapping.json", "vqc": "vqc.pt"}
SYNTHETIC_KINDS = {"synthetic_test_fixture", "synthetic_random"}
BUNDLE_MODEL_TYPE = "qaoa_vqc_bundle"


class BundleError(Exception):
    """Bundle is missing, corrupted, or incompatible. ``code`` is machine-readable."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class LoadedBundle:
    path: Path
    manifest: dict
    upstream: UpstreamWeights
    mapping: dict
    vqc: Any                      # VQCClassifier

    @property
    def synthetic(self) -> bool:
        return self.manifest["training_data"]["kind"] in SYNTHETIC_KINDS

    def provenance(self) -> dict:
        return {
            "bundle_id": self.manifest["bundle_id"],
            "version": self.manifest["version"],
            "training_data": self.manifest["training_data"],
            "synthetic": self.synthetic,
            "file_checksums_sha256": {k: v["sha256"] for k, v in self.manifest["files"].items()},
        }

    def quantum_configuration(self) -> dict:
        d = self.manifest["dims"]
        return {
            "vqc": {"n_qubits": d["n_qubits"], "depth": d["vqc_depth"],
                    "input_dim": d["vqc_input_dim"], "num_classes": d["num_classes"],
                    "backend": self.vqc.backend},
            "qaoa": self.mapping.get("qaoa", {}),
            "fused_dim": self.mapping["fused_dim"],
            "candidate_indices": self.mapping["candidate_indices"],
            "selected_indices": self.mapping["selected_indices"],
        }


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def make_mapping(
    fused_dim: int, candidate_indices: list[int], selected_local: list[int],
    train_selected: np.ndarray, qaoa_summary: dict, feature_names: Optional[list[str]] = None,
) -> dict:
    """Build the saved QAOA feature mapping.

    ``selected_local`` indexes into ``candidate_indices`` (QAOA's variable numbering);
    ``train_selected`` is the (N, k) TRAINING-set matrix of the selected fused dims used to
    derive the standardisation applied at inference.
    """
    sel_global = [int(candidate_indices[i]) for i in selected_local]
    names = feature_names or [f"fused_{g}" for g in sel_global]
    return {
        "schema_version": SCHEMA_VERSION,
        "fused_dim": int(fused_dim),
        "candidate_indices": [int(i) for i in candidate_indices],
        "selected_indices": sel_global,
        "selected_feature_names": names,
        "mean": train_selected.mean(axis=0).astype(float).tolist(),
        "std": train_selected.std(axis=0).astype(float).tolist(),
        "qaoa": qaoa_summary,
    }


def save_bundle(
    bundle_dir: Path, *, cnn: SimpleCNN, lstm: TemporalLSTM, fusion: FusionModule,
    mapping: dict, vqc, training_data: dict, class_names: list[str], version: str,
    frame_hw: tuple[int, int] = (112, 112),
    frame_mean: np.ndarray = IMAGENET_MEAN, frame_std: np.ndarray = IMAGENET_STD,
) -> Path:
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": SCHEMA_VERSION,
        "cnn": {"config": {"in_channels": 3, "feature_dim": cnn.feature_dim},
                "state_dict": {k: v.cpu() for k, v in cnn.state_dict().items()}},
        "lstm": {"config": {"input_dim": lstm.input_dim, "hidden_dim": lstm.hidden_dim,
                            "num_layers": lstm.num_layers, "dropout": 0.0},
                 "state_dict": {k: v.cpu() for k, v in lstm.state_dict().items()}},
        "fusion": {"config": {"visual_dim": fusion.visual_dim, "structured_dim": fusion.structured_dim,
                              "output_dim": fusion.output_dim, "use_structured": fusion.use_structured},
                   "state_dict": {k: v.cpu() for k, v in fusion.state_dict().items()}},
    }, str(bundle_dir / FILE_NAMES["upstream"]))
    (bundle_dir / FILE_NAMES["mapping"]).write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    vqc.save(bundle_dir / FILE_NAMES["vqc"])

    files = {}
    for key, name in FILE_NAMES.items():
        p = bundle_dir / name
        files[key] = {"name": name, "sha256": sha256_file(p), "size_bytes": p.stat().st_size}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "bundle_id": str(uuid.uuid4()),
        "version": version,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "training_data": training_data,
        "files": files,
        "dims": {
            "frame_hw": list(frame_hw),
            "cnn_feature_dim": cnn.feature_dim,
            "lstm_hidden_dim": lstm.hidden_dim,
            "lstm_layers": lstm.num_layers,
            "fusion_output_dim": fusion.output_dim,
            "vqc_input_dim": vqc.input_dim,
            "n_qubits": vqc.n_qubits,
            "vqc_depth": vqc.depth,
            "num_classes": vqc.num_classes,
        },
        "class_names": list(class_names),
        "frame_normalization": {"mean": np.asarray(frame_mean).tolist(),
                                "std": np.asarray(frame_std).tolist(), "color_order": "RGB"},
        "uses_structured_features": bool(fusion.use_structured),
    }
    (bundle_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return bundle_dir


# ---------------------------------------------------------------------------
# Loading / validation
# ---------------------------------------------------------------------------
def read_manifest(bundle_dir: Path) -> dict:
    mp = Path(bundle_dir) / MANIFEST_NAME
    if not mp.exists():
        raise BundleError("missing_files", f"{MANIFEST_NAME} not found in {bundle_dir}")
    try:
        manifest = json.loads(mp.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BundleError("manifest_invalid", f"Manifest is not valid JSON: {exc}") from exc
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise BundleError("manifest_invalid", f"Unsupported bundle schema {manifest.get('schema_version')!r}")
    for key in ("bundle_id", "version", "training_data", "files", "dims", "class_names"):
        if key not in manifest:
            raise BundleError("manifest_invalid", f"Manifest missing '{key}'")
    return manifest


def verify_files(bundle_dir: Path, manifest: dict) -> None:
    for key, name in FILE_NAMES.items():
        entry = manifest["files"].get(key)
        p = Path(bundle_dir) / name
        if entry is None or not p.exists():
            raise BundleError("missing_files", f"Bundle file missing: {name}")
        if sha256_file(p) != entry["sha256"]:
            raise BundleError("checksum_mismatch", f"Checksum mismatch for {name}; file changed after packaging")


def load_bundle(bundle_dir: Path, settings, allow_synthetic: Optional[bool] = None) -> LoadedBundle:
    """Fully validate and load a bundle. Raises :class:`BundleError`."""
    bundle_dir = Path(bundle_dir)
    allow_synthetic = settings.allow_synthetic_model_inference if allow_synthetic is None else allow_synthetic
    manifest = read_manifest(bundle_dir)
    verify_files(bundle_dir, manifest)

    kind = manifest["training_data"].get("kind", "unknown")
    if kind in SYNTHETIC_KINDS and not allow_synthetic:
        raise BundleError(
            "synthetic_checkpoint_refused",
            f"Bundle was trained on '{kind}' data and is refused for real-video inference "
            "(set ALLOW_SYNTHETIC_MODEL_INFERENCE=true only for plumbing tests).",
        )
    d = manifest["dims"]
    if manifest.get("uses_structured_features"):
        raise BundleError("structured_features_required",
                          "Bundle's fusion module needs structured features; none are supplied for video-only inference.")
    if tuple(d["frame_hw"]) != (settings.frame_height, settings.frame_width):
        raise BundleError("incompatible_dimensions",
                          f"Bundle expects {tuple(d['frame_hw'])} crops; configured size is "
                          f"{(settings.frame_height, settings.frame_width)}")

    try:
        blob = torch.load(str(bundle_dir / FILE_NAMES["upstream"]), map_location="cpu", weights_only=True)
        cnn = SimpleCNN(**blob["cnn"]["config"])
        lstm = TemporalLSTM(**blob["lstm"]["config"])
        fusion = FusionModule(**blob["fusion"]["config"])
        cnn.load_state_dict(blob["cnn"]["state_dict"], strict=True)
        lstm.load_state_dict(blob["lstm"]["state_dict"], strict=True)
        fusion.load_state_dict(blob["fusion"]["state_dict"], strict=True)
    except Exception as exc:
        raise BundleError("incompatible_dimensions", f"Upstream weights do not match their config: {exc}") from exc
    for m in (cnn, lstm, fusion):
        m.eval()

    mapping = json.loads((bundle_dir / FILE_NAMES["mapping"]).read_text(encoding="utf-8"))
    sel = mapping["selected_indices"]
    k = len(sel)
    chain_errors = []
    if cnn.feature_dim != lstm.input_dim:
        chain_errors.append(f"CNN dim {cnn.feature_dim} != LSTM input {lstm.input_dim}")
    if lstm.hidden_dim != fusion.visual_dim:
        chain_errors.append(f"LSTM hidden {lstm.hidden_dim} != fusion visual dim {fusion.visual_dim}")
    if fusion.output_dim != mapping["fused_dim"]:
        chain_errors.append(f"fusion output {fusion.output_dim} != mapping fused_dim {mapping['fused_dim']}")
    if k == 0 or len(set(sel)) != k or min(sel) < 0 or max(sel) >= mapping["fused_dim"]:
        chain_errors.append(f"selected_indices {sel} invalid for fused_dim {mapping['fused_dim']}")
    if not set(sel) <= set(mapping["candidate_indices"]):
        chain_errors.append("selected_indices are not a subset of candidate_indices")
    if len(mapping["mean"]) != k or len(mapping["std"]) != k:
        chain_errors.append("mean/std length differs from number of selected features")
    elif not (np.isfinite(mapping["mean"]).all() and np.isfinite(mapping["std"]).all()):
        chain_errors.append("mean/std contain non-finite values")
    declared = (d["cnn_feature_dim"], d["lstm_hidden_dim"], d["fusion_output_dim"], d["vqc_input_dim"])
    actual = (cnn.feature_dim, lstm.hidden_dim, fusion.output_dim, k)
    if declared != actual:
        chain_errors.append(f"manifest dims {declared} != actual {actual}")
    if chain_errors:
        raise BundleError("incompatible_dimensions", "; ".join(chain_errors))

    try:
        from src.models.quantum.vqc import VQCClassifier
    except ImportError as exc:
        raise BundleError("quantum_unavailable", f"PennyLane is not installed: {exc}") from exc
    try:
        vqc = VQCClassifier.load(bundle_dir / FILE_NAMES["vqc"])
    except Exception as exc:
        raise BundleError("incompatible_checkpoint", f"VQC checkpoint cannot be loaded: {exc}") from exc
    vqc.eval()
    if vqc.input_dim != k:
        raise BundleError("incompatible_dimensions",
                          f"VQC input_dim {vqc.input_dim} != number of QAOA-selected features {k}")
    if vqc.num_classes != len(manifest["class_names"]):
        raise BundleError("incompatible_dimensions",
                          f"VQC outputs {vqc.num_classes} classes but manifest lists {len(manifest['class_names'])}")

    norm = manifest.get("frame_normalization", {})
    upstream = UpstreamWeights(
        cnn, lstm, fusion,
        provenance={"kind": "trained_bundle", "bundle_id": manifest["bundle_id"],
                    "training_data_kind": kind, "upstream_sha256": manifest["files"]["upstream"]["sha256"]},
        frame_mean=np.asarray(norm.get("mean", IMAGENET_MEAN), dtype=np.float32),
        frame_std=np.asarray(norm.get("std", IMAGENET_STD), dtype=np.float32),
    )
    return LoadedBundle(bundle_dir, manifest, upstream, mapping, vqc)


# ---------------------------------------------------------------------------
# Quantum head
# ---------------------------------------------------------------------------
def map_features(fused: np.ndarray, mapping: dict) -> np.ndarray:
    """Apply the saved QAOA selection + training-set standardisation to a fused vector."""
    if fused.ndim != 1 or fused.shape[0] != mapping["fused_dim"]:
        raise BundleError("incompatible_dimensions",
                          f"fused vector has shape {tuple(fused.shape)}; mapping expects ({mapping['fused_dim']},)")
    idx = np.asarray(mapping["selected_indices"], dtype=int)
    mean = np.asarray(mapping["mean"], dtype=np.float32)
    std = np.asarray(mapping["std"], dtype=np.float32)
    x = (fused[idx] - mean) / (std + 1e-8)
    if not np.isfinite(x).all():
        raise BundleError("incompatible_dimensions", "mapped features are non-finite")
    return x.astype(np.float32)


@torch.no_grad()
def predict_with_bundle(fused: np.ndarray, bundle: LoadedBundle) -> dict:
    x = map_features(fused, bundle.mapping)
    logits = bundle.vqc(torch.from_numpy(x).unsqueeze(0))
    probs = torch.softmax(logits, dim=-1).squeeze(0)
    if not torch.isfinite(probs).all():
        raise BundleError("inference_failed", "VQC produced non-finite probabilities")
    cls = int(probs.argmax().item())
    names = bundle.manifest["class_names"]
    return {
        "predicted_class": cls,
        "predicted_label": names[cls] if cls < len(names) else str(cls),
        "class_probabilities": [float(p) for p in probs.tolist()],
        "mapped_input": [float(v) for v in x.tolist()],
    }
