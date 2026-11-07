"""Build a SYNTHETIC-TEST pipeline bundle (upstream weights + QAOA mapping + VQC).

Everything here is explicitly labelled ``synthetic_test_fixture``.  It is trained on random
noise "crops" with arbitrary labels purely to exercise the QAOA -> VQC inference path and the
compatibility checks.  It has NO behavioural meaning and must never be reported as a model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.models.cnn.extractor import SimpleCNN
from src.models.fusion.fusion import FusionModule
from src.models.lstm.temporal import TemporalLSTM
from src.models.quantum.qaoa import QAOAConfig, QAOAFeatureSelector
from src.models.quantum.vqc import VQCClassifier
from src.pipeline.bundle import make_mapping, save_bundle
from src.pipeline.features import CropSequence, UpstreamWeights, run_cnn_lstm_fusion


def build_synthetic_bundle(
    out_dir: Path, settings, *, n_candidates: int = 4, k: int = 2, n_qubits: int = 4,
    depth: int = 1, n_per_class: int = 5, seq_len: int = 5, epochs: int = 2, seed: int = 0,
    qaoa_max_iter: int = 40, corrupt_dims: bool = False,
) -> Path:
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    cnn = SimpleCNN(feature_dim=settings.cnn_feature_dim).eval()
    lstm = TemporalLSTM(settings.cnn_feature_dim, settings.lstm_hidden_dim,
                        settings.lstm_num_layers, settings.lstm_dropout).eval()
    fusion = FusionModule(visual_dim=settings.lstm_hidden_dim, output_dim=settings.fusion_output_dim,
                          use_structured=False).eval()
    up = UpstreamWeights(cnn, lstm, fusion, provenance={"kind": "synthetic_fixture"})

    # synthetic sequences: class c has base brightness 40 + 70c (so fused vectors differ by class)
    H, W = settings.frame_height, settings.frame_width
    fused, labels = [], []
    for c in range(settings.num_classes):
        for _ in range(n_per_class):
            imgs = np.clip(rng.normal(40 + 70 * c, 25, (seq_len, H, W, 3)), 0, 255).astype(np.uint8)
            seq = CropSequence(imgs, [i * 500.0 for i in range(seq_len)], list(range(seq_len)), list(range(seq_len)))
            fused.append(run_cnn_lstm_fusion(seq, up, settings.sequence_length).fused)
            labels.append(c)
    X, y = np.stack(fused), np.asarray(labels)

    # candidates = top dims by between-class variance; QAOA picks k of them
    probe = QAOAFeatureSelector(QAOAConfig(n_qubits=n_qubits, target_k=k))
    rel_all = probe.compute_relevance_scores(X, y)
    candidates = np.argsort(-rel_all)[:n_candidates].tolist()
    sel = QAOAFeatureSelector(QAOAConfig(n_qubits=n_qubits, depth=depth, target_k=k,
                                         max_iter=qaoa_max_iter, random_seed=seed))
    qres = sel.run(X[:, candidates], y, [f"fused_{c}" for c in candidates])
    sel_global = [candidates[i] for i in qres.selected_indices]
    Xs = X[:, sel_global]
    mapping = make_mapping(
        settings.fusion_output_dim, candidates, qres.selected_indices, Xs,
        {"n_qubits": n_qubits, "depth": depth, "backend": qres.backend,
         "objective_value": qres.objective_value, "target_k": k, "relevance": qres.relevance,
         "gamma": qres.gamma_params, "beta": qres.beta_params, "optimizer": "COBYLA",
         "qubo_value_selected": qres.qubo_value_selected, "qubo_value_optimal": qres.qubo_value_optimal,
         "matches_exhaustive_optimum": qres.matches_exhaustive_optimum},
    )
    Z = torch.tensor((Xs - np.asarray(mapping["mean"])) / (np.asarray(mapping["std"]) + 1e-8), dtype=torch.float32)
    vqc = VQCClassifier(input_dim=len(sel_global) + (1 if corrupt_dims else 0), n_qubits=n_qubits,
                        depth=depth, num_classes=settings.num_classes)
    opt = torch.optim.Adam(vqc.parameters(), lr=0.05)
    lossf = torch.nn.CrossEntropyLoss()
    vqc.train()
    zz = torch.cat([Z, torch.zeros(len(Z), 1)], 1) if corrupt_dims else Z
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(vqc(zz), torch.as_tensor(y, dtype=torch.long))
        loss.backward()
        opt.step()
    vqc.eval()
    return save_bundle(
        out_dir, cnn=cnn, lstm=lstm, fusion=fusion, mapping=mapping, vqc=vqc,
        training_data={"kind": "synthetic_test_fixture", "n_samples": int(len(y)),
                       "description": "random noise crops with arbitrary labels â€” plumbing test only"},
        class_names=list(settings.class_names), version="synthetic-test-0",
        frame_hw=(H, W),
    )
