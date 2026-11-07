"""Genuine Variational Quantum Circuit (VQC) classifier using PennyLane + PyTorch."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from backend.app.core.logging_config import get_logger
from backend.app.core.errors import CheckpointError

logger = get_logger(__name__)


def build_vqc_circuit(n_qubits: int, depth: int, backend: str = "default.qubit"):
    """Build and return a PennyLane QNode for the VQC.

    Architecture:
    1. AngleEncoding: RY gates encoding normalized input features.
    2. Trainable variational layers: RZ, RY, CZ entanglement.
    3. Measurement: Pauli-Z expectation on each qubit.
    """
    dev = qml.device(backend, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list:
        """inputs: (n_qubits,) encoded features, weights: (depth, n_qubits, 3) rotations."""
        # Angle encoding
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)

        # Variational layers
        for d in range(depth):
            # Trainable rotations
            for i in range(n_qubits):
                qml.RZ(weights[d, i, 0], wires=i)
                qml.RY(weights[d, i, 1], wires=i)
                qml.RZ(weights[d, i, 2], wires=i)
            # Entanglement: CZ ring
            for i in range(n_qubits):
                qml.CZ(wires=[i, (i + 1) % n_qubits])

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


class VQCClassifier(nn.Module):
    """Hybrid VQC + classical head classifier.

    Input: (B, n_features) fused feature vector.
    Output: (B, num_classes) logits.

    The input features are projected to n_qubits dimensions and
    encoded into quantum states. Trainable weights parametrize
    the variational circuit. Measurement outcomes feed a classical
    linear head.
    """

    MODEL_ID = "vqc_classifier_v1"

    def __init__(
        self,
        input_dim: int,
        n_qubits: int = 4,
        depth: int = 3,
        num_classes: int = 3,
        backend: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.depth = depth
        self.num_classes = num_classes
        self.backend = backend

        # Classical input encoder: projects input_dim -> n_qubits
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, n_qubits * 2),
            nn.Tanh(),
            nn.Linear(n_qubits * 2, n_qubits),
            nn.Tanh(),
        )

        # Variational weights: (depth, n_qubits, 3 rotations)
        weight_shape = (depth, n_qubits, 3)
        self.quantum_weights = nn.Parameter(
            torch.randn(weight_shape) * 0.1
        )

        # Build quantum circuit
        self.circuit = build_vqc_circuit(n_qubits, depth, backend)

        # Classical output head: n_qubits measurements -> num_classes
        self.output_head = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, input_dim) -> (B, num_classes) logits."""
        B = x.shape[0]
        # Encode to n_qubits angles in [-pi, pi]
        encoded = self.input_encoder(x) * np.pi  # (B, n_qubits)

        # Run quantum circuit for each sample in batch
        measurements = []
        for i in range(B):
            exp_vals = self.circuit(encoded[i], self.quantum_weights)
            if isinstance(exp_vals, (list, tuple)):
                meas = torch.stack([v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32) for v in exp_vals])
            elif isinstance(exp_vals, torch.Tensor):
                meas = exp_vals.float()
            else:
                meas = torch.tensor(exp_vals, dtype=torch.float32)
            measurements.append(meas)

        quantum_out = torch.stack(measurements, dim=0).float()  # (B, n_qubits) — cast to float32
        return self.output_head(quantum_out)

    def save(self, path: Path) -> None:
        """Save model weights and configuration."""
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model_id": self.MODEL_ID,
            "state_dict": {k: v.cpu() for k, v in self.state_dict().items()},
            "config": {
                "input_dim": self.input_dim,
                "n_qubits": self.n_qubits,
                "depth": self.depth,
                "num_classes": self.num_classes,
                "backend": self.backend,
            },
        }
        torch.save(state, str(path))
        logger.info("Saved VQC checkpoint to %s", path)

    @classmethod
    def load(cls, path: Path) -> "VQCClassifier":
        """Load model from checkpoint."""
        if not path.exists():
            raise CheckpointError(f"VQC checkpoint not found: {path}")
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        if state.get("model_id") != cls.MODEL_ID:
            raise CheckpointError(
                f"Incompatible checkpoint: expected {cls.MODEL_ID}, "
                f"got {state.get('model_id')}"
            )
        cfg = state["config"]
        model = cls(**cfg)
        model.load_state_dict(state["state_dict"])
        logger.info("Loaded VQC from %s", path)
        return model

    def save_config(self, path: Path) -> None:
        cfg = {
            "model_id": self.MODEL_ID,
            "input_dim": self.input_dim,
            "n_qubits": self.n_qubits,
            "depth": self.depth,
            "num_classes": self.num_classes,
            "backend": self.backend,
        }
        path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
