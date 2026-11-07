"""Classical MLP baseline for comparison experiments."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from backend.app.core.logging_config import get_logger

logger = get_logger(__name__)


class ClassicalMLP(nn.Module):
    """Simple multi-layer perceptron classifier.

    Used as a classical baseline to compare against the quantum VQC.

    Input: (B, input_dim) feature vectors
    Output: (B, num_classes) logits
    """

    MODEL_ID = "classical_mlp_v1"

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_id": self.MODEL_ID,
            "state_dict": {k: v.cpu() for k, v in self.state_dict().items()},
            "config": {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "num_classes": self.num_classes,
            },
        }, str(path))

    @classmethod
    def load(cls, path: Path) -> "ClassicalMLP":
        state = torch.load(str(path), map_location="cpu", weights_only=False)
        cfg = state["config"]
        model = cls(**cfg)
        model.load_state_dict(state["state_dict"])
        return model
