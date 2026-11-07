"""CNN spatial feature extractor using PyTorch."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from backend.app.core.logging_config import get_logger

logger = get_logger(__name__)


class SimpleCNN(nn.Module):
    """Lightweight CNN feature extractor for CPU-feasible development.
    Input: (B, C, H, W) float32 normalised frames.
    Output: (B, feature_dim) feature vectors.
    """

    def __init__(self, in_channels: int = 3, feature_dim: int = 256) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112->56
            nn.Dropout2d(0.1),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56->28
            nn.Dropout2d(0.1),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28->14
            nn.Dropout2d(0.1),
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14->7
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> (B, feature_dim)"""
        feats = self.backbone(x)
        pooled = self.pool(feats)
        return self.projection(pooled)


def build_cnn(
    backbone: str = "simple_cnn",
    feature_dim: int = 256,
    pretrained: bool = False,
) -> SimpleCNN:
    """Factory for CNN models. Currently only simple_cnn is supported offline."""
    if backbone == "simple_cnn":
        model = SimpleCNN(feature_dim=feature_dim)
        logger.info("Built SimpleCNN with feature_dim=%d", feature_dim)
        return model
    raise ValueError(f"Unknown CNN backbone: {backbone}. Supported: simple_cnn")


def save_cnn_config(model: SimpleCNN, path: Path) -> None:
    cfg = {
        "backbone": "simple_cnn",
        "feature_dim": model.feature_dim,
        "in_channels": 3,
    }
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def load_cnn_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
