"""Frame normalization and sequence construction for model input."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ImageNet-style mean and std for RGB
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FrameNormalizer:
    """Normalize BGR frames from OpenCV into float tensors for the CNN."""

    def __init__(
        self,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        target_size: tuple[int, int] = (112, 112),
    ) -> None:
        self.mean = mean if mean is not None else IMAGENET_MEAN
        self.std = std if std is not None else IMAGENET_STD
        self.target_size = target_size

    def normalize_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Convert BGR uint8 HxWxC -> float32 CxHxW normalized."""
        # BGR -> RGB
        frame_rgb = frame_bgr[..., ::-1].copy()
        # [0, 255] -> [0, 1]
        frame_f = frame_rgb.astype(np.float32) / 255.0
        # Normalize with mean/std
        frame_f = (frame_f - self.mean) / self.std
        # HxWxC -> CxHxW
        return frame_f.transpose(2, 0, 1)

    def build_sequence_tensor(
        self,
        frames: np.ndarray,
        padding_mask: Optional[np.ndarray] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build sequence tensor (T, C, H, W) and padding mask (T,).

        Args:
            frames: (T, H, W, C) uint8 BGR array.
            padding_mask: (T,) boolean where True = real, False = padding.

        Returns:
            sequence: torch.Tensor (T, C, H, W) float32
            mask: torch.Tensor (T,) bool
        """
        T = frames.shape[0]
        normalized = np.stack([self.normalize_frame(f) for f in frames], axis=0)
        sequence = torch.from_numpy(normalized)

        if padding_mask is None:
            padding_mask = np.ones(T, dtype=bool)
        mask = torch.from_numpy(padding_mask.astype(bool))
        return sequence, mask

    def save_config(self, path: Path) -> None:
        cfg = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "target_size": list(self.target_size),
        }
        path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    @classmethod
    def from_config(cls, path: Path) -> "FrameNormalizer":
        cfg = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            mean=np.array(cfg["mean"], dtype=np.float32),
            std=np.array(cfg["std"], dtype=np.float32),
            target_size=tuple(cfg["target_size"]),
        )
