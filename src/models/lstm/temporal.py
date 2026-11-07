"""LSTM temporal model for sequence modelling over CNN feature vectors."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from backend.app.core.logging_config import get_logger

logger = get_logger(__name__)


class TemporalLSTM(nn.Module):
    """LSTM module that accepts CNN feature sequences.
    
    Input: (B, T, feature_dim) CNN features
    Mask:  (B, T) boolean (True = real frame)
    Output: (B, hidden_dim) temporal representation
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: (B, T, D), mask: (B, T) bool -> (B, hidden_dim)"""
        B, T, D = x.shape

        if mask is not None:
            # Compute actual lengths from mask
            lengths = mask.sum(dim=1).long().clamp(min=1).cpu()
            packed = pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            packed_out, (h_n, _) = self.lstm(packed)
            # Use last hidden state from top layer
            out = h_n[-1]  # (B, hidden_dim)
        else:
            _, (h_n, _) = self.lstm(x)
            out = h_n[-1]

        out = self.dropout(out)
        out = self.layer_norm(out)
        return out  # (B, hidden_dim)


def save_lstm_config(model: TemporalLSTM, path: Path) -> None:
    cfg = {
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
    }
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def load_lstm_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
