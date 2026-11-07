"""Feature fusion module combining visual temporal features with structured behavioural features."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from backend.app.core.logging_config import get_logger

logger = get_logger(__name__)

# Documented schema for structured (tabular) input features
STRUCTURED_FEATURE_SCHEMA = [
    "screen_time_hours_per_day",   # Separately supplied measurement
    "age_years",
    "sleep_hours_per_night",
    "physical_activity_score",
    "caregiver_reported_attention_score",
]
STRUCTURED_FEATURE_DIM = len(STRUCTURED_FEATURE_SCHEMA)


class FusionModule(nn.Module):
    """Fuses CNN-LSTM temporal representation with optional structured features.

    Inputs:
        visual_repr: (B, visual_dim) from LSTM output
        structured_feats (optional): (B, structured_dim) normalised features
    Output:
        (B, output_dim) fused representation
    """

    def __init__(
        self,
        visual_dim: int = 128,
        structured_dim: int = STRUCTURED_FEATURE_DIM,
        output_dim: int = 64,
        use_structured: bool = True,
    ) -> None:
        super().__init__()
        self.visual_dim = visual_dim
        self.structured_dim = structured_dim
        self.output_dim = output_dim
        self.use_structured = use_structured

        in_dim = visual_dim + (structured_dim if use_structured else 0)
        self.fusion_head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        visual_repr: torch.Tensor,
        structured_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_structured and structured_feats is not None:
            if structured_feats.shape[-1] != self.structured_dim:
                raise ValueError(
                    f"Expected {self.structured_dim} structured features, "
                    f"got {structured_feats.shape[-1]}"
                )
            combined = torch.cat([visual_repr, structured_feats], dim=-1)
        elif not self.use_structured:
            combined = visual_repr
        else:
            logger.warning("use_structured=True but structured_feats not supplied; using zeros.")
            zeros = torch.zeros(
                visual_repr.shape[0], self.structured_dim,
                dtype=visual_repr.dtype, device=visual_repr.device,
            )
            combined = torch.cat([visual_repr, zeros], dim=-1)

        return self.fusion_head(combined)

    def save_config(self, path: Path) -> None:
        cfg = {
            "visual_dim": self.visual_dim,
            "structured_dim": self.structured_dim,
            "output_dim": self.output_dim,
            "use_structured": self.use_structured,
            "structured_feature_schema": STRUCTURED_FEATURE_SCHEMA,
            "version": "1.0",
        }
        path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    @classmethod
    def from_config(cls, path: Path) -> "FusionModule":
        cfg = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            visual_dim=cfg["visual_dim"],
            structured_dim=cfg["structured_dim"],
            output_dim=cfg["output_dim"],
            use_structured=cfg["use_structured"],
        )
