"""Dataset manifest loading, validation, and splitting."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from backend.app.core.logging_config import get_logger
from backend.app.core.errors import DatasetError

logger = get_logger(__name__)

REQUIRED_COLUMNS = {"video_path", "label", "participant_id"}
VALID_SPLITS = {"train", "val", "test"}


@dataclass
class DatasetReport:
    total_videos: int
    total_participants: int
    class_distribution: dict[str, int]
    split_counts: dict[str, int]
    missing_files: list[str]
    invalid_labels: list[str]
    duplicate_videos: list[str]
    is_valid: bool
    errors: list[str]

    def to_json(self) -> str:
        return json.dumps({
            "total_videos": self.total_videos,
            "total_participants": self.total_participants,
            "class_distribution": self.class_distribution,
            "split_counts": self.split_counts,
            "missing_files": self.missing_files,
            "invalid_labels": self.invalid_labels,
            "duplicate_videos": self.duplicate_videos,
            "is_valid": self.is_valid,
            "errors": self.errors,
        }, indent=2)


class DatasetManager:
    """Load, validate, split, and inspect labelled video datasets."""

    def __init__(self, class_names: list[str], random_seed: int = 42) -> None:
        self.class_names = class_names
        self.random_seed = random_seed
        self._df: Optional[pd.DataFrame] = None

    def load_manifest(self, manifest_path: Path) -> pd.DataFrame:
        """Load a CSV manifest and return the validated DataFrame."""
        if not manifest_path.exists():
            raise DatasetError(f"Manifest not found: {manifest_path}")
        df = pd.read_csv(manifest_path)
        missing_cols = REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            raise DatasetError(f"Missing required columns: {missing_cols}")
        self._df = df
        return df

    def validate(self, base_dir: Optional[Path] = None) -> DatasetReport:
        """Run full validation and return a DatasetReport."""
        if self._df is None:
            raise DatasetError("No manifest loaded. Call load_manifest() first.")
        df = self._df.copy()
        errors: list[str] = []

        # Missing files
        missing_files: list[str] = []
        for p in df["video_path"]:
            check = Path(p) if base_dir is None else base_dir / p
            if not check.exists():
                missing_files.append(str(p))

        # Invalid labels
        invalid_labels: list[str] = [
            str(lbl) for lbl in df["label"].unique() if str(lbl) not in self.class_names
        ]

        # Duplicates
        dup_mask = df.duplicated(subset=["video_path"], keep=False)
        duplicate_videos = df.loc[dup_mask, "video_path"].tolist()

        if missing_files:
            errors.append(f"{len(missing_files)} video files not found")
        if invalid_labels:
            errors.append(f"Invalid labels: {invalid_labels}")
        if duplicate_videos:
            errors.append(f"{len(duplicate_videos)} duplicate video references")

        class_dist = df["label"].value_counts().to_dict()
        split_counts: dict[str, int] = {}
        if "split" in df.columns:
            split_counts = df["split"].value_counts().to_dict()

        return DatasetReport(
            total_videos=len(df),
            total_participants=df["participant_id"].nunique(),
            class_distribution={str(k): int(v) for k, v in class_dist.items()},
            split_counts={str(k): int(v) for k, v in split_counts.items()},
            missing_files=missing_files,
            invalid_labels=invalid_labels,
            duplicate_videos=list(set(duplicate_videos)),
            is_valid=len(errors) == 0,
            errors=errors,
        )

    def participant_split(
        self,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
    ) -> pd.DataFrame:
        """Assign participant-level train/val/test splits (no subject leakage)."""
        if self._df is None:
            raise DatasetError("No manifest loaded.")
        rng = random.Random(self.random_seed)
        participants = list(self._df["participant_id"].unique())
        rng.shuffle(participants)
        n = len(participants)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        train_pts = set(participants[:n_train])
        val_pts = set(participants[n_train: n_train + n_val])

        def assign(pid):
            if pid in train_pts:
                return "train"
            if pid in val_pts:
                return "val"
            return "test"

        df = self._df.copy()
        df["split"] = df["participant_id"].apply(assign)
        df["split_seed"] = self.random_seed
        self._df = df
        logger.info(
            "Participant split: train=%d val=%d test=%d (seed=%d)",
            (df["split"] == "train").sum(),
            (df["split"] == "val").sum(),
            (df["split"] == "test").sum(),
            self.random_seed,
        )
        return df

    @property
    def dataframe(self) -> Optional[pd.DataFrame]:
        return self._df

    def get_split(self, split: str) -> pd.DataFrame:
        if self._df is None or "split" not in self._df.columns:
            raise DatasetError("Run participant_split() first.")
        return self._df[self._df["split"] == split].copy()
