"""Dataset validation and status endpoints."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.db import get_db
from backend.app.schemas.schemas import DatasetValidationResult, DatasetStatusResponse
from backend.app.core.logging_config import get_logger
from src.config.settings import get_settings

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])


@router.get("/status", response_model=DatasetStatusResponse)
async def dataset_status(db: Session = Depends(get_db)) -> DatasetStatusResponse:
    """Report whether a labelled dataset is available."""
    settings = get_settings()
    manifest_path = Path("data/manifests/dataset_manifest.csv")

    if not manifest_path.exists():
        return DatasetStatusResponse(
            dataset_available=False,
            total_samples=0,
            message=(
                "No dataset manifest found at data/manifests/dataset_manifest.csv. "
                "See docs/dataset_preparation.md for import instructions."
            ),
        )

    try:
        import pandas as pd
        df = pd.read_csv(str(manifest_path))
        total = len(df)
        dist = None
        if "label" in df.columns:
            dist = df["label"].value_counts().to_dict()
        return DatasetStatusResponse(
            dataset_available=True,
            total_samples=total,
            class_distribution=dist,
            manifest_path=str(manifest_path),
            message=f"Dataset available: {total} records.",
        )
    except Exception as exc:
        return DatasetStatusResponse(
            dataset_available=False,
            message=f"Manifest found but could not be parsed: {exc}",
        )


@router.post("/validate", response_model=DatasetValidationResult)
async def validate_dataset(db: Session = Depends(get_db)) -> DatasetValidationResult:
    """Validate the dataset manifest and referenced video files."""
    manifest_path = Path("data/manifests/dataset_manifest.csv")

    if not manifest_path.exists():
        return DatasetValidationResult(
            valid=False,
            errors=["Manifest not found: data/manifests/dataset_manifest.csv"],
            warnings=["Create the manifest using the template in data/manifests/template.csv"],
        )

    try:
        import pandas as pd
        df = pd.read_csv(str(manifest_path))
    except Exception as exc:
        return DatasetValidationResult(
            valid=False,
            errors=[f"Cannot parse manifest: {exc}"],
        )

    required_cols = {"video_path", "label", "participant_id"}
    missing_cols = required_cols - set(df.columns)
    errors = []
    warnings = []

    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    missing_videos = 0
    if "video_path" in df.columns:
        for vp in df["video_path"].dropna():
            if not Path(vp).exists():
                missing_videos += 1

    if missing_videos:
        warnings.append(f"{missing_videos} video file(s) referenced in manifest but not found on disk")

    dist = {}
    if "label" in df.columns:
        dist = df["label"].value_counts().to_dict()

    valid = len(errors) == 0
    return DatasetValidationResult(
        valid=valid,
        total_records=len(df),
        valid_records=len(df) - missing_videos,
        invalid_records=missing_videos,
        missing_videos=missing_videos,
        class_distribution=dist,
        errors=errors,
        warnings=warnings,
    )
