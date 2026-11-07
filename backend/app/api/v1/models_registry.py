"""Model registry and federated status endpoints."""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.db import get_db
from backend.app.models import ModelRegistry
from backend.app.schemas.schemas import ModelInfo
from backend.app.services.model_service import describe_record
from backend.app.core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["models"])


def _to_model_info(m: ModelRegistry) -> ModelInfo:
    d = describe_record(m)
    return ModelInfo(
        id=m.id,
        model_type=m.model_type,
        version=m.version,
        is_active=m.is_active,
        metrics=m.metrics,
        created_at=m.created_at,
        feature_schema_version=m.feature_schema_version,
        selected_feature_names=m.selected_feature_names,
        usable_for_video_inference=d["usable"],
        status_detail=d["detail"],
        synthetic=d["synthetic"],
    )


@router.get("/models", response_model=List[ModelInfo])
async def list_models(db: Session = Depends(get_db)) -> List[ModelInfo]:
    """List all registered model checkpoints."""
    models = db.query(ModelRegistry).order_by(ModelRegistry.created_at.desc()).all()
    if not models:
        return []
    return [_to_model_info(m) for m in models]


federated_router = APIRouter(prefix="/api/v1/federated", tags=["federated"])


@federated_router.get("/status")
async def federated_status(db: Session = Depends(get_db)) -> dict:
    """Report federated learning experiment status."""
    from backend.app.models import Experiment
    fed_exps = (
        db.query(Experiment)
        .filter(Experiment.experiment_type == "federated")
        .order_by(Experiment.created_at.desc())
        .limit(5)
        .all()
    )
    return {
        "federated_experiments": [
            {
                "id": e.id,
                "name": e.name,
                "status": e.status,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in fed_exps
        ],
        "note": (
            "Federated learning is simulated locally (single machine, multiple data partitions). "
            "Raw video data is never transmitted; only model parameter updates are shared."
        ),
    }
