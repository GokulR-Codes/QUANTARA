"""Experiment management endpoints."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.db import get_db
from backend.app.models import Experiment
from backend.app.schemas.schemas import (
    ExperimentCreateRequest, ExperimentResponse,
    ExperimentStatus, ExperimentType
)
from backend.app.services.experiment_service import run_experiment_task
from backend.app.core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/experiments", tags=["experiments"])


def _to_response(exp: Experiment) -> ExperimentResponse:
    return ExperimentResponse(
        experiment_id=exp.id,
        name=exp.name,
        experiment_type=ExperimentType(exp.experiment_type),
        status=ExperimentStatus(exp.status),
        config=exp.config,
        created_at=exp.created_at,
        started_at=exp.started_at,
        completed_at=exp.completed_at,
        results=exp.results,
        notes=exp.notes,
    )


@router.post("", response_model=ExperimentResponse, status_code=202)
async def create_experiment(
    req: ExperimentCreateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> ExperimentResponse:
    """Create and queue an experiment."""
    exp = Experiment(
        id=str(uuid.uuid4()),
        name=req.name,
        experiment_type=req.experiment_type.value,
        status=ExperimentStatus.queued.value,
        config=req.config or {},
        created_at=datetime.now(timezone.utc),
    )
    db.add(exp)
    db.commit()
    db.refresh(exp)

    background_tasks.add_task(run_experiment_task, exp.id, req.experiment_type.value, req.config or {})

    return _to_response(exp)


@router.get("", response_model=List[ExperimentResponse])
async def list_experiments(db: Session = Depends(get_db)) -> List[ExperimentResponse]:
    """List all experiments."""
    exps = db.query(Experiment).order_by(Experiment.created_at.desc()).all()
    return [_to_response(e) for e in exps]


@router.get("/{experiment_id}/status", response_model=ExperimentResponse)
async def get_experiment_status(
    experiment_id: str,
    db: Session = Depends(get_db),
) -> ExperimentResponse:
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    return _to_response(exp)


@router.get("/{experiment_id}/results", response_model=ExperimentResponse)
async def get_experiment_results(
    experiment_id: str,
    db: Session = Depends(get_db),
) -> ExperimentResponse:
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    if exp.status != ExperimentStatus.completed.value:
        raise HTTPException(
            status_code=202,
            detail=f"Experiment not yet complete. Status: {exp.status}"
        )
    return _to_response(exp)
