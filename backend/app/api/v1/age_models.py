"""Age-group-specific model configuration and registry endpoints."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.core.logging_config import get_logger
from backend.app.core.security import get_current_guardian, get_current_verified_guardian
from backend.app.db import get_db
from backend.app.models_auth import AGE_GROUP_LABELS, VALID_AGE_GROUPS, AgeModelConfig, Guardian
from src.models.age_profiles import PROFILES, get_profile

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/models/age-groups", tags=["age-models"])


class AgeGroupStatusResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    age_group: str
    label: str
    has_active_model: bool
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    is_approved: bool = False
    validation_metrics: Optional[dict] = None
    processing_config_summary: dict = {}
    research_notes: str = ""


@router.get("", response_model=list[AgeGroupStatusResponse])
async def list_age_group_statuses(
    db: Session = Depends(get_db),
    guardian: Guardian = Depends(get_current_guardian),
):
    """Return the model status for each of the three age groups."""
    result = []
    for age_group in VALID_AGE_GROUPS:
        profile = get_profile(age_group)
        config = (
            db.query(AgeModelConfig)
            .filter(AgeModelConfig.age_group == age_group, AgeModelConfig.is_active == True)  # noqa: E712
            .order_by(AgeModelConfig.created_at.desc())
            .first()
        )
        result.append(AgeGroupStatusResponse(
            age_group=age_group,
            label=AGE_GROUP_LABELS[age_group],
            has_active_model=config is not None and config.is_approved,
            model_id=config.id if config else None,
            model_version=config.version if config else None,
            is_approved=config.is_approved if config else False,
            validation_metrics=config.validation_metrics if config else None,
            processing_config_summary={
                "qaoa_n_qubits": profile.qaoa_n_qubits,
                "qaoa_depth": profile.qaoa_depth,
                "qaoa_target_k": profile.qaoa_target_k,
                "vqc_n_qubits": profile.vqc_n_qubits,
                "vqc_depth": profile.vqc_depth,
                "sequence_length": profile.sequence_length,
                "min_tracked_frames": profile.min_tracked_frames,
            },
            research_notes=profile.research_notes,
        ))
    return result


@router.get("/{age_group}", response_model=AgeGroupStatusResponse)
async def get_age_group_status(
    age_group: str,
    db: Session = Depends(get_db),
    guardian: Guardian = Depends(get_current_guardian),
):
    """Return the model status for a specific age group."""
    if age_group not in VALID_AGE_GROUPS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid age_group. Must be one of: {', '.join(VALID_AGE_GROUPS)}",
        )
    profile = get_profile(age_group)
    config = (
        db.query(AgeModelConfig)
        .filter(AgeModelConfig.age_group == age_group, AgeModelConfig.is_active == True)  # noqa: E712
        .order_by(AgeModelConfig.created_at.desc())
        .first()
    )
    return AgeGroupStatusResponse(
        age_group=age_group,
        label=AGE_GROUP_LABELS[age_group],
        has_active_model=config is not None and config.is_approved,
        model_id=config.id if config else None,
        model_version=config.version if config else None,
        is_approved=config.is_approved if config else False,
        validation_metrics=config.validation_metrics if config else None,
        processing_config_summary={
            "qaoa_n_qubits": profile.qaoa_n_qubits,
            "qaoa_depth": profile.qaoa_depth,
            "qaoa_target_k": profile.qaoa_target_k,
            "vqc_n_qubits": profile.vqc_n_qubits,
            "vqc_depth": profile.vqc_depth,
            "sequence_length": profile.sequence_length,
            "min_tracked_frames": profile.min_tracked_frames,
        },
        research_notes=profile.research_notes,
    )
