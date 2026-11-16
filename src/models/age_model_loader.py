"""Age-aware model loading — select the correct pipeline bundle for a given age group.

This module is called by the inference pipeline to load the age-appropriate model
configuration when a job is associated with a child profile.

If no age-specific model is registered:
  - Returns None (caller must report model_not_ready with the specific age group)
  - NEVER silently falls back to a different age group's model
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from backend.app.core.logging_config import get_logger
from backend.app.models_auth import AgeModelConfig, VALID_AGE_GROUPS
from src.models.age_profiles import AgeGroupProfile, get_profile

logger = get_logger(__name__)


def get_active_age_model(db: Session, age_group: str) -> Optional[AgeModelConfig]:
    """Return the active, approved AgeModelConfig for the given age group.

    Returns None if no approved model is available.
    NEVER returns a model for a different age group.
    """
    if age_group not in VALID_AGE_GROUPS:
        logger.error("get_active_age_model: unknown age_group=%s", age_group)
        return None

    config = (
        db.query(AgeModelConfig)
        .filter(
            AgeModelConfig.age_group == age_group,
            AgeModelConfig.is_active == True,   # noqa: E712
            AgeModelConfig.is_approved == True,  # noqa: E712
        )
        .order_by(AgeModelConfig.created_at.desc())
        .first()
    )

    if config is None:
        logger.info(
            "No active approved model for age_group=%s — job will complete without prediction",
            age_group,
        )
    else:
        logger.info(
            "Selected age model: age_group=%s model_id=%s version=%s",
            age_group, config.id, config.version,
        )
    return config


def get_age_group_for_job(db: Session, job_id: str) -> Optional[str]:
    """Look up the age group associated with a job via assessment_sessions.

    Returns the age group string, or None if the job is not linked to an assessment.
    """
    from backend.app.models_auth import AssessmentSession
    session = (
        db.query(AssessmentSession)
        .filter(AssessmentSession.job_id == job_id)
        .first()
    )
    if session:
        return session.age_group
    return None


def describe_age_model_status(db: Session, age_group: Optional[str]) -> dict:
    """Return a structured status dict for the age group model availability.

    Used by the processing worker and /ready endpoint to report model status.
    """
    if age_group is None:
        return {
            "age_group": None,
            "age_model_available": False,
            "detail": "No age group associated with this job (not linked to a child profile). "
                      "Using default model selection.",
        }

    if age_group not in VALID_AGE_GROUPS:
        return {
            "age_group": age_group,
            "age_model_available": False,
            "detail": f"Unknown age group '{age_group}'.",
        }

    profile = get_profile(age_group)
    config = get_active_age_model(db, age_group)

    if config is None:
        return {
            "age_group": age_group,
            "age_group_label": profile.label,
            "age_model_available": False,
            "detail": (
                f"No trained and approved model available for age group '{profile.label}'. "
                f"Upload and register a labelled dataset, then train the age-specific model "
                f"before attempting inference for this age group."
            ),
        }

    return {
        "age_group": age_group,
        "age_group_label": profile.label,
        "age_model_available": True,
        "model_id": config.id,
        "model_version": config.version,
        "model_registry_id": config.model_registry_id,
        "detail": f"Age-specific model available: v{config.version}",
        "processing_config": {
            "qaoa_n_qubits": profile.qaoa_n_qubits,
            "qaoa_target_k": profile.qaoa_target_k,
            "vqc_n_qubits": profile.vqc_n_qubits,
            "sequence_length": profile.sequence_length,
        },
    }
