"""Assessment session endpoints — link guardian + child + processing job."""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.core.logging_config import get_logger
from backend.app.core.security import get_current_verified_guardian
from backend.app.db import get_db
from backend.app.models_auth import (
    AGE_GROUP_LABELS, VALID_AGE_GROUPS,
    AssessmentSession, ChildProfile, Guardian,
)
from backend.app.services.auth_service import _audit
from backend.app.services.job_service import get_job

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/assessments", tags=["assessments"])


class CreateAssessmentRequest(BaseModel):
    child_id: str
    job_id: Optional[str] = None      # may be linked after upload
    notes: Optional[str] = None


class AssessmentResponse(BaseModel):
    model_config = {"from_attributes": True, "protected_namespaces": ()}
    id: str
    guardian_id: str
    child_id: str
    child_name: Optional[str] = None
    job_id: Optional[str] = None
    age_group: str
    age_group_label: str
    age_at_assessment: Optional[float] = None
    status: str
    model_version: Optional[str] = None
    result_summary: Optional[dict] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    notes: Optional[str] = None


def _to_response(session: AssessmentSession, child: Optional[ChildProfile] = None) -> AssessmentResponse:
    return AssessmentResponse(
        id=session.id,
        guardian_id=session.guardian_id,
        child_id=session.child_id,
        child_name=child.display_name if child else None,
        job_id=session.job_id,
        age_group=session.age_group,
        age_group_label=AGE_GROUP_LABELS.get(session.age_group, session.age_group),
        age_at_assessment=session.age_at_assessment,
        status=session.status,
        model_version=session.model_version,
        result_summary=session.result_summary,
        created_at=session.created_at,
        completed_at=session.completed_at,
        notes=session.notes,
    )


def _calculate_age(dob_str: Optional[str]) -> Optional[float]:
    if not dob_str:
        return None
    try:
        dob = date.fromisoformat(dob_str)
        return (date.today() - dob).days / 365.25
    except (ValueError, TypeError):
        return None


@router.post("", status_code=status.HTTP_201_CREATED, response_model=AssessmentResponse)
async def create_assessment(
    req: CreateAssessmentRequest,
    guardian: Guardian = Depends(get_current_verified_guardian),
    db: Session = Depends(get_db),
):
    """Create an assessment session for a registered child."""
    # Verify child belongs to guardian
    child = db.query(ChildProfile).filter(
        ChildProfile.id == req.child_id,
        ChildProfile.guardian_id == guardian.id,
    ).first()
    if not child:
        raise HTTPException(status_code=404, detail="Child profile not found.")
    if not child.is_active:
        raise HTTPException(status_code=400, detail="Cannot create assessment for a deactivated profile.")
    if not child.guardian_consent_given:
        raise HTTPException(
            status_code=400,
            detail="Guardian consent has not been recorded for this child profile. "
                   "Update the profile to add consent before starting an assessment.",
        )

    # Optionally validate job_id belongs to this guardian's scope
    if req.job_id:
        job = get_job(db, req.job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {req.job_id} not found.")

    session = AssessmentSession(
        guardian_id=guardian.id,
        child_id=child.id,
        job_id=req.job_id,
        age_group=child.age_group,
        age_at_assessment=_calculate_age(child.date_of_birth),
        status="created",
        notes=req.notes,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    _audit(db, guardian.id, "assessment_started",
           f"Assessment {session.id} for child {child.id} age_group={child.age_group}")
    logger.info("Assessment session created: %s child=%s", session.id, child.id)
    return _to_response(session, child)


@router.get("", response_model=list[AssessmentResponse])
async def list_assessments(
    guardian: Guardian = Depends(get_current_verified_guardian),
    db: Session = Depends(get_db),
    child_id: Optional[str] = None,
    age_group: Optional[str] = None,
):
    """List all assessment sessions for the authenticated guardian."""
    query = db.query(AssessmentSession).filter(AssessmentSession.guardian_id == guardian.id)
    if child_id:
        query = query.filter(AssessmentSession.child_id == child_id)
    if age_group:
        if age_group not in VALID_AGE_GROUPS:
            raise HTTPException(status_code=422, detail=f"Invalid age_group: {age_group}")
        query = query.filter(AssessmentSession.age_group == age_group)
    sessions = query.order_by(AssessmentSession.created_at.desc()).all()

    result = []
    for s in sessions:
        child = db.query(ChildProfile).filter(ChildProfile.id == s.child_id).first()
        result.append(_to_response(s, child))
    return result


@router.get("/{assessment_id}", response_model=AssessmentResponse)
async def get_assessment(
    assessment_id: str,
    guardian: Guardian = Depends(get_current_verified_guardian),
    db: Session = Depends(get_db),
):
    session, child = _require_own_assessment(db, guardian, assessment_id)
    return _to_response(session, child)


@router.patch("/{assessment_id}/link-job")
async def link_job_to_assessment(
    assessment_id: str,
    job_id: str,
    guardian: Guardian = Depends(get_current_verified_guardian),
    db: Session = Depends(get_db),
):
    """Link a video processing job to an existing assessment session."""
    session, child = _require_own_assessment(db, guardian, assessment_id)
    job = get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    session.job_id = job_id
    db.commit()
    return {"assessment_id": assessment_id, "job_id": job_id, "message": "Job linked to assessment."}


@router.delete("/{assessment_id}")
async def delete_assessment(
    assessment_id: str,
    guardian: Guardian = Depends(get_current_verified_guardian),
    db: Session = Depends(get_db),
):
    session, _ = _require_own_assessment(db, guardian, assessment_id)
    db.delete(session)
    db.commit()
    _audit(db, guardian.id, "assessment_deleted", f"Assessment {assessment_id} deleted")
    return {"message": "Assessment record deleted.", "assessment_id": assessment_id}


def _require_own_assessment(
    db: Session, guardian: Guardian, assessment_id: str
) -> tuple[AssessmentSession, Optional[ChildProfile]]:
    session = db.query(AssessmentSession).filter(AssessmentSession.id == assessment_id).first()
    if not session or session.guardian_id != guardian.id:
        raise HTTPException(status_code=404, detail="Assessment not found.")
    child = db.query(ChildProfile).filter(ChildProfile.id == session.child_id).first()
    return session, child
