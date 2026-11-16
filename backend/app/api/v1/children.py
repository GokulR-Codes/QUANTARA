"""Child profile management endpoints — guardian-scoped."""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.core.logging_config import get_logger
from backend.app.core.security import get_current_verified_guardian
from backend.app.db import get_db
from backend.app.models_auth import (
    AGE_GROUP_A, AGE_GROUP_B, AGE_GROUP_C,
    AGE_GROUP_LABELS, AGE_GROUP_RANGES, VALID_AGE_GROUPS,
    AssessmentSession, ChildProfile, ConsentRecord, Guardian,
)
from backend.app.services.auth_service import _audit

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/children", tags=["children"])


# ---------------------------------------------------------------------------
# Age group validation
# ---------------------------------------------------------------------------

def _calculate_age(dob_str: str) -> Optional[float]:
    """Calculate age in years from ISO 8601 date string. Returns None on parse error."""
    try:
        dob = date.fromisoformat(dob_str)
        today = date.today()
        return (today - dob).days / 365.25
    except (ValueError, TypeError):
        return None


def _validate_age_group(age_group: str, dob_str: Optional[str]) -> tuple[str, str]:
    """Validate age_group selection. Returns (verified_status, warning_message)."""
    if age_group not in VALID_AGE_GROUPS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid age_group '{age_group}'. Must be one of: {', '.join(VALID_AGE_GROUPS)}",
        )

    if not dob_str:
        return "guardian_selected", ""

    age = _calculate_age(dob_str)
    if age is None:
        return "guardian_selected", "Could not parse date of birth; using guardian-selected category."

    low, high = AGE_GROUP_RANGES[age_group]
    # For GROUP_C the upper bound is inclusive
    in_range = (low <= age < high) if age_group != AGE_GROUP_C else (low <= age <= high)

    if age < 5.0:
        raise HTTPException(
            status_code=422,
            detail=f"Child age ({age:.1f} years) is below the supported minimum of 5 years.",
        )
    if age > 12.0:
        raise HTTPException(
            status_code=422,
            detail=f"Child age ({age:.1f} years) is above the supported maximum of 12 years.",
        )

    if not in_range:
        # Determine correct group
        if 5.0 <= age < 7.0:
            correct = AGE_GROUP_A
        elif 7.0 <= age < 10.0:
            correct = AGE_GROUP_B
        else:
            correct = AGE_GROUP_C
        raise HTTPException(
            status_code=422,
            detail=(
                f"The selected age group '{AGE_GROUP_LABELS[age_group]}' does not match the "
                f"calculated age ({age:.1f} years). Based on the date of birth, the correct "
                f"group is '{AGE_GROUP_LABELS[correct]}'. Please correct the selection."
            ),
        )

    return "dob_matches", ""


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ChildCreateRequest(BaseModel):
    display_name: str = Field(..., min_length=1, max_length=100)
    age_group: str = Field(..., description="One of: 5_7, 7_10, 10_12")
    date_of_birth: Optional[str] = Field(None, description="ISO 8601 date (YYYY-MM-DD), optional")
    gender: Optional[str] = None
    notes: Optional[str] = None
    guardian_consent_given: bool = Field(False)
    # Consent choices
    consent_video_assessment: bool = False
    consent_retain_video: bool = False
    consent_research_training: bool = False
    consent_biometric_research: bool = False


class ChildUpdateRequest(BaseModel):
    display_name: Optional[str] = Field(None, min_length=1, max_length=100)
    age_group: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    notes: Optional[str] = None
    is_active: Optional[bool] = None


class ChildResponse(BaseModel):
    id: str
    guardian_id: str
    display_name: str
    age_group: str
    age_group_label: str
    date_of_birth: Optional[str] = None
    age_verified_status: str
    gender: Optional[str] = None
    notes: Optional[str] = None
    is_active: bool
    guardian_consent_given: bool
    consent_given_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    assessment_count: int = 0

    model_config = {"from_attributes": True}


def _child_to_response(child: ChildProfile, db: Session) -> ChildResponse:
    count = db.query(AssessmentSession).filter(AssessmentSession.child_id == child.id).count()
    return ChildResponse(
        id=child.id,
        guardian_id=child.guardian_id,
        display_name=child.display_name,
        age_group=child.age_group,
        age_group_label=AGE_GROUP_LABELS.get(child.age_group, child.age_group),
        date_of_birth=child.date_of_birth,
        age_verified_status=child.age_verified_status,
        gender=child.gender,
        notes=child.notes,
        is_active=child.is_active,
        guardian_consent_given=child.guardian_consent_given,
        consent_given_at=child.consent_given_at,
        created_at=child.created_at,
        updated_at=child.updated_at,
        assessment_count=count,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("", status_code=status.HTTP_201_CREATED, response_model=ChildResponse)
async def create_child(
    req: ChildCreateRequest,
    guardian: Guardian = Depends(get_current_verified_guardian),
    db: Session = Depends(get_db),
):
    """Create a new child profile under the authenticated guardian's account."""
    verified_status, _warning = _validate_age_group(req.age_group, req.date_of_birth)

    child = ChildProfile(
        guardian_id=guardian.id,
        display_name=req.display_name,
        age_group=req.age_group,
        date_of_birth=req.date_of_birth,
        age_verified_status=verified_status,
        gender=req.gender,
        notes=req.notes,
        guardian_consent_given=req.guardian_consent_given,
        consent_given_at=datetime.now(timezone.utc) if req.guardian_consent_given else None,
    )
    db.add(child)
    db.flush()  # get child.id

    # Create consent record
    consent = ConsentRecord(
        child_id=child.id,
        guardian_id=guardian.id,
        consent_video_assessment=req.consent_video_assessment,
        consent_retain_video=req.consent_retain_video,
        consent_research_training=req.consent_research_training,
        consent_biometric_research=req.consent_biometric_research,
    )
    db.add(consent)
    db.commit()
    db.refresh(child)

    _audit(db, guardian.id, "child_created", f"Child profile created: {child.id} name={req.display_name}")
    logger.info("Child profile created: id=%s guardian=%s name=%s", child.id, guardian.id, req.display_name)
    return _child_to_response(child, db)


@router.get("", response_model=list[ChildResponse])
async def list_children(
    guardian: Guardian = Depends(get_current_verified_guardian),
    db: Session = Depends(get_db),
    include_inactive: bool = False,
):
    """List all child profiles under the authenticated guardian."""
    query = db.query(ChildProfile).filter(ChildProfile.guardian_id == guardian.id)
    if not include_inactive:
        query = query.filter(ChildProfile.is_active == True)  # noqa: E712
    children = query.order_by(ChildProfile.created_at.desc()).all()
    return [_child_to_response(c, db) for c in children]


@router.get("/{child_id}", response_model=ChildResponse)
async def get_child(
    child_id: str,
    guardian: Guardian = Depends(get_current_verified_guardian),
    db: Session = Depends(get_db),
):
    """Get a specific child profile (must belong to the authenticated guardian)."""
    child = _require_own_child(db, guardian, child_id)
    return _child_to_response(child, db)


@router.put("/{child_id}", response_model=ChildResponse)
async def update_child(
    child_id: str,
    req: ChildUpdateRequest,
    guardian: Guardian = Depends(get_current_verified_guardian),
    db: Session = Depends(get_db),
):
    """Update a child profile."""
    child = _require_own_child(db, guardian, child_id)

    if req.display_name is not None:
        child.display_name = req.display_name
    if req.age_group is not None:
        dob = req.date_of_birth or child.date_of_birth
        verified_status, _ = _validate_age_group(req.age_group, dob)
        child.age_group = req.age_group
        child.age_verified_status = verified_status
    if req.date_of_birth is not None:
        child.date_of_birth = req.date_of_birth
    if req.gender is not None:
        child.gender = req.gender
    if req.notes is not None:
        child.notes = req.notes
    if req.is_active is not None:
        child.is_active = req.is_active
    child.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(child)
    _audit(db, guardian.id, "child_updated", f"Child {child_id} updated")
    return _child_to_response(child, db)


@router.delete("/{child_id}", status_code=status.HTTP_200_OK)
async def delete_child(
    child_id: str,
    permanent: bool = False,
    guardian: Guardian = Depends(get_current_verified_guardian),
    db: Session = Depends(get_db),
):
    """Delete or deactivate a child profile.

    By default, deactivates (soft delete). Pass `permanent=true` for full removal
    including all assessment records.
    """
    child = _require_own_child(db, guardian, child_id)
    if permanent:
        db.delete(child)
        db.commit()
        _audit(db, guardian.id, "child_permanently_deleted", f"Child {child_id} permanently deleted")
        return {"message": "Child profile and all associated records permanently deleted.", "child_id": child_id}
    else:
        child.is_active = False
        child.updated_at = datetime.now(timezone.utc)
        db.commit()
        _audit(db, guardian.id, "child_deactivated", f"Child {child_id} deactivated")
        return {"message": "Child profile deactivated.", "child_id": child_id}


@router.get("/{child_id}/assessments")
async def list_child_assessments(
    child_id: str,
    guardian: Guardian = Depends(get_current_verified_guardian),
    db: Session = Depends(get_db),
):
    """List all assessment sessions for a specific child."""
    child = _require_own_child(db, guardian, child_id)
    assessments = (
        db.query(AssessmentSession)
        .filter(AssessmentSession.child_id == child.id)
        .order_by(AssessmentSession.created_at.desc())
        .all()
    )
    return [
        {
            "id": a.id,
            "child_id": a.child_id,
            "job_id": a.job_id,
            "age_group": a.age_group,
            "age_group_label": AGE_GROUP_LABELS.get(a.age_group, a.age_group),
            "age_at_assessment": a.age_at_assessment,
            "status": a.status,
            "model_version": a.model_version,
            "result_summary": a.result_summary,
            "created_at": a.created_at.isoformat() if a.created_at else None,
            "completed_at": a.completed_at.isoformat() if a.completed_at else None,
        }
        for a in assessments
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_own_child(db: Session, guardian: Guardian, child_id: str) -> ChildProfile:
    """Get a child profile that belongs to the given guardian. Raises 404/403 on failure."""
    child = db.query(ChildProfile).filter(ChildProfile.id == child_id).first()
    if child is None:
        raise HTTPException(status_code=404, detail=f"Child profile {child_id} not found.")
    if child.guardian_id != guardian.id:
        # Return 404 to avoid revealing existence to unauthorized parties
        raise HTTPException(status_code=404, detail=f"Child profile {child_id} not found.")
    return child
