"""Additional SQLAlchemy ORM models for Q-CARE guardian/child/assessment upgrade.

These tables are ADDITIVE — the existing Job, Experiment, ModelRegistry tables in
models.py are preserved unchanged.  Import both models.py and this module so that
Base.metadata.create_all() creates all tables.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer,
    JSON, String, Text, ForeignKey, Enum as SAEnum,
)
from sqlalchemy.orm import relationship

from backend.app.db import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Age group enum values (stored as strings for SQLite compatibility)
# ---------------------------------------------------------------------------
# GROUP_A: 5.00 ≤ age < 7.00
# GROUP_B: 7.00 ≤ age < 10.00
# GROUP_C: 10.00 ≤ age ≤ 12.00
AGE_GROUP_A = "5_7"
AGE_GROUP_B = "7_10"
AGE_GROUP_C = "10_12"
VALID_AGE_GROUPS = {AGE_GROUP_A, AGE_GROUP_B, AGE_GROUP_C}

AGE_GROUP_LABELS = {
    AGE_GROUP_A: "5–7 Years",
    AGE_GROUP_B: "7–10 Years",
    AGE_GROUP_C: "10–12 Years",
}

AGE_GROUP_RANGES = {
    AGE_GROUP_A: (5.0, 7.0),   # inclusive lower, exclusive upper
    AGE_GROUP_B: (7.0, 10.0),
    AGE_GROUP_C: (10.0, 12.0),  # inclusive upper
}


class Guardian(Base):
    """A registered parent/guardian account."""
    __tablename__ = "guardians"

    id = Column(String, primary_key=True, default=_uuid)
    email = Column(String, nullable=False, unique=True, index=True)
    username = Column(String, nullable=False, unique=True, index=True)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=False)
    mobile = Column(String, nullable=True)
    relationship_to_child = Column(String, nullable=True)
    country = Column(String, nullable=True)

    # Account state
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    role = Column(String, default="guardian", nullable=False)  # guardian | admin

    # Auth
    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)
    last_login_at = Column(DateTime, nullable=True)

    # Login throttling
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime, nullable=True)

    # Password reset token (hashed)
    reset_token_hash = Column(String, nullable=True)
    reset_token_expires = Column(DateTime, nullable=True)

    # Relationships
    children = relationship("ChildProfile", back_populates="guardian", cascade="all, delete-orphan")
    otp_records = relationship("OTPRecord", back_populates="guardian", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="guardian")


class OTPRecord(Base):
    """OTP verification codes (stored hashed)."""
    __tablename__ = "otp_records"

    id = Column(String, primary_key=True, default=_uuid)
    guardian_id = Column(String, ForeignKey("guardians.id", ondelete="CASCADE"), nullable=False, index=True)
    purpose = Column(String, nullable=False)  # email_verification | password_reset | login_2fa
    otp_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=_now)
    expires_at = Column(DateTime, nullable=False)
    used_at = Column(DateTime, nullable=True)
    attempts = Column(Integer, default=0, nullable=False)
    invalidated = Column(Boolean, default=False, nullable=False)

    guardian = relationship("Guardian", back_populates="otp_records")


class ChildProfile(Base):
    """A child profile registered under a guardian account."""
    __tablename__ = "child_profiles"

    id = Column(String, primary_key=True, default=_uuid)
    guardian_id = Column(String, ForeignKey("guardians.id", ondelete="CASCADE"), nullable=False, index=True)

    display_name = Column(String, nullable=False)
    date_of_birth = Column(String, nullable=True)   # ISO 8601 date, optional
    age_group = Column(String, nullable=False)       # AGE_GROUP_A/B/C
    # age_verified_at_assessment: stored per assessment, not on the profile
    age_verified_status = Column(String, default="guardian_selected", nullable=False)
    # guardian_selected | dob_matches | dob_conflicts

    gender = Column(String, nullable=True)           # optional, only if justified
    notes = Column(Text, nullable=True)

    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)

    # Consent
    guardian_consent_given = Column(Boolean, default=False, nullable=False)
    consent_given_at = Column(DateTime, nullable=True)

    # Relationships
    guardian = relationship("Guardian", back_populates="children")
    assessments = relationship("AssessmentSession", back_populates="child", cascade="all, delete-orphan")
    consent_records = relationship("ConsentRecord", back_populates="child", cascade="all, delete-orphan")


class ConsentRecord(Base):
    """Per-child, per-purpose consent record."""
    __tablename__ = "consent_records"

    id = Column(String, primary_key=True, default=_uuid)
    child_id = Column(String, ForeignKey("child_profiles.id", ondelete="CASCADE"), nullable=False, index=True)
    guardian_id = Column(String, ForeignKey("guardians.id", ondelete="CASCADE"), nullable=False)

    # Separate consent choices
    consent_video_assessment = Column(Boolean, default=False, nullable=False)
    consent_retain_video = Column(Boolean, default=False, nullable=False)
    consent_research_training = Column(Boolean, default=False, nullable=False)
    consent_biometric_research = Column(Boolean, default=False, nullable=False)

    version = Column(String, default="1.0", nullable=False)
    created_at = Column(DateTime, default=_now)
    withdrawn_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)

    child = relationship("ChildProfile", back_populates="consent_records")


class AssessmentSession(Base):
    """Links a processing job to a child profile + guardian + age category at assessment time."""
    __tablename__ = "assessment_sessions"

    id = Column(String, primary_key=True, default=_uuid)
    guardian_id = Column(String, ForeignKey("guardians.id", ondelete="CASCADE"), nullable=False, index=True)
    child_id = Column(String, ForeignKey("child_profiles.id", ondelete="CASCADE"), nullable=False, index=True)
    job_id = Column(String, nullable=True, index=True)  # references jobs.id (not FK to allow separate creation)

    # Age category AT ASSESSMENT TIME (never updated after recording)
    age_group = Column(String, nullable=False)
    age_at_assessment = Column(Float, nullable=True)    # calculated from DoB if available

    # Model used
    model_version = Column(String, nullable=True)
    model_id = Column(String, nullable=True)

    # Status & result
    status = Column(String, default="created", nullable=False)
    result_summary = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=_now)
    completed_at = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)

    child = relationship("ChildProfile", back_populates="assessments")


class AgeModelConfig(Base):
    """Age-group-specific model configuration and registry entry."""
    __tablename__ = "age_model_configs"

    id = Column(String, primary_key=True, default=_uuid)
    age_group = Column(String, nullable=False, index=True)   # AGE_GROUP_A/B/C
    model_registry_id = Column(String, nullable=True)        # references model_registry.id
    version = Column(String, nullable=False)
    is_active = Column(Boolean, default=False, nullable=False)
    is_approved = Column(Boolean, default=False, nullable=False)

    # Processing profile config (JSON)
    processing_config = Column(JSON, nullable=True)
    normalization_params = Column(JSON, nullable=True)
    qaoa_config = Column(JSON, nullable=True)
    vqc_config = Column(JSON, nullable=True)
    thresholds = Column(JSON, nullable=True)

    # Training metadata
    dataset_id = Column(String, nullable=True)
    dataset_version = Column(String, nullable=True)
    training_timestamp = Column(DateTime, nullable=True)
    hyperparameters = Column(JSON, nullable=True)
    validation_metrics = Column(JSON, nullable=True)
    test_metrics = Column(JSON, nullable=True)
    quantum_backend = Column(String, nullable=True)

    created_at = Column(DateTime, default=_now)
    notes = Column(Text, nullable=True)


class AuditLog(Base):
    """Security and access event log."""
    __tablename__ = "audit_logs"

    id = Column(String, primary_key=True, default=_uuid)
    guardian_id = Column(String, ForeignKey("guardians.id", ondelete="SET NULL"), nullable=True, index=True)
    event_type = Column(String, nullable=False, index=True)
    # e.g.: login_success, login_failed, logout, register, otp_sent, otp_verified,
    #       child_created, child_deleted, assessment_started, password_reset_requested
    detail = Column(Text, nullable=True)
    ip_address = Column(String, nullable=True)
    created_at = Column(DateTime, default=_now, index=True)

    guardian = relationship("Guardian", back_populates="audit_logs")
