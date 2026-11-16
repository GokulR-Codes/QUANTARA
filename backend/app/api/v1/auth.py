"""Guardian authentication API endpoints."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from backend.app.core.logging_config import get_logger
from backend.app.core.security import get_current_guardian, get_current_verified_guardian
from backend.app.db import get_db
from backend.app.models_auth import Guardian
from backend.app.services.auth_service import (
    AuthError,
    initiate_password_reset,
    login_guardian,
    register_guardian,
    reset_password_with_otp,
    send_otp,
    verify_otp,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


# ---------------------------------------------------------------------------
# Pydantic schemas (auth-specific)
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    email: str = Field(..., description="Guardian email address")
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    confirm_password: str
    full_name: str = Field(..., min_length=2, max_length=100)
    mobile: Optional[str] = None
    relationship_to_child: Optional[str] = None
    country: Optional[str] = None


class LoginRequest(BaseModel):
    email_or_username: str
    password: str


class OTPRequest(BaseModel):
    otp_code: str = Field(..., min_length=6, max_length=6)
    purpose: str = Field(default="email_verification")


class ResendOTPRequest(BaseModel):
    purpose: str = Field(default="email_verification")


class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    email: str
    otp_code: str = Field(..., min_length=6, max_length=6)
    new_password: str = Field(..., min_length=8)
    confirm_password: str


class GuardianPublicResponse(BaseModel):
    id: str
    email: str
    username: str
    full_name: str
    mobile: Optional[str] = None
    relationship_to_child: Optional[str] = None
    country: Optional[str] = None
    is_verified: bool
    role: str

    model_config = {"from_attributes": True}


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    guardian: GuardianPublicResponse


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ip(request: Request) -> Optional[str]:
    xff = request.headers.get("X-Forwarded-For")
    return xff.split(",")[0].strip() if xff else request.client.host if request.client else None


def _handle_auth_error(exc: AuthError) -> None:
    raise HTTPException(status_code=exc.status_code, detail=exc.message)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest, request: Request, db: Session = Depends(get_db)):
    """Register a new guardian account. A verification OTP is sent (or printed in dev mode)."""
    try:
        guardian = register_guardian(
            db,
            email=req.email,
            username=req.username,
            password=req.password,
            confirm_password=req.confirm_password,
            full_name=req.full_name,
            mobile=req.mobile,
            relationship_to_child=req.relationship_to_child,
            country=req.country,
            ip_address=_ip(request),
        )
    except AuthError as exc:
        _handle_auth_error(exc)

    return {
        "message": "Registration successful. Check your email (or terminal in dev mode) for a 6-digit verification code.",
        "guardian_id": guardian.id,
        "email": guardian.email,
        "verification_required": True,
    }


@router.post("/login", response_model=AuthResponse)
async def login(req: LoginRequest, request: Request, db: Session = Depends(get_db)):
    """Login with email/username and password. Returns JWT access token."""
    try:
        guardian, token = login_guardian(
            db,
            email_or_username=req.email_or_username,
            password=req.password,
            ip_address=_ip(request),
        )
    except AuthError as exc:
        _handle_auth_error(exc)

    return AuthResponse(
        access_token=token,
        guardian=GuardianPublicResponse.model_validate(guardian),
    )


@router.post("/logout")
async def logout(guardian: Guardian = Depends(get_current_guardian)):
    """Logout (client should discard the token; no server-side session to invalidate for JWT)."""
    # For JWT-based auth, the client discards the token.
    # Stateful session revocation can be added later via a denylist table.
    return {"message": "Logged out successfully.", "guardian_id": guardian.id}


@router.post("/verify-otp")
async def verify_otp_endpoint(
    req: OTPRequest,
    guardian: Guardian = Depends(get_current_guardian),
    db: Session = Depends(get_db),
):
    """Verify the OTP sent to the guardian's email."""
    try:
        verify_otp(db, guardian, req.otp_code, req.purpose)
    except AuthError as exc:
        _handle_auth_error(exc)
    return {"message": "Verification successful.", "is_verified": guardian.is_verified}


@router.post("/resend-otp")
async def resend_otp_endpoint(
    req: ResendOTPRequest,
    guardian: Guardian = Depends(get_current_guardian),
    db: Session = Depends(get_db),
):
    """Re-send OTP (rate limited to once per 60 seconds)."""
    try:
        send_otp(db, guardian, req.purpose)
    except AuthError as exc:
        _handle_auth_error(exc)
    return {"message": "A new verification code has been sent (or printed in dev mode)."}


@router.post("/forgot-password")
async def forgot_password(req: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """Request a password reset OTP. Always returns 200 to prevent email enumeration."""
    initiate_password_reset(db, req.email)
    return {
        "message": (
            "If an account with that email exists, a reset code has been sent "
            "(or printed in dev mode)."
        )
    }


@router.post("/reset-password")
async def reset_password(req: ResetPasswordRequest, db: Session = Depends(get_db)):
    """Reset password using OTP."""
    try:
        reset_password_with_otp(
            db, req.email, req.otp_code, req.new_password, req.confirm_password
        )
    except AuthError as exc:
        _handle_auth_error(exc)
    return {"message": "Password reset successfully. You may now log in."}


@router.get("/me", response_model=GuardianPublicResponse)
async def get_me(guardian: Guardian = Depends(get_current_guardian)):
    """Return the currently authenticated guardian's profile."""
    return GuardianPublicResponse.model_validate(guardian)
