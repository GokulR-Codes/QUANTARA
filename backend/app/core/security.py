"""FastAPI security dependencies — JWT token extraction and guardian resolution."""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy.orm import Session

from backend.app.db import get_db
from backend.app.models_auth import Guardian
from backend.app.services.auth_service import decode_access_token, get_guardian_by_id

_bearer = HTTPBearer(auto_error=False)


def _extract_guardian(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: Session = Depends(get_db),
) -> Optional[Guardian]:
    """Return guardian from JWT token, or None if no/invalid token."""
    if credentials is None:
        return None
    try:
        payload = decode_access_token(credentials.credentials)
        guardian_id: str = payload.get("sub", "")
    except JWTError:
        return None
    return get_guardian_by_id(db, guardian_id)


def get_current_guardian(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: Session = Depends(get_db),
) -> Guardian:
    """Require a valid JWT. Raises 401 if missing/invalid/expired."""
    guardian = _extract_guardian(credentials, db)
    if guardian is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not guardian.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is deactivated.")
    return guardian


def get_current_verified_guardian(guardian: Guardian = Depends(get_current_guardian)) -> Guardian:
    """Require a verified guardian (email OTP confirmed)."""
    if not guardian.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email verification required. Check your email for the verification code.",
        )
    return guardian


def get_current_admin(guardian: Guardian = Depends(get_current_guardian)) -> Guardian:
    """Require admin role."""
    if guardian.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required.")
    return guardian
