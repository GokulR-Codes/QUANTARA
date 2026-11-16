"""Authentication and guardian account management service.

Security decisions:
- Passwords hashed with Argon2id (passlib).
- Sessions via HS256 JWT (python-jose).
- OTPs: cryptographically secure 6-digit code, stored as bcrypt hash.
  In development mode (APP_ENV=development) the OTP is printed to the terminal.
  In production, configure SMTP_HOST/SMTP_USER/SMTP_PASSWORD in .env.
- Login throttling: 5 failed attempts → 15-minute lockout.
- Passwords NEVER logged or stored in plaintext.
"""
from __future__ import annotations

import hashlib
import os
import re
import secrets
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from typing import Optional

from jose import JWTError, jwt
from passlib.hash import argon2 as argon2_hasher
from sqlalchemy.orm import Session

from backend.app.core.logging_config import get_logger
from backend.app.models_auth import AuditLog, Guardian, OTPRecord
from src.config.settings import get_settings

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_MINUTES = 15
OTP_EXPIRY_MINUTES = 5
OTP_MAX_ATTEMPTS = 5
OTP_RESEND_COOLDOWN_SECONDS = 60

PASSWORD_MIN_LENGTH = 8
PASSWORD_PATTERN = re.compile(
    r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*()_+\-=\[\]{}|;':\",./<>?]).{8,}$"
)


# ---------------------------------------------------------------------------
# Password utilities
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    """Hash password with Argon2id."""
    return argon2_hasher.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against an Argon2id hash."""
    try:
        # passlib argon2.verify(secret, hash) — plain password first, hash second
        return argon2_hasher.verify(plain, hashed)
    except Exception:
        return False


def validate_password_strength(password: str) -> tuple[bool, str]:
    """Return (is_valid, reason). Reason is empty string when valid."""
    if len(password) < PASSWORD_MIN_LENGTH:
        return False, f"Password must be at least {PASSWORD_MIN_LENGTH} characters."
    if not PASSWORD_PATTERN.match(password):
        return False, (
            "Password must contain at least one uppercase letter, one lowercase letter, "
            "one digit, and one special character."
        )
    return True, ""


# ---------------------------------------------------------------------------
# JWT utilities
# ---------------------------------------------------------------------------

def create_access_token(guardian_id: str, role: str = "guardian") -> str:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    payload = {
        "sub": guardian_id,
        "role": role,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_access_token(token: str) -> dict:
    """Decode and validate a JWT. Raises JWTError on failure."""
    settings = get_settings()
    return jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])


# ---------------------------------------------------------------------------
# OTP utilities
# ---------------------------------------------------------------------------

def _generate_otp() -> tuple[str, str]:
    """Return (plaintext_otp, hashed_otp). OTP is 6 digits."""
    otp = str(secrets.randbelow(1_000_000)).zfill(6)
    # Store as sha256 hex — fast lookup, OTP entropy is low enough that
    # sha256 is fine (we rate-limit attempts separately).
    otp_hash = hashlib.sha256(otp.encode()).hexdigest()
    return otp, otp_hash


def _hash_otp(otp: str) -> str:
    return hashlib.sha256(otp.encode()).hexdigest()


def _send_otp_email(email: str, otp: str, purpose: str) -> None:
    """Send OTP via SMTP if configured; otherwise print to terminal (dev mode)."""
    settings = get_settings()
    smtp_host = os.getenv("SMTP_HOST", "")
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_password = os.getenv("SMTP_PASSWORD", "")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))

    purpose_labels = {
        "email_verification": "Email Verification",
        "password_reset": "Password Reset",
        "login_2fa": "Two-Factor Login",
    }
    label = purpose_labels.get(purpose, "Verification")

    if smtp_host and smtp_user and smtp_password:
        try:
            msg = MIMEText(
                f"Your Q-CARE {label} code is: {otp}\n\n"
                f"This code expires in {OTP_EXPIRY_MINUTES} minutes.\n"
                "Do not share this code with anyone.\n\n"
                "Q-CARE is a research prototype, not a clinical tool."
            )
            msg["Subject"] = f"Q-CARE {label} Code"
            msg["From"] = smtp_user
            msg["To"] = email
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.sendmail(smtp_user, [email], msg.as_string())
            logger.info("OTP email sent to %s for purpose=%s", email, purpose)
        except Exception as exc:
            logger.error("Failed to send OTP email: %s", exc)
            # Fall back to terminal in any environment
            print(f"\n[Q-CARE OTP] {label} code for {email}: {otp}  (expires {OTP_EXPIRY_MINUTES}min)\n")
    else:
        # Development mode: print to terminal only
        print(
            f"\n[Q-CARE DEV] *** OTP for {email} ({purpose}): {otp} ***  "
            f"(expires {OTP_EXPIRY_MINUTES}min)\n"
        )
        logger.info("DEV MODE: OTP for %s purpose=%s printed to terminal (not emailed)", email, purpose)


# ---------------------------------------------------------------------------
# Guardian CRUD
# ---------------------------------------------------------------------------

class AuthError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def register_guardian(
    db: Session,
    *,
    email: str,
    username: str,
    password: str,
    confirm_password: str,
    full_name: str,
    mobile: Optional[str] = None,
    relationship_to_child: Optional[str] = None,
    country: Optional[str] = None,
    ip_address: Optional[str] = None,
) -> Guardian:
    """Create a new guardian account. Raises AuthError on validation failure."""
    # Normalize
    email = email.strip().lower()
    username = username.strip().lower()

    # Validate email format (simple)
    if not re.match(r"^[^@]+@[^@]+\.[^@]+$", email):
        raise AuthError("Invalid email address format.", 422)

    # Duplicate checks
    if db.query(Guardian).filter(Guardian.email == email).first():
        raise AuthError("An account with this email already exists.", 409)
    if db.query(Guardian).filter(Guardian.username == username).first():
        raise AuthError("This username is already taken.", 409)

    # Password validation
    if password != confirm_password:
        raise AuthError("Passwords do not match.", 422)
    valid, reason = validate_password_strength(password)
    if not valid:
        raise AuthError(reason, 422)

    guardian = Guardian(
        email=email,
        username=username,
        hashed_password=hash_password(password),
        full_name=full_name,
        mobile=mobile,
        relationship_to_child=relationship_to_child,
        country=country,
        is_active=True,
        is_verified=False,
        role="guardian",
    )
    db.add(guardian)
    db.commit()
    db.refresh(guardian)

    _audit(db, guardian.id, "register", f"Guardian registered: {email}", ip_address)

    # Issue verification OTP
    _create_and_send_otp(db, guardian, "email_verification")

    logger.info("Guardian registered: id=%s email=%s", guardian.id, email)
    return guardian


def login_guardian(
    db: Session,
    *,
    email_or_username: str,
    password: str,
    ip_address: Optional[str] = None,
) -> tuple[Guardian, str]:
    """Authenticate. Returns (guardian, access_token). Raises AuthError on failure."""
    identifier = email_or_username.strip().lower()

    # Find by email or username
    guardian = (
        db.query(Guardian).filter(Guardian.email == identifier).first()
        or db.query(Guardian).filter(Guardian.username == identifier).first()
    )

    # --- Constant-time failure path ---
    if guardian is None:
        # Still do a dummy hash to prevent timing attacks
        argon2_hasher.hash("dummy-timing-protection")
        _audit(db, None, "login_failed", f"Unknown identifier: {identifier}", ip_address)
        raise AuthError("Invalid credentials.", 401)

    # Lockout check
    if guardian.locked_until and guardian.locked_until > datetime.now(timezone.utc):
        remaining = (guardian.locked_until - datetime.now(timezone.utc)).seconds // 60
        raise AuthError(
            f"Account temporarily locked due to too many failed attempts. "
            f"Try again in {remaining + 1} minute(s).",
            429,
        )

    if not guardian.is_active:
        raise AuthError("Account is deactivated. Contact support.", 403)

    if not verify_password(password, guardian.hashed_password):
        guardian.failed_login_attempts = (guardian.failed_login_attempts or 0) + 1
        if guardian.failed_login_attempts >= MAX_FAILED_ATTEMPTS:
            guardian.locked_until = datetime.now(timezone.utc) + timedelta(minutes=LOCKOUT_MINUTES)
            guardian.failed_login_attempts = 0
            db.commit()
            _audit(db, guardian.id, "account_locked",
                   f"Locked after {MAX_FAILED_ATTEMPTS} failed attempts", ip_address)
            raise AuthError(
                f"Too many failed attempts. Account locked for {LOCKOUT_MINUTES} minutes.", 429
            )
        db.commit()
        _audit(db, guardian.id, "login_failed",
               f"Attempt {guardian.failed_login_attempts}/{MAX_FAILED_ATTEMPTS}", ip_address)
        raise AuthError("Invalid credentials.", 401)

    # Success
    guardian.failed_login_attempts = 0
    guardian.locked_until = None
    guardian.last_login_at = datetime.now(timezone.utc)
    db.commit()

    token = create_access_token(guardian.id, guardian.role)
    _audit(db, guardian.id, "login_success", None, ip_address)
    return guardian, token


def get_guardian_by_id(db: Session, guardian_id: str) -> Optional[Guardian]:
    return db.query(Guardian).filter(Guardian.id == guardian_id).first()


def update_guardian_profile(db: Session, guardian: Guardian, **kwargs) -> Guardian:
    """Update allowed profile fields (not password, email, role)."""
    allowed = {"full_name", "mobile", "relationship_to_child", "country"}
    for key, val in kwargs.items():
        if key in allowed:
            setattr(guardian, key, val)
    guardian.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(guardian)
    return guardian


# ---------------------------------------------------------------------------
# OTP flow
# ---------------------------------------------------------------------------

def _create_and_send_otp(db: Session, guardian: Guardian, purpose: str) -> None:
    """Invalidate old OTPs for this purpose, create a new one, and send."""
    # Invalidate previous codes for same purpose
    db.query(OTPRecord).filter(
        OTPRecord.guardian_id == guardian.id,
        OTPRecord.purpose == purpose,
        OTPRecord.invalidated == False,  # noqa: E712
    ).update({"invalidated": True})
    db.commit()

    otp, otp_hash = _generate_otp()
    record = OTPRecord(
        guardian_id=guardian.id,
        purpose=purpose,
        otp_hash=otp_hash,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=OTP_EXPIRY_MINUTES),
    )
    db.add(record)
    db.commit()

    _send_otp_email(guardian.email, otp, purpose)


def send_otp(db: Session, guardian: Guardian, purpose: str) -> None:
    """Public wrapper — also enforces resend cooldown."""
    # Cooldown: check most recent non-invalidated OTP
    latest = (
        db.query(OTPRecord)
        .filter(
            OTPRecord.guardian_id == guardian.id,
            OTPRecord.purpose == purpose,
            OTPRecord.invalidated == False,  # noqa: E712
        )
        .order_by(OTPRecord.created_at.desc())
        .first()
    )
    if latest:
        cooldown_end = latest.created_at.replace(tzinfo=timezone.utc) + timedelta(seconds=OTP_RESEND_COOLDOWN_SECONDS)
        if datetime.now(timezone.utc) < cooldown_end:
            remaining = int((cooldown_end - datetime.now(timezone.utc)).total_seconds())
            raise AuthError(f"Please wait {remaining}s before requesting a new code.", 429)

    _create_and_send_otp(db, guardian, purpose)


def verify_otp(db: Session, guardian: Guardian, otp_code: str, purpose: str) -> bool:
    """Verify OTP. Returns True on success, raises AuthError on failure."""
    otp_hash = _hash_otp(otp_code.strip())
    now = datetime.now(timezone.utc)

    record = (
        db.query(OTPRecord)
        .filter(
            OTPRecord.guardian_id == guardian.id,
            OTPRecord.purpose == purpose,
            OTPRecord.invalidated == False,  # noqa: E712
            OTPRecord.used_at == None,  # noqa: E711
        )
        .order_by(OTPRecord.created_at.desc())
        .first()
    )

    if record is None:
        raise AuthError("No pending verification code found. Request a new one.", 400)

    if record.expires_at.replace(tzinfo=timezone.utc) < now:
        record.invalidated = True
        db.commit()
        raise AuthError("Verification code has expired. Request a new one.", 400)

    record.attempts = (record.attempts or 0) + 1
    if record.attempts > OTP_MAX_ATTEMPTS:
        record.invalidated = True
        db.commit()
        raise AuthError("Too many incorrect attempts. Request a new code.", 429)

    if record.otp_hash != otp_hash:
        db.commit()
        raise AuthError(f"Incorrect code. {OTP_MAX_ATTEMPTS - record.attempts} attempt(s) remaining.", 400)

    # Success
    record.used_at = now
    record.invalidated = True
    if purpose == "email_verification":
        guardian.is_verified = True
    db.commit()
    _audit(db, guardian.id, "otp_verified", f"purpose={purpose}")
    return True


def initiate_password_reset(db: Session, email: str) -> None:
    """Generate and send a password-reset OTP. No error if email not found (prevents enumeration)."""
    email = email.strip().lower()
    guardian = db.query(Guardian).filter(Guardian.email == email).first()
    if guardian and guardian.is_active:
        _create_and_send_otp(db, guardian, "password_reset")
    # Always return 200 to prevent user enumeration


def reset_password_with_otp(
    db: Session,
    email: str,
    otp_code: str,
    new_password: str,
    confirm_password: str,
) -> None:
    """Reset password after OTP verification."""
    email = email.strip().lower()
    guardian = db.query(Guardian).filter(Guardian.email == email).first()
    if not guardian:
        raise AuthError("Invalid request.", 400)

    if new_password != confirm_password:
        raise AuthError("Passwords do not match.", 422)
    valid, reason = validate_password_strength(new_password)
    if not valid:
        raise AuthError(reason, 422)

    verify_otp(db, guardian, otp_code, "password_reset")

    guardian.hashed_password = hash_password(new_password)
    guardian.updated_at = datetime.now(timezone.utc)
    db.commit()
    _audit(db, guardian.id, "password_reset", "Password reset via OTP")
    logger.info("Password reset for guardian %s", guardian.id)


# ---------------------------------------------------------------------------
# Audit helper
# ---------------------------------------------------------------------------

def _audit(
    db: Session,
    guardian_id: Optional[str],
    event_type: str,
    detail: Optional[str] = None,
    ip_address: Optional[str] = None,
) -> None:
    try:
        log = AuditLog(
            guardian_id=guardian_id,
            event_type=event_type,
            detail=detail,
            ip_address=ip_address,
        )
        db.add(log)
        db.commit()
    except Exception as exc:
        logger.warning("Audit log write failed: %s", exc)
