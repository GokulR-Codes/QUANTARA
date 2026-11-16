"""Create the first admin guardian account directly in the database.

Usage (from project root, with venv active):
    python scripts/create_admin.py

Or with explicit credentials:
    python scripts/create_admin.py --email admin@qcare.local --username admin --password "SecurePass1!"

The script is idempotent: if the email already exists it prints the existing
guardian's ID and exits without modifying the record.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path regardless of invocation directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _prompt(label: str, hidden: bool = False) -> str:
    if hidden:
        import getpass
        return getpass.getpass(f"  {label}: ")
    return input(f"  {label}: ").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the first Q-CARE admin guardian account.")
    parser.add_argument("--email", default="", help="Guardian email address")
    parser.add_argument("--username", default="", help="Guardian username (login handle)")
    parser.add_argument("--full-name", default="", dest="full_name", help="Guardian full name")
    parser.add_argument("--password", default="", help="Password (prompted securely if omitted)")
    args = parser.parse_args()

    print("\n=== Q-CARE Admin Guardian Setup ===\n")

    email = args.email or _prompt("Email address")
    username = args.username or _prompt("Username")
    full_name = args.full_name or _prompt("Full name (optional, press Enter to skip)")
    password = args.password or _prompt("Password", hidden=True)
    if not args.password:
        confirm = _prompt("Confirm password", hidden=True)
        if password != confirm:
            print("\n[ERROR] Passwords do not match. Aborting.")
            sys.exit(1)

    # --- initialise DB and import services ---
    from backend.app.db import init_db, SessionLocal
    from backend.app.models_auth import Guardian
    from backend.app.services.auth_service import (
        hash_password,
        validate_password_strength,
    )

    init_db()

    is_valid, reason = validate_password_strength(password)
    if not is_valid:
        print(f"\n[ERROR] Weak password: {reason}")
        print("Requirements: ≥8 chars, uppercase, lowercase, digit, special character.")
        sys.exit(1)

    db = SessionLocal()
    try:
        # Idempotency check
        existing = db.query(Guardian).filter(Guardian.email == email.lower()).first()
        if existing:
            print(f"\n[INFO] Guardian with email '{email}' already exists.")
            print(f"  Guardian ID : {existing.id}")
            print(f"  Username    : {existing.username}")
            print(f"  Role        : {existing.role}")
            print(f"  Verified    : {existing.is_verified}")
            print("\nNo changes made.")
            return

        guardian = Guardian(
            email=email.lower().strip(),
            username=username.strip(),
            full_name=full_name.strip() or None,
            hashed_password=hash_password(password),
            role="admin",
            is_verified=True,   # admin accounts are pre-verified
            is_active=True,
            relationship_to_child=None,
        )
        db.add(guardian)
        db.commit()
        db.refresh(guardian)

        print(f"\n[SUCCESS] Admin guardian created.")
        print(f"  Guardian ID : {guardian.id}")
        print(f"  Email       : {guardian.email}")
        print(f"  Username    : {guardian.username}")
        print(f"  Role        : {guardian.role}")
        print(f"  Verified    : {guardian.is_verified}")
        print("\nYou can now log in at the Q-CARE frontend with these credentials.")

    except Exception as exc:
        db.rollback()
        print(f"\n[ERROR] Failed to create guardian: {exc}")
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
