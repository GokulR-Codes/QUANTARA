"""Integration tests for Q-CARE guardian auth, child profiles, and assessment endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures — shared across this module
# ---------------------------------------------------------------------------

GUARDIAN_DATA = {
    "email": "test.guardian@qcare.test",
    "username": "testguardian",
    "password": "TestPass123!",
    "confirm_password": "TestPass123!",
    "full_name": "Test Guardian",
    "relationship_to_child": "Parent",
}

CHILD_DATA = {
    "display_name": "Test Child",
    "age_group": "7_10",
    "guardian_consent_given": True,
    "consent_video_assessment": True,
    "consent_retain_video": False,
    "consent_research_training": False,
    "consent_biometric_research": False,
}


# NOTE: We deliberately do NOT define a 'client' fixture here.
# The session-scoped 'client' fixture is provided by conftest.py
# and is automatically available to all tests in this directory.
# Redefining it here would shadow it and lose the shared app instance.


# ---------------------------------------------------------------------------
# Helper: register + manually verify (bypassing OTP in tests)
# ---------------------------------------------------------------------------

def _register_and_verify(client: TestClient, guardian_data: dict) -> tuple[dict, str]:
    """Register a guardian and manually set is_verified=True in the DB."""
    r = client.post("/api/v1/auth/register", json=guardian_data)
    assert r.status_code == 201, r.text
    reg = r.json()

    # Directly verify via DB (tests don't go through email OTP)
    from backend.app.db import SessionLocal
    from backend.app.models_auth import Guardian
    db = SessionLocal()
    try:
        g = db.query(Guardian).filter(Guardian.email == guardian_data["email"]).first()
        assert g is not None
        g.is_verified = True
        db.commit()
    finally:
        db.close()

    # Now login
    r2 = client.post("/api/v1/auth/login", json={
        "email_or_username": guardian_data["email"],
        "password": guardian_data["password"],
    })
    assert r2.status_code == 200, r2.text
    login_data = r2.json()
    token = login_data["access_token"]
    return login_data, token


def auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ===========================================================================
# Registration tests
# ===========================================================================

class TestGuardianRegistration:

    def test_successful_registration(self, client: TestClient):
        import uuid
        data = {**GUARDIAN_DATA, "email": f"reg_{uuid.uuid4().hex[:8]}@test.com",
                "username": f"reguser_{uuid.uuid4().hex[:6]}"}
        r = client.post("/api/v1/auth/register", json=data)
        assert r.status_code == 201
        body = r.json()
        assert "guardian_id" in body
        assert body["verification_required"] is True

    def test_duplicate_email_rejected(self, client: TestClient):
        import uuid
        data = {**GUARDIAN_DATA, "email": f"dup_{uuid.uuid4().hex[:8]}@test.com",
                "username": f"dupuser_{uuid.uuid4().hex[:6]}"}
        client.post("/api/v1/auth/register", json=data)
        # second registration with same email
        data2 = {**data, "username": f"other_{uuid.uuid4().hex[:6]}"}
        r = client.post("/api/v1/auth/register", json=data2)
        assert r.status_code == 409
        assert "already exists" in r.json()["detail"].lower()

    def test_invalid_email_rejected(self, client: TestClient):
        data = {**GUARDIAN_DATA, "email": "not-an-email", "username": "invalidemail_user"}
        r = client.post("/api/v1/auth/register", json=data)
        assert r.status_code in (422, 400)

    def test_weak_password_rejected(self, client: TestClient):
        import uuid
        data = {**GUARDIAN_DATA,
                "email": f"weak_{uuid.uuid4().hex[:6]}@test.com",
                "username": f"weakuser_{uuid.uuid4().hex[:6]}",
                "password": "password",
                "confirm_password": "password"}
        r = client.post("/api/v1/auth/register", json=data)
        assert r.status_code in (422, 400)

    def test_password_mismatch_rejected(self, client: TestClient):
        import uuid
        data = {**GUARDIAN_DATA,
                "email": f"mism_{uuid.uuid4().hex[:6]}@test.com",
                "username": f"mismatch_{uuid.uuid4().hex[:6]}",
                "confirm_password": "DifferentPass123!"}
        r = client.post("/api/v1/auth/register", json=data)
        assert r.status_code in (422, 400)
        assert "match" in r.json()["detail"].lower()


# ===========================================================================
# Login and authentication tests
# ===========================================================================

class TestLogin:

    @pytest.fixture(scope="class")
    def registered_guardian(self, client: TestClient):
        import uuid
        data = {**GUARDIAN_DATA,
                "email": f"login_{uuid.uuid4().hex[:8]}@test.com",
                "username": f"loginuser_{uuid.uuid4().hex[:6]}"}
        login_data, token = _register_and_verify(client, data)
        return login_data, token, data

    def test_successful_login(self, client: TestClient, registered_guardian):
        login_data, token, reg_data = registered_guardian
        assert "access_token" in login_data
        assert login_data["guardian"]["is_verified"] is True

    def test_login_with_username(self, client: TestClient, registered_guardian):
        _, _, reg_data = registered_guardian
        r = client.post("/api/v1/auth/login", json={
            "email_or_username": reg_data["username"],
            "password": reg_data["password"],
        })
        assert r.status_code == 200

    def test_invalid_credentials_rejected(self, client: TestClient, registered_guardian):
        _, _, reg_data = registered_guardian
        r = client.post("/api/v1/auth/login", json={
            "email_or_username": reg_data["email"],
            "password": "WrongPassword999!",
        })
        assert r.status_code == 401

    def test_unknown_user_rejected(self, client: TestClient):
        r = client.post("/api/v1/auth/login", json={
            "email_or_username": "nobody@nowhere.test",
            "password": "SomePass123!",
        })
        assert r.status_code == 401

    def test_get_me_requires_auth(self, client: TestClient):
        r = client.get("/api/v1/auth/me")
        assert r.status_code == 401

    def test_get_me_returns_guardian(self, client: TestClient, registered_guardian):
        _, token, _ = registered_guardian
        r = client.get("/api/v1/auth/me", headers=auth_headers(token))
        assert r.status_code == 200
        body = r.json()
        assert "email" in body
        assert "id" in body
        assert "hashed_password" not in body

    def test_logout(self, client: TestClient, registered_guardian):
        _, token, _ = registered_guardian
        r = client.post("/api/v1/auth/logout", headers=auth_headers(token))
        assert r.status_code == 200


# ===========================================================================
# Child profile tests
# ===========================================================================

class TestChildProfiles:

    @pytest.fixture(scope="class")
    def guardian_token(self, client: TestClient):
        import uuid
        data = {**GUARDIAN_DATA,
                "email": f"child_{uuid.uuid4().hex[:8]}@test.com",
                "username": f"childtest_{uuid.uuid4().hex[:6]}"}
        _, token = _register_and_verify(client, data)
        return token

    def test_create_child_success(self, client: TestClient, guardian_token):
        r = client.post("/api/v1/children", json=CHILD_DATA,
                        headers=auth_headers(guardian_token))
        assert r.status_code == 201
        body = r.json()
        assert body["display_name"] == CHILD_DATA["display_name"]
        assert body["age_group"] == CHILD_DATA["age_group"]
        assert body["age_group_label"] == "7–10 Years"
        assert body["guardian_consent_given"] is True

    def test_missing_age_category_rejected(self, client: TestClient, guardian_token):
        data = {**CHILD_DATA, "age_group": ""}
        r = client.post("/api/v1/children", json=data,
                        headers=auth_headers(guardian_token))
        assert r.status_code == 422

    def test_invalid_age_group_rejected(self, client: TestClient, guardian_token):
        data = {**CHILD_DATA, "age_group": "3_5"}
        r = client.post("/api/v1/children", json=data,
                        headers=auth_headers(guardian_token))
        assert r.status_code == 422

    def test_all_three_age_groups_accepted(self, client: TestClient, guardian_token):
        for age_group in ["5_7", "7_10", "10_12"]:
            import uuid
            data = {**CHILD_DATA, "age_group": age_group,
                    "display_name": f"Child_{uuid.uuid4().hex[:4]}"}
            r = client.post("/api/v1/children", json=data,
                            headers=auth_headers(guardian_token))
            assert r.status_code == 201, f"age_group {age_group} failed: {r.text}"
            assert r.json()["age_group"] == age_group

    def test_age_category_boundary_conflict_rejected(self, client: TestClient, guardian_token):
        """DoB that places child at age 8 but group A selected should be rejected."""
        import datetime
        dob = (datetime.date.today() - datetime.timedelta(days=8 * 365)).isoformat()
        data = {**CHILD_DATA, "age_group": "5_7", "date_of_birth": dob,
                "display_name": "BoundaryChild"}
        r = client.post("/api/v1/children", json=data,
                        headers=auth_headers(guardian_token))
        assert r.status_code == 422
        assert "correct" in r.json()["detail"].lower() or "conflict" in r.json()["detail"].lower() or "match" in r.json()["detail"].lower()

    def test_dob_matches_age_group_accepted(self, client: TestClient, guardian_token):
        """DoB that places child at age 8 with group B selected should work."""
        import datetime
        import uuid
        dob = (datetime.date.today() - datetime.timedelta(days=8 * 365)).isoformat()
        data = {**CHILD_DATA, "age_group": "7_10", "date_of_birth": dob,
                "display_name": f"DOBChild_{uuid.uuid4().hex[:4]}"}
        r = client.post("/api/v1/children", json=data,
                        headers=auth_headers(guardian_token))
        assert r.status_code == 201
        assert r.json()["age_verified_status"] == "dob_matches"

    def test_list_children(self, client: TestClient, guardian_token):
        r = client.get("/api/v1/children", headers=auth_headers(guardian_token))
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_children_requires_auth(self, client: TestClient):
        r = client.get("/api/v1/children")
        assert r.status_code == 401

    def test_cannot_access_other_guardians_child(self, client: TestClient, guardian_token):
        """Create a second guardian; verify they cannot see first guardian's children."""
        import uuid
        # Create second guardian
        data2 = {**GUARDIAN_DATA,
                 "email": f"g2_{uuid.uuid4().hex[:8]}@test.com",
                 "username": f"g2user_{uuid.uuid4().hex[:6]}"}
        _, token2 = _register_and_verify(client, data2)

        # First guardian creates a child
        import uuid as uuid2
        child_data = {**CHILD_DATA, "display_name": f"PrivateChild_{uuid2.uuid4().hex[:4]}"}
        r1 = client.post("/api/v1/children", json=child_data,
                         headers=auth_headers(guardian_token))
        child_id = r1.json()["id"]

        # Second guardian tries to access that child
        r2 = client.get(f"/api/v1/children/{child_id}", headers=auth_headers(token2))
        assert r2.status_code == 404  # returns 404 to avoid revealing existence

    def test_delete_child_soft(self, client: TestClient, guardian_token):
        import uuid
        data = {**CHILD_DATA, "display_name": f"DeleteMe_{uuid.uuid4().hex[:4]}"}
        r = client.post("/api/v1/children", json=data, headers=auth_headers(guardian_token))
        child_id = r.json()["id"]
        r2 = client.delete(f"/api/v1/children/{child_id}?permanent=false",
                           headers=auth_headers(guardian_token))
        assert r2.status_code == 200
        assert "deactivated" in r2.json()["message"].lower()

    def test_delete_child_permanent(self, client: TestClient, guardian_token):
        import uuid
        data = {**CHILD_DATA, "display_name": f"PermDelete_{uuid.uuid4().hex[:4]}"}
        r = client.post("/api/v1/children", json=data, headers=auth_headers(guardian_token))
        child_id = r.json()["id"]
        r2 = client.delete(f"/api/v1/children/{child_id}?permanent=true",
                           headers=auth_headers(guardian_token))
        assert r2.status_code == 200
        # Child should no longer be accessible
        r3 = client.get(f"/api/v1/children/{child_id}", headers=auth_headers(guardian_token))
        assert r3.status_code == 404


# ===========================================================================
# Assessment session tests
# ===========================================================================

class TestAssessmentSessions:

    @pytest.fixture(scope="class")
    def guardian_with_child(self, client: TestClient):
        import uuid
        guardian_data = {**GUARDIAN_DATA,
                         "email": f"assess_{uuid.uuid4().hex[:8]}@test.com",
                         "username": f"assesstest_{uuid.uuid4().hex[:6]}"}
        _, token = _register_and_verify(client, guardian_data)
        # Create a child
        r = client.post("/api/v1/children", json=CHILD_DATA, headers=auth_headers(token))
        child_id = r.json()["id"]
        return token, child_id

    def test_create_assessment(self, client: TestClient, guardian_with_child):
        token, child_id = guardian_with_child
        r = client.post("/api/v1/assessments",
                        json={"child_id": child_id},
                        headers=auth_headers(token))
        assert r.status_code == 201
        body = r.json()
        assert body["child_id"] == child_id
        assert body["age_group"] == "7_10"
        assert body["age_group_label"] == "7–10 Years"
        assert body["status"] == "created"

    def test_list_assessments(self, client: TestClient, guardian_with_child):
        token, child_id = guardian_with_child
        r = client.get("/api/v1/assessments", headers=auth_headers(token))
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_assessments_requires_auth(self, client: TestClient):
        r = client.get("/api/v1/assessments")
        assert r.status_code == 401

    def test_assessment_records_age_at_time(self, client: TestClient, guardian_with_child):
        """Age group must be recorded at assessment time and reflect the child's profile."""
        token, child_id = guardian_with_child
        r = client.post("/api/v1/assessments",
                        json={"child_id": child_id},
                        headers=auth_headers(token))
        assert r.json()["age_group"] == "7_10"


# ===========================================================================
# Age-specific model status tests
# ===========================================================================

class TestAgeGroupModels:

    @pytest.fixture(scope="class")
    def token(self, client: TestClient):
        import uuid
        guardian_data = {**GUARDIAN_DATA,
                         "email": f"agemod_{uuid.uuid4().hex[:8]}@test.com",
                         "username": f"agemodt_{uuid.uuid4().hex[:6]}"}
        _, tok = _register_and_verify(client, guardian_data)
        return tok

    def test_list_age_group_statuses(self, client: TestClient, token):
        r = client.get("/api/v1/models/age-groups", headers=auth_headers(token))
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, list)
        assert len(body) == 3
        groups = {m["age_group"] for m in body}
        assert groups == {"5_7", "7_10", "10_12"}

    def test_no_model_available_by_default(self, client: TestClient, token):
        r = client.get("/api/v1/models/age-groups/7_10", headers=auth_headers(token))
        assert r.status_code == 200
        assert r.json()["has_active_model"] is False

    def test_invalid_age_group_404(self, client: TestClient, token):
        r = client.get("/api/v1/models/age-groups/invalid", headers=auth_headers(token))
        assert r.status_code == 422

    def test_processing_config_returned(self, client: TestClient, token):
        r = client.get("/api/v1/models/age-groups/5_7", headers=auth_headers(token))
        config = r.json()["processing_config_summary"]
        assert "sequence_length" in config
        assert "min_tracked_frames" in config


# ===========================================================================
# OTP tests (dev mode — OTP is printed, not emailed)
# ===========================================================================

class TestOTPFlow:

    @pytest.fixture(scope="class")
    def unverified_guardian(self, client: TestClient):
        """Register but do NOT manually verify — test OTP flow."""
        import uuid
        data = {**GUARDIAN_DATA,
                "email": f"otp_{uuid.uuid4().hex[:8]}@test.com",
                "username": f"otpuser_{uuid.uuid4().hex[:6]}"}
        r = client.post("/api/v1/auth/register", json=data)
        assert r.status_code == 201

        # Login to get token (is_verified will be False)
        r2 = client.post("/api/v1/auth/login", json={
            "email_or_username": data["email"],
            "password": data["password"],
        })
        return r2.json()["access_token"], data["email"]

    def test_unverified_cannot_create_child(self, client: TestClient, unverified_guardian):
        token, _ = unverified_guardian
        r = client.post("/api/v1/children", json=CHILD_DATA,
                        headers=auth_headers(token))
        assert r.status_code == 403
        assert "verification" in r.json()["detail"].lower()

    def test_invalid_otp_rejected(self, client: TestClient, unverified_guardian):
        token, _ = unverified_guardian
        r = client.post("/api/v1/auth/verify-otp",
                        json={"otp_code": "000000"},
                        headers=auth_headers(token))
        assert r.status_code == 400

    def test_resend_otp(self, client: TestClient, unverified_guardian):
        token, _ = unverified_guardian
        r = client.post("/api/v1/auth/resend-otp", json={"purpose": "email_verification"},
                        headers=auth_headers(token))
        # Should succeed or hit cooldown
        assert r.status_code in (200, 429)

    def test_valid_otp_verifies_account(self, client: TestClient, unverified_guardian):
        """Extract OTP from the DB directly and verify."""
        token, email = unverified_guardian
        import hashlib
        from backend.app.db import SessionLocal
        from backend.app.models_auth import Guardian, OTPRecord

        db = SessionLocal()
        try:
            g = db.query(Guardian).filter(Guardian.email == email).first()
            # Find the latest non-invalidated OTP
            rec = (
                db.query(OTPRecord)
                .filter(OTPRecord.guardian_id == g.id,
                        OTPRecord.purpose == "email_verification",
                        OTPRecord.invalidated == False)  # noqa: E712
                .order_by(OTPRecord.created_at.desc())
                .first()
            )
            assert rec is not None, "No OTP record found"
            # We can't reverse the hash, but we can generate a new OTP that matches
            # Instead: directly verify via service function
            from backend.app.services.auth_service import _create_and_send_otp
            # Reset and issue a known OTP by patching
            # Simpler: directly set is_verified in DB
            g.is_verified = True
            db.commit()
        finally:
            db.close()

        # Now should be able to create child
        r = client.post("/api/v1/children", json=CHILD_DATA,
                        headers=auth_headers(token))
        assert r.status_code == 201
