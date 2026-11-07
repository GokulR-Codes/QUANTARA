"""Health and readiness API endpoints."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.db import get_db
from backend.app.schemas.schemas import ComponentStatus, HealthResponse, ReadinessResponse

router = APIRouter(prefix="/api/v1", tags=["health"])

# --------------------------------------------------------------------------
# Component probe helpers — cached after first successful call but patchable
# by tests via the module-level sentinel pattern.
# --------------------------------------------------------------------------

_PENNYLANE_CACHE: tuple[bool, str] | None = None
_PYTORCH_CACHE: tuple[bool, str] | None = None
_TRACKER_CACHE: tuple[bool, str, str] | None = None


def _clear_component_cache() -> None:
    """Reset all probe caches (used by tests that monkeypatch dependencies)."""
    global _PENNYLANE_CACHE, _PYTORCH_CACHE, _TRACKER_CACHE
    _PENNYLANE_CACHE = None
    _PYTORCH_CACHE = None
    _TRACKER_CACHE = None


def _check_pennylane() -> tuple[bool, str]:
    global _PENNYLANE_CACHE
    if _PENNYLANE_CACHE is None:
        try:
            import pennylane as qml
            _PENNYLANE_CACHE = (True, f"version {qml.__version__}")
        except Exception as exc:
            _PENNYLANE_CACHE = (False, f"{type(exc).__name__}: {exc}")
    return _PENNYLANE_CACHE


def _check_pytorch() -> tuple[bool, str]:
    global _PYTORCH_CACHE
    if _PYTORCH_CACHE is None:
        try:
            import torch
            _PYTORCH_CACHE = (True, f"version {torch.__version__}")
        except Exception as exc:
            _PYTORCH_CACHE = (False, str(exc))
    return _PYTORCH_CACHE


def _check_tracker() -> tuple[bool, str, str]:
    """Return (tracker_ok, component_status, detail). Cached after first call."""
    global _TRACKER_CACHE
    if _TRACKER_CACHE is None:
        try:
            from src.video.tracker import create_cv_tracker
            _, backend = create_cv_tracker()
            detail = f"tracker backend: {backend}"
            status = "ok"
            if backend == "MIL":
                detail += " (CSRT/KCF unavailable; install opencv-contrib-python-headless)"
                status = "degraded"
            _TRACKER_CACHE = (True, status, detail)
        except Exception as exc:
            _TRACKER_CACHE = (False, "unavailable", str(exc))
    return _TRACKER_CACHE


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness: returns 200 whenever the process is serving requests."""
    return HealthResponse(status="ok", timestamp=datetime.now(timezone.utc), version="1.0.0")


@router.get("/ready", response_model=ReadinessResponse)
async def readiness(db: Session = Depends(get_db)) -> ReadinessResponse:
    """Readiness: reports, per component, what can actually run right now."""
    components: list[ComponentStatus] = []

    db_ok = True
    try:
        db.execute(text("SELECT 1"))
        components.append(ComponentStatus(name="database", status="ok"))
    except Exception as exc:
        db_ok = False
        components.append(ComponentStatus(name="database", status="unavailable", detail=str(exc)))

    # PennyLane — re-checked every request (tests patch sys.modules["pennylane"])
    quantum_ok = True
    try:
        import pennylane as qml
        components.append(ComponentStatus(name="pennylane", status="ok", detail=f"version {qml.__version__}"))
    except Exception as exc:
        quantum_ok = False
        components.append(ComponentStatus(
            name="pennylane", status="unavailable", detail=f"{type(exc).__name__}: {exc}"))

    # PyTorch — cached (DLL load is expensive but not monkeypatched in tests)
    torch_ok, torch_detail = _check_pytorch()
    components.append(ComponentStatus(
        name="pytorch",
        status="ok" if torch_ok else "unavailable",
        detail=torch_detail,
    ))

    # Tracker — re-checked every request (tests monkeypatch create_cv_tracker)
    tracker_ok = True
    try:
        from src.video.tracker import create_cv_tracker
        _, backend = create_cv_tracker()
        detail = f"tracker backend: {backend}"
        t_status = "ok"
        if backend == "MIL":
            detail += " (CSRT/KCF unavailable; install opencv-contrib-python-headless)"
            t_status = "degraded"
        components.append(ComponentStatus(name="opencv_tracker", status=t_status, detail=detail))
    except Exception as exc:
        tracker_ok = False
        components.append(ComponentStatus(name="opencv_tracker", status="unavailable", detail=str(exc)))

    model_ok = False
    try:
        from backend.app.services.model_service import describe_record, get_active_bundle_record
        rec = get_active_bundle_record(db)
        if rec is None:
            components.append(ComponentStatus(
                name="active_model", status="unavailable",
                detail="No pipeline bundle (upstream weights + QAOA mapping + VQC) is registered; "
                       "videos will be processed up to feature extraction without a prediction."))
        else:
            d = describe_record(rec)
            model_ok = d["usable"]
            components.append(ComponentStatus(
                name="active_model", status="ok" if model_ok else "unavailable",
                detail=f"id={rec.id}; {d['detail']}" + ("; synthetic-trained" if d["synthetic"] else "")))
    except Exception as exc:
        components.append(ComponentStatus(name="active_model", status="unavailable", detail=str(exc)))

    ready = db_ok and torch_ok and tracker_ok
    return ReadinessResponse(
        ready=ready, api_operational=True, database_operational=db_ok,
        model_available=model_ok, quantum_available=quantum_ok, components=components,
        message=(
            "System operational — inference pipeline ready." if (ready and model_ok)
            else "API operational; trained model not available — processing will complete "
                 "without a prediction." if ready
            else "API operational; one or more required components are unavailable."
        ),
    )
