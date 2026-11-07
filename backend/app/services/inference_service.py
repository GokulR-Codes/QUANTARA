"""Inference service — re-run the model stage (features -> QAOA mapping -> VQC) for a job.

Works from the job's persisted target crops, so it does not need the raw video (which may
have been deleted per the retention policy).  Any previous prediction is cleared first, so a
failed or unavailable re-run can never leave a stale prediction behind.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from backend.app.core.errors import AppError
from backend.app.core.logging_config import get_logger
from backend.app.models import Job, ModelRegistry
from backend.app.services.job_service import job_dir_for, merge_result, update_job_status
from backend.app.services.model_service import describe_record, get_active_bundle_record
from src.config.settings import get_settings
from src.pipeline.bundle import BUNDLE_MODEL_TYPE
from src.pipeline.runner import run_model_stage
from src.video.target_pipeline import TRACKING_FILE

logger = get_logger(__name__)

_STALE_RESULT_KEYS = (
    "prediction", "active_model_id", "model_version", "feature_schema_version",
    "selected_feature_names", "quantum_configuration", "model_provenance", "model_input_features",
)
_MODEL_WARNING_PREFIXES = (
    "Model unavailable", "Inference failed", "Prediction comes from",
    "CNN/LSTM/fusion ran with untrained", "Insufficient tracked frames",
)


def run_inference(db: Session, job: Job, model_id: Optional[str] = None) -> None:
    settings = get_settings()
    job_dir = job_dir_for(job.id)
    prior_status = job.status

    if not (job_dir / TRACKING_FILE).exists():
        result = merge_result(job, {
            "model_status": "frames_not_available",
            "model_status_detail": "Target crops are not available for this job (it was never "
                                   "processed successfully, or its artifacts were deleted).",
        }, drop=_STALE_RESULT_KEYS)
        update_job_status(db, job, "completed_without_prediction", result=result)
        return

    record: Optional[ModelRegistry]
    if model_id:
        record = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
        if record is None:
            raise AppError(f"Model {model_id} not found", status_code=404)
    else:
        record = get_active_bundle_record(db)

    bundle_dir: Optional[Path] = None
    no_model_detail = "No trained pipeline bundle is registered as active."
    if record is not None:
        if record.model_type == BUNDLE_MODEL_TYPE and record.checkpoint_path:
            bundle_dir = Path(record.checkpoint_path)
        else:
            no_model_detail = describe_record(record)["detail"]

    update_job_status(db, job, "inference")
    try:
        outcome = run_model_stage(job_dir, settings, bundle_dir, no_model_detail=no_model_detail)
    except Exception:
        job.status = prior_status        # never leave a job stuck in "inference"
        db.commit()
        raise

    updates = {
        "feature_extraction_status": outcome.feature_extraction_status,
        "feature_extraction": outcome.feature_info,
        "model_status": outcome.model_status,
        "model_status_detail": outcome.model_status_detail,
    }
    if outcome.predicted:
        updates["prediction"] = outcome.prediction
        updates["active_model_id"] = record.id if record else None
        updates.update({k: v for k, v in outcome.model_meta.items() if k != "mapped_input"})
        updates["model_input_features"] = outcome.model_meta.get("mapped_input")
    result = merge_result(job, updates, drop=_STALE_RESULT_KEYS)
    stages = dict(result.get("pipeline_stages") or {})
    stages.update({"feature_extraction": outcome.feature_extraction_status,
                   "quantum_inference": outcome.model_status})
    result["pipeline_stages"] = stages

    warnings = [w for w in (job.warnings or []) if not w.startswith(_MODEL_WARNING_PREFIXES)]
    warnings.extend(w for w in outcome.warnings if w not in warnings)
    final = "completed" if outcome.predicted else "completed_without_prediction"
    update_job_status(db, job, final, result=result, warnings=warnings)
    logger.info("Inference re-run for job %s: %s (%s)", job.id, final, outcome.model_status)
