"""Job CRUD operations — create, read, update, cancel, delete — and response building."""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from sqlalchemy.orm import Session

from backend.app.core.errors import JobCancelled, JobConflictError
from backend.app.core.logging_config import get_logger
from backend.app.models import Job
from backend.app.schemas.schemas import (
    FrameProcessingSummary, JobResponse, JobStatus,
    PredictionResult, TargetTrackingSummary, VideoMetadata,
)
from src.config.settings import get_settings

logger = get_logger(__name__)

TERMINAL_STATES = {"completed", "completed_without_prediction", "failed", "cancelled"}
ACTIVE_STATES = {"queued", "processing", "tracking", "feature_extraction", "inference"}


def job_dir_for(job_id: str) -> Path:
    """Directory holding every derived artifact of a job (crops, tracking.json, features)."""
    return Path(get_settings().frames_dir) / job_id


def create_job(
    db: Session, job_id: str, original_filename: str, stored_path: str, file_size_bytes: int,
    video_meta: Optional[dict] = None, warnings: Optional[list[str]] = None,
) -> Job:
    now = datetime.now(timezone.utc)
    job = Job(
        id=job_id, status=JobStatus.uploaded.value, original_filename=original_filename,
        stored_path=stored_path, file_size_bytes=file_size_bytes, created_at=now, updated_at=now,
        warnings=warnings or None,
    )
    if video_meta:
        job.video_duration_s = video_meta["duration_s"]
        job.video_fps = video_meta["fps"]
        job.video_width = video_meta["width"]
        job.video_height = video_meta["height"]
        job.video_codec = video_meta["codec"]
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_job(db: Session, job_id: str) -> Optional[Job]:
    return db.query(Job).filter(Job.id == job_id).first()


def current_status(job_id: str) -> Optional[str]:
    """Status as currently committed in the DB, read through a fresh session."""
    from backend.app.db import SessionLocal
    with SessionLocal() as s:
        return s.query(Job.status).filter(Job.id == job_id).scalar()


def is_cancelled(job_id: str) -> bool:
    return current_status(job_id) == "cancelled"


def update_job_status(
    db: Session, job: Job, status: str,
    stage_progress: Optional[dict] = None, result: Optional[dict] = None,
    error_message: Optional[str] = None, warnings: Optional[list[str]] = None,
) -> None:
    """Persist a status change.

    Raises :class:`JobCancelled` if the operator cancelled the job in the meantime, so a
    running worker can never overwrite ``cancelled`` with a later stage.
    """
    if status != "cancelled" and current_status(job.id) == "cancelled":
        raise JobCancelled()
    job.status = status
    job.updated_at = datetime.now(timezone.utc)
    if stage_progress is not None:
        job.stage_progress = stage_progress
    if result is not None:
        job.result = result
    if error_message is not None:
        job.error_message = error_message
    if warnings is not None:
        job.warnings = warnings or None
    if status in TERMINAL_STATES:
        job.completed_at = datetime.now(timezone.utc)
    db.commit()


def cancel_job(db: Session, job: Job) -> bool:
    if job.status in TERMINAL_STATES:
        return False
    update_job_status(db, job, "cancelled")
    return True


def remove_job_artifacts(job_id: str) -> bool:
    d = job_dir_for(job_id)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
        return True
    return False


def delete_job_record(db: Session, job: Job) -> list[str]:
    """Explicit delete: removes the raw video (regardless of retention), all derived artifacts
    and the DB row.  Refuses while the job is being processed (cancel it first)."""
    if job.status in ACTIVE_STATES:
        raise JobConflictError(
            f"Job is {job.status}; cancel it before deleting."
        )
    removed: list[str] = []
    if job.stored_path:
        p = Path(job.stored_path)
        if p.exists():
            p.unlink()
            removed.append("raw_video")
    if remove_job_artifacts(job.id):
        removed.append("job_artifacts")
    db.delete(job)
    db.commit()
    return removed


def merge_result(job: Job, updates: dict, drop: Iterable[str] = ()) -> dict:
    """New result dict (JSON columns do not detect in-place mutation)."""
    merged = {k: v for k, v in (job.result or {}).items() if k not in set(drop)}
    merged.update(updates)
    return merged


def _filtered(model_cls, data: dict):
    return model_cls(**{k: v for k, v in data.items() if k in model_cls.model_fields})


def job_to_response(job: Job) -> JobResponse:
    """Convert a Job ORM object to a JobResponse schema (everything comes from stored data)."""
    result_data = job.result or {}
    stage = job.stage_progress or {}

    video_meta = None
    if job.video_width or job.video_fps or job.video_duration_s:
        video_meta = VideoMetadata(
            duration_s=job.video_duration_s, fps=job.video_fps, width=job.video_width,
            height=job.video_height, codec=job.video_codec,
            file_size_bytes=job.file_size_bytes, original_filename=job.original_filename,
        )

    target_selection = None
    if job.target_bbox_x is not None:
        target_selection = {
            "bbox": {"x": job.target_bbox_x, "y": job.target_bbox_y,
                     "w": job.target_bbox_w, "h": job.target_bbox_h},
            "frame_index": job.target_frame_index,
            "frame_time_ms": job.target_frame_time_ms,
            "frame_width": job.video_width,
            "frame_height": job.video_height,
            "selected_at": job.target_selected_at.isoformat() if job.target_selected_at else None,
        }

    frame_summary = _filtered(FrameProcessingSummary, stage["frame_processing"]) if "frame_processing" in stage else None
    tracking_summary = None
    if "tracking" in stage:
        tr = dict(stage["tracking"])
        tr.setdefault("initial_bbox", (tr.get("selection") or {}).get("bbox"))
        tracking_summary = _filtered(TargetTrackingSummary, tr)

    prediction = _filtered(PredictionResult, result_data["prediction"]) if result_data.get("prediction") else None

    return JobResponse(
        job_id=job.id, recording_id=job.recording_id, status=JobStatus(job.status),
        created_at=job.created_at, updated_at=job.updated_at, completed_at=job.completed_at,
        video_metadata=video_meta, frame_processing_summary=frame_summary,
        target_tracking_summary=tracking_summary, target_selection=target_selection,
        feature_extraction_status=result_data.get("feature_extraction_status"),
        feature_extraction=result_data.get("feature_extraction"),
        model_status=result_data.get("model_status"),
        model_status_detail=result_data.get("model_status_detail"),
        model_provenance=result_data.get("model_provenance"),
        active_model_id=result_data.get("active_model_id"),
        model_version=result_data.get("model_version"),
        feature_schema_version=result_data.get("feature_schema_version"),
        selected_feature_names=result_data.get("selected_feature_names"),
        quantum_configuration=result_data.get("quantum_configuration"),
        prediction=prediction, experiment_metrics=result_data.get("experiment_metrics"),
        warnings=job.warnings, errors=job.error_message,
        artifact_references=result_data.get("artifact_references"),
        processing_duration_s=job.processing_duration_s,
    )
