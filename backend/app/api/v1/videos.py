"""Video upload, participant selection, processing, and result retrieval endpoints."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from backend.app.core.errors import VideoValidationError
from backend.app.core.logging_config import get_logger
from backend.app.db import get_db
from backend.app.schemas.schemas import (
    JobResponse, JobStatus, TargetSelectionRequest, TargetSelectionResponse, UploadResponse,
)
from backend.app.services.job_service import (
    ACTIVE_STATES, TERMINAL_STATES, cancel_job, create_job, delete_job_record, get_job,
    job_dir_for, job_to_response,
)
from backend.app.services.video_service import save_upload_streaming
from backend.app.workers.processing_worker import run_processing_job
from src.config.settings import get_settings
from src.video.extractor import probe_video, read_frame_at
from src.video.geometry import BBoxError, validate_bbox
from src.video.target_pipeline import TRACKING_FILE

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/videos", tags=["videos"])

# States in which the participant may still be (re)selected / processing may start.
SELECTABLE_STATES = {"uploaded", "failed", "cancelled"}
STARTABLE_STATES = {"uploaded", "failed", "cancelled", "completed_without_prediction"}


def _require_job(db: Session, job_id: str):
    job = get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


def _require_raw_video(job) -> Path:
    if not job.stored_path or not Path(job.stored_path).exists():
        raise HTTPException(
            status_code=410,
            detail="The raw video is no longer available (deleted per the retention policy "
                   "or removed). Re-upload to select a participant or reprocess.",
        )
    return Path(job.stored_path)


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_video(file: UploadFile = File(...), db: Session = Depends(get_db)) -> UploadResponse:
    """Upload a video recording. Returns a job_id; the participant must then be selected."""
    settings = get_settings()
    filename = file.filename or "upload.mp4"
    job_id = str(uuid.uuid4())
    stored_path = Path(settings.upload_dir) / f"{job_id}{Path(filename).suffix.lower()}"

    size = await save_upload_streaming(
        file, stored_path, filename, settings.max_upload_size_bytes, settings.accepted_video_formats,
    )
    try:
        meta = await run_in_threadpool(probe_video, stored_path)
        if meta["duration_s"] > settings.max_video_duration_seconds:
            raise VideoValidationError(
                f"Video is {meta['duration_s']:.0f}s long; the limit is "
                f"{settings.max_video_duration_seconds}s"
            )
    except Exception:
        stored_path.unlink(missing_ok=True)
        raise

    create_job(db, job_id, filename, str(stored_path), size, video_meta=meta, warnings=meta["warnings"])
    logger.info("Video uploaded: job_id=%s file=%s size=%d bytes %dx%d", job_id, filename, size,
                meta["width"], meta["height"])
    return UploadResponse(
        job_id=job_id, status=JobStatus.uploaded,
        message="Video uploaded. Select the participant (select-target), then start processing (process).",
        original_filename=filename, file_size_bytes=size,
    )


@router.get("/{job_id}/frame-info")
async def frame_info(job_id: str, t_ms: float = Query(0.0, ge=0.0), db: Session = Depends(get_db)) -> dict:
    """Resolve a time to the exact source frame the operator will see and select on."""
    job = _require_job(db, job_id)
    video = _require_raw_video(job)
    try:
        df = await run_in_threadpool(read_frame_at, video, t_ms)
    except VideoValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.message)
    h, w = df.frame.shape[:2]
    return {"job_id": job_id, "frame_index": df.index, "time_ms": df.t_ms, "width": w, "height": h,
            "duration_s": job.video_duration_s}


@router.get("/{job_id}/preview-frame")
async def preview_frame(
    job_id: str, frame_index: Optional[int] = Query(None, ge=0), t_ms: float = Query(0.0, ge=0.0),
    db: Session = Depends(get_db),
) -> Response:
    """JPEG of one source frame at full resolution (coordinates = source pixels)."""
    video = _require_raw_video(_require_job(db, job_id))
    try:
        df = await run_in_threadpool(read_frame_at, video, t_ms, frame_index)
    except VideoValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.message)
    ok, buf = cv2.imencode(".jpg", df.frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode preview frame")
    h, w = df.frame.shape[:2]
    return Response(
        content=buf.tobytes(), media_type="image/jpeg",
        headers={"Cache-Control": "no-store", "X-Frame-Index": str(df.index),
                 "X-Frame-Time-Ms": f"{df.t_ms:.3f}", "X-Frame-Width": str(w), "X-Frame-Height": str(h)},
    )


@router.post("/{job_id}/select-target", response_model=TargetSelectionResponse)
async def select_target(job_id: str, req: TargetSelectionRequest, db: Session = Depends(get_db)):
    """Record the operator-selected participant box.

    The box is validated against the video's *actual* decoded frame size (not against
    client-supplied dimensions, which must merely agree with it); invalid or out-of-bounds
    boxes are rejected with 422 and nothing is stored.
    """
    settings = get_settings()
    job = _require_job(db, job_id)
    if req.job_id is not None and req.job_id != job_id:
        raise HTTPException(status_code=422, detail="job_id in body does not match the URL")
    if job.status not in SELECTABLE_STATES:
        raise HTTPException(status_code=409,
                            detail=f"Participant cannot be changed while job is '{job.status}'.")
    video = _require_raw_video(job)

    try:
        df = await run_in_threadpool(read_frame_at, video, 0.0, req.frame_index)
    except VideoValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.message)
    actual_h, actual_w = df.frame.shape[:2]
    if (req.frame_width, req.frame_height) != (actual_w, actual_h):
        raise HTTPException(
            status_code=422,
            detail=(f"Frame size mismatch: selection was made on {req.frame_width}x{req.frame_height} "
                    f"but the video frames are {actual_w}x{actual_h}. Coordinates must be in "
                    "source-frame pixels."),
        )
    try:
        bbox = validate_bbox(req.bbox_x, req.bbox_y, req.bbox_w, req.bbox_h,
                             actual_w, actual_h, settings.min_box_px)
    except BBoxError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    job.target_bbox_x, job.target_bbox_y, job.target_bbox_w, job.target_bbox_h = bbox.as_tuple()
    job.target_frame_index = df.index
    job.target_frame_time_ms = df.t_ms
    job.target_selected_at = datetime.now(timezone.utc)
    job.video_width, job.video_height = actual_w, actual_h
    db.commit()
    logger.info("Target selected: job=%s bbox=%s frame=%d t=%.1fms", job_id, bbox.as_dict(), df.index, df.t_ms)
    return TargetSelectionResponse(
        job_id=job_id, target_set=True, bbox=bbox.as_dict(), frame_index=df.index,
        frame_time_ms=df.t_ms, frame_width=actual_w, frame_height=actual_h,
    )


@router.post("/{job_id}/process", status_code=status.HTTP_202_ACCEPTED)
async def process_video(job_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)) -> dict:
    """Queue processing. A participant must have been selected explicitly first."""
    job = _require_job(db, job_id)
    if job.status in ACTIVE_STATES:
        raise HTTPException(status_code=409, detail=f"Job is already {job.status}")
    if job.status == JobStatus.completed.value:
        raise HTTPException(status_code=409, detail="Job already completed")
    if job.status not in STARTABLE_STATES:
        raise HTTPException(status_code=409, detail=f"Job in state '{job.status}' cannot be processed")
    if None in (job.target_bbox_x, job.target_bbox_y, job.target_bbox_w, job.target_bbox_h):
        raise HTTPException(
            status_code=409,
            detail="No participant selected. Select the target participant (select-target) before processing.",
        )
    if not job.stored_path or not Path(job.stored_path).exists():
        raise HTTPException(
            status_code=409,
            detail="The raw video is no longer available (deleted per the retention policy); "
                   "re-upload to reprocess. Use /predict to re-run inference on stored crops.",
        )
    job.status = JobStatus.queued.value
    job.updated_at = datetime.now(timezone.utc)
    db.commit()
    background_tasks.add_task(run_processing_job, job_id)
    return {"job_id": job_id, "status": JobStatus.queued.value,
            "message": "Processing started. Poll /status for progress."}


@router.get("/{job_id}/status", response_model=JobResponse)
async def get_job_status(job_id: str, db: Session = Depends(get_db)) -> JobResponse:
    return job_to_response(_require_job(db, job_id))


@router.get("/{job_id}/result", response_model=JobResponse)
async def get_job_result(job_id: str, db: Session = Depends(get_db)):
    """Complete stored result of a finished job; 202 (with status) while still running."""
    job = _require_job(db, job_id)
    if job.status not in TERMINAL_STATES:
        return JSONResponse(
            status_code=202,
            content={"job_id": job_id, "status": job.status,
                     "detail": f"Job is not yet complete. Current status: {job.status}"},
        )
    return job_to_response(job)


@router.get("/{job_id}/tracking")
async def get_tracking_record(job_id: str, db: Session = Depends(get_db)) -> dict:
    """Per-sample tracking record (timestamps, boxes, status, crop file) for audit."""
    job = _require_job(db, job_id)
    path = job_dir_for(job_id) / TRACKING_FILE
    if not path.exists():
        raise HTTPException(status_code=404, detail="No tracking record exists for this job")
    return json.loads(path.read_text(encoding="utf-8"))


@router.post("/{job_id}/cancel", status_code=status.HTTP_200_OK)
async def cancel_job_endpoint(job_id: str, db: Session = Depends(get_db)) -> dict:
    job = _require_job(db, job_id)
    cancelled = cancel_job(db, job)
    return {"job_id": job_id, "cancelled": cancelled, "status": job.status}


@router.delete("/{job_id}", status_code=status.HTTP_200_OK)
async def delete_job(job_id: str, db: Session = Depends(get_db)) -> dict:
    job = _require_job(db, job_id)
    removed = delete_job_record(db, job)
    return {"job_id": job_id, "deleted": True, "artifacts_removed": removed}
