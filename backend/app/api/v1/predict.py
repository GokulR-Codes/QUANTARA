"""Prediction endpoint — runs the quantum inference pipeline."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.core.errors import AppError
from backend.app.db import get_db
from backend.app.schemas.schemas import PredictRequest, JobResponse
from backend.app.services.job_service import get_job, job_to_response
from backend.app.services.inference_service import run_inference
from backend.app.core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["predict"])


@router.post("/predict", response_model=JobResponse)
async def predict(
    req: PredictRequest,
    db: Session = Depends(get_db),
) -> JobResponse:
    """
    Run the quantum inference pipeline on a processed job.

    Requires:
    - The job must have finished processing (target crops are read from disk; the raw video
      is not needed).
    - A compatible pipeline bundle (upstream weights + QAOA mapping + VQC checkpoint).

    Returns:
    - Updated job result with prediction (if model is available).
    - Or status=completed_without_prediction with explicit model_status if not.

    This endpoint NEVER returns a fake prediction.
    """
    job = get_job(db, req.job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {req.job_id} not found")

    if job.status == "cancelled":
        raise HTTPException(status_code=409, detail="Job was cancelled; inference is not available.")
    completed_statuses = (
        "completed", "completed_without_prediction", "failed"
    )
    if job.status not in completed_statuses:
        raise HTTPException(
            status_code=409,
            detail=f"Job has not finished processing. Current status: {job.status}. "
                   "Wait for processing to complete before requesting inference."
        )

    if job.status == "failed":
        raise HTTPException(
            status_code=422,
            detail="Job failed during processing; inference is not available."
        )

    try:
        run_inference(db, job, model_id=req.model_id)
    except AppError:
        raise
    except Exception as exc:
        logger.error("Inference failed for job %s: %s", req.job_id, exc, exc_info=True)
        raise HTTPException(status_code=503, detail=f"Inference failed: {exc}")

    db.refresh(job)
    return job_to_response(job)
