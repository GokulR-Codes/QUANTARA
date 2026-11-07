"""Background video processing worker.

Pipeline (each stage's output is persisted; nothing is reported unless it actually ran):

    validate video + selection  ->  target tracking & timestamp-aligned crops
    ->  CNN -> LSTM -> fusion   ->  QAOA mapping -> VQC (only with a compatible bundle)
    ->  persist result          ->  retention cleanup (only now)

Cancellation is cooperative: every status write and the tracking loop re-check the DB, and a
cancelled job stops with its derived artifacts removed.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

from backend.app.core.errors import JobCancelled, VideoProcessingError, VideoValidationError
from backend.app.core.logging_config import get_logger
from backend.app.db import SessionLocal
from backend.app.models import Job
from backend.app.services.job_service import (
    is_cancelled, job_dir_for, remove_job_artifacts, update_job_status,
)
from backend.app.services.model_service import get_active_bundle_record
from src.config.settings import get_settings
from src.pipeline.features import FEATURES_FILE, FeatureExtractionError
from src.pipeline.runner import run_model_stage
from src.video.extractor import probe_video
from src.video.geometry import BBoxError, validate_bbox
from src.video.target_pipeline import CROPS_DIR, TRACKING_FILE, extract_target_crops
from src.video.tracker import TrackerParams

logger = get_logger(__name__)

# Test hook: a callable returning a tracker object (init/update). None -> real OpenCV tracker.
TRACKER_FACTORY: Optional[Callable] = None


def _fail(db, job: Optional[Job], message: str) -> None:
    """Mark the job failed and remove partial derived artifacts.

    The raw video is deliberately kept so the operator can retry; it is removed by an explicit
    DELETE of the job (see docs/data_retention.md).
    """
    if job is None:
        return
    logger.error("Job %s failed: %s", job.id, message)
    remove_job_artifacts(job.id)
    try:
        job.frames_dir = None
        update_job_status(db, job, "failed", error_message=message)
    except JobCancelled:
        pass


def run_processing_job(job_id: str) -> None:
    db = SessionLocal()
    job: Optional[Job] = None
    try:
        settings = get_settings()
        job = db.query(Job).filter(Job.id == job_id).first()
        if job is None:
            logger.error("Job %s not found in DB", job_id)
            return
        if job.status == "cancelled":
            logger.info("Job %s was cancelled before processing started", job_id)
            return

        t_start = time.monotonic()
        job_dir = job_dir_for(job_id)
        remove_job_artifacts(job_id)                      # (re)processing always starts clean
        job_dir.mkdir(parents=True, exist_ok=True)
        job.frames_dir = str(job_dir)
        job.error_message = None
        job.completed_at = None
        job.result = None
        stage: dict = {}
        warnings: list[str] = []
        db.commit()

        # ── Stage 1: validate raw video and the operator's selection ─────────────────
        update_job_status(db, job, "processing", stage_progress={"stage": "validating"})
        video_path = Path(job.stored_path) if job.stored_path else None
        if video_path is None or not video_path.exists():
            _fail(db, job, "Stored video file not found (it may have been deleted per the "
                           "retention policy); re-upload to reprocess.")
            return
        if None in (job.target_bbox_x, job.target_bbox_y, job.target_bbox_w, job.target_bbox_h):
            _fail(db, job, "No participant was selected for this job; select the target "
                           "participant before processing.")
            return
        try:
            meta = probe_video(video_path)
            bbox = validate_bbox(job.target_bbox_x, job.target_bbox_y, job.target_bbox_w,
                                 job.target_bbox_h, meta["width"], meta["height"], settings.min_box_px)
        except (VideoValidationError, BBoxError) as exc:
            _fail(db, job, f"Validation failed: {exc}")
            return
        job.video_duration_s, job.video_fps = meta["duration_s"], meta["fps"]
        job.video_width, job.video_height, job.video_codec = meta["width"], meta["height"], meta["codec"]
        warnings.extend(meta["warnings"])
        stage["video_metadata"] = {k: meta[k] for k in
                                   ("duration_s", "fps", "width", "height", "codec", "timestamp_source")}
        db.commit()

        # ── Stage 2: tracking + timestamp-aligned target crops ───────────────────────
        update_job_status(db, job, "tracking", stage_progress={**stage, "stage": "tracking"},
                          warnings=warnings)
        params = TrackerParams(
            min_similarity=settings.track_min_similarity,
            resume_similarity=settings.track_resume_similarity,
            max_jump_ratio=settings.track_max_jump_ratio,
            max_area_ratio=settings.track_max_area_ratio,
            min_box_px=settings.min_box_px,
            allow_reacquisition=settings.track_allow_reacquisition,
        )
        tracking = extract_target_crops(
            video_path, job_dir, bbox, job.target_frame_index or 0,
            sample_fps=settings.frame_sample_fps, tracking_fps=settings.tracking_fps,
            max_samples=settings.max_frames_per_video,
            crop_size=(settings.frame_height, settings.frame_width),
            context_pad=settings.crop_context_pad, params=params,
            tracker_factory=TRACKER_FACTORY,
            should_cancel=lambda: is_cancelled(job_id),
            video_duration_s=meta["duration_s"],
        )
        ts = tracking.summary
        if ts["intervals"]:
            warnings.append(
                f"Target not trusted for {ts['frames_invalid']} of {ts['samples_total']} sampled "
                f"frames ({len(ts['intervals'])} invalid interval(s)); no crops were generated for them."
            )
        if ts["truncated"]:
            warnings.append(f"Only the first {settings.max_frames_per_video} sampled frames were processed.")
        seq_len = settings.sequence_length
        n_crops = ts["crops_generated"]
        stage["tracking"] = ts
        stage["frame_processing"] = {
            "frames_decoded": ts["frames_decoded"],
            "frames_sampled": ts["samples_total"],
            "frames_accepted": n_crops,
            "frames_rejected": ts["frames_invalid"],
            "frames_used": min(n_crops, seq_len),
            "padding_applied": max(0, seq_len - n_crops) if n_crops else 0,
            "truncation_applied": bool(ts["truncated"]),
            "subsampled": n_crops > seq_len,
        }
        update_job_status(db, job, "tracking", stage_progress={**stage, "stage": "tracking_done"},
                          warnings=warnings)

        # ── Stages 3-4: CNN -> LSTM -> fusion, then QAOA mapping -> VQC if possible ──
        record = get_active_bundle_record(db)
        bundle_dir = Path(record.checkpoint_path) if record and record.checkpoint_path else None
        outcome = run_model_stage(
            job_dir, settings, bundle_dir,
            on_stage=lambda st: update_job_status(db, job, st, stage_progress={**stage, "stage": st}),
        )
        warnings.extend(w for w in outcome.warnings if w not in warnings)
        stage["feature_extraction"] = {"status": outcome.feature_extraction_status, **outcome.feature_info}

        artifacts = {
            "job_dir": str(job_dir),
            "tracking": str(job_dir / TRACKING_FILE),
            "crops_dir": str(job_dir / CROPS_DIR),
            "raw_video": "retained",
        }
        if (job_dir / FEATURES_FILE).exists():
            artifacts["features"] = str(job_dir / FEATURES_FILE)

        result = {
            "feature_extraction_status": outcome.feature_extraction_status,
            "feature_extraction": outcome.feature_info,
            "model_status": outcome.model_status,
            "model_status_detail": outcome.model_status_detail,
            "pipeline_stages": {
                "tracking": "completed",
                "feature_extraction": outcome.feature_extraction_status,
                "quantum_inference": outcome.model_status,
            },
            "artifact_references": artifacts,
        }
        if outcome.predicted:
            result["prediction"] = outcome.prediction
            result["active_model_id"] = record.id if record else None
            result.update({k: v for k, v in outcome.model_meta.items() if k != "mapped_input"})
            result["model_input_features"] = outcome.model_meta.get("mapped_input")

        job.processing_duration_s = time.monotonic() - t_start
        final = "completed" if outcome.predicted else "completed_without_prediction"

        # Persist the finished result FIRST ...
        update_job_status(db, job, final, stage_progress={**stage, "stage": "complete"},
                          result=result, warnings=warnings)

        # ── Stage 5: retention cleanup — only now that every stage that reads the raw
        #    video (tracking, crops, features, inference) has completed and is persisted.
        if not settings.retain_raw_video:
            try:
                video_path.unlink()
                job.stored_path = None
                artifacts["raw_video"] = "deleted_per_retention_policy"
            except OSError as exc:
                logger.warning("Could not delete raw video %s: %s", video_path, exc)
                artifacts["raw_video"] = f"retained (delete failed: {exc})"
            job.result = {**result, "artifact_references": artifacts}
            db.commit()
        logger.info("Job %s %s in %.2fs: tracked=%d/%d model_status=%s", job_id, final,
                    job.processing_duration_s, ts["frames_tracked"], ts["samples_total"],
                    outcome.model_status)

    except JobCancelled:
        logger.info("Job %s cancelled; removing derived artifacts (raw video kept)", job_id)
        remove_job_artifacts(job_id)
        if job is not None:
            try:
                job.frames_dir = None
                db.commit()
            except Exception:
                db.rollback()
    except (VideoValidationError, VideoProcessingError, FeatureExtractionError) as exc:
        db.rollback()
        _fail(db, job, f"{type(exc).__name__}: {getattr(exc, 'message', None) or exc}")
    except Exception as exc:
        logger.error("Unexpected error in job %s: %s", job_id, exc, exc_info=True)
        db.rollback()
        _fail(db, job, f"Unexpected processing error: {exc}")
    finally:
        db.close()
