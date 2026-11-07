"""Pydantic schemas for API request/response validation."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    uploaded = "uploaded"
    queued = "queued"
    processing = "processing"
    tracking = "tracking"
    feature_extraction = "feature_extraction"
    inference = "inference"
    completed = "completed"
    completed_without_prediction = "completed_without_prediction"
    failed = "failed"
    cancelled = "cancelled"


class ExperimentType(str, Enum):
    classical = "classical"
    qaoa_vqc = "qaoa_vqc"
    federated = "federated"


class ExperimentStatus(str, Enum):
    created = "created"
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


# ---------------------------------------------------------------------------
# Video / Job schemas
# ---------------------------------------------------------------------------

class VideoMetadata(BaseModel):
    duration_s: Optional[float] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    codec: Optional[str] = None
    file_size_bytes: Optional[int] = None
    original_filename: Optional[str] = None


class FrameProcessingSummary(BaseModel):
    """Counts for the sampled-frame stage (one sample per 1/sample_fps slot)."""
    frames_decoded: int = 0       # every frame decoded from the video up to the stop point
    frames_sampled: int = 0       # sampled slots (each gets a tracking record)
    frames_accepted: int = 0      # samples where the target was trusted -> crop produced
    frames_rejected: int = 0      # samples with no trusted target (lost/ambiguous/out of frame)
    frames_used: int = 0          # real crops fed to the model (<= sequence_length)
    padding_applied: int = 0      # mask-only padding positions (never real frames)
    truncation_applied: bool = False
    subsampled: bool = False      # more crops than sequence_length -> evenly subsampled


class TargetTrackingSummary(BaseModel):
    tracking_enabled: bool = False
    initial_bbox: Optional[dict] = None
    selection: Optional[dict] = None           # {frame_index, time_ms, bbox}
    tracker_backend: Optional[str] = None      # the tracker that actually ran (CSRT/KCF/MIL)
    timestamp_source: Optional[str] = None
    samples_total: int = 0
    frames_tracked: int = 0
    frames_lost: int = 0
    frames_ambiguous: int = 0
    frames_out_of_frame: int = 0
    frames_invalid: int = 0
    tracking_coverage_pct: Optional[float] = None
    tracking_failures: int = 0                 # lost / out-of-frame events
    ambiguity_events: int = 0
    resumed_after_ambiguity: int = 0
    first_loss_ms: Optional[float] = None
    crops_generated: int = 0
    intervals: list[dict] = []                 # invalid intervals (lost / ambiguous / out_of_frame)
    reacquisition_enabled: bool = False
    truncated: bool = False
    tracking_error: Optional[str] = None


class PredictionResult(BaseModel):
    predicted_class: int
    predicted_label: str
    class_probabilities: Optional[list[float]] = None
    synthetic_model: bool = False
    disclaimer: str = (
        "This is an experimental research prototype. "
        "Predictions are NOT clinical diagnoses and must not be used as such."
    )


class JobResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    job_id: str
    recording_id: Optional[str] = None
    status: JobStatus
    created_at: datetime
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    video_metadata: Optional[VideoMetadata] = None
    frame_processing_summary: Optional[FrameProcessingSummary] = None
    target_tracking_summary: Optional[TargetTrackingSummary] = None
    target_selection: Optional[dict] = None
    feature_extraction_status: Optional[str] = None
    feature_extraction: Optional[dict] = None
    model_status: Optional[str] = None
    model_status_detail: Optional[str] = None
    model_provenance: Optional[dict] = None
    active_model_id: Optional[str] = None
    model_version: Optional[str] = None
    feature_schema_version: Optional[str] = None
    selected_feature_names: Optional[list[str]] = None
    quantum_configuration: Optional[dict] = None
    prediction: Optional[PredictionResult] = None
    experiment_metrics: Optional[dict] = None
    warnings: Optional[list[str]] = None
    errors: Optional[str] = None
    artifact_references: Optional[dict] = None
    processing_duration_s: Optional[float] = None


class UploadResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str
    original_filename: str
    file_size_bytes: int


class TargetSelectionRequest(BaseModel):
    """Operator-drawn participant box, in SOURCE-frame pixel coordinates.

    ``frame_width``/``frame_height`` are the dimensions of the image the operator drew on and
    must equal the video's real dimensions (checked server-side); ``frame_index`` is the
    source frame shown to the operator (as returned by the preview-frame endpoint).
    """
    job_id: Optional[str] = None
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    frame_width: int
    frame_height: int
    frame_index: int = Field(0, ge=0)


class TargetSelectionResponse(BaseModel):
    job_id: str
    target_set: bool
    bbox: dict
    frame_index: int
    frame_time_ms: float
    frame_width: int
    frame_height: int


# ---------------------------------------------------------------------------
# Health / Readiness
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "1.0.0"


class ComponentStatus(BaseModel):
    name: str
    status: str  # ok | degraded | unavailable
    detail: Optional[str] = None


class ReadinessResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    ready: bool
    api_operational: bool
    database_operational: bool
    model_available: bool
    quantum_available: bool
    components: list[ComponentStatus] = []
    message: str


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    model_config = {"protected_namespaces": ()}
    id: str
    model_type: str
    version: str
    is_active: bool
    metrics: Optional[dict] = None
    created_at: datetime
    feature_schema_version: Optional[str] = None
    selected_feature_names: Optional[list[str]] = None
    # Whether this record can serve real-video inference, and why not if it cannot.
    usable_for_video_inference: bool = False
    status_detail: Optional[str] = None
    synthetic: Optional[bool] = None


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    job_id: str
    model_id: Optional[str] = None  # if None, use active model


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

class ExperimentCreateRequest(BaseModel):
    name: str
    experiment_type: ExperimentType
    config: Optional[dict] = None


class ExperimentResponse(BaseModel):
    experiment_id: str
    name: str
    experiment_type: ExperimentType
    status: ExperimentStatus
    config: Optional[dict] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: Optional[dict] = None
    notes: Optional[list[str]] = None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DatasetValidationResult(BaseModel):
    valid: bool
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    missing_videos: int = 0
    class_distribution: Optional[dict] = None
    errors: list[str] = []
    warnings: list[str] = []


class DatasetStatusResponse(BaseModel):
    dataset_available: bool
    total_samples: int = 0
    class_distribution: Optional[dict] = None
    manifest_path: Optional[str] = None
    message: str
