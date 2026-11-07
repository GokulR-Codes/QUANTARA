"""SQLAlchemy ORM models (DB tables)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer,
    JSON, String, Text, ForeignKey,
)
from sqlalchemy.orm import relationship

from backend.app.db import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class Job(Base):
    """Tracks a video processing job from upload to completion."""
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=_uuid)
    recording_id = Column(String, nullable=True, index=True)
    status = Column(String, nullable=False, default="uploaded")
    # statuses: uploaded | queued | processing | tracking |
    #           feature_extraction | inference | completed |
    #           completed_without_prediction | failed | cancelled
    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)
    completed_at = Column(DateTime, nullable=True)

    original_filename = Column(String, nullable=True)
    stored_path = Column(String, nullable=True)  # internal path only
    frames_dir = Column(String, nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    video_duration_s = Column(Float, nullable=True)
    video_fps = Column(Float, nullable=True)
    video_width = Column(Integer, nullable=True)
    video_height = Column(Integer, nullable=True)
    video_codec = Column(String, nullable=True)

    # Target selection — operator-supplied bounding box
    target_bbox_x = Column(Integer, nullable=True)
    target_bbox_y = Column(Integer, nullable=True)
    target_bbox_w = Column(Integer, nullable=True)
    target_bbox_h = Column(Integer, nullable=True)
    # Frame the operator drew the box on (source frame index + its timestamp)
    target_frame_index = Column(Integer, nullable=True)
    target_frame_time_ms = Column(Float, nullable=True)
    target_selected_at = Column(DateTime, nullable=True)

    # Processing progress / stage metadata (JSON)
    stage_progress = Column(JSON, nullable=True)

    # Result
    result = Column(JSON, nullable=True)

    # Errors & warnings
    error_message = Column(Text, nullable=True)
    warnings = Column(JSON, nullable=True)

    processing_duration_s = Column(Float, nullable=True)


class Experiment(Base):
    """Research experiment tracking."""
    __tablename__ = "experiments"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False)
    experiment_type = Column(String, nullable=False)  # classical | qaoa_vqc | federated
    status = Column(String, nullable=False, default="created")
    config = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=_now)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    results = Column(JSON, nullable=True)
    notes = Column(JSON, nullable=True)


class ModelRegistry(Base):
    """Saved model checkpoints and metadata."""
    __tablename__ = "model_registry"

    id = Column(String, primary_key=True, default=_uuid)
    model_type = Column(String, nullable=False)  # cnn_lstm | vqc | classical
    version = Column(String, nullable=False)
    checkpoint_path = Column(String, nullable=True)
    config = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=_now)
    is_active = Column(Boolean, default=False)
    feature_schema_version = Column(String, nullable=True)
    selected_feature_names = Column(JSON, nullable=True)
