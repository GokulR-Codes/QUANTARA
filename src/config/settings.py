"""Typed application configuration loaded from environment variables."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_DEFAULT_SECRET = "dev-secret-key-change-in-production"


class Settings(BaseSettings):
    """All application configuration. Loaded from .env or environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_env: Literal["development", "production", "test"] = "development"
    secret_key: str = _DEFAULT_SECRET
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # API
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    frontend_url: str = "http://localhost:5173"

    # Database
    database_url: str = "sqlite:///./artifacts/app.db"

    # Storage
    upload_dir: str = "data/uploads"
    frames_dir: str = "data/frames"
    artifacts_dir: str = "artifacts"
    max_upload_size_mb: int = 500
    max_video_duration_seconds: int = 600
    retain_raw_video: bool = False

    # Video processing
    frame_sample_fps: float = 2.0
    max_frames_per_video: int = 150
    frame_height: int = 112
    frame_width: int = 112
    sequence_length: int = 30
    accepted_video_formats: list[str] = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

    # Target tracking / crops
    # Tracker runs at (at least) tracking_fps; every sampled frame is also tracked.
    tracking_fps: float = 10.0
    # Minimum number of successfully tracked crops needed to attempt feature extraction.
    min_tracked_frames: int = 5
    # If False (default) a target reported lost is NOT re-acquired automatically:
    # the remainder of the video is marked invalid rather than risk following someone else.
    track_allow_reacquisition: bool = False
    # HSV-histogram correlation against the operator-selected template.
    track_min_similarity: float = 0.30      # below this -> ambiguous
    track_resume_similarity: float = 0.45   # must exceed this to resume after ambiguity
    # Plausibility limits between consecutive tracker steps.
    track_max_jump_ratio: float = 1.0       # centre displacement / box diagonal
    track_max_area_ratio: float = 3.0       # box area change factor
    min_box_px: int = 8                     # smallest accepted box side (pixels)
    crop_context_pad: float = 0.10          # context padding around the box (fraction)

    # CNN
    cnn_backbone: str = "simple_cnn"
    cnn_feature_dim: int = 256
    cnn_pretrained: bool = False

    # LSTM
    lstm_hidden_dim: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.3

    # Fusion
    fusion_output_dim: int = 64

    # Quantum
    quantum_n_qubits: int = 4
    qaoa_depth: int = 2
    vqc_depth: int = 3
    quantum_backend: str = "default.qubit"
    quantum_shots: int = 1024
    quantum_analytic: bool = True

    # Training
    batch_size: int = 8
    learning_rate: float = 0.001
    num_epochs: int = 10
    random_seed: int = 42

    # Federated
    fed_rounds: int = 5
    fed_clients: int = 3
    fed_local_epochs: int = 2

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"

    # Labels
    class_names: list[str] = ["Low", "Moderate", "High"]
    num_classes: int = 3

    # Model-use policy
    # Checkpoints trained on synthetic/test data are refused for real-video inference
    # unless this is explicitly enabled (used by the automated integration tests only).
    allow_synthetic_model_inference: bool = False

    @field_validator("quantum_n_qubits")
    @classmethod
    def validate_qubits(cls, v: int) -> int:
        if v < 2 or v > 20:
            raise ValueError(f"quantum_n_qubits must be between 2 and 20, got {v}")
        return v

    @field_validator("sequence_length")
    @classmethod
    def validate_sequence_length(cls, v: int) -> int:
        if v < 1 or v > 500:
            raise ValueError(f"sequence_length must be between 1 and 500, got {v}")
        return v

    @model_validator(mode="after")
    def validate_consistency(self) -> "Settings":
        if len(self.class_names) != self.num_classes:
            raise ValueError(
                f"num_classes ({self.num_classes}) does not match "
                f"len(class_names) ({len(self.class_names)})"
            )
        if self.app_env == "production" and (
            self.secret_key == _DEFAULT_SECRET or len(self.secret_key) < 32
        ):
            raise ValueError(
                "SECRET_KEY must be set to a random value of >= 32 characters "
                "when APP_ENV=production"
            )
        return self

    @model_validator(mode="after")
    def validate_paths(self) -> "Settings":
        for path_attr in ["upload_dir", "frames_dir", "artifacts_dir"]:
            path = Path(getattr(self, path_attr))
            path.mkdir(parents=True, exist_ok=True)
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def models_dir(self) -> Path:
        return Path(self.artifacts_dir) / "models"

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
