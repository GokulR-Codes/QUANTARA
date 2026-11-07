# Architecture Overview

## System Layers

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Browser / Frontend                           │
│  React 19 + TypeScript + Vite                                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌──────────┐ │
│  │ Dashboard│ │ Camera   │ │ Upload   │ │ Results   │ │Experiments│ │
│  └──────────┘ └──────────┘ └──────────┘ └───────────┘ └──────────┘ │
│  useCamera (MediaRecorder) · useJob (polling) · axios API client     │
└──────────────────────────────────────┬───────────────────────────────┘
                                       │ HTTP/REST (localhost:8000)
                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                               │
│  Python 3.12 · Uvicorn ASGI · SQLAlchemy ORM · Pydantic v2          │
│                                                                      │
│  Routers:                                                            │
│    /api/v1/health        → liveness + readiness                     │
│    /api/v1/videos        → upload, select-target, process, status   │
│    /api/v1/experiments   → create, poll, results                    │
│    /api/v1/models        → registry, federated status               │
│    /api/v1/datasets      → manifest validation                      │
│    /api/v1/predict       → direct inference endpoint                │
│                                                                      │
│  Services:  VideoService · JobService · InferenceService            │
│             ExperimentService · ModelService                         │
│  Worker:    BackgroundThread (processing_worker.py)                 │
│  DB:        SQLite (artifacts/app.db)                               │
└──────────────────────────────────────┬───────────────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                      ▼
      ┌─────────────────────────┐          ┌──────────────────────────┐
      │   src/video/            │          │  src/models/             │
      │   extractor.py          │          │  cnn/extractor.py        │
      │   tracker.py            │          │  lstm/temporal.py        │
      │   target_pipeline.py    │          │  fusion/fusion.py        │
      │   geometry.py           │          │  quantum/qaoa.py         │
      └─────────────────────────┘          │  quantum/vqc.py          │
                                           │  baselines/classical.py  │
                                           └──────────────────────────┘
                                                        │
                                           ┌────────────┴────────────┐
                                           │    src/federated/       │
                                           │    fedavg.py            │
                                           └─────────────────────────┘
```

## Request Lifecycle — Video Processing

```
POST /api/v1/videos/upload
  └─► Validate file (container, codec, duration, size)
  └─► Probe true metadata (decode-scan for WebM/live mode)
  └─► Store to data/uploads/{job_id}.{ext}
  └─► Create Job record (status=uploaded)
  └─► Return 202 with job_id

POST /api/v1/videos/{job_id}/select-target
  └─► Validate bbox against actual frame dimensions
  └─► Scale display coordinates to source coordinates
  └─► Store selection (target_frame_index, bbox)
  └─► Update Job (status=target_selected)

POST /api/v1/videos/{job_id}/process
  └─► Spawn background thread → processing_worker.py
  └─► Return 202

[Background Worker]
  └─► Stage: frame_extraction
      └─► FrameReader: decode all frames, extract at sample_fps timestamps
      └─► Store metadata (fps, duration, frame_count)
  └─► Stage: tracking
      └─► Initialize CSRT tracker at operator-selected frame
      └─► Run at tracking_fps; classify: TRACKED / AMBIGUOUS / LOST / OUT_OF_FRAME
      └─► Save crops for TRACKED frames only
  └─► Stage: feature_extraction
      └─► CNN: 112×112 crops → 256-d embeddings
      └─► LSTM: sequence of embeddings → 128-d representation
      └─► Fusion: cross-attention → 64-d fused vector
  └─► Stage: inference
      └─► Load active pipeline bundle (if available)
      └─► QAOA: select features from 64-d vector
      └─► VQC: classify → risk_level {low, moderate, high}
      └─► Store result + per-frame tracking doc
  └─► Update Job (status=completed | completed_without_prediction | failed)

GET /api/v1/videos/{job_id}/result    → full inference result
GET /api/v1/videos/{job_id}/tracking  → per-frame tracking document
```

## Database Schema

```sql
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,           -- UUID job identifier
    status TEXT,                   -- uploaded | target_selected | processing | completed | ...
    video_path TEXT,               -- path to uploaded video
    video_metadata TEXT,           -- JSON: fps, duration, size, frame_count
    target_frame_index INTEGER,    -- operator-selected frame index
    target_bbox TEXT,              -- JSON: {x, y, w, h}
    result TEXT,                   -- JSON: full processing result
    warnings TEXT,                 -- JSON array of warning strings
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    target_selected_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE experiments (
    id TEXT PRIMARY KEY,           -- UUID experiment identifier
    name TEXT,
    experiment_type TEXT,          -- classical | qaoa_vqc | federated
    status TEXT,                   -- pending | running | completed | failed
    config TEXT,                   -- JSON: training configuration
    results TEXT,                  -- JSON: metrics, accuracy, parameters
    created_at TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE model_registry (
    id INTEGER PRIMARY KEY,
    name TEXT,
    bundle_path TEXT,              -- path to pipeline bundle directory
    is_active BOOLEAN,             -- only one active at a time
    is_synthetic BOOLEAN,          -- True if trained on synthetic data
    metadata TEXT,                 -- JSON: architecture, QAOA dims, etc.
    registered_at TIMESTAMP
);
```

## Configuration Flow

```
.env file
  ↓
src/config/settings.py (Pydantic BaseSettings)
  ↓ injected via Depends(get_settings)
FastAPI routers + workers
```

Key settings: `MAX_UPLOAD_SIZE_MB`, `MAX_VIDEO_DURATION_SECONDS`, `FRAME_SAMPLE_FPS`, `TRACKING_FPS`, `SEQUENCE_LENGTH`, `RETAIN_RAW_VIDEO`, `ALLOW_SYNTHETIC_MODEL_INFERENCE`.
