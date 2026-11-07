# Quantum-Centric Federated Learning for Screen-Time-Associated Behavioural Risk Assessment in Children

### A QAOA–VQC Framework

---

## Overview

This project implements a research pipeline that combines classical computer vision, temporal modelling, quantum machine learning, and federated learning for behavioural risk assessment in children based on screen-time-associated behavioural cues observed in video recordings.

> **Research context only.** No real children's data is used, collected, or inferred. All experiments run on synthetic data. Predictions are only produced when a real trained pipeline bundle is registered. The system never invents or fabricates model outputs.

---

## Architecture

```
Browser Camera
      │
      ▼ (WebRTC / MediaRecorder)
┌─────────────────────────────────────────────────────────────┐
│                  React + TypeScript Frontend                │
│   Camera Page → Participant Selection → Results Display    │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP / REST
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 FastAPI Backend (Python 3.12)               │
│                                                             │
│  /api/v1/videos      → Upload, Select-Target, Process      │
│  /api/v1/experiments → Run QAOA+VQC / Classical / Federated │
│  /api/v1/models      → Registry + Federated Status         │
│  /api/v1/health      → Liveness + Readiness                │
└───────┬────────────────────────────────────────────────────-┘
        │
        ▼ (Background Worker)
┌─────────────────────────────────────────────────────────────┐
│               Video Processing Pipeline                     │
│                                                             │
│  Frame Extraction  →  CSRT Tracker  →  Target Crops        │
│  CNN (ResNet-like) →  LSTM (128-d)  →  Feature Fusion      │
│  QAOA Feature Selection → VQC Classification               │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
SQLite (artifacts/app.db)  +  Frame store (data/frames/)
```

### Key Components

| Layer | Module | Description |
|---|---|---|
| Video Extraction | `src/video/extractor.py` | Timestamp-based frame sampling, WebM-safe probing |
| Participant Tracking | `src/video/tracker.py` | OpenCV CSRT tracker with appearance/motion guards |
| CNN Features | `src/models/cnn/extractor.py` | Spatial feature extraction (256-d embeddings) |
| LSTM Temporal | `src/models/lstm/temporal.py` | Temporal modelling (128-d sequences) |
| Feature Fusion | `src/models/fusion/fusion.py` | Cross-attention fusion (64-d vectors) |
| QAOA Selection | `src/models/quantum/qaoa.py` | Variational quantum eigensolver for feature selection |
| VQC Classification | `src/models/quantum/vqc.py` | Variational Quantum Classifier (PennyLane) |
| Federated Learning | `src/federated/fedavg.py` | Federated Averaging across simulated clients |

---

## Requirements

- **Python** 3.12
- **Node.js** ≥ 20, npm ≥ 10
- **OS**: Windows 11 / Linux (tested on both)
- **GPU**: Not required. CPU-only PyTorch is used by default.
- **Quantum hardware**: Not required. PennyLane's `lightning.qubit` CPU simulator is used.

### Python Dependencies

```
torch==2.4.1+cpu
pennylane==0.38.0
pennylane-lightning==0.38.0
autoray==0.6.12          # critical: 0.11.x breaks PennyLane 0.38
opencv-contrib-python-headless==4.10.0.84
fastapi==0.115.4
uvicorn==0.32.0
sqlalchemy==2.0.36
pydantic==2.9.2
numpy==1.26.4
scikit-learn==1.5.2
```

---

## Quick Start

### 1. Install dependencies

```powershell
# Clone / extract the project
Set-Location "d:\Final\quantum_federated_behaviour_ai"

# Python dependencies
pip install -r requirements.txt

# Frontend dependencies
Set-Location frontend
npm install
Set-Location ..
```

### 2. Verify environment

```powershell
python scripts\check_environment.py
```

Expected output shows: Python 3.12, torch (CPU), pennylane 0.38.0, CSRT tracker available.

### 3. Start the backend

```powershell
# From project root
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Visit: http://127.0.0.1:8000/api/v1/health

### 4. Start the frontend

```powershell
Set-Location frontend
npm run dev
```

Visit: http://localhost:5173

---

## Camera Workflow

The application supports a complete operator-driven recording workflow:

1. **Camera Page** — Click **Start Camera** to request permission via `navigator.mediaDevices.getUserMedia`
2. **Live Preview** — Video preview shown; camera-active indicator visible
3. **Start Recording** — Click to begin. Elapsed timer shown. Recording stored via `MediaRecorder`.
4. **Stop Recording** — Click to end. Preview of the recorded clip shown.
5. **Retake** — Discard and re-record without sending to backend.
6. **Save & Process** — Sends the recording to `/api/v1/videos/upload`.
7. **Select Participant** — Draw a bounding box around the target child on a video frame preview.
8. **Process** — Background worker runs: tracking → CNN → LSTM → Fusion → (QAOA+VQC if model available).
9. **Results** — Retrieved from `/api/v1/videos/{job_id}/result`.

> **Upload fallback**: An Upload Page allows processing pre-recorded video files (MP4, WebM, AVI, MOV) without using the camera.

---

## Processing States

The system distinguishes these states explicitly — they are never interchangeable:

| State | Meaning |
|---|---|
| `uploaded` | Video received and validated |
| `target_selected` | Operator drew bounding box |
| `processing` | Background worker running |
| `completed_without_prediction` | Feature extraction succeeded; no trained model registered |
| `completed` | Full pipeline including VQC classification succeeded |
| `failed` | Error during processing |
| `cancelled` | Operator cancelled the job |

A `model_not_ready` result **never** produces a fake prediction. It reports honestly that no trained checkpoint is available.

---

## Quantum Components

### QAOA Feature Selection (`src/models/quantum/qaoa.py`)

Uses a Quantum Approximate Optimization Algorithm to solve a QUBO (Quadratic Unconstrained Binary Optimization) problem that selects the most relevant features from the 64-d fused representation.

- Hamiltonian: Ising-form with exact coefficients derived from feature correlations
- Backend: PennyLane `lightning.qubit` CPU simulator
- Output: Binary selection mask over feature dimensions

### VQC Classification (`src/models/quantum/vqc.py`)

A Variational Quantum Classifier trained to distinguish behavioural risk levels.

- Ansatz: Layer of Ry rotations + CNOT entanglement per layer
- Measurement: Pauli-Z expectation value → binary classification
- Optimization: Classical gradient via PennyLane's parameter-shift rule

> **No trained model is shipped.** To produce real predictions, train on a labelled dataset and register the bundle using `scripts/register_bundle.py`.

---

## Federated Learning

The federated learning module (`src/federated/fedavg.py`) implements Federated Averaging (FedAvg) across simulated clients. In the current demo mode, synthetic data is distributed across N clients; real federated deployment would require separate client processes.

Run via the Experiments page: **Experiment Type → Federated Learning**.

---

## Running Experiments

### Via Frontend
Navigate to **Experiments** → Select type → **Create Experiment**.

### Via API

```bash
# Create a QAOA+VQC experiment (uses synthetic data)
curl -X POST http://127.0.0.1:8000/api/v1/experiments \
  -H "Content-Type: application/json" \
  -d '{"name": "qaoa_test", "experiment_type": "qaoa_vqc", "config": {"epochs": 5}}'

# Poll for status
curl http://127.0.0.1:8000/api/v1/experiments/{experiment_id}/status

# Get results
curl http://127.0.0.1:8000/api/v1/experiments/{experiment_id}/results
```

> **Important:** Experiments run on **synthetic data**. Results are clearly labelled `synthetic: true` and the trained checkpoint is never used for real video inference unless `ALLOW_SYNTHETIC_MODEL_INFERENCE=true` is set in `.env`.

---

## Tests

```powershell
# Full suite (209 tests)
python -m pytest tests/ -v --tb=short

# Unit tests only
python -m pytest tests/unit/ -v

# Integration tests only
python -m pytest tests/integration/ -v

# With coverage
python -m pytest tests/ --cov=src --cov=backend --cov-report=term-missing
```

**Expected: 209 passed, 0 failed.**

---

## Project Structure

```
quantum_federated_behaviour_ai/
├── backend/
│   └── app/
│       ├── api/v1/          # FastAPI routers
│       ├── services/        # Business logic
│       ├── workers/         # Background processing worker
│       ├── core/            # Error handling, logging
│       ├── db.py            # SQLAlchemy engine
│       ├── models.py        # ORM tables
│       ├── schemas/         # Pydantic request/response models
│       └── main.py          # FastAPI app entry point
├── frontend/
│   └── src/
│       ├── pages/           # React page components
│       ├── hooks/           # useCamera, useJob
│       ├── api/             # Axios API client
│       └── types/           # TypeScript types
├── src/
│   ├── config/settings.py   # Pydantic settings
│   ├── data/dataset.py      # Dataset loader
│   ├── federated/fedavg.py  # Federated Averaging
│   ├── models/
│   │   ├── cnn/             # CNN feature extractor
│   │   ├── lstm/            # LSTM temporal model
│   │   ├── fusion/          # Feature fusion
│   │   ├── quantum/         # QAOA + VQC
│   │   └── baselines/       # Classical MLP baseline
│   ├── pipeline/            # Bundle save/load, inference pipeline
│   ├── preprocessing/       # Normalizer
│   ├── training/trainer.py  # Training loop
│   ├── evaluation/metrics.py
│   └── video/               # Extractor, tracker, geometry, pipeline
├── tests/
│   ├── unit/                # Unit tests
│   ├── integration/         # API + end-to-end tests
│   ├── fixtures/            # Synthetic test data
│   └── helpers/             # Test utilities
├── scripts/
│   ├── check_environment.py
│   ├── create_test_fixtures.py
│   ├── register_bundle.py
│   ├── package_delivery.py
│   ├── run_backend.ps1
│   └── run_frontend.ps1
├── data/
│   ├── uploads/             # Uploaded video files (auto-cleaned)
│   ├── frames/              # Extracted frames per job
│   └── manifests/           # Dataset manifest template
├── artifacts/
│   ├── app.db               # SQLite database
│   └── checkpoints/         # Trained model bundles
├── conftest.py
├── pyproject.toml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Configuration

Copy `.env.example` to `.env` and edit as needed:

```env
APP_ENV=development
DATABASE_URL=sqlite:///./artifacts/app.db
UPLOAD_DIR=data/uploads
FRAMES_DIR=data/frames
ARTIFACTS_DIR=artifacts
LOG_FILE=logs/app.log

# Video limits
MAX_UPLOAD_SIZE_MB=500
MAX_VIDEO_DURATION_SECONDS=600

# Processing
FRAME_SAMPLE_FPS=2.0
TRACKING_FPS=10.0
SEQUENCE_LENGTH=32
FRAME_WIDTH=112
FRAME_HEIGHT=112

# Safety
RETAIN_RAW_VIDEO=false
ALLOW_SYNTHETIC_MODEL_INFERENCE=false
```

---

## Known Limitations

| Limitation | Impact | Workaround |
|---|---|---|
| No real labelled dataset | No trained model; VQC outputs synthetic labels only | Provide labelled video dataset and train |
| No trained CNN/LSTM weights | CNN uses random init; features are not meaningful for real classification | Train on labelled dataset |
| CSRT tracker box shrinks on flat colours | Bounding box accuracy degrades (~IoU 0.35 on solid colours) | Works better on real textured clothing |
| Appearance guard is HSV-histogram heuristic | Cannot guarantee identity if two people have similar clothing | Operator must verify target visually |
| No real browser camera test | useCamera/MediaRecorder was never run in an actual browser in CI | Test manually on localhost |
| No authentication | API is localhost-only | Do not expose beyond localhost |

---

## Privacy and Safety

- **No facial recognition** — the system does not detect, identify, or age-estimate any person
- **Operator-selected target** — the operator draws a bounding box to select the participant
- **Synthetic data only** — all experiments use programmatically generated data, not real children's data
- **Raw video deletion** — uploaded video is deleted after frame extraction by default (`RETAIN_RAW_VIDEO=false`)
- **Honest predictions** — the system never invents a prediction when no trained model is registered

---

## License

MIT — See `LICENSE` file.

---

## Citation

If you use this framework in academic work:

```
@article{qcfl2026,
  title={Quantum-Centric Federated Learning for Screen-Time-Associated Behavioural Risk Assessment in Children: A QAOA-VQC Framework},
  year={2026}
}
```
