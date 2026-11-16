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
│   Auth → Guardian Dashboard → Child Profile → Assessment   │
│   Camera Page → Participant Selection → Results Display    │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP / REST (JWT Bearer Auth)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 FastAPI Backend (Python 3.12)               │
│                                                             │
│  /api/v1/auth       → Register, Login, OTP, Password Reset │
│  /api/v1/children   → Child profiles (age categories)      │
│  /api/v1/assessments→ Assessment sessions (child+job+age)  │
│  /api/v1/videos     → Upload, Select-Target, Process       │
│  /api/v1/experiments→ Run QAOA+VQC / Classical / Federated │
│  /api/v1/models     → Registry + Age-Group Status          │
│  /api/v1/health     → Liveness + Readiness                 │
└───────┬────────────────────────────────────────────────────-┘
        │
        ▼ (Background Worker)
┌─────────────────────────────────────────────────────────────┐
│               Video Processing Pipeline                     │
│                                                             │
│  Frame Extraction  →  CSRT Tracker  →  Target Crops        │
│  CNN (ResNet-like) →  LSTM (128-d)  →  Feature Fusion      │
│  QAOA Feature Selection → VQC Classification               │
│  (age-specific configs per group: 5–7, 7–10, 10–12)       │
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
Set-Location "D:\Final\QCFL_project_delivery\quantum_federated_behaviour_ai"

# Python dependencies (use the venv)
.\.venv\Scripts\pip.exe install -r requirements.txt

# Frontend dependencies
Set-Location frontend
npm install
Set-Location ..
```

### 2. Verify environment

```powershell
.\.venv\Scripts\python.exe scripts\check_environment.py
```

Expected output shows: Python 3.12, torch (CPU), pennylane 0.38.0, CSRT tracker available.

### 3. Create first admin account (one-time)

```powershell
.\.venv\Scripts\python.exe scripts\create_admin.py --email admin@qcare.local --username admin --password "YourSecurePass1!"
```

### 4. Start the backend

```powershell
# From project root
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Visit: http://127.0.0.1:8000/api/v1/health

### 5. Start the frontend

```powershell
Set-Location frontend
cmd /c "npm run dev"
```

Visit: http://localhost:5173 → Redirects to `/auth`

### 6. First-time usage

1. **Register** — Guardian creates an account at `/auth` (Register tab)
2. **Verify** — Check the backend terminal for the 6-digit OTP (dev mode), enter at `/verify-email`
3. **Add child** — From the dashboard, add a child profile with a mandatory age category (5–7, 7–10, or 10–12)
4. **Start assessment** — Click a child's assessment button → Camera or Upload page
5. **Record/upload** → **Select participant** → **Process** → **View results**

> **OTP in development mode:** OTPs are printed to the backend terminal (look for `[Q-CARE DEV]` lines). In production, configure `SMTP_HOST`, `SMTP_USER`, `SMTP_PASSWORD` in `.env` for email delivery.

---

## Assessment Workflow

The application supports a guardian-authenticated, age-adaptive assessment workflow:

1. **Login** — Guardian logs in at `/auth` with credentials + OTP verification
2. **Select Child** — From the Guardian Dashboard, select a child profile (with its mandatory age category)
3. **Camera Page** — Click **Start Camera** to request permission via `navigator.mediaDevices.getUserMedia`. The selected child's name and age group are shown.
4. **Live Preview** — Video preview shown; camera-active indicator visible
5. **Start Recording** — Click to begin. Elapsed timer shown. Recording stored via `MediaRecorder`.
6. **Stop Recording** — Click to end. Preview of the recorded clip shown.
7. **Retake** — Discard and re-record without sending to backend.
8. **Save & Process** — Sends the recording to `/api/v1/videos/upload`. An assessment session is automatically created linking the job to the child profile and recording the age group at assessment time.
9. **Select Participant** — Draw a bounding box around the target child on a video frame preview.
10. **Process** — Background worker runs: tracking → CNN → LSTM → Fusion → (QAOA+VQC if age-specific model available).
11. **Results** — Retrieved from `/api/v1/videos/{job_id}/result`. The age group and child profile context are displayed.

> **Upload fallback**: An Upload Page allows processing pre-recorded video files (MP4, WebM, AVI, MOV) without using the camera. Both Camera and Upload pages accept `?child=<childId>` to pre-select a child profile.

> **Age categories**: The system uses three age groups — **Group A (5–7)**, **Group B (7–10)**, and **Group C (10–12)**. Each group has separate processing parameters and model configurations. The age category is **always selected by the guardian** — the system never infers age from video.

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
# Full suite (244 tests — 209 existing + 35 auth/child-profile tests)
.\.venv\Scripts\python.exe -m pytest tests/ -v --tb=short -p no:randomly

# Unit tests only
.\.venv\Scripts\python.exe -m pytest tests/unit/ -v

# Integration tests only (includes auth tests)
.\.venv\Scripts\python.exe -m pytest tests/integration/ -v

# Auth and child profile tests only
.\.venv\Scripts\python.exe -m pytest tests/integration/test_auth_children.py -v

# With coverage
.\.venv\Scripts\python.exe -m pytest tests/ --cov=src --cov=backend --cov-report=term-missing
```

**Expected: 244 passed, 0 failed.**

---

## Project Structure

```
quantum_federated_behaviour_ai/
├── backend/
│   └── app/
│       ├── api/v1/          # FastAPI routers (auth, children, assessments, age_models, videos, experiments)
│       ├── services/        # Business logic (auth_service, model_service)
│       ├── workers/         # Background processing worker
│       ├── core/            # Error handling, logging, security (JWT deps)
│       ├── db.py            # SQLAlchemy engine
│       ├── models.py        # ORM tables (jobs, experiments, model_registry)
│       ├── models_auth.py   # ORM tables (guardians, children, assessments, OTP, audit)
│       ├── schemas/         # Pydantic request/response models
│       └── main.py          # FastAPI app entry point
├── frontend/
│   └── src/
│       ├── pages/           # React pages (AuthPage, GuardianDashboard, ChildProfile, AgeModels, Camera, Upload, Results, ...)
│       ├── components/      # ProtectedRoute, shared components
│       ├── contexts/        # AuthContext (React context + useAuth hook)
│       ├── hooks/           # useCamera, useJob
│       ├── api/             # client.ts (public), authClient.ts (authenticated)
│       └── types/           # TypeScript types (auth.ts, index.ts)
├── src/
│   ├── config/settings.py   # Pydantic settings
│   ├── data/dataset.py      # Dataset loader
│   ├── federated/fedavg.py  # Federated Averaging
│   ├── models/
│   │   ├── cnn/             # CNN feature extractor
│   │   ├── lstm/            # LSTM temporal model
│   │   ├── fusion/          # Feature fusion
│   │   ├── quantum/         # QAOA + VQC
│   │   ├── baselines/       # Classical MLP baseline
│   │   ├── age_profiles.py  # Age-specific processing configs (5–7, 7–10, 10–12)
│   │   └── age_model_loader.py # Age-aware model loading (never cross-substitutes)
│   ├── pipeline/            # Bundle save/load, inference pipeline
│   ├── preprocessing/       # Normalizer
│   ├── training/trainer.py  # Training loop
│   ├── evaluation/metrics.py
│   └── video/               # Extractor, tracker, geometry, pipeline
├── tests/
│   ├── unit/                # Unit tests
│   ├── integration/         # API + end-to-end + auth tests
│   ├── fixtures/            # Synthetic test data
│   └── helpers/             # Test utilities
├── scripts/
│   ├── check_environment.py
│   ├── create_admin.py      # Create first admin guardian account
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
│   ├── app.db               # SQLite database (10 tables)
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

# Auth
SECRET_KEY=change-this-to-a-random-secret-key-before-deployment
ACCESS_TOKEN_EXPIRE_MINUTES=60

# OTP / Email (dev mode prints to terminal if SMTP not set)
SMTP_HOST=
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=

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

## API Endpoints Summary

| Group | Endpoints | Auth Required |
|---|---|---|
| `POST /api/v1/auth/register` | Guardian registration | No |
| `POST /api/v1/auth/login` | Login → JWT token | No |
| `POST /api/v1/auth/verify-otp` | OTP verification | Yes (token) |
| `POST /api/v1/auth/resend-otp` | Resend OTP | Yes |
| `GET /api/v1/auth/me` | Current guardian info | Yes |
| `POST,GET /api/v1/children` | CRUD child profiles | Yes (verified) |
| `POST,GET /api/v1/assessments` | Assessment sessions | Yes (verified) |
| `GET /api/v1/models/age-groups` | Age model status | Yes |
| `POST /api/v1/videos/upload` | Video upload | No* |
| `POST /api/v1/videos/{id}/select-target` | Participant selection | No* |
| `POST /api/v1/videos/{id}/process` | Start processing | No* |
| `GET /api/v1/videos/{id}/result` | Get results | No* |
| `GET /api/v1/health` | Health check | No |

\* Video endpoints do not require auth for backward compatibility but the assessment linkage requires a valid token.

---

## Age Categories

The system implements three mandatory age categories, each with distinct processing parameters:

| Category | Range | Key Differences |
|---|---|---|
| **Group A** (`5_7`) | 5.0 ≤ age < 7.0 | Shorter sequences (20), gentler normalization, lower QAOA depth |
| **Group B** (`7_10`) | 7.0 ≤ age < 10.0 | Standard sequences (30), balanced config |
| **Group C** (`10_12`) | 10.0 ≤ age ≤ 12.0 | Longer sequences (40), higher QAOA/VQC depth |

- Age categories are **always guardian-selected** — never inferred from video
- If a DoB is provided, it is cross-checked against the selected category
- A model trained on one age group is **never substituted** for another
- Age group is **recorded at assessment time** (immutable snapshot)

---

## Known Limitations

| Limitation | Impact | Workaround |
|---|---|---|
| No real labelled dataset | No trained model; VQC outputs synthetic labels only | Provide labelled video dataset and train |
| No trained CNN/LSTM weights | CNN uses random init; features are not meaningful for real classification | Train on labelled dataset |
| CSRT tracker box shrinks on flat colours | Bounding box accuracy degrades (~IoU 0.35 on solid colours) | Works better on real textured clothing |
| Appearance guard is HSV-histogram heuristic | Cannot guarantee identity if two people have similar clothing | Operator must verify target visually |
| No real browser camera test | useCamera/MediaRecorder was never run in an actual browser in CI | Test manually on localhost |
| OTP prints to terminal in dev mode | Not emailed unless SMTP configured | Configure SMTP vars in .env for production |
| Age models not pre-trained | Default config per group; system returns `model_not_ready` | Train per-group models with labelled data |

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
