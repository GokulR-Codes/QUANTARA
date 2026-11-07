# Resume instructions

**Project:** `quantum_federated_behaviour_ai` — research prototype: camera/video → operator-selected participant →
tracking → target crops → CNN → LSTM → fusion → QAOA-selected features → VQC → persisted result (FastAPI + React).
It is NOT a clinical tool; no real trained model exists in the repo.

## Recreate the environment (the sandbox venv is NOT part of this archive)
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt     # torch 2.4.1 from PyPI is the CUDA build (~3 GB); CPU users: pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu first
cp .env.example .env    # then set SECRET_KEY to a random value
cd frontend && npm ci && cd ..
```
## Verify
```bash
python -m pytest tests -q            # expected: 209 passed (≈1 min on 1 CPU)
cd frontend && npm run build && npm test && npm run typecheck:tests
```
## Run
`python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload` and `cd frontend && npm run dev` (or `scripts/run_*.sh|ps1`).

## State at last checkpoint
See `PROJECT_STATUS.md` (per-requirement status) and `CHANGELOG.md` (what changed and why). Pending list: `TODO_NEXT_STEPS.md`.
The live HTTP run was in progress: a 640×480 synthetic clip was uploaded, a participant selected, and processing finished as
`completed_without_prediction` (`model_not_ready`) with 12 tracked crops. Next action: retrieve the same result through an independent
request, then write README/docs, run `pip check`, rerun all tests/build, and package with `python scripts/package_delivery.py`.

## Key files
`src/video/{extractor,geometry,tracker,target_pipeline}.py`, `src/pipeline/{features,bundle,runner}.py`, `src/models/quantum/qaoa.py`,
`backend/app/workers/processing_worker.py`, `backend/app/api/v1/videos.py`, `backend/app/services/{job,inference,model}_service.py`,
`frontend/src/pages/TargetSelectionPage.tsx`, `frontend/src/utils/bbox.ts`, `tests/integration/test_pipeline_end_to_end.py`.

## Backup
Latest verified ZIP: recorded in `PROJECT_STATUS.md` (section "Backups").
