# Troubleshooting

## Backend Won't Start

**Symptom**: `ModuleNotFoundError: No module named 'backend'` or similar

**Fix**: Always run from the project root, not from `backend/`:
```powershell
Set-Location "d:\Final\quantum_federated_behaviour_ai"
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

---

## PennyLane Import Error: `NumpyMimic`

**Symptom**: `AttributeError: module 'autoray' has no attribute 'NumpyMimic'`

**Cause**: `autoray` version ≥ 0.7 removed `NumpyMimic`. PennyLane 0.38 requires `autoray==0.6.12`.

**Fix**:
```powershell
pip install "autoray==0.6.12" "pennylane-lightning==0.38.0"
```

---

## VQC TypeError: Float Mismatch

**Symptom**: `RuntimeError: expected scalar type Float but found Double`

**Cause**: PennyLane's `lightning.qubit` returns `float64`; PyTorch linear layers need `float32`.

**Fix**: Already applied in `src/models/quantum/vqc.py` — the forward pass does `.float()` after stacking measurements. If you see this error, check that you haven't overwritten `vqc.py`.

---

## Tracker Falls Back to MIL

**Symptom**: `/api/v1/ready` shows `opencv_tracker: degraded` with `"CSRT/KCF unavailable"`

**Cause**: Only `opencv-python-headless` is installed; CSRT requires `opencv-contrib-python-headless`.

**Fix**:
```powershell
pip uninstall opencv-python-headless -y
pip install "opencv-contrib-python-headless==4.10.0.84"
```

---

## Frontend Camera Permission Denied

**Symptom**: Browser shows "Permission denied" when clicking "Start Camera"

**Cause**: Browsers only allow camera access on HTTPS or `localhost`. If you're accessing the frontend on a non-localhost IP, the camera API will be blocked.

**Fix**: Access via `http://localhost:5173`, not via IP address.

---

## npm install Fails

**Symptom**: `npm install` fails or `node_modules` is missing

**Fix**:
```powershell
Set-Location "d:\Final\quantum_federated_behaviour_ai\frontend"
npm install
```

If npm scripts fail with "cannot be loaded because running scripts is disabled":
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Or use `cmd /c` to bypass:
```powershell
cmd /c "npm install"
cmd /c "npm run dev"
```

---

## Video Upload Returns 422

**Common causes**:
1. File is not a valid video container
2. Video duration exceeds `MAX_VIDEO_DURATION_SECONDS` (default: 600s)
3. File size exceeds `MAX_UPLOAD_SIZE_MB` (default: 500MB)
4. Video codec cannot be decoded by OpenCV

**Check**: The 422 response body includes a `detail` field explaining the specific rejection reason.

---

## Job Stuck in `processing`

**Symptom**: Job status stays `processing` and never completes

**Causes**:
1. Background worker crashed (check `logs/app.log`)
2. Very long video — check `FRAME_SAMPLE_FPS` and `MAX_FRAMES_PER_VIDEO` settings

**Fix**: Cancel the job via `POST /api/v1/videos/{job_id}/cancel`, then check logs.

---

## No Prediction in Result

**Symptom**: Job completes with `status: completed_without_prediction`

**This is expected** when no trained pipeline bundle is registered. To get predictions:
1. Run a QAOA+VQC experiment via the Experiments page
2. Set `ALLOW_SYNTHETIC_MODEL_INFERENCE=true` in `.env` to use the synthetic bundle
3. Or provide a real trained bundle and register it with `scripts/register_bundle.py`

---

## `tsc` Not Found in PATH (Windows)

**Symptom**: `npm run build` fails with `'tsc' is not recognized`

**Fix**: Use the local binary via cmd:
```powershell
cd frontend
cmd /c "node_modules\.bin\tsc -b"
cmd /c "node_modules\.bin\vite build"
# or simply:
cmd /c "npm run build"
```

---

## Tests Fail Due to Test Order

**Symptom**: Tests pass individually but fail when run together

**Cause**: Some tests use `monkeypatch` to patch module-level variables; if test ordering is randomized, patches may leak.

**Fix**: Run with deterministic ordering:
```powershell
python -m pytest tests/ -p no:randomly
```

This is a known issue with test isolation in `test_pipeline_end_to_end.py`. All tests pass in isolation.
