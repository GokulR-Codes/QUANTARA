# Video Processing Pipeline

## Frame Extraction

### Overview

The video processing pipeline starts with `src/video/extractor.py`. It uses OpenCV to decode video frames and selects frames at regular timestamp intervals (not frame indices).

### Timestamp-Based Sampling

Unlike stride-based sampling (every N-th frame), timestamp sampling handles:
- Variable frame rates (VFR video from MediaRecorder/WebM)
- Dropped frames
- Live-mode WebM files with no container duration

**`FrameSampler`** — Divides the video timeline into slots of `1/sample_fps` seconds. The first frame in each slot is selected. Gaps in timestamps do not cause burst-sampling.

### WebM / Live-Mode Robustness

Browser `MediaRecorder` outputs WebM files in "live mode" — the container may not contain a valid `duration` header, and frame count metadata is unreliable (can be `1.4e17`).

**`probe_video()`** handles this by:
1. Attempting to read container metadata (fast)
2. If `frame_count <= 0` or `duration <= 0`, performing a decode-scan to count actual frames and measure true duration from packet timestamps
3. Setting `scanned=True` and adding a warning in the result

### FrameReader

A context-manager iterator that decodes frames one at a time:

```python
with FrameReader(path) as reader:
    for decoded_frame in reader:
        # decoded_frame.index    — sequential frame number
        # decoded_frame.t_ms     — timestamp in milliseconds
        # decoded_frame.frame    — BGR numpy array
```

---

## Participant Tracking

### Design Philosophy

The tracker does **not** detect or identify people. It follows a region selected by the human operator. This is intentional:

- No facial recognition
- No age estimation  
- No automatic multi-person detection
- The operator is responsible for selecting the correct participant

### OpenCV CSRT Tracker

The system uses the **CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability)** tracker from `opencv-contrib-python-headless`. If CSRT is unavailable, it falls back to KCF, then MIL — and reports which backend is actually in use.

Readiness endpoint (`/api/v1/ready`) explicitly shows tracker backend status. A `degraded` status (MIL fallback) is reported when CSRT is unavailable.

### Tracking Guards

The tracker wraps the raw OpenCV output with three heuristic guards:

| Guard | Threshold | Purpose |
|---|---|---|
| Appearance similarity | HSV histogram correlation ≥ 0.30 | Detect replacement by a different-looking object |
| Motion plausibility | Jump ≤ 1.0 × box diagonal | Detect teleportation |
| Scale plausibility | Area change ≤ 3× | Detect box collapse or explosion |
| Visibility | ≥ 50% inside frame | Detect exit from frame |

When any guard fails, the frame is classified `AMBIGUOUS` (not `TRACKED`) and no crop is generated for that frame.

### Track Status States

```
TRACKED      — all guards pass; crop generated
AMBIGUOUS    — tracker reports success, but a guard failed; no crop generated
LOST         — tracker reported failure; terminal by default (no re-acquisition)
OUT_OF_FRAME — box mostly outside frame; treated like LOST
```

### Re-acquisition

By default, `allow_reacquisition=False`. Once the tracker reports failure, all subsequent frames are marked `LOST`. Enabling re-acquisition requires `confirm_frames` (default: 3) consecutive trusted frames before trusting again.

### Tracking Report

After processing, a per-frame tracking document is stored as JSON at `data/frames/{job_id}/tracking.json` and accessible via:

```
GET /api/v1/videos/{job_id}/tracking
```

The document includes:
- `summary.tracker_backend` — which OpenCV backend was used
- `summary.tracking_coverage_pct` — percentage of frames tracked
- `samples[]` — per-sample: status, bbox, similarity, crop_file, timestamp_ms

---

## Feature Extraction

### CNN (Spatial Features)

- Input: 112×112 RGB crop
- Output: 256-dimensional embedding
- Architecture: ResNet-like with configurable depth
- **Status**: Weights are randomly initialised unless a trained bundle is registered

### LSTM (Temporal Features)

- Input: Sequence of CNN embeddings (length = `SEQUENCE_LENGTH`)
- Output: 128-dimensional hidden state
- Padding: Short sequences are left-padded with zeros; a mask is applied so padded positions do not contribute to the hidden state

### Feature Fusion

- Input: CNN embedding + LSTM hidden state
- Method: Cross-attention fusion
- Output: 64-dimensional fused vector

---

## Crops and File Organization

```
data/frames/{job_id}/
├── tracking.json          — full tracking document
├── features.npz           — CNN embeddings, LSTM output, fused vector
├── sample_0000.jpg        — crop for sample 0 (if tracked)
├── sample_0001.jpg        — crop for sample 1
└── ...
```

Crop filenames match `sample_index` in the tracking document.

---

## Data Retention Policy

- **Default**: Raw uploaded video is deleted after frame extraction
- **Configure**: Set `RETAIN_RAW_VIDEO=true` in `.env` to keep the original upload
- **On failure or cancellation**: Raw video is always kept (may be needed for retry)
- **On deletion**: `DELETE /api/v1/videos/{job_id}` refuses to delete active jobs; 410 is returned for already-deleted jobs
