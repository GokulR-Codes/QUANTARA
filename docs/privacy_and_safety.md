# Privacy and Safety

## Scope

This document describes the privacy and safety design decisions in the QAOA–VQC framework.

---

## No Facial Recognition

The system **does not** detect, recognize, or identify any person's face. The operator selects the target participant by drawing a bounding box on a video frame. The tracker then follows that bounding box region — it has no knowledge of who is inside the box.

**Technical basis**: OpenCV CSRT is an appearance-based tracker using color and texture correlation. It cannot match identities across sessions.

## No Age Estimation

The system does not estimate, infer, or store any participant's age. The research framing is that the operator knows they are recording a child — the system does not verify this.

## No Biometric Data

No biometric identifiers (face embeddings, voice prints, gait signatures, etc.) are extracted or stored. CNN embeddings extracted from crops are model-specific intermediate representations — they are not face embeddings and are not cross-session identifiable.

## Synthetic Data Only

All experiments in the software run on programmatically generated synthetic data. No real video of children is bundled with or required by the software. The `tests/fixtures/videos/` directory contains programmatically generated solid-colour AVI files.

## Raw Video Deletion

By default (`RETAIN_RAW_VIDEO=false` in `.env`), the uploaded video file is deleted after frame extraction completes. The extracted crops (112×112 px JPEG files) are stored only for the duration of the job.

On failure or cancellation, the raw video is retained to allow retry without re-upload.

## Honest Predictions

The system **never** invents or fabricates a prediction. When no trained model is registered:

- The job completes with status `completed_without_prediction`
- The result includes `"prediction": null` and `"model_status": "model_not_ready"`
- The VQC is **not** called with randomly initialized weights and the result is **not** reported as a real prediction

When an experiment runs on synthetic data, the trained checkpoint is flagged `is_synthetic=True` and is refused for video inference unless `ALLOW_SYNTHETIC_MODEL_INFERENCE=true` is set.

## API Security

- The API is designed for **localhost-only** use during development and demonstration
- No authentication is implemented in the current version
- Do not expose port 8000 beyond localhost in any setting involving real video of children
- All uploaded files are stored server-side and accessible to anyone with localhost access

## GDPR / Data Minimization

The system is designed for research demonstration only, not for production deployment. For any deployment involving real video of real children:

1. Obtain explicit informed consent from parents/guardians
2. Conduct a DPIA (Data Protection Impact Assessment)
3. Implement proper authentication, authorization, and audit logging
4. Apply data minimization — delete all data as soon as it is no longer needed
5. Store nothing in the clear — encrypt the database and file store at rest
