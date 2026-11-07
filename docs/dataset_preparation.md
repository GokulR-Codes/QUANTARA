# Dataset Preparation

> **Important**: No real dataset is required to run the software or tests. All experiments use synthetic data generated automatically. This guide describes what a real dataset would look like and how to integrate it.

## Dataset Format

The pipeline expects a CSV manifest file listing videos and their labels.

### Manifest Schema

```csv
video_id,label,age_group,screen_time_hours,notes
child_001,low_risk,6-8,1.5,
child_002,high_risk,9-12,6.0,
child_003,moderate_risk,8-10,3.5,
```

| Column | Type | Description |
|---|---|---|
| `video_id` | string | Unique identifier for the recording |
| `label` | string | `low_risk` / `moderate_risk` / `high_risk` |
| `age_group` | string | Age range, e.g. `6-8`, `9-12` |
| `screen_time_hours` | float | Self-reported daily screen time |
| `notes` | string | Optional annotation notes |

A template is provided at `data/manifests/template.csv`.

## Video Requirements

| Property | Requirement |
|---|---|
| Format | MP4 (H.264), WebM (VP8/VP9), AVI |
| Resolution | Minimum 224×224 px |
| Frame rate | 10–30 fps recommended |
| Duration | 30 seconds – 10 minutes |
| Lighting | Well-lit, single child clearly visible |
| Camera angle | Stable, front-facing or side view |

## Labelling Guidelines

Labels should be assigned by a qualified child psychologist based on standardized screening instruments, NOT by the automated system. The automated system assists observation — it does not replace clinical judgment.

Recommended tools for ground-truth labelling:
- **Screen Time Questionnaire (STQ)**
- **Child Behaviour Checklist (CBCL)**
- **Strengths and Difficulties Questionnaire (SDQ)**

## Creating Synthetic Test Fixtures

For testing the pipeline without real data:

```powershell
python scripts\create_test_fixtures.py
```

This creates:
- `tests/fixtures/videos/solid_colour_*.avi` — tiny synthetic clips
- `tests/fixtures/videos/short_5s.avi` — 5-second test clip
- `tests/fixtures/synthetic_manifest.csv` — synthetic dataset manifest

## Training on Real Data

Once a labelled dataset is available:

1. Place videos in a directory, e.g. `data/real_videos/`
2. Create a manifest CSV at `data/manifests/dataset.csv`
3. Run the QAOA+VQC experiment with the dataset path configured:
   ```json
   {
     "name": "real_training",
     "experiment_type": "qaoa_vqc",
     "config": {
       "epochs": 50,
       "dataset_path": "data/manifests/dataset.csv",
       "video_dir": "data/real_videos/"
     }
   }
   ```
4. After training completes, register the bundle:
   ```powershell
   python scripts\register_bundle.py --bundle-path artifacts/checkpoints/your_bundle/
   ```

## Data Validation

The API provides a validation endpoint:

```bash
POST /api/v1/datasets/validate
Content-Type: application/json
{"manifest_path": "data/manifests/dataset.csv"}
```

This checks:
- Manifest file exists and is parseable
- Required columns are present
- Labels are from the expected set
- Video files (if `video_dir` is provided) are accessible
