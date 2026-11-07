# Test Fixtures

> **IMPORTANT**: All fixtures in this directory are SYNTHETIC and are NOT derived from real children's data.
> They exist solely to enable automated pipeline testing without requiring a real dataset.
> Do not interpret synthetic fixture labels as clinically meaningful.

## Contents

- `synthetic_manifest.csv` — Synthetic dataset manifest for pipeline testing
- `videos/` — Tiny programmatically generated video files (created by `scripts/create_test_fixtures.py`)

## Creating Test Videos

Run:
```powershell
python scripts/create_test_fixtures.py
```

This generates small solid-colour MP4 files used by unit and integration tests.
