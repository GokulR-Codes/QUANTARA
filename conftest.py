# conftest.py — shared pytest configuration.
#
# All storage (DB, uploads, frames, artifacts, logs) is redirected to a temporary directory
# BEFORE any project module is imported, so running the tests never touches the repository's
# data/, artifacts/ or logs/ directories and never leaves generated files behind.
import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

_TMP = Path(tempfile.mkdtemp(prefix="qfba_tests_"))
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)

os.environ.update({
    "APP_ENV": "test",
    "DATABASE_URL": f"sqlite:///{_TMP / 'test.db'}",
    "UPLOAD_DIR": str(_TMP / "uploads"),
    "FRAMES_DIR": str(_TMP / "frames"),
    "ARTIFACTS_DIR": str(_TMP / "artifacts"),
    "LOG_FILE": str(_TMP / "logs" / "app.log"),
    "RETAIN_RAW_VIDEO": "false",
    "ALLOW_SYNTHETIC_MODEL_INFERENCE": "false",
})

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def tmp_root() -> Path:
    return _TMP
