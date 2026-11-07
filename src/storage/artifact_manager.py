"""Artifact management: saving, loading, and validating model artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from backend.app.core.logging_config import get_logger
from backend.app.core.errors import ArtifactError

logger = get_logger(__name__)


def compute_checksum(path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def save_artifact_manifest(artifacts: dict[str, Path], manifest_path: Path) -> None:
    """Save artifact manifest with paths and checksums."""
    manifest = {}
    for name, path in artifacts.items():
        if path.exists():
            manifest[name] = {
                "path": str(path),
                "checksum_sha256": compute_checksum(path),
                "size_bytes": path.stat().st_size,
            }
        else:
            manifest[name] = {"path": str(path), "exists": False}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Saved artifact manifest to %s", manifest_path)


def validate_artifact(path: Path, expected_checksum: Optional[str] = None) -> None:
    """Validate that an artifact exists and optionally verify its checksum."""
    if not path.exists():
        raise ArtifactError(f"Required artifact not found: {path}")
    if path.stat().st_size == 0:
        raise ArtifactError(f"Artifact file is empty: {path}")
    if expected_checksum:
        actual = compute_checksum(path)
        if actual != expected_checksum:
            raise ArtifactError(
                f"Artifact checksum mismatch for {path}. "
                f"Expected {expected_checksum}, got {actual}."
            )
    logger.info("Artifact validated: %s", path)


class ArtifactManager:
    """Manages model artifacts, checkpoints, and experiment outputs."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def artifact_path(self, *parts: str) -> Path:
        return self.base_dir.joinpath(*parts)

    def save_json(self, data: dict, *parts: str) -> Path:
        path = self.artifact_path(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    def load_json(self, *parts: str) -> dict:
        path = self.artifact_path(*parts)
        if not path.exists():
            raise ArtifactError(f"Artifact not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def list_experiments(self) -> list[str]:
        exp_dir = self.base_dir / "experiments"
        if not exp_dir.exists():
            return []
        return [d.name for d in exp_dir.iterdir() if d.is_dir()]
