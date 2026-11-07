"""Model service — active-bundle lookup, registration, and usability reporting."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from backend.app.core.logging_config import get_logger
from backend.app.models import ModelRegistry
from src.config.settings import get_settings
from src.pipeline.bundle import (
    BUNDLE_MODEL_TYPE, SYNTHETIC_KINDS, BundleError, load_bundle, read_manifest, verify_files,
)

logger = get_logger(__name__)


def get_active_model_info(db: Session) -> Optional[ModelRegistry]:
    """Newest active registry record of any type (informational)."""
    return (
        db.query(ModelRegistry).filter(ModelRegistry.is_active == True)  # noqa: E712
        .order_by(ModelRegistry.created_at.desc()).first()
    )


def get_active_bundle_record(db: Session) -> Optional[ModelRegistry]:
    """Newest active *pipeline bundle* — the only record type video inference can use."""
    return (
        db.query(ModelRegistry)
        .filter(ModelRegistry.model_type == BUNDLE_MODEL_TYPE, ModelRegistry.is_active == True)  # noqa: E712
        .order_by(ModelRegistry.created_at.desc()).first()
    )


def register_model(
    db: Session, model_type: str, version: str, checkpoint_path: str,
    config: Optional[dict] = None, metrics: Optional[dict] = None,
    feature_schema_version: Optional[str] = None, selected_feature_names: Optional[list] = None,
    make_active: bool = True,
) -> ModelRegistry:
    """Register a model checkpoint and optionally make it the active one of its type."""
    if make_active:
        db.query(ModelRegistry).filter(
            ModelRegistry.model_type == model_type, ModelRegistry.is_active == True,  # noqa: E712
        ).update({"is_active": False})
    record = ModelRegistry(
        model_type=model_type, version=version, checkpoint_path=checkpoint_path,
        config=config or {}, metrics=metrics or {}, is_active=make_active,
        feature_schema_version=feature_schema_version, selected_feature_names=selected_feature_names,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    logger.info("Registered model: type=%s version=%s active=%s", model_type, version, make_active)
    return record


def register_bundle(db: Session, bundle_dir: Path, make_active: bool = True) -> ModelRegistry:
    """Register a pipeline bundle after verifying its manifest and file checksums.

    Full dimension/compatibility validation happens in :func:`load_bundle` when the bundle is
    used (and in ``/ready``).  Raises :class:`BundleError` if the bundle is unreadable.
    """
    bundle_dir = Path(bundle_dir).resolve()
    manifest = read_manifest(bundle_dir)
    verify_files(bundle_dir, manifest)
    mapping_path = bundle_dir / manifest["files"]["mapping"]["name"]
    import json
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    return register_model(
        db, BUNDLE_MODEL_TYPE, manifest["version"], str(bundle_dir),
        config={"dims": manifest["dims"], "training_data": manifest["training_data"],
                "bundle_id": manifest["bundle_id"]},
        metrics={},
        feature_schema_version=f"bundle_schema_{manifest['schema_version']}",
        selected_feature_names=mapping.get("selected_feature_names"),
        make_active=make_active,
    )


def describe_record(rec: ModelRegistry) -> dict:
    """Whether a registry record can serve real-video inference, and why not otherwise."""
    settings = get_settings()
    if rec.model_type != BUNDLE_MODEL_TYPE:
        training = (rec.config or {}).get("training_data", {})
        return {
            "usable": False,
            "synthetic": training.get("kind") in SYNTHETIC_KINDS if training else None,
            "detail": (
                "Experiment checkpoint (raw synthetic features). It has no upstream "
                "CNN/LSTM/fusion weights and no QAOA mapping over fused features, so it cannot "
                "be used for video inference."
            ),
        }
    if not rec.checkpoint_path:
        return {"usable": False, "synthetic": None, "detail": "No bundle path recorded"}
    try:
        bundle = load_bundle(Path(rec.checkpoint_path), settings)
        return {"usable": True, "synthetic": bundle.synthetic, "detail": "Bundle verified and loadable"}
    except BundleError as exc:
        synthetic = (rec.config or {}).get("training_data", {}).get("kind") in SYNTHETIC_KINDS
        return {"usable": False, "synthetic": synthetic, "detail": f"{exc.code}: {exc.message}"}
