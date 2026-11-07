"""Model stage: crops -> CNN -> LSTM -> fusion -> (QAOA mapping -> VQC) with explicit status."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from backend.app.core.logging_config import get_logger
from src.pipeline.bundle import BundleError, LoadedBundle, load_bundle, predict_with_bundle
from src.pipeline.features import (
    FeatureExtractionError, build_untrained_upstream, load_crop_sequence,
    run_cnn_lstm_fusion, save_features,
)

logger = get_logger(__name__)

DISCLAIMER = (
    "This is an experimental research prototype. "
    "Predictions are NOT clinical diagnoses and must not be used as such."
)
SYNTHETIC_DISCLAIMER = (
    " The model that produced this output was trained on SYNTHETIC test data; "
    "the output demonstrates the software path only and has no behavioural meaning."
)

# BundleError.code -> model_status exposed through the API
STATUS_BY_CODE = {
    "missing_files": "model_not_ready",
    "manifest_invalid": "bundle_incompatible",
    "checksum_mismatch": "bundle_corrupt",
    "synthetic_checkpoint_refused": "synthetic_checkpoint_refused",
    "structured_features_required": "bundle_incompatible",
    "incompatible_dimensions": "bundle_incompatible",
    "incompatible_checkpoint": "bundle_incompatible",
    "quantum_unavailable": "quantum_unavailable",
    "inference_failed": "inference_failed",
}


@dataclass
class ModelStageOutcome:
    feature_extraction_status: str
    feature_info: dict = field(default_factory=dict)
    model_status: str = "model_not_ready"
    model_status_detail: str = ""
    prediction: Optional[dict] = None
    model_meta: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def predicted(self) -> bool:
        return self.prediction is not None


def run_model_stage(
    job_dir: Path,
    settings,
    bundle_dir: Optional[Path],
    *,
    no_model_detail: str = "No trained pipeline bundle is registered as active.",
    on_stage: Optional[Callable[[str], None]] = None,
) -> ModelStageOutcome:
    """Run feature extraction and, when a compatible bundle exists, the quantum head.

    Never raises for model-availability problems (they become ``model_status``);
    raises :class:`FeatureExtractionError` for genuine extraction failures.
    """
    job_dir = Path(job_dir)
    seq = load_crop_sequence(job_dir)
    n_tracked = len(seq.images)
    if n_tracked < settings.min_tracked_frames:
        return ModelStageOutcome(
            feature_extraction_status="insufficient_tracked_frames",
            feature_info={"n_crops_available": n_tracked, "min_required": settings.min_tracked_frames},
            model_status="insufficient_tracked_frames",
            model_status_detail=(
                f"Only {n_tracked} trusted target crop(s) were produced; at least "
                f"{settings.min_tracked_frames} are required. No features or prediction were computed."
            ),
            warnings=[f"Insufficient tracked frames ({n_tracked} < {settings.min_tracked_frames})."],
        )

    notify = on_stage or (lambda _stage: None)
    notify("feature_extraction")
    bundle: Optional[LoadedBundle] = None
    status_code: Optional[str] = None
    detail = no_model_detail
    warnings: list[str] = []
    if bundle_dir is not None:
        try:
            bundle = load_bundle(Path(bundle_dir), settings)
        except BundleError as exc:
            status_code, detail = exc.code, exc.message
            warnings.append(f"Model unavailable ({exc.code}): {exc.message}")
            logger.warning("Bundle %s not usable: %s (%s)", bundle_dir, exc.message, exc.code)
    else:
        status_code = "missing_files"

    upstream = bundle.upstream if bundle is not None else build_untrained_upstream(settings)
    out = run_cnn_lstm_fusion(seq, upstream, settings.sequence_length)   # raises on failure
    save_features(job_dir, out)
    info = out.summary()
    fe_status = "completed" if upstream.trained else "completed_untrained_weights"
    if not upstream.trained:
        warnings.append(
            "CNN/LSTM/fusion ran with untrained, randomly initialised weights: "
            "embeddings demonstrate the pipeline only and cannot support a prediction."
        )

    if bundle is None:
        return ModelStageOutcome(
            feature_extraction_status=fe_status, feature_info=info,
            model_status=STATUS_BY_CODE.get(status_code or "", "model_not_ready"),
            model_status_detail=detail, warnings=warnings,
        )

    notify("inference")
    try:
        pred = predict_with_bundle(out.fused, bundle)
    except BundleError as exc:
        return ModelStageOutcome(
            fe_status, info, STATUS_BY_CODE.get(exc.code, "inference_failed"), exc.message,
            warnings=warnings + [f"Inference failed ({exc.code}): {exc.message}"],
        )
    except Exception as exc:  # PennyLane / torch runtime error
        logger.error("VQC inference error: %s", exc, exc_info=True)
        return ModelStageOutcome(
            fe_status, info, "inference_failed", f"{type(exc).__name__}: {exc}",
            warnings=warnings + [f"Inference failed: {exc}"],
        )

    mapped = pred.pop("mapped_input")
    synthetic = bundle.synthetic
    prediction = {**pred, "disclaimer": DISCLAIMER + (SYNTHETIC_DISCLAIMER if synthetic else ""),
                  "synthetic_model": synthetic}
    if synthetic:
        warnings.append("Prediction comes from a synthetic-trained test bundle; it has no behavioural meaning.")
    return ModelStageOutcome(
        feature_extraction_status=fe_status, feature_info=info,
        model_status="inference_completed", model_status_detail="",
        prediction=prediction, warnings=warnings,
        model_meta={
            "bundle_id": bundle.manifest["bundle_id"],
            "model_version": bundle.manifest["version"],
            "feature_schema_version": f"bundle_schema_{bundle.manifest['schema_version']}",
            "selected_feature_names": bundle.mapping["selected_feature_names"],
            "quantum_configuration": bundle.quantum_configuration(),
            "model_provenance": bundle.provenance(),
            "mapped_input": mapped,
        },
    )


__all__ = ["ModelStageOutcome", "run_model_stage", "FeatureExtractionError", "STATUS_BY_CODE"]
