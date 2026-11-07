"""Evaluation metrics for model comparison experiments."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score,
)

from backend.app.core.logging_config import get_logger

logger = get_logger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[list[str]] = None,
) -> dict:
    """Compute classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    result = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "classification_report": report,
        "n_samples": int(len(y_true)),
    }

    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            n_classes = y_proba.shape[1]
            if n_classes == 2:
                auc = roc_auc_score(y_true, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            result["roc_auc"] = float(auc)
        except Exception as e:
            logger.warning("AUC computation failed: %s", e)

    if class_names:
        result["class_names"] = class_names

    return result


def compare_models(results: dict[str, dict]) -> dict:
    """Generate a model comparison table from individual metric dicts."""
    comparison = {}
    metrics_to_compare = ["accuracy", "f1_macro", "f1_weighted"]
    for model_name, metrics in results.items():
        comparison[model_name] = {
            m: metrics.get(m, None) for m in metrics_to_compare
        }
    return comparison


def save_evaluation_report(
    metrics: dict,
    path: Path,
    model_name: str = "model",
    dataset_split: str = "test",
) -> None:
    report = {
        "model_name": model_name,
        "dataset_split": dataset_split,
        "metrics": metrics,
        "disclaimer": (
            "These results are from an experimental system. "
            "Predictions are behavioural risk screenings only, not clinical diagnoses."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved evaluation report to %s", path)
