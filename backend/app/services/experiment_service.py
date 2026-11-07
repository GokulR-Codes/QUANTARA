"""Experiment runner service — executes QAOA, VQC, classical, and federated experiments."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from backend.app.core.logging_config import get_logger

logger = get_logger(__name__)


def _get_db():
    """Get a new DB session (used inside background tasks)."""
    from backend.app.db import SessionLocal
    return SessionLocal()


def run_experiment_task(
    experiment_id: str,
    experiment_type: str,
    config: dict,
) -> None:
    """Background task: run an experiment and persist results."""
    db = _get_db()
    try:
        from backend.app.models import Experiment
        exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if not exp:
            return

        exp.status = "running"
        exp.started_at = datetime.now(timezone.utc)
        db.commit()

        if experiment_type == "classical":
            results = _run_classical_experiment(config)
        elif experiment_type == "qaoa_vqc":
            results = _run_qaoa_vqc_experiment(config)
        elif experiment_type == "federated":
            results = _run_federated_experiment(config)
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")

        exp.status = "completed"
        exp.results = results
        exp.completed_at = datetime.now(timezone.utc)
        db.commit()
        logger.info("Experiment %s completed: %s", experiment_id, experiment_type)

    except Exception as exc:
        logger.error("Experiment %s failed: %s", experiment_id, exc, exc_info=True)
        try:
            from backend.app.models import Experiment
            exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
            if exp:
                exp.status = "failed"
                exp.results = {"error": str(exc)}
                exp.completed_at = datetime.now(timezone.utc)
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


def _make_synthetic_data(n_samples: int = 60, n_features: int = 8, n_classes: int = 3, seed: int = 42):
    """Generate synthetic data for smoke-testing. Clearly marked as synthetic."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    y = rng.randint(0, n_classes, size=n_samples)
    split = int(0.7 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


def _run_classical_experiment(config: dict) -> dict:
    """Run a classical baseline experiment on synthetic data."""
    t0 = time.monotonic()
    n_features = config.get("n_features", 8)
    n_classes = config.get("n_classes", 3)
    n_samples = config.get("n_samples", 60)

    X_tr, y_tr, X_val, y_val = _make_synthetic_data(n_samples, n_features, n_classes)

    import torch
    import torch.nn as nn
    from src.training.trainer import ClassificationTrainer
    from src.models.baselines.classical import ClassicalMLP

    model = ClassicalMLP(input_dim=n_features, num_classes=n_classes)
    trainer = ClassificationTrainer(
        model=model,
        num_classes=n_classes,
        num_epochs=config.get("epochs", 5),
        model_name="classical_mlp",
    )
    history = trainer.fit(X_tr, y_tr, X_val, y_val)

    return {
        "experiment_type": "classical",
        "synthetic_data": True,
        "disclaimer": "Results from synthetic data only — not indicative of real-world performance.",
        "n_train": len(X_tr),
        "n_val": len(X_val),
        "best_val_accuracy": history.best_val_accuracy,
        "best_epoch": history.best_epoch,
        "duration_s": time.monotonic() - t0,
        "train_losses": history.train_losses,
        "val_accuracies": history.val_accuracies,
    }


def _run_qaoa_vqc_experiment(config: dict) -> dict:
    """Run QAOA feature selection + VQC training on synthetic data."""
    t0 = time.monotonic()
    n_features = config.get("n_features", 4)
    n_qubits = config.get("n_qubits", 4)
    n_classes = config.get("n_classes", 3)
    n_samples = config.get("n_samples", 40)

    X_tr, y_tr, X_val, y_val = _make_synthetic_data(n_samples, n_features, n_classes)

    # QAOA feature selection
    from src.models.quantum.qaoa import QAOAConfig, QAOAFeatureSelector
    qaoa_config = QAOAConfig(
        n_qubits=min(n_qubits, n_features),
        depth=config.get("qaoa_depth", 2),
        target_k=config.get("target_k", min(2, n_features)),
        random_seed=42,
    )
    selector = QAOAFeatureSelector(qaoa_config)
    feature_names = [f"f{i}" for i in range(n_features)]
    qaoa_result = selector.run(X_tr, y_tr, feature_names)

    selected_idx = qaoa_result.selected_indices
    X_tr_sel = X_tr[:, selected_idx] if selected_idx else X_tr
    X_val_sel = X_val[:, selected_idx] if selected_idx else X_val
    input_dim = X_tr_sel.shape[1]

    # VQC training
    import torch
    import torch.nn as nn
    from src.models.quantum.vqc import VQCClassifier
    from src.training.trainer import ClassificationTrainer

    vqc = VQCClassifier(
        input_dim=input_dim,
        n_qubits=min(n_qubits, input_dim),
        depth=config.get("vqc_depth", 2),
        num_classes=n_classes,
    )
    trainer = ClassificationTrainer(
        model=vqc,
        num_classes=n_classes,
        num_epochs=config.get("epochs", 3),
        model_name="vqc",
    )
    history = trainer.fit(X_tr_sel, y_tr, X_val_sel, y_val)

    # Save checkpoint
    from src.config.settings import get_settings
    ckpt_dir = Path(get_settings().artifacts_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "vqc_experiment_best.pt"
    vqc.save(ckpt_path)

    # Register in DB
    from backend.app.db import SessionLocal
    db = SessionLocal()
    try:
        from backend.app.services.model_service import register_model
        register_model(
            db=db,
            model_type="vqc",
            version="experiment_v1",
            checkpoint_path=str(ckpt_path),
            config={
                "input_dim": input_dim,
                "n_qubits": vqc.n_qubits,
                "depth": vqc.depth,
                "num_classes": n_classes,
                # Provenance: random labels, raw synthetic features. Never usable on real video.
                "training_data": {"kind": "synthetic_random",
                                  "description": "random Gaussian features with random labels"},
                "video_pipeline_compatible": False,
            },
            metrics={"best_val_accuracy_synthetic": history.best_val_accuracy},
            feature_schema_version="synthetic_v1",
            selected_feature_names=qaoa_result.selected_names,
            make_active=True,
        )
    finally:
        db.close()

    return {
        "experiment_type": "qaoa_vqc",
        "synthetic_data": True,
        "disclaimer": "Results from synthetic data only — not indicative of real-world performance.",
        "qaoa": {
            "selected_indices": qaoa_result.selected_indices,
            "selected_names": qaoa_result.selected_names,
            "objective_value": qaoa_result.objective_value,
            "optimization_duration_s": qaoa_result.optimization_duration_s,
            "n_qubits": qaoa_result.n_qubits,
            "depth": qaoa_result.depth,
            "qubo_value_selected": qaoa_result.qubo_value_selected,
            "qubo_value_optimal": qaoa_result.qubo_value_optimal,
            "matches_exhaustive_optimum": qaoa_result.matches_exhaustive_optimum,
            "decoded_probability": qaoa_result.decoded_probability,
        },
        "vqc": {
            "best_val_accuracy": history.best_val_accuracy,
            "best_epoch": history.best_epoch,
            "train_losses": history.train_losses,
        },
        "checkpoint_path": str(ckpt_path),
        "duration_s": time.monotonic() - t0,
    }


def _run_federated_experiment(config: dict) -> dict:
    """Run simulated federated learning experiment on synthetic data."""
    t0 = time.monotonic()
    n_features = config.get("n_features", 8)
    n_classes = config.get("n_classes", 3)
    n_clients = config.get("n_clients", 3)
    n_rounds = config.get("n_rounds", 3)
    n_samples = config.get("n_samples", 90)

    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    y = rng.randint(0, n_classes, n_samples)

    val_split = int(0.2 * n_samples)
    X_val, y_val = X[:val_split], y[:val_split]
    X_train, y_train = X[val_split:], y[val_split:]

    # Split data into client partitions
    partition_size = len(X_train) // n_clients

    from src.config.settings import get_settings
    from src.models.baselines.classical import ClassicalMLP
    from src.federated.fedavg import SimulatedFederatedClient, FederatedServer

    global_model = ClassicalMLP(input_dim=n_features, num_classes=n_classes)
    clients = []
    for i in range(n_clients):
        start = i * partition_size
        end = start + partition_size
        clients.append(SimulatedFederatedClient(
            client_id=f"client_{i}",
            X_local=X_train[start:end],
            y_local=y_train[start:end],
            local_epochs=config.get("local_epochs", 2),
        ))

    server = FederatedServer(
        global_model=global_model,
        clients=clients,
        n_rounds=n_rounds,
        checkpoint_dir=Path(get_settings().artifacts_dir) / "checkpoints" / "federated",
    )
    fed_result = server.run(X_val=X_val, y_val=y_val)

    return {
        "experiment_type": "federated",
        "synthetic_data": True,
        "disclaimer": (
            "Simulated federated learning on a single machine using synthetic data. "
            "Raw video/frames are never transmitted. Not differential-privacy protected."
        ),
        "n_clients": n_clients,
        "n_rounds": n_rounds,
        "final_accuracy": fed_result.final_accuracy,
        "rounds": [
            {
                "round": r.round_idx,
                "val_accuracy": r.global_val_accuracy,
                "client_losses": r.client_losses,
                "duration_s": r.duration_s,
            }
            for r in fed_result.rounds
        ],
        "total_duration_s": fed_result.total_duration_s,
        "notes": fed_result.notes,
    }
