"""FedAvg implementation for classical and hybrid model parameters."""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from backend.app.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class FederatedRoundResult:
    round_idx: int
    participating_clients: list[str]
    n_clients: int
    global_val_accuracy: Optional[float]
    client_losses: dict[str, float] = field(default_factory=dict)
    communication_payload_bytes: int = 0
    duration_s: float = 0.0


@dataclass
class FederatedExperimentResult:
    rounds: list[FederatedRoundResult] = field(default_factory=list)
    final_accuracy: Optional[float] = None
    total_duration_s: float = 0.0
    config: dict = field(default_factory=dict)
    model_type: str = ""
    federates_quantum: bool = False
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def validate_update(update: dict[str, torch.Tensor]) -> bool:
    """Reject NaN/Inf model updates."""
    for k, v in update.items():
        if torch.isnan(v).any() or torch.isinf(v).any():
            logger.warning("Invalid (NaN/Inf) values in parameter %s", k)
            return False
    return True


def fedavg_aggregate(
    global_model: nn.Module,
    client_updates: list[dict[str, torch.Tensor]],
    client_sample_counts: list[int],
) -> nn.Module:
    """FedAvg: sample-weighted average of client model parameters.
    
    Formula: w_global = sum_i (n_i / N) * w_i
    where n_i = client i sample count, N = total samples.
    
    Only valid client updates (non-NaN/Inf) are included.
    """
    total_samples = sum(client_sample_counts)
    if total_samples == 0:
        raise ValueError("Total sample count is zero; cannot aggregate.")

    valid_updates = []
    valid_counts = []
    for update, count in zip(client_updates, client_sample_counts):
        if validate_update(update):
            valid_updates.append(update)
            valid_counts.append(count)
        else:
            logger.warning("Rejecting invalid client update.")

    if not valid_updates:
        raise ValueError("All client updates are invalid; cannot aggregate.")

    valid_total = sum(valid_counts)
    new_state = copy.deepcopy(global_model.state_dict())

    # Zero out parameters to accumulate
    for k in new_state:
        new_state[k] = torch.zeros_like(new_state[k], dtype=torch.float32)

    for update, count in zip(valid_updates, valid_counts):
        weight = count / valid_total
        for k in new_state:
            if k in update:
                new_state[k] += weight * update[k].float()
            else:
                logger.warning("Parameter %s missing from client update; using global.", k)
                # Fall back to current global parameter
                new_state[k] += weight * global_model.state_dict()[k].float()

    global_model.load_state_dict(new_state)
    return global_model


class SimulatedFederatedClient:
    """Simulates a federated client that trains on a local data partition."""

    def __init__(
        self,
        client_id: str,
        X_local: np.ndarray,
        y_local: np.ndarray,
        local_epochs: int = 2,
        learning_rate: float = 0.001,
        batch_size: int = 8,
        random_seed: int = 42,
    ) -> None:
        self.client_id = client_id
        self.X_local = X_local
        self.y_local = y_local
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.n_samples = len(X_local)

    def local_train(
        self,
        global_model: nn.Module,
    ) -> tuple[dict[str, torch.Tensor], float]:
        """Train a copy of global_model on local data and return updated parameters."""
        local_model = copy.deepcopy(global_model)
        optimizer = torch.optim.Adam(local_model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        X = torch.from_numpy(self.X_local.astype(np.float32))
        y = torch.from_numpy(self.y_local.astype(np.int64))
        from torch.utils.data import TensorDataset, DataLoader
        ds = TensorDataset(X, y)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        local_model.train()
        epoch_losses = []
        for epoch in range(self.local_epochs):
            for Xb, yb in loader:
                optimizer.zero_grad()
                logits = local_model(Xb)
                loss = criterion(logits, yb)
                if torch.isnan(loss):
                    continue
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        update = {k: v.detach().clone() for k, v in local_model.state_dict().items()}
        logger.info("Client %s: local training done, avg_loss=%.4f", self.client_id, avg_loss)
        return update, avg_loss


class FederatedServer:
    """Orchestrates FedAvg rounds across simulated clients."""

    def __init__(
        self,
        global_model: nn.Module,
        clients: list[SimulatedFederatedClient],
        n_rounds: int = 5,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_interval: int = 1,
    ) -> None:
        self.global_model = global_model
        self.clients = clients
        self.n_rounds = n_rounds
        if checkpoint_dir is None:
            from src.config.settings import get_settings
            checkpoint_dir = Path(get_settings().artifacts_dir) / "checkpoints" / "federated"
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval

    def run(
        self,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> FederatedExperimentResult:
        """Run FedAvg for n_rounds. Optionally evaluate on validation set after each round."""
        t0 = time.monotonic()
        experiment = FederatedExperimentResult(
            model_type=self.global_model.__class__.__name__,
            config={"n_rounds": self.n_rounds, "n_clients": len(self.clients)},
            notes=[
                "This is a simulated federated experiment (single machine, multiple data partitions).",
                "Raw video data is never transmitted; only model parameter updates are shared.",
                "FedAvg does not constitute cryptographic secure aggregation.",
            ],
        )

        for round_idx in range(1, self.n_rounds + 1):
            round_t0 = time.monotonic()
            logger.info("=== Federated Round %d/%d ===", round_idx, self.n_rounds)

            client_updates = []
            client_sample_counts = []
            client_losses = {}
            payload_bytes = 0

            for client in self.clients:
                try:
                    update, loss = client.local_train(self.global_model)
                    client_updates.append(update)
                    client_sample_counts.append(client.n_samples)
                    client_losses[client.client_id] = loss
                    # Approximate payload: sum of parameter bytes
                    payload_bytes += sum(v.numel() * 4 for v in update.values())
                except Exception as e:
                    logger.warning("Client %s failed: %s", client.client_id, e)

            if client_updates:
                self.global_model = fedavg_aggregate(
                    self.global_model, client_updates, client_sample_counts
                )

            # Evaluate
            val_acc = None
            if X_val is not None and y_val is not None:
                val_acc = self._evaluate(X_val, y_val)
                logger.info("Round %d | global_val_acc=%.4f", round_idx, val_acc)

            round_result = FederatedRoundResult(
                round_idx=round_idx,
                participating_clients=[c.client_id for c in self.clients],
                n_clients=len(self.clients),
                global_val_accuracy=val_acc,
                client_losses=client_losses,
                communication_payload_bytes=payload_bytes,
                duration_s=time.monotonic() - round_t0,
            )
            experiment.rounds.append(round_result)

            if round_idx % self.checkpoint_interval == 0:
                self._save_checkpoint(round_idx)

        experiment.final_accuracy = experiment.rounds[-1].global_val_accuracy if experiment.rounds else None
        experiment.total_duration_s = time.monotonic() - t0
        return experiment

    @torch.no_grad()
    def _evaluate(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        self.global_model.eval()
        X = torch.from_numpy(X_val.astype(np.float32))
        y = torch.from_numpy(y_val.astype(np.int64))
        logits = self.global_model(X)
        preds = logits.argmax(dim=-1)
        return float((preds == y).float().mean().item())

    def _save_checkpoint(self, round_idx: int) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"global_round_{round_idx:03d}.pt"
        torch.save(self.global_model.state_dict(), str(path))
        logger.info("Saved global checkpoint: %s", path)
