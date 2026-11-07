"""Training loops for CNN-LSTM, VQC, and classical baseline models."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from backend.app.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingHistory:
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    train_accuracies: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)
    best_val_accuracy: float = 0.0
    best_epoch: int = 0
    total_duration_s: float = 0.0
    model_type: str = ""
    config: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class ClassificationTrainer:
    """Generic trainer for PyTorch classification models."""

    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 3,
        learning_rate: float = 0.001,
        batch_size: int = 8,
        num_epochs: int = 10,
        random_seed: int = 42,
        checkpoint_dir: Optional[Path] = None,
        model_name: str = "model",
    ) -> None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        if checkpoint_dir is None:
            from src.config.settings import get_settings
            checkpoint_dir = Path(get_settings().artifacts_dir) / "checkpoints"
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.history = TrainingHistory(model_type=model_name)

    def _compute_accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        preds = logits.argmax(dim=-1)
        return (preds == labels).float().mean().item()

    def train_epoch(
        self,
        loader: DataLoader,
    ) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0
        for X, y in loader:
            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.criterion(logits, y)
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("NaN/Inf loss detected; skipping batch.")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            total_acc += self._compute_accuracy(logits, y)
            n_batches += 1
        if n_batches == 0:
            return 0.0, 0.0
        return total_loss / n_batches, total_acc / n_batches

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0
        for X, y in loader:
            logits = self.model(X)
            loss = self.criterion(logits, y)
            total_loss += loss.item()
            total_acc += self._compute_accuracy(logits, y)
            n_batches += 1
        if n_batches == 0:
            return 0.0, 0.0
        return total_loss / n_batches, total_acc / n_batches

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TrainingHistory:
        """Train model on numpy arrays."""
        X_tr = torch.from_numpy(X_train.astype(np.float32))
        y_tr = torch.from_numpy(y_train.astype(np.int64))
        X_v = torch.from_numpy(X_val.astype(np.float32))
        y_v = torch.from_numpy(y_val.astype(np.int64))

        train_ds = TensorDataset(X_tr, y_tr)
        val_ds = TensorDataset(X_v, y_v)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        t0 = time.monotonic()
        best_val_acc = 0.0
        best_epoch = 0

        for epoch in range(1, self.num_epochs + 1):
            tr_loss, tr_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.eval_epoch(val_loader)

            self.history.train_losses.append(tr_loss)
            self.history.val_losses.append(val_loss)
            self.history.train_accuracies.append(tr_acc)
            self.history.val_accuracies.append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                self._save_best_checkpoint()

            logger.info(
                "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
                epoch, self.num_epochs, tr_loss, tr_acc, val_loss, val_acc,
            )

        self.history.best_val_accuracy = best_val_acc
        self.history.best_epoch = best_epoch
        self.history.total_duration_s = time.monotonic() - t0
        logger.info("Training complete. Best val_acc=%.4f at epoch %d", best_val_acc, best_epoch)
        return self.history

    def _save_best_checkpoint(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"{self.model_name}_best.pt"
        torch.save(self.model.state_dict(), str(path))


class HybridCNNLSTMModel(nn.Module):
    """Full CNN-LSTM-Fusion hybrid model for end-to-end training."""

    MODEL_ID = "cnn_lstm_fusion_v1"

    def __init__(
        self,
        cnn,
        lstm,
        fusion,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.cnn = cnn
        self.lstm = lstm
        self.fusion = fusion
        self.classifier = nn.Linear(fusion.output_dim, num_classes)
        self.num_classes = num_classes

    def forward(
        self,
        frames: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        structured: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """frames: (B, T, C, H, W) -> (B, num_classes) logits."""
        B, T, C, H, W = frames.shape
        # Flatten batch and time for CNN
        frames_flat = frames.view(B * T, C, H, W)
        cnn_feats = self.cnn(frames_flat)  # (B*T, feature_dim)
        cnn_feats = cnn_feats.view(B, T, -1)  # (B, T, feature_dim)
        # LSTM
        if mask is not None:
            lstm_mask = mask  # (B, T)
        else:
            lstm_mask = None
        temporal_repr = self.lstm(cnn_feats, lstm_mask)  # (B, hidden_dim)
        # Fusion
        fused = self.fusion(temporal_repr, structured)  # (B, output_dim)
        return self.classifier(fused)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_id": self.MODEL_ID,
            "state_dict": {k: v.cpu() for k, v in self.state_dict().items()},
        }, str(path))
        logger.info("Saved HybridCNNLSTM to %s", path)
