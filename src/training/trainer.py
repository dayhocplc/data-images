"""
src/training/trainer.py
TrilemmaTrainer — training loop with ATWS, early stopping,
5-fold cross-validation, and per-epoch fairness monitoring.

Shared hyperparameters (paper Section 4.3, applied identically
to all 11 configurations):
  AdamW, LR 1e-4 → 1e-5 cosine annealing
  Weight decay 0.01, batch size 32
  Early stopping: patience=10 on val-F1, max 150 epochs
  Balanced class-weighted BCE
"""

from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold

from src.evaluation.metrics import evaluate, TrilemmaMetrics
from src.training.atws import AdaptiveTrilemmaWeightScheduler
from src.training.losses import TrilemmaLoss

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping on val-F1 with patience."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = -float("inf")
        self.counter    = 0
        self.best_weights: Optional[dict] = None

    def step(self, score: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if score > self.best_score + self.min_delta:
            self.best_score   = score
            self.counter      = 0
            self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience


class TrilemmaTrainer:
    """
    Training loop for all 11 benchmark configurations.

    Supports:
      - ATWS phased weight scheduling (configs with atws=True)
      - Per-epoch EOD + F1 monitoring for ATWS overrides
      - Early stopping on val-F1
      - Cosine LR annealing
      - 5-fold cross-validation (for statistical testing)
      - Checkpoint saving (best val-F1 model)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: TrilemmaLoss,
        cfg: dict,
        device: torch.device,
        output_dir: Path,
        atws: Optional[AdaptiveTrilemmaWeightScheduler] = None,
    ):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.loss_fn      = loss_fn
        self.cfg          = cfg
        self.device       = device
        self.output_dir   = Path(output_dir)
        self.atws         = atws

        train_cfg = cfg.get("training", {})
        self.max_epochs  = train_cfg.get("max_epochs", 150)
        self.patience    = train_cfg.get("early_stopping", {}).get("patience", 10)

        # Optimizer — shared across all configs (Section 4.3)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=train_cfg.get("lr", 1e-4),
            weight_decay=train_cfg.get("weight_decay", 0.01),
        )

        # LR scheduler: cosine annealing 1e-4 → 1e-5
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_epochs,
            eta_min=train_cfg.get("lr_min", 1e-5),
        )

        self.early_stopping = EarlyStopping(patience=self.patience)
        self.best_model: Optional[nn.Module] = None
        self.history: List[Dict] = []

    # ── Main training loop ────────────────────────────────────────────────

    def fit(self) -> nn.Module:
        """
        Run training until early stopping or max_epochs.

        Returns:
            Best model (highest val-F1).
        """
        logger.info(
            f"[Trainer] Starting: max_epochs={self.max_epochs}, "
            f"patience={self.patience}, "
            f"ATWS={'on' if self.atws else 'off'}"
        )
        start = time.time()

        for epoch in range(self.max_epochs):
            # ── Training step ─────────────────────────────────────────
            train_losses = self._train_epoch(epoch)

            # ── Validation ────────────────────────────────────────────
            val_metrics = evaluate(
                self.model, self.val_loader, self.device,
                n_bootstrap=0,  # Skip bootstrap during training
            )

            # ── ATWS weight update ────────────────────────────────────
            if self.atws is not None:
                atws_weights = self.atws.step(
                    epoch=epoch,
                    val_eod_gender=val_metrics.eod_gender,
                    val_eod_ethnicity=val_metrics.eod_ethnicity,
                    val_f1=val_metrics.f1,
                )
                self.loss_fn.update_weights(
                    alpha=atws_weights.alpha,
                    beta=atws_weights.beta,
                    gamma=atws_weights.gamma,
                )

            # ── LR step ───────────────────────────────────────────────
            self.scheduler.step()

            # ── Early stopping check ──────────────────────────────────
            stop = self.early_stopping.step(val_metrics.f1, self.model)

            # ── Logging ───────────────────────────────────────────────
            record = {
                "epoch":       epoch,
                "val_f1":      val_metrics.f1,
                "val_eod_g":   val_metrics.eod_gender,
                "val_eod_e":   val_metrics.eod_ethnicity,
                "val_sens":    val_metrics.sensitivity,
                "lr":          self.scheduler.get_last_lr()[0],
                **{f"train_{k}": v for k, v in train_losses.items()},
            }
            if self.atws:
                aw = self.atws.current_weights if hasattr(self.atws, "current_weights") \
                     else self.atws._current_weights
                record.update({"alpha": aw.alpha, "beta": aw.beta, "gamma": aw.gamma})

            self.history.append(record)

            if epoch % 10 == 0 or stop:
                logger.info(
                    f"Epoch {epoch:3d}/{self.max_epochs} | "
                    f"val_F1={val_metrics.f1:.4f} | "
                    f"EOD_g={val_metrics.eod_gender*100:.1f}% | "
                    f"EOD_e={val_metrics.eod_ethnicity*100:.1f}% | "
                    f"lr={self.scheduler.get_last_lr()[0]:.2e}"
                )

            if stop:
                logger.info(
                    f"Early stopping at epoch {epoch}. "
                    f"Best val-F1: {self.early_stopping.best_score:.4f}"
                )
                break

        elapsed = time.time() - start
        logger.info(f"Training complete in {elapsed/60:.1f} min")

        # Restore best weights
        if self.early_stopping.best_weights is not None:
            self.model.load_state_dict(self.early_stopping.best_weights)

        self.best_model = self.model

        # Save training history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return self.best_model

    # ── Training epoch ────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """One training epoch with trilemma loss."""
        self.model.train()
        total_loss = 0.0
        total_l_acc = 0.0
        total_l_fair = 0.0
        n_batches = 0

        for batch in self.train_loader:
            images      = batch["image"].to(self.device)
            labels      = batch["label"].to(self.device)
            gender_ids  = batch["gender"].to(self.device)
            eth_ids     = batch["ethnicity"].to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(images)
            loss_dict = self.loss_fn(
                logits, labels, gender_ids, eth_ids
            )

            loss = loss_dict["loss"]
            loss.backward()

            # Gradient clipping (stability)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss  += loss.item()
            total_l_acc  += loss_dict["l_acc"].item()
            total_l_fair += loss_dict["l_fair"].item()
            n_batches += 1

        return {
            "loss":   total_loss  / max(n_batches, 1),
            "l_acc":  total_l_acc  / max(n_batches, 1),
            "l_fair": total_l_fair / max(n_batches, 1),
        }

    # ── 5-fold cross-validation ───────────────────────────────────────────

    def cross_validate(
        self,
        full_dataset,
        n_folds: int = 5,
    ) -> List[TrilemmaMetrics]:
        """
        5-fold stratified CV on the training set (Section 4.5.3).

        Used for pairwise Wilcoxon signed-rank testing.
        Provides mean ± std F1 and fairness estimates.

        Args:
            full_dataset: ASDFaceDataset with split="train"
            n_folds: Number of folds (default 5, paper Section 4.5.3)

        Returns:
            List of TrilemmaMetrics, one per fold.
        """
        logger.info(f"[CV] Starting {n_folds}-fold cross-validation")

        labels = full_dataset.df["label"].values
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        fold_metrics: List[TrilemmaMetrics] = []
        batch_size = self.cfg["training"]["batch_size"]

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(
            range(len(full_dataset)), labels
        )):
            logger.info(f"[CV] Fold {fold_idx + 1}/{n_folds}")

            # Rebuild model from scratch for each fold
            from src.models.backbone import build_model
            fold_model = build_model(
                backbone=self.cfg["model"]["backbone"],
                pretrained=self.cfg["model"]["pretrained"],
                dropout=self.cfg["model"]["head"]["dropout"],
                device=self.device,
            )

            # Fold-specific loaders
            train_loader_fold = DataLoader(
                full_dataset,
                batch_size=batch_size,
                sampler=SubsetRandomSampler(train_idx),
                num_workers=4, pin_memory=True,
            )
            val_loader_fold = DataLoader(
                full_dataset,
                batch_size=batch_size,
                sampler=SubsetRandomSampler(val_idx),
                num_workers=4, pin_memory=True,
            )

            # Train fold
            fold_trainer = TrilemmaTrainer(
                model=fold_model,
                train_loader=train_loader_fold,
                val_loader=val_loader_fold,
                loss_fn=copy.deepcopy(self.loss_fn),
                cfg=self.cfg,
                device=self.device,
                output_dir=self.output_dir / f"fold_{fold_idx}",
                atws=copy.deepcopy(self.atws) if self.atws else None,
            )
            fold_trainer.fit()

            # Evaluate on val fold
            fold_m = evaluate(
                fold_trainer.best_model,
                val_loader_fold,
                self.device,
                n_bootstrap=0,
            )
            fold_metrics.append(fold_m)

            logger.info(
                f"  Fold {fold_idx+1}: F1={fold_m.f1:.4f}, "
                f"EOD_g={fold_m.eod_gender*100:.1f}%, "
                f"EOD_e={fold_m.eod_ethnicity*100:.1f}%"
            )

        # Summary
        import numpy as np
        f1_values = [m.f1 for m in fold_metrics]
        eod_g_values = [m.eod_gender for m in fold_metrics]
        logger.info(
            f"[CV] {n_folds}-fold summary: "
            f"F1={np.mean(f1_values):.4f}±{np.std(f1_values):.4f}, "
            f"EOD_g={np.mean(eod_g_values)*100:.1f}%±{np.std(eod_g_values)*100:.1f}%"
        )

        return fold_metrics
