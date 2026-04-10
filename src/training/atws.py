"""
src/training/atws.py
Adaptive Trilemma Weight Scheduler (ATWS) — Algorithm 1 from the paper.

Phases objective weights (α, β, γ) across three training periods to
prevent accuracy collapse under premature fairness regularization.

Motivation: gradient conflict between L_acc and L_fair when fairness
regularization is applied before stable discriminative representations
are learned [Chen et al. GradNorm, ICML 2018].

ATWS improvement over fixed-weight training (paper Table 7):
  B1: +0.013 F1, −0.7pp EOD_gender
  C2: +0.013 F1, −0.6pp EOD_gender
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class ATWSWeights:
    """Objective weights for the trilemma scalarization (Eq. 9)."""
    alpha: float  # Accuracy weight (L_acc)
    beta: float   # Fairness weight (L_fair)
    gamma: float  # Efficiency weight (L_eff)

    def normalize(self) -> "ATWSWeights":
        """ℓ₁-normalize weights (Algorithm 1, line 8)."""
        total = self.alpha + self.beta + self.gamma
        if total < 1e-9:
            return ATWSWeights(1/3, 1/3, 1/3)
        return ATWSWeights(
            self.alpha / total,
            self.beta  / total,
            self.gamma / total,
        )

    def as_dict(self) -> Dict[str, float]:
        return {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma}


# Default phase weights from paper (Algorithm 1)
PHASE1_WEIGHTS = ATWSWeights(alpha=0.7, beta=0.2, gamma=0.1)
PHASE2_WEIGHTS = ATWSWeights(alpha=0.4, beta=0.4, gamma=0.2)
PHASE3_WEIGHTS = ATWSWeights(alpha=0.3, beta=0.4, gamma=0.3)


class AdaptiveTrilemmaWeightScheduler:
    """
    Adaptive Trilemma Weight Scheduler (ATWS).

    Implements the three-phase scheduling with two dynamic overrides:
      Override 1: Fairness violation guard — β += 0.10 if val-EOD > τ_fair
      Override 2: Accuracy collapse guard — reset to Phase I if val-F1 < 0.80

    All weight vectors are ℓ₁-normalized after each update.

    Args:
        max_epochs: Total training epochs (E in paper)
        phase1_end: Fraction of E where Phase I ends (default 0.40)
        phase2_end: Fraction of E where Phase II ends (default 0.70)
        override_eod_threshold: τ for Override 1 (default 0.10)
        override_f1_threshold: τ for Override 2 (default 0.80)
        override_f1_reset_epochs: How many epochs to hold Phase I (default 5)

    Usage:
        scheduler = AdaptiveTrilemmaWeightScheduler(max_epochs=87)
        for epoch in range(max_epochs):
            weights = scheduler.step(
                epoch=epoch,
                val_eod_gender=0.05,
                val_eod_ethnicity=0.04,
                val_f1=0.88,
            )
            loss = compute_trilemma_loss(outputs, targets, **weights.as_dict())
    """

    def __init__(
        self,
        max_epochs: int,
        phase1_end: float = 0.40,
        phase2_end: float = 0.70,
        override_eod_threshold: float = 0.10,
        override_f1_threshold: float = 0.80,
        override_f1_reset_epochs: int = 5,
        verbose: bool = True,
    ):
        self.max_epochs = max_epochs
        self.phase1_end_epoch = int(phase1_end * max_epochs)
        self.phase2_end_epoch = int(phase2_end * max_epochs)
        self.override_eod_threshold = override_eod_threshold
        self.override_f1_threshold = override_f1_threshold
        self.override_f1_reset_epochs = override_f1_reset_epochs
        self.verbose = verbose

        # Internal state
        self._current_weights = PHASE1_WEIGHTS
        self._f1_reset_countdown = 0
        self._history: list = []

    def step(
        self,
        epoch: int,
        val_eod_gender: Optional[float] = None,
        val_eod_ethnicity: Optional[float] = None,
        val_f1: Optional[float] = None,
    ) -> ATWSWeights:
        """
        Compute objective weights for the current epoch.

        Algorithm 1 from paper:
          Lines 2-5: Phase scheduling
          Lines 5-6: Override 1 (fairness violation)
          Lines 7-8: Override 2 (accuracy collapse)
          Line 8: ℓ₁-normalization
        """
        # ── Phase scheduling (Algorithm 1, lines 2–4) ──────────────────────
        if self._f1_reset_countdown > 0:
            # Override 2 active: hold Phase I weights
            weights = PHASE1_WEIGHTS
            self._f1_reset_countdown -= 1
        elif epoch <= self.phase1_end_epoch:
            weights = PHASE1_WEIGHTS
        elif epoch <= self.phase2_end_epoch:
            weights = PHASE2_WEIGHTS
        else:
            weights = PHASE3_WEIGHTS

        # ── Override 1: fairness violation (Algorithm 1, lines 5–6) ────────
        if val_eod_gender is not None and val_eod_ethnicity is not None:
            eod_violated = (
                val_eod_gender   > self.override_eod_threshold or
                val_eod_ethnicity > self.override_eod_threshold
            )
            if eod_violated:
                # β += 0.10, redistributed from α and γ proportionally
                beta_increment = 0.10
                remaining_increment = beta_increment / 2
                weights = ATWSWeights(
                    alpha=weights.alpha - remaining_increment,
                    beta =weights.beta  + beta_increment,
                    gamma=weights.gamma - remaining_increment,
                )
                if self.verbose and epoch % 5 == 0:
                    logger.info(
                        f"[ATWS] Epoch {epoch}: Override 1 active "
                        f"(EOD_g={val_eod_gender:.3f}, "
                        f"EOD_e={val_eod_ethnicity:.3f} > "
                        f"{self.override_eod_threshold}). "
                        f"β += {beta_increment}"
                    )

        # ── Override 2: accuracy collapse (Algorithm 1, lines 7–8) ─────────
        if val_f1 is not None and val_f1 < self.override_f1_threshold:
            weights = PHASE1_WEIGHTS
            self._f1_reset_countdown = self.override_f1_reset_epochs
            if self.verbose:
                logger.warning(
                    f"[ATWS] Epoch {epoch}: Override 2 active "
                    f"(val_F1={val_f1:.3f} < "
                    f"{self.override_f1_threshold}). "
                    f"Reset to Phase I for {self.override_f1_reset_epochs} epochs."
                )

        # ── ℓ₁-normalization (Algorithm 1, line 8) ──────────────────────────
        weights = weights.normalize()
        self._current_weights = weights

        # Log history
        record = {"epoch": epoch, **weights.as_dict()}
        if val_f1 is not None:
            record["val_f1"] = val_f1
        if val_eod_gender is not None:
            record["val_eod_gender"] = val_eod_gender
        self._history.append(record)

        return weights

    @property
    def current_phase(self) -> int:
        """Current phase (1, 2, or 3) based on most recent epoch."""
        if not self._history:
            return 1
        epoch = self._history[-1]["epoch"]
        if epoch <= self.phase1_end_epoch:
            return 1
        elif epoch <= self.phase2_end_epoch:
            return 2
        return 3

    @property
    def history(self) -> list:
        return self._history

    def __repr__(self) -> str:
        return (
            f"ATWS(max_epochs={self.max_epochs}, "
            f"phase1_end={self.phase1_end_epoch}, "
            f"phase2_end={self.phase2_end_epoch})"
        )
