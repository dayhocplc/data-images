"""
src/training/losses.py
Multi-component trilemma loss function (Eq. 9 from paper).

L_scalar(θ; α, β, γ) = α·L_acc(θ) + β·L_fair(θ) + γ·L_eff(θ)

All three components normalized to [0,1] so weights (α,β,γ)
function as genuine priority coefficients, not scale-correction factors.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AccuracyLoss(nn.Module):
    """
    L_acc(θ) = 1 − F1(θ)  ∈ [0, 1]   (Eq. 2)

    Uses a differentiable approximation of F1 for training.
    For evaluation, hard-threshold F1 is computed via metrics.py.
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B,) raw logits before sigmoid
            labels: (B,) binary labels {0, 1}
            class_weights: optional per-class weights for imbalance

        Returns:
            L_acc ∈ [0, 1]
        """
        probs = torch.sigmoid(logits)

        # Differentiable F1 approximation (soft TP, FP, FN)
        tp = (probs * labels).sum()
        fp = (probs * (1 - labels)).sum()
        fn = ((1 - probs) * labels).sum()

        precision = tp / (tp + fp + self.smooth)
        recall    = tp / (tp + fn + self.smooth)

        f1_soft = 2 * precision * recall / (precision + recall + self.smooth)
        return 1.0 - f1_soft


class FairnessLoss(nn.Module):
    """
    L_fair(θ) = ½ (EOD_gen(θ) + EOD_eth(θ))  ∈ [0, 1]   (Eq. 4)

    EOD_g(θ) = |TPR_g(majority) − TPR_g(minority)|  ∈ [0, 1]   (Eq. 3)

    Uses differentiable TPR approximation for training.
    Hard EOD is computed via evaluation/fairness.py for reporting.
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def _soft_tpr(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable TPR = TP / (TP + FN) for a subgroup mask."""
        positive_mask = mask & (labels == 1)
        tp = (probs[positive_mask]).sum()
        fn = ((1 - probs[positive_mask])).sum()
        return tp / (tp + fn + self.smooth)

    def _attribute_eod(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        attr_ids: torch.Tensor,
        majority_id: int,
        minority_id: int,
    ) -> torch.Tensor:
        """Compute EOD for one protected attribute."""
        mask_maj = attr_ids == majority_id
        mask_min = attr_ids == minority_id

        tpr_maj = self._soft_tpr(probs, labels, mask_maj)
        tpr_min = self._soft_tpr(probs, labels, mask_min)

        return torch.abs(tpr_maj - tpr_min)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        gender_ids: torch.Tensor,
        ethnicity_ids: torch.Tensor,
        majority_gender: int = 0,    # male
        minority_gender: int = 1,    # female
        majority_ethnicity: int = 0, # white
        minority_ethnicity: int = 3, # dark
    ) -> torch.Tensor:
        """
        Args:
            logits: (B,) raw model outputs
            labels: (B,) binary labels
            gender_ids: (B,) gender group index
            ethnicity_ids: (B,) ethnicity group index

        Returns:
            L_fair = ½(EOD_gender + EOD_ethnicity)  ∈ [0, 1]
        """
        probs = torch.sigmoid(logits)

        eod_gender = self._attribute_eod(
            probs, labels, gender_ids,
            majority_gender, minority_gender,
        )
        eod_eth = self._attribute_eod(
            probs, labels, ethnicity_ids,
            majority_ethnicity, minority_ethnicity,
        )

        return 0.5 * (eod_gender + eod_eth)


class EfficiencyLoss(nn.Module):
    """
    L_eff(θ) = w_s · S̃(θ) + w_t · T̃(θ)   (Eq. 6)

    where S̃ = S(θ)/S*, T̃ = T_E3(θ)/T*

    L_eff > 1 when constraints are violated (intentional — encodes
    hard-constraint violation in loss magnitude, Eq. 6 note).

    Not differentiable w.r.t. θ in the strict sense.
    Used as a proxy signal in ATWS scheduling and reporting.
    During training, this is approximated via FLOPs regularization.
    """

    def __init__(
        self,
        size_constraint_mb: float = 10.0,
        latency_constraint_ms: float = 300.0,
        w_size: float = 0.5,
        w_latency: float = 0.5,
    ):
        super().__init__()
        self.s_star = size_constraint_mb
        self.t_star = latency_constraint_ms
        self.w_s = w_size
        self.w_t = w_latency

    def from_measurements(
        self,
        size_mb: float,
        latency_ms: float,
    ) -> float:
        """
        Compute L_eff from measured size and latency (non-differentiable).
        Used in Pareto analysis and DFZ evaluation.
        """
        s_norm = size_mb    / self.s_star
        t_norm = latency_ms / self.t_star
        return self.w_s * s_norm + self.w_t * t_norm

    def forward(
        self,
        flops: Optional[torch.Tensor] = None,
        flops_target: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Differentiable proxy via FLOPs regularization (training only).
        For evaluation, use from_measurements() with actual TFLite size + E3 latency.
        """
        if flops is None:
            return torch.tensor(0.0)
        if flops_target is None:
            return torch.tensor(0.0)
        return torch.clamp(flops / flops_target - 1.0, min=0.0)


class TrilemmaLoss(nn.Module):
    """
    Scalarized trilemma loss (Eq. 9):

      L_scalar(θ; α, β, γ) = α·L_acc(θ) + β·L_fair(θ) + γ·L_eff(θ)

    with α + β + γ = 1, α, β, γ ≥ 0.

    Because all three component losses are normalized, weights function
    as genuine priority coefficients (not scale-correction factors).

    Args:
        alpha: accuracy weight
        beta:  fairness weight
        gamma: efficiency weight (set to 0 if no differentiable proxy)
        class_weights: for balanced BCE
    """

    def __init__(
        self,
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.2,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.class_weights = class_weights

        self.loss_acc  = AccuracyLoss()
        self.loss_fair = FairnessLoss()
        self.loss_eff  = EfficiencyLoss()

    def update_weights(self, alpha: float, beta: float, gamma: float):
        """Update weights from ATWS scheduler."""
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        gender_ids: torch.Tensor,
        ethnicity_ids: torch.Tensor,
        flops: Optional[torch.Tensor] = None,
        flops_target: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with total loss and per-component losses for logging.
        """
        l_acc  = self.loss_acc(logits, labels, self.class_weights)
        l_fair = self.loss_fair(logits, labels, gender_ids, ethnicity_ids)
        l_eff  = self.loss_eff(flops, flops_target)

        total = (
            self.alpha * l_acc
            + self.beta  * l_fair
            + self.gamma * l_eff
        )

        return {
            "loss":   total,
            "l_acc":  l_acc.detach(),
            "l_fair": l_fair.detach(),
            "l_eff":  l_eff.detach(),
            "alpha":  torch.tensor(self.alpha),
            "beta":   torch.tensor(self.beta),
            "gamma":  torch.tensor(self.gamma),
        }
