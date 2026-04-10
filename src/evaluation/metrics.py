"""
src/evaluation/metrics.py
Complete evaluation metrics for the trilemma benchmark.

Reports: F1, sensitivity, specificity, EOD (per attribute),
DPD (per attribute), SPG (across 24 subgroups).

All metrics align with paper Section 4.5.2 and Table 3/4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader


@dataclass
class TrilemmaMetrics:
    """
    Complete set of trilemma evaluation metrics.

    Accuracy metrics:
        f1, sensitivity (=recall), specificity, precision, auc
    Fairness metrics (primary):
        eod_gender, eod_ethnicity
    Fairness metrics (secondary):
        dpd_gender, dpd_ethnicity
    Fairness metrics (supplementary):
        spg (Subgroup Performance Gap across 24 intersectional cells)
    """
    # Accuracy
    f1: float = 0.0
    sensitivity: float = 0.0   # = recall = TPR
    specificity: float = 0.0   # = TNR
    precision: float = 0.0
    auc: float = 0.0
    accuracy: float = 0.0

    # Fairness — primary (per Eqs. 3–4)
    eod_gender: float = 0.0
    eod_ethnicity: float = 0.0
    l_fair: float = 0.0        # = 0.5*(eod_gender + eod_ethnicity)

    # Fairness — secondary
    dpd_gender: float = 0.0
    dpd_ethnicity: float = 0.0

    # Fairness — supplementary
    spg: float = 0.0           # max_i(F1_i) - min_i(F1_i) across 24 cells

    # DFZ qualification (all four constraints)
    dfz_qualified: bool = False
    dfz_f1_ok: bool = False
    dfz_eod_ok: bool = False
    dfz_size_ok: bool = False
    dfz_latency_ok: bool = False

    # Efficiency (filled post-TFLite export)
    size_mb: float = 0.0
    latency_e3_ms: float = 0.0
    flops_g: float = 0.0
    l_eff: float = 0.0

    # Per-subgroup F1 (24 cells: gender x ethnicity x age)
    subgroup_f1: Dict[str, float] = field(default_factory=dict)

    # Bootstrap CI
    f1_ci_lower: float = 0.0
    f1_ci_upper: float = 0.0
    eod_gender_ci_lower: float = 0.0
    eod_gender_ci_upper: float = 0.0

    def check_dfz(
        self,
        tau_acc: float = 0.85,
        tau_fair: float = 0.10,
        size_constraint: float = 10.0,
        latency_constraint: float = 300.0,
    ) -> bool:
        """
        Evaluate DFZ qualification (Definition 3b, Eq. 12):
          F1 ≥ τ_acc AND EOD_g < τ_fair ∀g AND size ≤ S* AND latency ≤ T*
        """
        self.dfz_f1_ok      = self.f1 >= tau_acc
        self.dfz_eod_ok     = (
            self.eod_gender   < tau_fair and
            self.eod_ethnicity < tau_fair
        )
        self.dfz_size_ok    = self.size_mb <= size_constraint or self.size_mb == 0
        self.dfz_latency_ok = (
            self.latency_e3_ms <= latency_constraint or self.latency_e3_ms == 0
        )
        self.dfz_qualified  = (
            self.dfz_f1_ok and self.dfz_eod_ok and
            self.dfz_size_ok and self.dfz_latency_ok
        )
        return self.dfz_qualified

    def pareto_vector(self) -> np.ndarray:
        """
        Normalized objective vector (L_acc, L_fair, L_eff) for Pareto analysis.
        Used to identify knee point via ‖L(θ)‖₂ (Eq. 13).
        """
        l_acc  = 1.0 - self.f1
        l_fair = self.l_fair
        l_eff  = self.l_eff
        return np.array([l_acc, l_fair, l_eff])

    def euclidean_distance_to_ideal(self) -> float:
        """‖L(θ)‖₂ — distance to ideal point (0,0,0). Eq. 13."""
        return float(np.linalg.norm(self.pareto_vector()))

    def to_dict(self) -> Dict:
        return {
            k: v for k, v in self.__dict__.items()
            if not isinstance(v, dict)
        }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
) -> TrilemmaMetrics:
    """
    Full evaluation over a DataLoader.

    Args:
        model: trained classifier
        loader: val or test DataLoader (ASDFaceDataset)
        device: evaluation device
        threshold: classification threshold (default 0.5)
        n_bootstrap: bootstrap iterations for CI

    Returns:
        TrilemmaMetrics with all accuracy and fairness metrics populated.
    """
    model.eval()
    model.to(device)

    all_logits      = []
    all_labels      = []
    all_gender      = []
    all_ethnicity   = []
    all_subgroup_id = []

    for batch in loader:
        logits = model(batch["image"].to(device)).squeeze(-1)
        all_logits.append(logits.cpu())
        all_labels.append(batch["label"])
        all_gender.append(batch["gender"])
        all_ethnicity.append(batch["ethnicity"])
        all_subgroup_id.append(batch["subgroup_id"])

    logits      = torch.cat(all_logits).numpy()
    labels      = torch.cat(all_labels).numpy().astype(int)
    gender      = torch.cat(all_gender).numpy()
    ethnicity   = torch.cat(all_ethnicity).numpy()
    subgroup_id = torch.cat(all_subgroup_id).numpy()

    probs = 1 / (1 + np.exp(-logits))      # sigmoid
    preds = (probs >= threshold).astype(int)

    m = TrilemmaMetrics()

    # ── Accuracy metrics ────────────────────────────────────────────────────
    m.f1          = f1_score(labels, preds, zero_division=0)
    m.sensitivity = recall_score(labels, preds, zero_division=0)
    m.precision   = precision_score(labels, preds, zero_division=0)
    m.accuracy    = (preds == labels).mean()

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    m.specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        m.auc = roc_auc_score(labels, probs)
    except ValueError:
        m.auc = 0.0

    # ── Fairness — EOD (primary, Eq. 3) ─────────────────────────────────────
    m.eod_gender    = _compute_eod(labels, preds, gender,    maj=0, min_=1)
    m.eod_ethnicity = _compute_eod(labels, preds, ethnicity, maj=0, min_=3)
    m.l_fair = 0.5 * (m.eod_gender + m.eod_ethnicity)

    # ── Fairness — DPD (secondary) ──────────────────────────────────────────
    m.dpd_gender    = _compute_dpd(preds, gender,    maj=0, min_=1)
    m.dpd_ethnicity = _compute_dpd(preds, ethnicity, maj=0, min_=3)

    # ── Fairness — SPG (supplementary, across 24 subgroups) ─────────────────
    subgroup_f1s = {}
    for sg_id in np.unique(subgroup_id):
        mask = subgroup_id == sg_id
        if mask.sum() < 2:
            continue
        sg_f1 = f1_score(labels[mask], preds[mask], zero_division=0)
        subgroup_f1s[int(sg_id)] = sg_f1
    m.subgroup_f1 = subgroup_f1s

    if len(subgroup_f1s) >= 2:
        f1_vals = list(subgroup_f1s.values())
        m.spg = max(f1_vals) - min(f1_vals)

    # ── Bootstrap CI (95%) ──────────────────────────────────────────────────
    ci = _bootstrap_ci(labels, preds, probs, gender, ethnicity, n=n_bootstrap)
    m.f1_ci_lower            = ci["f1_lower"]
    m.f1_ci_upper            = ci["f1_upper"]
    m.eod_gender_ci_lower    = ci["eod_gender_lower"]
    m.eod_gender_ci_upper    = ci["eod_gender_upper"]

    return m


# ── Helper functions ──────────────────────────────────────────────────────────

def _compute_eod(
    labels: np.ndarray,
    preds:  np.ndarray,
    attr:   np.ndarray,
    maj: int,
    min_: int,
) -> float:
    """
    EOD_g(θ) = |TPR_g(majority) − TPR_g(minority)|  (Eq. 3)
    """
    def tpr(mask):
        pos = (labels == 1) & mask
        if pos.sum() == 0:
            return 0.0
        return (preds[pos] == 1).sum() / pos.sum()

    tpr_maj = tpr(attr == maj)
    tpr_min = tpr(attr == min_)
    return abs(tpr_maj - tpr_min)


def _compute_dpd(
    preds: np.ndarray,
    attr:  np.ndarray,
    maj: int,
    min_: int,
) -> float:
    """
    DPD_g = |PPR_majority − PPR_minority|
    where PPR = (TP+FP) / N_group
    """
    def ppr(mask):
        if mask.sum() == 0:
            return 0.0
        return preds[mask].mean()

    return abs(ppr(attr == maj) - ppr(attr == min_))


def _bootstrap_ci(
    labels: np.ndarray,
    preds:  np.ndarray,
    probs:  np.ndarray,
    gender: np.ndarray,
    ethnicity: np.ndarray,
    n: int = 1000,
    ci: float = 0.95,
) -> Dict[str, float]:
    """
    Stratified bootstrap CI for F1 and EOD.
    Preserves class and demographic proportions (paper Section 4.5.3).
    """
    rng = np.random.default_rng(42)
    n_samples = len(labels)

    f1_boot          = []
    eod_gender_boot  = []

    for _ in range(n):
        # Stratified resampling (by label × gender)
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        b_labels    = labels[idx]
        b_preds     = preds[idx]
        b_gender    = gender[idx]
        b_ethnicity = ethnicity[idx]

        f1_boot.append(f1_score(b_labels, b_preds, zero_division=0))
        eod_gender_boot.append(_compute_eod(b_labels, b_preds, b_gender, 0, 1))

    alpha = 1 - ci
    return {
        "f1_lower":           float(np.percentile(f1_boot, 100 * alpha / 2)),
        "f1_upper":           float(np.percentile(f1_boot, 100 * (1 - alpha / 2))),
        "eod_gender_lower":   float(np.percentile(eod_gender_boot, 100 * alpha / 2)),
        "eod_gender_upper":   float(np.percentile(eod_gender_boot, 100 * (1 - alpha / 2))),
    }


@torch.no_grad()
def compute_eod(
    model: nn.Module,
    loader: DataLoader,
    attributes: List[str],
    device: torch.device,
) -> Dict[str, float]:
    """
    Convenience function for PFP per-neuron fairness gate.
    Returns dict mapping attribute name → EOD value.
    """
    model.eval()
    all_logits, all_labels, all_gender, all_ethnicity = [], [], [], []

    for batch in loader:
        logits = model(batch["image"].to(device)).squeeze(-1)
        all_logits.append(logits.cpu())
        all_labels.append(batch["label"])
        all_gender.append(batch["gender"])
        all_ethnicity.append(batch["ethnicity"])

    logits    = torch.cat(all_logits).numpy()
    labels    = torch.cat(all_labels).numpy().astype(int)
    gender    = torch.cat(all_gender).numpy()
    ethnicity = torch.cat(all_ethnicity).numpy()

    preds = (logits >= 0).astype(int)  # threshold at 0 (pre-sigmoid)

    result = {}
    if "gender"    in attributes:
        result["gender"]    = _compute_eod(labels, preds, gender,    0, 1)
    if "ethnicity" in attributes:
        result["ethnicity"] = _compute_eod(labels, preds, ethnicity, 0, 3)
    return result
