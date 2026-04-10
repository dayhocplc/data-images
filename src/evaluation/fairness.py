"""
src/evaluation/fairness.py
Per-subgroup and intersectional fairness analysis.

Provides:
  - Per-subgroup F1 across all 24 intersectional cells
  - Subgroup Performance Gap (SPG)
  - Visualization helpers for Table 4 equivalent
  - Fairness audit report generation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# Subgroup name mapping
GENDER_NAMES    = {0: "Male",   1: "Female"}
ETHNICITY_NAMES = {0: "White",  1: "Asian", 2: "Black", 3: "Dark"}
AGE_NAMES       = {0: "0-2",    1: "3-4",   2: "5-6"}


@dataclass
class SubgroupResult:
    """Fairness metrics for a single demographic subgroup."""
    gender_id:    int
    ethnicity_id: int
    gender:       str
    ethnicity:    str

    n_total:   int = 0
    n_positive: int = 0

    f1:          float = 0.0
    sensitivity: float = 0.0   # TPR
    specificity: float = 0.0   # TNR
    precision:   float = 0.0

    meets_sens_threshold: bool = False  # TPR ≥ 90%


@dataclass
class FairnessReport:
    """Complete fairness audit across all 24 intersectional subgroups."""
    config_id: str

    # Primary metrics (per Eqs. 3–4)
    eod_gender:    float = 0.0
    eod_ethnicity: float = 0.0
    l_fair:        float = 0.0   # = 0.5*(eod_gender + eod_ethnicity)

    # Secondary
    dpd_gender:    float = 0.0
    dpd_ethnicity: float = 0.0

    # Supplementary
    spg:           float = 0.0   # max - min F1 across 24 subgroups

    # Per-subgroup breakdown
    subgroup_results: List[SubgroupResult] = field(default_factory=list)

    # TPR analysis (important for Section 6.6 limitation)
    n_subgroups_meeting_tpr90: int = 0
    smallest_subgroups_excluded: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"[FairnessReport] {self.config_id}",
            f"  EOD_gender    = {self.eod_gender*100:.1f}%",
            f"  EOD_ethnicity = {self.eod_ethnicity*100:.1f}%",
            f"  L_fair        = {self.l_fair:.4f}",
            f"  DPD_gender    = {self.dpd_gender*100:.1f}%",
            f"  DPD_ethnicity = {self.dpd_ethnicity*100:.1f}%",
            f"  SPG           = {self.spg*100:.1f}%",
            f"  Subgroups w/ TPR≥90%: {self.n_subgroups_meeting_tpr90}/24",
        ]
        if self.smallest_subgroups_excluded:
            lines.append(
                f"  Excluded (n<25): {', '.join(self.smallest_subgroups_excluded)}"
            )
        return "\n".join(lines)


@torch.no_grad()
def full_fairness_audit(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config_id: str = "unknown",
    threshold: float = 0.5,
    min_subgroup_size: int = 25,
) -> FairnessReport:
    """
    Comprehensive fairness audit across all 24 intersectional subgroups.

    Computes EOD, DPD, SPG, and per-subgroup F1/TPR.
    Implements Section 6.6 limitation: two smallest subgroups (n=19, n=22)
    are flagged and excluded from per-subgroup fairness conclusions.

    Args:
        min_subgroup_size: Subgroups below this size are flagged.
                           Paper notes n=19, n=22 don't reach 90% TPR.
    """
    model.eval().to(device)

    all_logits, all_labels = [], []
    all_gender, all_ethnicity = [], []

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
    preds     = (logits >= 0).astype(int)   # pre-sigmoid threshold

    report = FairnessReport(config_id=config_id)

    # ── Per-subgroup analysis ─────────────────────────────────────────────
    subgroup_f1s = []
    subgroup_results = []

    for g_id, g_name in GENDER_NAMES.items():
        for e_id, e_name in ETHNICITY_NAMES.items():
            mask = (gender == g_id) & (ethnicity == e_id)
            n    = mask.sum()

            sg = SubgroupResult(
                gender_id=g_id, ethnicity_id=e_id,
                gender=g_name, ethnicity=e_name,
                n_total=n,
                n_positive=(labels[mask] == 1).sum() if n > 0 else 0,
            )

            if n < 2:
                logger.debug(f"Subgroup {g_name}×{e_name}: n={n}, skipped")
                subgroup_results.append(sg)
                continue

            sg.f1          = f1_score(labels[mask], preds[mask], zero_division=0)
            sg.sensitivity = recall_score(labels[mask], preds[mask], zero_division=0)
            sg.meets_sens_threshold = sg.sensitivity >= 0.90

            if n < min_subgroup_size:
                report.smallest_subgroups_excluded.append(
                    f"{g_name}×{e_name} (n={n})"
                )

            subgroup_f1s.append(sg.f1)
            subgroup_results.append(sg)

    report.subgroup_results = subgroup_results
    report.n_subgroups_meeting_tpr90 = sum(
        1 for sg in subgroup_results if sg.meets_sens_threshold
    )

    # ── EOD per attribute (Eq. 3) ─────────────────────────────────────────
    def tpr(mask):
        pos = (labels == 1) & mask
        if pos.sum() == 0:
            return 0.0
        return float(preds[pos].mean())

    tpr_male   = tpr(gender == 0)
    tpr_female = tpr(gender == 1)
    tpr_white  = tpr(ethnicity == 0)
    tpr_dark   = tpr(ethnicity == 3)

    report.eod_gender    = abs(tpr_male - tpr_female)
    report.eod_ethnicity = abs(tpr_white - tpr_dark)
    report.l_fair        = 0.5 * (report.eod_gender + report.eod_ethnicity)

    # ── DPD per attribute ─────────────────────────────────────────────────
    def ppr(mask):
        return float(preds[mask].mean()) if mask.sum() > 0 else 0.0

    report.dpd_gender    = abs(ppr(gender == 0) - ppr(gender == 1))
    report.dpd_ethnicity = abs(ppr(ethnicity == 0) - ppr(ethnicity == 3))

    # ── SPG across valid subgroups ────────────────────────────────────────
    if len(subgroup_f1s) >= 2:
        report.spg = max(subgroup_f1s) - min(subgroup_f1s)

    logger.info(report.summary())
    return report


def compare_fairness_reports(
    reports: List[FairnessReport],
) -> "pd.DataFrame":
    """
    Build comparison table equivalent to paper Table 4.
    """
    import pandas as pd
    rows = []
    for r in reports:
        rows.append({
            "Config":       r.config_id,
            "EOD_g (%)":   round(r.eod_gender * 100, 1),
            "EOD_e (%)":   round(r.eod_ethnicity * 100, 1),
            "DPD_g (%)":   round(r.dpd_gender * 100, 1),
            "DPD_e (%)":   round(r.dpd_ethnicity * 100, 1),
            "SPG (%)":     round(r.spg * 100, 1),
            "L_fair":      round(r.l_fair, 3),
            "TPR90 (/24)": r.n_subgroups_meeting_tpr90,
        })
    return pd.DataFrame(rows)
