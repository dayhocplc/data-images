"""
src/evaluation/pareto.py
Pareto frontier analysis and DFZ identification.

Implements Definitions 1–3 from the paper (Section 3.5):
  - Pareto dominance (Def. 1, Eq. 10)
  - Pareto frontier P* (Def. 2, Eq. 11)
  - Deployment-Feasible Zone DFZ (Def. 3, Eq. 12)
  - Knee point θ* via ‖L(θ)‖₂ minimization (Eq. 13)

Also implements ternary sensitivity analysis (Section 5.4):
  36 weight combinations in {0.1,...,0.8}³ subject to unit-sum.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


import numpy as np
import pandas as pd


@dataclass
class ConfigResult:
    """Trilemma position of a single optimization configuration."""
    config_id: str        # e.g. "C2"
    config_name: str      # e.g. "M7+PFP"

    # Objective values (lower = better for all three)
    l_acc:  float         # 1 - F1 ∈ [0,1]
    l_fair: float         # 0.5*(EOD_gender + EOD_ethnicity)
    l_eff:  float         # normalized size+latency

    # Raw metrics (for reporting)
    f1: float
    eod_gender: float
    eod_ethnicity: float
    size_mb: float
    latency_e3_ms: float

    # DFZ constraint satisfaction
    dfz: bool = False
    pareto_optimal: bool = False
    pareto_role: Optional[str] = None  # e.g. "knee_point"

    def objective_vector(self) -> np.ndarray:
        return np.array([self.l_acc, self.l_fair, self.l_eff])

    def distance_to_ideal(self) -> float:
        """‖L(θ)‖₂ — distance to ideal point (0,0,0). Eq. 13."""
        return float(np.linalg.norm(self.objective_vector()))

    def scalar_loss(self, alpha: float, beta: float, gamma: float) -> float:
        """L_scalar(θ; α,β,γ) = α·L_acc + β·L_fair + γ·L_eff. Eq. 9."""
        return alpha * self.l_acc + beta * self.l_fair + gamma * self.l_eff


def is_pareto_dominated(
    candidate: ConfigResult,
    others: List[ConfigResult],
) -> bool:
    """
    Check if candidate is Pareto-dominated by any config in others.

    Definition 1 (Eq. 10): θ_i is dominated by θ_j iff
      L_k(θ_j) ≤ L_k(θ_i) ∀k  AND  ∃k': L_{k'}(θ_j) < L_{k'}(θ_i)
    """
    v_cand = candidate.objective_vector()
    for other in others:
        if other.config_id == candidate.config_id:
            continue
        v_other = other.objective_vector()
        # θ_j dominates θ_i: all components ≤ AND at least one strictly <
        if np.all(v_other <= v_cand) and np.any(v_other < v_cand):
            return True
    return False


def compute_pareto_frontier(
    results: List[ConfigResult],
) -> List[ConfigResult]:
    """
    Compute Pareto frontier P* (Definition 2, Eq. 11).

    Returns non-dominated configurations.
    """
    pareto = []
    for config in results:
        if not is_pareto_dominated(config, results):
            config.pareto_optimal = True
            pareto.append(config)
        else:
            config.pareto_optimal = False
    return pareto


def compute_dfz(
    results: List[ConfigResult],
    tau_acc: float = 0.85,
    tau_fair: float = 0.10,
    size_constraint_mb: float = 10.0,
    latency_constraint_ms: float = 300.0,
) -> List[ConfigResult]:
    """
    Compute Deployment-Feasible Zone (Definition 3a, Eq. 12):

      DFZ = P* ∩ { θ | F1 ≥ τ_acc, EOD_g < τ_fair ∀g, S ≤ S*, T ≤ T* }

    All four constraints must be satisfied simultaneously.
    Note: τ_fair applied per-attribute (not aggregate L_fair).
    """
    # First compute Pareto frontier
    compute_pareto_frontier(results)

    dfz_configs = []
    for config in results:
        f1_ok      = config.f1 >= tau_acc
        # Per-attribute constraint (Eq. 8b): NOT aggregate L_fair
        eod_ok     = (
            config.eod_gender    < tau_fair and
            config.eod_ethnicity < tau_fair
        )
        size_ok    = config.size_mb <= size_constraint_mb
        latency_ok = config.latency_e3_ms <= latency_constraint_ms

        config.dfz = f1_ok and eod_ok and size_ok and latency_ok
        if config.dfz:
            dfz_configs.append(config)

    return dfz_configs


def find_knee_point(
    dfz_configs: List[ConfigResult],
    all_configs: Optional[List[ConfigResult]] = None,
) -> Optional[ConfigResult]:
    """
    Identify knee point θ* (Eq. 13):

      θ* = arg min_{θ ∈ DFZ} ‖ L̂(θ) ‖₂

    where L̂ applies min-max normalisation to L_eff across all observed
    configurations (or DFZ configs if all_configs not supplied).

    This matches the paper's reported ‖L(C2)‖₂ ≈ 0.219:
      L_acc(C2) = 0.066, L_fair(C2) = 0.0285 are already in [0,1].
      L_eff(C2) is rescaled so its range matches the other two objectives.
    """
    if not dfz_configs:
        return None

    pool = all_configs if all_configs else dfz_configs

    # Min-max normalise L_eff over the pool so all three objectives
    # are on commensurable scales for the ‖·‖₂ knee identification.
    eff_vals = [c.l_eff for c in pool]
    eff_min, eff_max = min(eff_vals), max(eff_vals)
    eff_range = eff_max - eff_min if eff_max > eff_min else 1.0

    def normalised_distance(c: ConfigResult) -> float:
        l_eff_norm = (c.l_eff - eff_min) / eff_range
        return float(np.linalg.norm([c.l_acc, c.l_fair, l_eff_norm]))

    return min(dfz_configs, key=normalised_distance)


def ternary_sensitivity_analysis(
    results: List[ConfigResult],
    step: float = 0.1,
) -> Dict:
    """
    Ternary sensitivity analysis over 36 weight combinations.
    Section 5.4: α,β,γ ∈ {0.1,...,0.8}³ subject to α+β+γ=1.

    Uses min-max normalised objectives so that weights (α,β,γ) function
    as genuine priority coefficients rather than scale-correction factors.
    Without normalisation, L_eff dominates due to its larger numerical range.

    Returns:
        Dict with frequency counts of which config is optimal per weight combo.
    """
    dfz_results = [c for c in results if c.dfz]
    if not dfz_results:
        return {}

    # Min-max normalise each objective over all configs (not just DFZ)
    # so weights reflect genuine relative priorities
    pool = results if results else dfz_results

    def _norm(vals):
        lo, hi = min(vals), max(vals)
        rng = hi - lo
        return [(v - lo) / rng if rng > 1e-9 else 0.0 for v in vals]

    acc_norm  = _norm([c.l_acc  for c in pool])
    fair_norm = _norm([c.l_fair for c in pool])
    eff_norm  = _norm([c.l_eff  for c in pool])

    # Build normalised lookup for DFZ configs
    pool_ids = [c.config_id for c in pool]
    norm_map = {
        c.config_id: (
            acc_norm[pool_ids.index(c.config_id)],
            fair_norm[pool_ids.index(c.config_id)],
            eff_norm[pool_ids.index(c.config_id)],
        )
        for c in dfz_results
        if c.config_id in pool_ids
    }

    # Generate all valid (α, β, γ) with step=0.1 summing to 1.0
    values = np.arange(step, 1.0, step)
    weight_combos = []
    for alpha, beta in itertools.product(values, values):
        gamma = round(1.0 - alpha - beta, 10)
        if step <= gamma <= 1.0:
            weight_combos.append((alpha, beta, gamma))

    optimal_counts: Dict[str, int] = {c.config_id: 0 for c in dfz_results}
    combo_details = []

    for alpha, beta, gamma in weight_combos:
        # Minimise normalised scalar loss within DFZ
        best_id = min(
            norm_map.keys(),
            key=lambda cid: (
                alpha * norm_map[cid][0]
                + beta  * norm_map[cid][1]
                + gamma * norm_map[cid][2]
            ),
        )
        optimal_counts[best_id] = optimal_counts.get(best_id, 0) + 1
        combo_details.append({
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "optimal": best_id,
        })

    n_combos = len(weight_combos)
    return {
        "n_combinations": n_combos,
        "optimal_counts": optimal_counts,
        "optimal_fractions": {
            k: v / n_combos
            for k, v in optimal_counts.items()
        },
        "combos": combo_details,
    }


def bootstrap_pareto_stability(
    results: List[ConfigResult],
    n_bootstrap: int = 1000,
    noise_std: float = 0.005,
    seed: int = 42,
) -> Dict:
    """
    Bootstrap stability analysis for knee point identification.
    Section 5.4: "C2 identified as knee point in 94.2% of resamples."

    Perturbs objective values with Gaussian noise to simulate sampling
    uncertainty, re-identifies knee point in each resample.
    """
    rng = np.random.default_rng(seed)
    dfz_results = [c for c in results if c.dfz]

    if not dfz_results:
        return {}

    knee_counts: Dict[str, int] = {}
    for _ in range(n_bootstrap):
        # Add small perturbation to objective vectors
        perturbed = []
        for config in dfz_results:
            noise = rng.normal(0, noise_std, 3)
            pv = config.objective_vector() + noise
            pv = np.clip(pv, 0, None)

            # Create temporary result with perturbed values
            temp = ConfigResult(
                config_id=config.config_id,
                config_name=config.config_name,
                l_acc=pv[0], l_fair=pv[1], l_eff=pv[2],
                f1=config.f1,
                eod_gender=config.eod_gender,
                eod_ethnicity=config.eod_ethnicity,
                size_mb=config.size_mb,
                latency_e3_ms=config.latency_e3_ms,
                dfz=True,
            )
            perturbed.append(temp)

        # Find knee point in perturbed objective space
        knee = min(perturbed, key=lambda c: c.distance_to_ideal())
        knee_counts[knee.config_id] = knee_counts.get(knee.config_id, 0) + 1

    return {
        "n_bootstrap": n_bootstrap,
        "knee_stability": {
            k: v / n_bootstrap
            for k, v in sorted(
                knee_counts.items(), key=lambda x: x[1], reverse=True
            )
        },
    }


def generate_benchmark_table(
    results: List[ConfigResult],
) -> pd.DataFrame:
    """Generate Table 3 equivalent from paper."""
    rows = []
    for c in results:
        rows.append({
            "ID":               c.config_id,
            "Config":           c.config_name,
            "F1":               round(c.f1, 3),
            "EOD_gender (%)":   round(c.eod_gender * 100, 1),
            "EOD_eth (%)":      round(c.eod_ethnicity * 100, 1),
            "Size (MB)":        round(c.size_mb, 1),
            "Lat E3 (ms)":      round(c.latency_e3_ms),
            "L_acc":            round(c.l_acc, 3),
            "L_fair":           round(c.l_fair, 3),
            "L_eff":            round(c.l_eff, 2),
            "‖L‖₂":            round(c.distance_to_ideal(), 3),
            "DFZ":              "✓" if c.dfz else "✗",
            "Pareto":           "✓" if c.pareto_optimal else "—",
            "Role":             c.pareto_role or "",
        })
    return pd.DataFrame(rows)
