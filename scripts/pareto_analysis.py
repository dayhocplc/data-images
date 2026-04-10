#!/usr/bin/env python3
"""
scripts/pareto_analysis.py
Post-run Pareto frontier and DFZ analysis.

Reads test_results.json from all 11 configuration output directories,
computes the Pareto frontier (Def. 2), DFZ (Def. 3), knee point (Eq. 13),
ternary sensitivity analysis, and bootstrap stability.

Reproduces paper Tables 3–5 and Section 5.4.

Usage:
    python scripts/pareto_analysis.py --results_dir outputs/
    python scripts/pareto_analysis.py --results_dir outputs/ --output analysis.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.pareto import (
    ConfigResult,
    bootstrap_pareto_stability,
    compute_dfz,
    compute_pareto_frontier,
    find_knee_point,
    generate_benchmark_table,
    ternary_sensitivity_analysis,
)

# DFZ thresholds from paper (Eq. 8a–8d)
TAU_ACC      = 0.85
TAU_FAIR     = 0.10
SIZE_MB      = 10.0
LATENCY_MS   = 300.0

# Efficiency normalization (Eq. 5)
S_STAR       = 10.0   # MB
T_STAR       = 300.0  # ms


def load_results(results_dir: str) -> list[ConfigResult]:
    """Load test_results.json from all config output directories."""
    results_dir = Path(results_dir)
    results = []

    for config_dir in sorted(results_dir.iterdir()):
        result_file = config_dir / "test_results.json"
        if not result_file.exists():
            continue

        with open(result_file) as f:
            data = json.load(f)

        # Compute L_eff from size and latency (Eq. 6)
        size_mb    = data.get("size_mb", 0.0)
        latency_ms = data.get("latency_e3_ms", 0.0)
        l_eff = 0.5 * (size_mb / S_STAR) + 0.5 * (latency_ms / T_STAR)

        eod_gender    = data.get("eod_gender", 0.0)
        eod_ethnicity = data.get("eod_ethnicity", 0.0)
        f1            = data.get("f1", 0.0)

        result = ConfigResult(
            config_id   = data.get("config_id", config_dir.name),
            config_name = data.get("config_name", config_dir.name),
            l_acc       = 1.0 - f1,
            l_fair      = 0.5 * (eod_gender + eod_ethnicity),
            l_eff       = l_eff,
            f1          = f1,
            eod_gender    = eod_gender,
            eod_ethnicity = eod_ethnicity,
            size_mb       = size_mb,
            latency_e3_ms = latency_ms,
        )
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--bootstrap_n", type=int, default=1000)
    args = parser.parse_args()

    print("Loading results...")
    results = load_results(args.results_dir)
    if not results:
        print(f"ERROR: No test_results.json found in {args.results_dir}")
        return 1

    print(f"Loaded {len(results)} configurations.")

    # ── Pareto frontier ──────────────────────────────────────────────────
    pareto = compute_pareto_frontier(results)
    print(f"\nPareto-optimal: {[c.config_id for c in pareto if c.pareto_optimal]}")

    # ── DFZ ──────────────────────────────────────────────────────────────
    dfz = compute_dfz(
        results,
        tau_acc=TAU_ACC,
        tau_fair=TAU_FAIR,
        size_constraint_mb=SIZE_MB,
        latency_constraint_ms=LATENCY_MS,
    )
    print(f"DFZ-qualified: {[c.config_id for c in dfz]}")

    # ── Knee point ───────────────────────────────────────────────────────
    knee = find_knee_point(dfz)
    if knee:
        knee.pareto_role = "knee_point"
        print(f"\nKnee point: {knee.config_id}")
        print(f"  F1        = {knee.f1:.3f}")
        print(f"  EOD_gen   = {knee.eod_gender*100:.1f}%")
        print(f"  EOD_eth   = {knee.eod_ethnicity*100:.1f}%")
        print(f"  Size      = {knee.size_mb:.1f} MB")
        print(f"  Latency   = {knee.latency_e3_ms:.0f} ms")
        print(f"  ‖L‖₂      = {knee.distance_to_ideal():.3f}")

    # ── Benchmark table (Table 3 equivalent) ─────────────────────────────
    table = generate_benchmark_table(results)
    print("\n" + "=" * 80)
    print("BENCHMARK TABLE (Table 3)")
    print("=" * 80)
    print(table.to_string(index=False))

    # ── Ternary sensitivity analysis ─────────────────────────────────────
    print("\nRunning ternary sensitivity analysis (36 weight combinations)...")
    sensitivity = ternary_sensitivity_analysis(results)
    print(f"  n_combinations: {sensitivity.get('n_combinations', 0)}")
    print("  Optimal fractions:")
    for cfg_id, frac in sorted(
        sensitivity.get("optimal_fractions", {}).items(),
        key=lambda x: x[1], reverse=True
    ):
        print(f"    {cfg_id}: {frac*100:.1f}%")

    # ── Bootstrap stability ───────────────────────────────────────────────
    print(f"\nRunning bootstrap stability (n={args.bootstrap_n})...")
    stability = bootstrap_pareto_stability(results, n_bootstrap=args.bootstrap_n)
    print("  Knee point stability:")
    for cfg_id, frac in sorted(
        stability.get("knee_stability", {}).items(),
        key=lambda x: x[1], reverse=True
    ):
        print(f"    {cfg_id}: {frac*100:.1f}%")

    # ── Save results ──────────────────────────────────────────────────────
    output = {
        "pareto_optimal":  [c.config_id for c in results if c.pareto_optimal],
        "dfz_qualified":   [c.config_id for c in results if c.dfz],
        "knee_point":      knee.config_id if knee else None,
        "knee_metrics": {
            "f1":           knee.f1           if knee else None,
            "eod_gender":   knee.eod_gender   if knee else None,
            "eod_eth":      knee.eod_ethnicity if knee else None,
            "size_mb":      knee.size_mb      if knee else None,
            "latency_ms":   knee.latency_e3_ms if knee else None,
            "dist_ideal":   knee.distance_to_ideal() if knee else None,
        },
        "sensitivity_analysis": sensitivity,
        "bootstrap_stability":  stability,
        "benchmark_table": table.to_dict(orient="records"),
    }

    out_path = args.output or str(Path(args.results_dir) / "pareto_analysis.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved analysis: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
