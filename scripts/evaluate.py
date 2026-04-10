#!/usr/bin/env python3
"""
scripts/evaluate.py
Post-training evaluation for all configurations.

Runs evaluation on the fixed test set (accessed once per config),
computes bootstrap CIs, pairwise Wilcoxon tests, and DFZ qualification.
Reproduces paper Tables 3–6.

Usage:
    python scripts/evaluate.py --config configs/C2_m7_pfp.yaml \
        --checkpoint outputs/C2/C2_final.pt

    python scripts/evaluate.py --all --results_dir outputs/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import ASDFaceDataset
from src.data.splits import create_fixed_splits
from src.evaluation.metrics import evaluate
from src.evaluation.efficiency import measure_model_size_mb, measure_flops
from src.models.backbone import build_model

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_single(
    config_path: str,
    checkpoint_path: str,
    output_dir: Path,
    device: torch.device,
) -> dict:
    """Evaluate one configuration on the test set."""
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if "_base_" in cfg:
        base_path = Path(config_path).parent / cfg["_base_"]
        with open(base_path) as f:
            base = yaml.safe_load(f)
        base.update({k: v for k, v in cfg.items() if k != "_base_"})
        cfg = base

    # Build model
    model = build_model(
        backbone=cfg["model"]["backbone"],
        pretrained=False,
        dropout=cfg["model"]["head"]["dropout"],
        device=device,
    )
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device)
    )

    # Test set loader
    splits = create_fixed_splits(cfg["data"]["root"])
    test_ds = ASDFaceDataset(splits["test"], split="test")
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=32, shuffle=False, num_workers=4
    )

    logger.info(
        f"Evaluating {cfg['config_id']} on test set "
        f"(n={len(test_ds)})..."
    )

    # Full evaluation with bootstrap CI
    metrics = evaluate(model, test_loader, device, n_bootstrap=1000)
    metrics.check_dfz(
        tau_acc=cfg["fairness"]["tau_acc"],
        tau_fair=cfg["fairness"]["tau_fair"],
    )

    # Efficiency
    metrics.size_mb     = measure_model_size_mb(model)
    metrics.flops_g     = measure_flops(model)

    result = metrics.to_dict()
    result.update({
        "config_id":   cfg["config_id"],
        "config_name": cfg.get("strategy", cfg["config_id"]),
    })

    # Save
    out_path = output_dir / "test_results.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    _print_result(cfg["config_id"], metrics)
    return result


def _print_result(config_id: str, metrics):
    logger.info("=" * 60)
    logger.info(f"[{config_id}] Test Results:")
    logger.info(
        f"  F1          = {metrics.f1:.3f} "
        f"[{metrics.f1_ci_lower:.3f}, {metrics.f1_ci_upper:.3f}]"
    )
    logger.info(f"  Sensitivity = {metrics.sensitivity:.3f}")
    logger.info(f"  Specificity = {metrics.specificity:.3f}")
    logger.info(f"  EOD_gender  = {metrics.eod_gender*100:.1f}%")
    logger.info(f"  EOD_eth     = {metrics.eod_ethnicity*100:.1f}%")
    logger.info(f"  DPD_gender  = {metrics.dpd_gender*100:.1f}%")
    logger.info(f"  SPG         = {metrics.spg*100:.1f}%")
    logger.info(f"  Size        = {metrics.size_mb:.1f} MB")
    logger.info(f"  DFZ         = {'✓' if metrics.dfz_qualified else '✗'}")
    logger.info("=" * 60)


def pairwise_wilcoxon_tests(cv_results: dict) -> dict:
    """
    Pairwise Wilcoxon signed-rank tests on 5-fold CV F1 values.
    Holm-Bonferroni correction for 6 pre-specified comparisons.
    Paper Table 6.

    cv_results: {config_id: [f1_fold1, f1_fold2, ..., f1_fold5]}
    """
    # Pre-specified pairs (paper Table 6)
    pairs = [
        ("C2", "A3", "C2 vs. A3 (fairness-aware vs. std. pruning)"),
        ("C2", "B1", "C2 vs. B1 (combined vs. compress-only)"),
        ("C2", "C4", "C2 vs. C4 (M7 vs. M1 data quality)"),
        ("B1", "A3", "B1 vs. A3 (PFP vs. std. pruning, equal size)"),
    ]

    results = []
    p_values = []

    for id1, id2, label in pairs:
        if id1 not in cv_results or id2 not in cv_results:
            continue
        f1_1 = np.array(cv_results[id1])
        f1_2 = np.array(cv_results[id2])

        stat, p = scipy_stats.wilcoxon(f1_1, f1_2, alternative="two-sided")
        delta = np.mean(f1_1) - np.mean(f1_2)

        # Cohen's d from fold-level distributions
        pooled_std = np.sqrt(
            (np.std(f1_1, ddof=1)**2 + np.std(f1_2, ddof=1)**2) / 2
        )
        cohens_d = delta / max(pooled_std, 1e-9)

        p_values.append(p)
        results.append({
            "comparison": label,
            "config1": id1,
            "config2": id2,
            "delta_f1": float(delta),
            "p_raw": float(p),
            "cohens_d": float(cohens_d),
        })

    # Holm-Bonferroni correction
    if p_values:
        _, p_corrected, _, _ = multipletests(p_values, method="holm")
        for i, r in enumerate(results):
            r["p_corrected"] = float(p_corrected[i])
            r["significant_05"] = p_corrected[i] < 0.05
            r["significant_01"] = p_corrected[i] < 0.01

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     help="YAML config path")
    parser.add_argument("--checkpoint", help="Model checkpoint path")
    parser.add_argument("--all",        action="store_true",
                        help="Evaluate all configs in results_dir")
    parser.add_argument("--results_dir", default="outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.all:
        # Evaluate all checkpoints found in results_dir
        results_dir = Path(args.results_dir)
        for config_dir in sorted(results_dir.iterdir()):
            ckpt_files = list(config_dir.glob("*_final.pt"))
            cfg_files  = list((Path("configs")).glob(f"{config_dir.name}*.yaml"))
            if ckpt_files and cfg_files:
                evaluate_single(
                    str(cfg_files[0]),
                    str(ckpt_files[0]),
                    config_dir,
                    device,
                )
    elif args.config and args.checkpoint:
        config_id = Path(args.config).stem.split("_")[0]
        output_dir = Path(args.results_dir) / config_id
        evaluate_single(args.config, args.checkpoint, output_dir, device)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
