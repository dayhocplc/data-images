#!/usr/bin/env python3
"""
scripts/train.py
Single-configuration training entry point.

Usage:
    python scripts/train.py --config configs/C2_m7_pfp.yaml --seed 42
    python scripts/train.py --config configs/B1_pfp.yaml --output_dir outputs/B1

All 11 configurations share the same hyperparameters (base.yaml).
Config-specific files override only augmentation and compression settings.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import ASDFaceDataset
from src.data.splits import create_fixed_splits
from src.evaluation.metrics import evaluate
from src.evaluation.pareto import ConfigResult, compute_dfz, find_knee_point
from src.models.backbone import build_model
from src.training.atws import AdaptiveTrilemmaWeightScheduler
from src.training.losses import TrilemmaLoss
from src.training.trainer import TrilemmaTrainer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML config, merging with base.yaml if specified."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if "_base_" in cfg:
        base_path = Path(config_path).parent / cfg["_base_"]
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f)
        # Merge: config overrides base
        base_cfg.update({k: v for k, v in cfg.items() if k != "_base_"})
        cfg = base_cfg

    return cfg


def build_dataloaders(cfg: dict, augment_fn=None):
    """Build train/val/test DataLoaders with fixed split."""
    split_metadata = create_fixed_splits(
        data_root=cfg["data"]["root"],
        train_frac=cfg["data"]["split"]["train"],
        val_frac=cfg["data"]["split"]["val"],
        seed=cfg["data"]["split"]["seed"],
        stratify_cols=cfg["data"]["split"]["stratify"],
    )

    train_ds = ASDFaceDataset(
        metadata_path=split_metadata["train"],
        split="train",
        augment_fn=augment_fn,
    )
    val_ds = ASDFaceDataset(
        metadata_path=split_metadata["val"],
        split="val",
    )
    test_ds = ASDFaceDataset(
        metadata_path=split_metadata["test"],
        split="test",
    )

    batch_size = cfg["training"]["batch_size"]
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    logger.info(
        f"Dataset: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )
    logger.info(f"Subgroup stats:\n{train_ds.get_subgroup_stats().to_string()}")
    return train_loader, val_loader, test_loader, train_ds


def build_augmentation(cfg: dict, device: torch.device):
    """Build augmentation function based on config."""
    aug_method = cfg.get("augmentation", {}).get("method", "m1_basic")

    if aug_method == "m7_3d_aware":
        from src.augmentation.aug_3d import AugConfig3D, ThreeDAwareAugmentation
        aug_cfg = AugConfig3D(
            minority_stratified=cfg["augmentation"].get("minority_stratified", True),
            expression_intensity=0.3,
        )
        logger.info("[Augmentation] M7: 3D-aware (DECA/FLAME)")
        return ThreeDAwareAugmentation(aug_cfg, device=device)

    elif aug_method == "m1_basic":
        from src.augmentation.standard_aug import BasicAugmentation
        logger.info("[Augmentation] M1: Basic 2D augmentation")
        return BasicAugmentation()

    elif aug_method == "none":
        logger.info("[Augmentation] No augmentation (baseline)")
        return None

    else:
        raise ValueError(f"Unknown augmentation method: {aug_method}")


def build_compression(cfg: dict, model, val_loader, device):
    """Apply compression method (PFP, INT8, KD) after base training."""
    compression_cfg = cfg.get("compression", {})
    method = compression_cfg.get("method", "none")

    if method == "pfp":
        from src.compression.pfp import PFPConfig, ProtectedFairnessPruning
        pfp_cfg = PFPConfig(
            target_sparsity=compression_cfg.get("sparsity", 0.80),
            eod_delta=compression_cfg.get("eod_threshold", 0.08),
        )
        logger.info(
            f"[Compression] PFP: sparsity={pfp_cfg.target_sparsity:.0%}, "
            f"δ={pfp_cfg.eod_delta}"
        )
        pfp = ProtectedFairnessPruning(model, val_loader, pfp_cfg, device)
        return pfp.prune()

    elif method == "int8":
        from src.compression.quantization import quantize_int8
        logger.info("[Compression] INT8 post-training quantization")
        return quantize_int8(model, val_loader, device)

    elif method == "pfp_int8":
        from src.compression.pfp import PFPConfig, ProtectedFairnessPruning
        from src.compression.quantization import quantize_int8
        # Sequential: PFP first, then INT8
        pfp_cfg = PFPConfig(target_sparsity=0.80, eod_delta=0.08)
        pfp = ProtectedFairnessPruning(model, val_loader, pfp_cfg, device)
        model = pfp.prune()
        logger.info("[Compression] B4: PFP → INT8 sequential pipeline")
        return quantize_int8(model, val_loader, device)

    elif method in ("none", ""):
        return model

    else:
        raise ValueError(f"Unknown compression method: {method}")


def main():
    parser = argparse.ArgumentParser(description="Train one trilemma configuration")
    parser.add_argument("--config", required=True, help="YAML config file path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--skip_test", action="store_true",
                        help="Skip test evaluation (for faster iteration)")
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Load config
    cfg = load_config(args.config)
    config_id = cfg.get("config_id", Path(args.config).stem)
    logger.info(f"Config: {config_id} — {cfg.get('description', '')}")

    # Output directory
    output_dir = Path(args.output_dir or f"outputs/{config_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        cfg["training"].get("device", "cuda")
        if torch.cuda.is_available()
        else "cpu"
    )
    logger.info(f"Device: {device}")

    # Build augmentation
    augment_fn = build_augmentation(cfg, device)

    # Build dataloaders
    train_loader, val_loader, test_loader, train_ds = build_dataloaders(
        cfg, augment_fn=augment_fn
    )

    # Build model
    model = build_model(
        backbone=cfg["model"]["backbone"],
        pretrained=cfg["model"]["pretrained"],
        dropout=cfg["model"]["head"]["dropout"],
    )

    # Build ATWS scheduler (if enabled)
    atws = None
    if cfg.get("training", {}).get("atws", False):
        max_epochs = cfg["training"]["max_epochs"]
        atws = AdaptiveTrilemmaWeightScheduler(
            max_epochs=max_epochs,
            override_eod_threshold=cfg["fairness"]["tau_fair"],
        )
        logger.info(f"[ATWS] Enabled. max_epochs={max_epochs}")

    # Build loss
    class_weights = train_ds.get_class_weights().to(device)
    loss_fn = TrilemmaLoss(
        alpha=0.7, beta=0.2, gamma=0.1,  # Phase I defaults
        class_weights=class_weights,
    )

    # Train
    trainer = TrilemmaTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        atws=atws,
        cfg=cfg,
        device=device,
        output_dir=output_dir,
    )
    trainer.fit()
    model = trainer.best_model

    # Apply compression (PFP, INT8, etc.)
    model = build_compression(cfg, model, val_loader, device)

    # Save model
    ckpt_path = output_dir / f"{config_id}_final.pt"
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"Saved model: {ckpt_path}")

    # ── Evaluate ──────────────────────────────────────────────────────────
    logger.info("Evaluating on validation set...")
    val_metrics = evaluate(model, val_loader, device, n_bootstrap=1000)
    val_metrics.check_dfz(
        tau_acc=cfg["fairness"]["tau_acc"],
        tau_fair=cfg["fairness"]["tau_fair"],
    )

    if not args.skip_test:
        logger.info(
            "Evaluating on TEST SET (accessed once, final results)..."
        )
        test_metrics = evaluate(model, test_loader, device, n_bootstrap=1000)
        test_metrics.check_dfz(
            tau_acc=cfg["fairness"]["tau_acc"],
            tau_fair=cfg["fairness"]["tau_fair"],
        )

        # Save test results
        results = test_metrics.to_dict()
        results["config_id"] = config_id
        results["config_name"] = cfg.get("strategy", config_id)
        results_path = output_dir / "test_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("=" * 60)
        logger.info(f"[{config_id}] TEST RESULTS:")
        logger.info(f"  F1          = {test_metrics.f1:.3f} "
                    f"[{test_metrics.f1_ci_lower:.3f}, {test_metrics.f1_ci_upper:.3f}]")
        logger.info(f"  Sensitivity = {test_metrics.sensitivity:.3f}")
        logger.info(f"  EOD_gender  = {test_metrics.eod_gender*100:.1f}%")
        logger.info(f"  EOD_eth     = {test_metrics.eod_ethnicity*100:.1f}%")
        logger.info(f"  L_fair      = {test_metrics.l_fair:.3f}")
        logger.info(f"  DFZ         = {'✓' if test_metrics.dfz_qualified else '✗'}")
        logger.info("=" * 60)
    else:
        logger.info("Test evaluation skipped (--skip_test).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
