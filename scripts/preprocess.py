#!/usr/bin/env python3
"""
scripts/preprocess.py
Run the 4-stage preprocessing pipeline on raw dataset images.

Paper Section 4.2:
  Stage 1: MediaPipe Face Mesh detection (confidence ≥ 0.9, pose ≤ 30°)
  Stage 2: Inter-ocular alignment (θ_r = arctan(Δy/Δx))
  Stage 3: Crop + pad to 224×224 with neutral gray (RGB: 128, 128, 128)
  Stage 4: Quality filter (blur-exposure score ≥ 0.6)
  Retention: 2,821 / 2,936 = 96.1%

Usage:
    python scripts/preprocess.py \\
        --data_dir data/raw \\
        --output_dir data/processed
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import run_preprocessing
from src.data.splits import create_fixed_splits

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    required=True,
                        help="Raw dataset root (kaggle/ + vietnamese/)")
    parser.add_argument("--output_dir",  required=True,
                        help="Processed output directory")
    parser.add_argument("--quality_threshold", type=float, default=0.6)
    parser.add_argument("--create_splits", action="store_true",
                        help="Also create fixed 80/10/10 splits after preprocessing")
    args = parser.parse_args()

    # Run preprocessing
    stats = run_preprocessing(
        data_root=args.data_dir,
        output_dir=args.output_dir,
        quality_threshold=args.quality_threshold,
    )

    # Save stats
    stats_path = Path(args.output_dir) / "preprocessing_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "n_processed": len(stats["processed"]),
            "n_rejected":  len(stats["rejected"]),
            "retention":   len(stats["processed"]) / max(
                len(stats["processed"]) + len(stats["rejected"]), 1
            ),
            "quality_threshold": args.quality_threshold,
        }, f, indent=2)

    logger.info(f"Preprocessing stats saved to {stats_path}")

    # Create splits
    if args.create_splits:
        logger.info("Creating fixed 80/10/10 stratified splits...")
        splits = create_fixed_splits(
            data_root=args.output_dir,
            train_frac=0.80,
            val_frac=0.10,
            seed=42,
            stratify_cols=["label", "gender", "ethnicity", "age_group"],
        )
        for split_name, path in splits.items():
            logger.info(f"  {split_name}: {path}")

    logger.info("Preprocessing complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
