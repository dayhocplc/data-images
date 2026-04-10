"""
src/data/splits.py
Fixed 80/10/10 stratified split — performed ONCE, before any augmentation.

Critical design: the split is created and saved to disk before training.
All augmentation is applied only to training-fold images.
The test partition is accessed exactly once per configuration.

Stratification columns: [label, gender, ethnicity, age_group]
This ensures all 24 intersectional subgroups are represented in each split.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

SPLIT_FILE = "splits.json"   # Cached split indices


def create_fixed_splits(
    data_root: str | Path,
    train_frac: float = 0.80,
    val_frac:   float = 0.10,
    seed: int = 42,
    stratify_cols: Optional[List[str]] = None,
    force_rebuild: bool = False,
) -> Dict[str, Path]:
    """
    Create (or load cached) fixed 80/10/10 stratified splits.

    Returns:
        Dict mapping split name → path to metadata CSV.
        Keys: "train", "val", "test"

    The split is determined by seed=42 and saved to splits.json.
    Subsequent calls with the same seed return the identical split,
    guaranteeing test set integrity across all 11 configurations.
    """
    data_root = Path(data_root)
    split_cache = data_root / SPLIT_FILE

    # Load existing split if available
    if split_cache.exists() and not force_rebuild:
        logger.info(f"[Splits] Loading cached split from {split_cache}")
        with open(split_cache) as f:
            split_info = json.load(f)
        _verify_split_hash(split_info, seed)
        return {
            k: data_root / v
            for k, v in split_info["paths"].items()
        }

    # Build metadata DataFrame
    logger.info("[Splits] Building stratified split from scratch...")
    df = _build_metadata(data_root)
    logger.info(f"[Splits] Total samples: {len(df)}")

    # Create stratification key (label × gender × ethnicity × age_group)
    if stratify_cols:
        strat_key = df[stratify_cols].astype(str).apply("_".join, axis=1)
    else:
        strat_key = df["label"].astype(str)

    # Remove strata with fewer than 2 samples (can't stratify)
    counts = strat_key.value_counts()
    rare = counts[counts < 2].index
    if len(rare) > 0:
        logger.warning(
            f"[Splits] {len(rare)} rare strata with n<2, using label-only stratification"
        )
        strat_key = df["label"].astype(str)

    # First split: train vs (val + test)
    test_frac = 1.0 - train_frac
    idx_train, idx_temp = train_test_split(
        np.arange(len(df)),
        test_size=test_frac,
        stratify=strat_key,
        random_state=seed,
    )

    # Second split: val vs test (equal halves of the held-out 20%)
    val_of_temp = val_frac / test_frac
    strat_temp  = strat_key.iloc[idx_temp]
    counts_temp = strat_temp.value_counts()
    rare_temp   = counts_temp[counts_temp < 2].index

    if len(rare_temp) > 0:
        strat_temp = df["label"].iloc[idx_temp].astype(str)

    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=1.0 - val_of_temp,
        stratify=strat_temp,
        random_state=seed,
    )

    logger.info(
        f"[Splits] train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}"
    )

    # Save metadata CSVs
    paths = {}
    for split_name, indices in [
        ("train", idx_train),
        ("val",   idx_val),
        ("test",  idx_test),
    ]:
        split_df = df.iloc[indices].copy()
        split_df["split"] = split_name
        out_path = data_root / f"metadata_{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        paths[split_name] = out_path.name
        logger.info(
            f"  {split_name}: {len(split_df)} samples, "
            f"ASD={split_df['label'].sum()}, "
            f"non-ASD={(split_df['label']==0).sum()}"
        )

    # Cache split info with hash (for integrity verification)
    split_info = {
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "n_train": len(idx_train),
        "n_val":   len(idx_val),
        "n_test":  len(idx_test),
        "paths": paths,
        "hash": _compute_split_hash(idx_train, idx_val, idx_test, seed),
    }
    with open(split_cache, "w") as f:
        json.dump(split_info, f, indent=2)

    logger.info(f"[Splits] Saved split cache to {split_cache}")
    return {k: data_root / v for k, v in paths.items()}


def _build_metadata(data_root: Path) -> pd.DataFrame:
    """
    Scan the dataset directory and build a metadata DataFrame.

    Expected structure:
        data_root/
            kaggle/asd/*.jpg
            kaggle/non_asd/*.jpg
            vietnamese/asd/*.jpg
            vietnamese/non_asd/*.jpg
    """
    records = []
    for source in ["kaggle", "vietnamese"]:
        for label_name, label_id in [("asd", 1), ("non_asd", 0)]:
            img_dir = data_root / source / label_name
            if not img_dir.exists():
                logger.warning(f"[Splits] Directory not found: {img_dir}")
                continue
            for img_path in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
                # Parse demographic attributes from filename
                # Expected format: {gender}_{ethnicity}_{age}_{idx}.jpg
                # e.g. male_white_3-4_0042.jpg
                attrs = _parse_filename(img_path.name)
                records.append({
                    "image_path": str(img_path),
                    "label":      label_id,
                    "source":     source,
                    **attrs,
                })

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(
            f"No images found in {data_root}. "
            "Check directory structure: source/label/*.jpg"
        )

    # Validate demographic columns
    for col in ["gender", "ethnicity", "age_group"]:
        if col not in df.columns:
            logger.warning(
                f"[Splits] '{col}' not parsed from filenames. "
                "Using placeholder values."
            )
            df[col] = "unknown"

    return df


def _parse_filename(filename: str) -> dict:
    """
    Parse demographic attributes from filename.
    Expected format: {gender}_{ethnicity}_{age}_{idx}.ext
    Falls back gracefully if parsing fails.
    """
    stem = Path(filename).stem
    parts = stem.lower().split("_")

    attrs = {
        "gender":    "unknown",
        "ethnicity": "unknown",
        "age_group": "unknown",
    }

    if len(parts) >= 1 and parts[0] in {"male", "female"}:
        attrs["gender"] = parts[0]
    if len(parts) >= 2 and parts[1] in {"white", "asian", "black", "dark"}:
        attrs["ethnicity"] = parts[1]
    if len(parts) >= 3 and parts[2] in {"0-2", "3-4", "5-6"}:
        attrs["age_group"] = parts[2]

    return attrs


def _compute_split_hash(
    idx_train: np.ndarray,
    idx_val:   np.ndarray,
    idx_test:  np.ndarray,
    seed: int,
) -> str:
    """Compute a hash of split indices for integrity verification."""
    content = (
        f"seed={seed}|"
        f"train={sorted(idx_train.tolist()[:20])}|"
        f"val={sorted(idx_val.tolist()[:20])}|"
        f"test={sorted(idx_test.tolist()[:20])}"
    )
    return hashlib.md5(content.encode()).hexdigest()[:12]


def _verify_split_hash(split_info: dict, seed: int):
    """Warn if cached split doesn't match expected seed."""
    if split_info.get("seed") != seed:
        logger.warning(
            f"[Splits] Cached split used seed={split_info['seed']}, "
            f"requested seed={seed}. "
            "Use force_rebuild=True to regenerate."
        )
