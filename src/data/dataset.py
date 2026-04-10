"""
src/data/dataset.py
ASD facial image dataset with intersectional demographic attributes.
Supports 24 subgroups: gender × ethnicity × age_group.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ── Demographic group definitions ───────────────────────────────────────────
GENDER_GROUPS   = {"male": 0, "female": 1}
ETHNICITY_GROUPS = {"white": 0, "asian": 1, "black": 2, "dark": 3}
AGE_GROUPS      = {"0-2": 0, "3-4": 1, "5-6": 2}

# Worst-case intersectional imbalance: 35.47:1
# Largest: Male×White×ASD (n=674), Smallest: Female×Dark×ASD (n=19)
MAJORITY_GROUP = {"gender": "male", "ethnicity": "white"}
MINORITY_GROUP = {"gender": "female", "ethnicity": "dark"}


class ASDFaceDataset(Dataset):
    """
    Facial image dataset for ASD binary classification.

    Demographic attributes enable per-subgroup fairness evaluation
    (EOD, DPD, SPG across 24 intersectional cells).

    Args:
        metadata_path: Path to CSV with columns:
            [image_path, label, gender, ethnicity, age_group, source]
        split: One of "train", "val", "test"
        transform: torchvision transforms
        augment_fn: Optional callable for 3D-aware augmentation (train only)
    """

    def __init__(
        self,
        metadata_path: str | Path,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        augment_fn=None,
    ):
        self.metadata_path = Path(metadata_path)
        self.split = split
        self.transform = transform or self._default_transform()
        self.augment_fn = augment_fn  # Applied only during training

        self.df = self._load_metadata()
        self._validate_metadata()

        # Pre-compute subgroup indices for efficient per-group evaluation
        self.subgroup_indices = self._build_subgroup_indices()

    # ── Loading ──────────────────────────────────────────────────────────────

    def _load_metadata(self) -> pd.DataFrame:
        df = pd.read_csv(self.metadata_path)
        df = df[df["split"] == self.split].reset_index(drop=True)

        # Encode demographics
        df["gender_id"]    = df["gender"].map(GENDER_GROUPS)
        df["ethnicity_id"] = df["ethnicity"].map(ETHNICITY_GROUPS)
        df["age_id"]       = df["age_group"].map(AGE_GROUPS)

        # Intersectional subgroup index (for SPG calculation)
        df["subgroup_id"] = (
            df["gender_id"] * len(ETHNICITY_GROUPS) * len(AGE_GROUPS)
            + df["ethnicity_id"] * len(AGE_GROUPS)
            + df["age_id"]
        )
        return df

    def _validate_metadata(self):
        required = {"image_path", "label", "gender", "ethnicity", "age_group", "split"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Metadata missing columns: {missing}")

        n = len(self.df)
        if self.split == "train" and not (2000 < n < 2600):
            import warnings
            warnings.warn(f"Unexpected train size: {n} (expected ~2258)")

    def _build_subgroup_indices(self) -> Dict[str, np.ndarray]:
        """Build index arrays for each (gender, ethnicity) subgroup."""
        indices = {}
        for g_name, g_id in GENDER_GROUPS.items():
            for e_name, e_id in ETHNICITY_GROUPS.items():
                key = f"{g_name}_{e_name}"
                mask = (self.df["gender_id"] == g_id) & (self.df["ethnicity_id"] == e_id)
                indices[key] = np.where(mask)[0]
        return indices

    # ── Default transforms ───────────────────────────────────────────────────

    @staticmethod
    def _default_transform() -> transforms.Compose:
        """ImageNet-normalized transform — applied to ALL splits."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # Load image
        img_path = row["image_path"]
        image = Image.open(img_path).convert("RGB")

        # 3D-aware augmentation (train split only, never val/test)
        if self.augment_fn is not None and self.split == "train":
            image = self.augment_fn(image, row.to_dict())

        image = self.transform(image)

        return {
            "image":        image,
            "label":        torch.tensor(row["label"], dtype=torch.float32),
            "gender":       torch.tensor(row["gender_id"], dtype=torch.long),
            "ethnicity":    torch.tensor(row["ethnicity_id"], dtype=torch.long),
            "age_group":    torch.tensor(row["age_id"], dtype=torch.long),
            "subgroup_id":  torch.tensor(row["subgroup_id"], dtype=torch.long),
            "source":       row.get("source", "unknown"),
            "image_path":   str(img_path),
        }

    # ── Utility ──────────────────────────────────────────────────────────────

    def get_class_weights(self) -> torch.Tensor:
        """Balanced class weights for BCE loss."""
        n_pos = (self.df["label"] == 1).sum()
        n_neg = (self.df["label"] == 0).sum()
        n_total = len(self.df)
        w_pos = n_total / (2 * n_pos)
        w_neg = n_total / (2 * n_neg)
        return torch.tensor([w_neg, w_pos], dtype=torch.float32)

    def get_subgroup_stats(self) -> pd.DataFrame:
        """Return per-subgroup sample counts for imbalance analysis."""
        stats = []
        for g_name, g_id in GENDER_GROUPS.items():
            for e_name, e_id in ETHNICITY_GROUPS.items():
                mask = (
                    (self.df["gender_id"] == g_id)
                    & (self.df["ethnicity_id"] == e_id)
                )
                sub = self.df[mask]
                stats.append({
                    "gender":    g_name,
                    "ethnicity": e_name,
                    "n_total":   len(sub),
                    "n_asd":     (sub["label"] == 1).sum(),
                    "n_non_asd": (sub["label"] == 0).sum(),
                })
        df_stats = pd.DataFrame(stats)
        df_stats["imbalance_ratio"] = (
            df_stats["n_total"].max() / df_stats["n_total"].clip(lower=1)
        )
        return df_stats

    def minority_indices(self) -> np.ndarray:
        """Return indices of minority-group samples (female × dark)."""
        return self.subgroup_indices.get("female_dark", np.array([]))

    def __repr__(self) -> str:
        n_pos = (self.df["label"] == 1).sum()
        n_neg = (self.df["label"] == 0).sum()
        return (
            f"ASDFaceDataset(split={self.split}, n={len(self.df)}, "
            f"ASD={n_pos}, non-ASD={n_neg}, "
            f"subgroups={self.df['subgroup_id'].nunique()})"
        )
