"""
src/compression/pruning.py
Standard magnitude-based unstructured pruning (configs A3, C1).

Magnitude pruning is the fairness-UNconstrained baseline that
demonstrates why PFP is needed. At 80% sparsity it achieves
identical size/latency as PFP (6.3 MB, 187 ms) but with
significantly worse fairness (EOD_gender 8.3% vs 2.1%).

Key finding (Proposition 1): magnitude pruning disproportionately
removes low-magnitude neurons that encode minority-group features,
because minority subgroups produce weaker gradient signal during
training on imbalanced data. PFP prevents this via a per-neuron
EOD impact check.

Paper Table 5: A3 and B1 have identical L_eff = 1.26,
confirming A3 is strictly Pareto-dominated by B1.
"""

from __future__ import annotations

import copy
import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MagnitudePruner:
    """
    Standard unstructured magnitude-based pruning.
    Removes the weights with smallest absolute magnitude globally.

    Used for configurations A3 and C1.
    Has NO fairness constraint — demonstrates the fairness degradation
    that motivates PFP (Section 6.2, Proposition 1).

    Args:
        model:           Trained PyTorch model
        target_sparsity: Fraction of weights to zero out (paper: 0.80)
    """

    def __init__(
        self,
        model: nn.Module,
        target_sparsity: float = 0.80,
    ):
        self.model = copy.deepcopy(model)
        self.target_sparsity = target_sparsity

    def prune(self) -> nn.Module:
        """
        Apply one-shot global magnitude pruning.

        Collects all prunable weight tensors, sorts by absolute magnitude,
        zeros out the bottom (target_sparsity * total_weights) fraction.

        Returns:
            Pruned model.
        """
        # Collect all weight values and their locations
        all_weights: list[tuple] = []   # (param, flat_idx, abs_value)
        total_weights = 0

        for name, param in self.model.named_parameters():
            if "weight" not in name or param.dim() < 2:
                continue
            flat = param.data.abs().flatten()
            for i, v in enumerate(flat):
                all_weights.append((name, i, v.item()))
            total_weights += flat.numel()

        # Sort ascending by magnitude
        all_weights.sort(key=lambda x: x[2])

        # Zero out bottom (target_sparsity %) by weight magnitude
        n_to_prune = int(self.target_sparsity * total_weights)
        param_dict = dict(self.model.named_parameters())

        for name, flat_idx, _ in all_weights[:n_to_prune]:
            param = param_dict[name]
            shape = param.shape
            # Convert flat index back to multi-dimensional index
            nd_idx = tuple(
                int(i) for i in
                torch.tensor(flat_idx).reshape(1).expand(len(shape))
            )
            # Use unravel_index
            import numpy as np
            nd_idx = np.unravel_index(flat_idx, shape)
            param.data[nd_idx] = 0.0

        actual_sparsity = self._measure_sparsity()
        logger.info(
            f"[MagPrune] Done. Target={self.target_sparsity:.0%}, "
            f"Actual={actual_sparsity:.2%}"
        )
        return self.model

    def _measure_sparsity(self) -> float:
        total, zeros = 0, 0
        for name, param in self.model.named_parameters():
            if "weight" in name:
                total += param.numel()
                zeros += (param.data == 0).sum().item()
        return zeros / total if total > 0 else 0.0
