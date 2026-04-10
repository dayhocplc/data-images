"""
src/compression/pfp.py
Protected Fairness Pruning (PFP) — Algorithm B.1 from the paper.

Per-neuron EOD impact check prevents the removal of neurons
disproportionately important for minority-subgroup performance.

Reference: Thanh et al. "Protected Fairness Pruning for mobile ASD
screening," IEEE J. Biomed. Health Inform., vol. 28, no. 5, 2024.

Key insight (Proposition 1): DenseNet-121 trained on imbalanced data
encodes minority-group discriminative features in low-magnitude neurons.
Magnitude pruning removes these first, worsening fairness.
PFP prevents this via a per-neuron fairness gate.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.fairness import compute_eod

logger = logging.getLogger(__name__)


@dataclass
class PFPConfig:
    """Configuration for Protected Fairness Pruning."""
    target_sparsity: float = 0.80    # 80% weight sparsity (paper Section 4.4)
    eod_delta: float = 0.08          # δ: max allowed EOD increase per neuron
    n_pruning_steps: int = 10        # Iterative pruning steps
    fairness_batch_size: int = 256   # Batch size for EOD evaluation
    attributes: List[str] = field(
        default_factory=lambda: ["gender", "ethnicity"]
    )
    verbose: bool = True


class NeuronFairnessGate:
    """
    Evaluates per-neuron fairness impact.

    For each candidate neuron, temporarily masks it and measures
    the change in EOD. If ΔEOD > δ, the neuron is protected.
    """

    def __init__(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        config: PFPConfig,
        device: torch.device,
    ):
        self.model = model
        self.val_loader = val_loader
        self.config = config
        self.device = device

    @torch.no_grad()
    def eod_impact(
        self,
        layer_name: str,
        neuron_idx: int,
        baseline_eod: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Measure EOD change when neuron at (layer_name, neuron_idx) is zeroed.

        Returns:
            delta_eod: dict mapping attribute → ΔEOD
        """
        # Temporarily zero the neuron weights
        param = dict(self.model.named_parameters())[layer_name]
        original_weight = param.data[neuron_idx].clone()
        param.data[neuron_idx] = 0.0

        # Measure EOD after zeroing
        new_eod = compute_eod(
            self.model, self.val_loader, self.config.attributes, self.device
        )

        # Restore original weight
        param.data[neuron_idx] = original_weight

        delta_eod = {
            attr: new_eod[attr] - baseline_eod[attr]
            for attr in self.config.attributes
        }
        return delta_eod

    def is_protected(
        self,
        layer_name: str,
        neuron_idx: int,
        baseline_eod: Dict[str, float],
    ) -> bool:
        """
        Returns True if the neuron should NOT be pruned.

        A neuron is protected if removing it would increase EOD
        for ANY protected attribute by more than δ.
        """
        delta = self.eod_impact(layer_name, neuron_idx, baseline_eod)
        return any(d > self.config.eod_delta for d in delta.values())


class ProtectedFairnessPruning:
    """
    Protected Fairness Pruning (PFP).

    Iterative magnitude-based unstructured pruning with per-neuron
    EOD impact check. Prevents pruning of neurons disproportionately
    important for minority-subgroup performance.

    Achieves identical size/latency as standard magnitude pruning
    at 80% sparsity but with significantly lower demographic fairness gaps
    (paper Table 5: EOD_gender 8.3% → 2.1% vs. standard pruning).

    Usage:
        pfp = ProtectedFairnessPruning(model, val_loader, config, device)
        pruned_model = pfp.prune()
    """

    def __init__(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        config: PFPConfig = PFPConfig(),
        device: Optional[torch.device] = None,
    ):
        self.model = copy.deepcopy(model)
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.gate = NeuronFairnessGate(
            self.model, val_loader, config, self.device
        )
        self._pruning_history: List[Dict] = []

    def prune(self) -> nn.Module:
        """
        Execute iterative PFP.

        Returns:
            Pruned model with ~80% sparsity, fairness-protected.
        """
        current_sparsity = self._measure_sparsity()
        sparsity_per_step = (
            self.config.target_sparsity - current_sparsity
        ) / self.config.n_pruning_steps

        if self.config.verbose:
            logger.info(
                f"[PFP] Starting iterative pruning: "
                f"target={self.config.target_sparsity:.0%}, "
                f"steps={self.config.n_pruning_steps}, "
                f"δ={self.config.eod_delta}"
            )

        for step in range(self.config.n_pruning_steps):
            target_step_sparsity = current_sparsity + sparsity_per_step
            step_stats = self._pruning_step(target_step_sparsity, step)
            self._pruning_history.append(step_stats)

            current_sparsity = self._measure_sparsity()

            if self.config.verbose:
                logger.info(
                    f"[PFP] Step {step + 1}/{self.config.n_pruning_steps} "
                    f"sparsity={current_sparsity:.2%} "
                    f"protected={step_stats['n_protected']} neurons "
                    f"EOD_gender={step_stats['eod_gender']:.4f}"
                )

        final_sparsity = self._measure_sparsity()
        logger.info(f"[PFP] Done. Final sparsity: {final_sparsity:.2%}")
        return self.model

    def _pruning_step(
        self, target_sparsity: float, step: int
    ) -> Dict:
        """One step of iterative pruning with fairness gate."""
        # Compute baseline EOD before pruning this step
        baseline_eod = compute_eod(
            self.model, self.val_loader, self.config.attributes, self.device
        )

        # Collect all unpruned weights with their magnitudes
        candidates = self._get_pruning_candidates()

        # Sort by magnitude (ascending — smallest first)
        candidates.sort(key=lambda x: x["magnitude"])

        # Current sparsity → compute how many weights to prune
        current_sparsity = self._measure_sparsity()
        total_weights = self._count_weights()
        n_to_prune = int(
            (target_sparsity - current_sparsity) * total_weights
        )

        n_pruned = 0
        n_protected = 0
        pruned_layers: Dict[str, List[int]] = {}

        for candidate in candidates:
            if n_pruned >= n_to_prune:
                break

            layer_name = candidate["layer"]
            neuron_idx = candidate["idx"]

            # Fairness gate: skip neuron if its removal worsens EOD > δ
            if self.gate.is_protected(layer_name, neuron_idx, baseline_eod):
                n_protected += 1
                continue

            # Prune (zero out) the weight
            param = dict(self.model.named_parameters())[layer_name]
            param.data[neuron_idx] = 0.0
            pruned_layers.setdefault(layer_name, []).append(neuron_idx)
            n_pruned += 1

        # Measure EOD after step
        post_eod = compute_eod(
            self.model, self.val_loader, self.config.attributes, self.device
        )

        return {
            "step": step,
            "n_pruned": n_pruned,
            "n_protected": n_protected,
            "eod_gender": post_eod.get("gender", 0.0),
            "eod_ethnicity": post_eod.get("ethnicity", 0.0),
            "sparsity": self._measure_sparsity(),
        }

    def _get_pruning_candidates(self) -> List[Dict]:
        """Return list of {layer, idx, magnitude} for all unpruned weights."""
        candidates = []
        for name, param in self.model.named_parameters():
            if "weight" not in name or param.dim() < 2:
                continue
            # Only prune Conv and Linear layers
            magnitudes = param.data.abs().view(param.shape[0], -1).mean(dim=1)
            for idx, mag in enumerate(magnitudes):
                # Skip already-zeroed neurons
                if param.data[idx].abs().max() < 1e-9:
                    continue
                candidates.append({
                    "layer": name,
                    "idx": idx,
                    "magnitude": mag.item(),
                })
        return candidates

    def _measure_sparsity(self) -> float:
        """Compute global weight sparsity."""
        total, zeros = 0, 0
        for name, param in self.model.named_parameters():
            if "weight" in name:
                total += param.numel()
                zeros += (param.data == 0).sum().item()
        return zeros / total if total > 0 else 0.0

    def _count_weights(self) -> int:
        """Count total prunable weights."""
        return sum(
            p.numel()
            for n, p in self.model.named_parameters()
            if "weight" in n
        )

    def summary(self) -> Dict:
        """Return pruning summary statistics."""
        if not self._pruning_history:
            return {}
        final = self._pruning_history[-1]
        total_protected = sum(h["n_protected"] for h in self._pruning_history)
        return {
            "final_sparsity": final["sparsity"],
            "total_neurons_protected": total_protected,
            "final_eod_gender": final["eod_gender"],
            "final_eod_ethnicity": final["eod_ethnicity"],
            "n_steps": len(self._pruning_history),
        }
