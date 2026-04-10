"""
tests/test_pfp.py
Unit tests for Protected Fairness Pruning.
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.compression.pfp import PFPConfig, ProtectedFairnessPruning


class TinyModel(nn.Module):
    """Minimal model for testing PFP."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def make_fake_loader(n=100, device="cpu"):
    """Create a fake DataLoader returning batches with demographic attributes."""
    dataset = []
    for i in range(n):
        dataset.append({
            "image":     torch.randn(3, 224, 224),
            "label":     torch.tensor(float(i % 2)),
            "gender":    torch.tensor(i % 2),
            "ethnicity": torch.tensor(i % 4),
        })
    return torch.utils.data.DataLoader(dataset, batch_size=16)


class TestPFPConfig:
    def test_defaults(self):
        cfg = PFPConfig()
        assert cfg.target_sparsity == 0.80
        assert cfg.eod_delta == 0.08
        assert "gender" in cfg.attributes
        assert "ethnicity" in cfg.attributes

    def test_custom_config(self):
        cfg = PFPConfig(target_sparsity=0.70, eod_delta=0.05)
        assert cfg.target_sparsity == 0.70
        assert cfg.eod_delta == 0.05


class TestProtectedFairnessPruning:
    def setup_method(self):
        self.device = torch.device("cpu")
        self.model = TinyModel()
        self.loader = make_fake_loader()
        self.cfg = PFPConfig(
            target_sparsity=0.50,
            n_pruning_steps=3,
            verbose=False,
        )

    def test_pfp_reduces_sparsity(self):
        """After PFP, model should have higher sparsity than before."""
        pfp = ProtectedFairnessPruning(
            self.model, self.loader, self.cfg, self.device
        )
        # Measure initial sparsity
        initial_sparsity = pfp._measure_sparsity()

        # Run pruning
        pruned = pfp.prune()

        final_sparsity = pfp._measure_sparsity()
        assert final_sparsity > initial_sparsity, (
            f"Expected sparsity to increase: {initial_sparsity:.2%} → {final_sparsity:.2%}"
        )

    def test_pfp_does_not_exceed_target(self):
        """PFP should not exceed target sparsity by more than 5pp."""
        pfp = ProtectedFairnessPruning(
            self.model, self.loader, self.cfg, self.device
        )
        pfp.prune()
        final = pfp._measure_sparsity()
        # Allow 5pp tolerance due to fairness protection
        assert final <= self.cfg.target_sparsity + 0.05, (
            f"Sparsity {final:.2%} exceeds target {self.cfg.target_sparsity:.2%}"
        )

    def test_pfp_summary_keys(self):
        """Summary should contain expected keys."""
        pfp = ProtectedFairnessPruning(
            self.model, self.loader, self.cfg, self.device
        )
        pfp.prune()
        summary = pfp.summary()
        for key in ["final_sparsity", "total_neurons_protected",
                    "final_eod_gender", "n_steps"]:
            assert key in summary, f"Missing key: {key}"

    def test_pfp_preserves_model_structure(self):
        """Pruned model should have same architecture."""
        pfp = ProtectedFairnessPruning(
            self.model, self.loader, self.cfg, self.device
        )
        pruned = pfp.prune()

        # Check layer names match
        orig_layers = set(n for n, _ in self.model.named_parameters())
        pruned_layers = set(n for n, _ in pruned.named_parameters())
        assert orig_layers == pruned_layers

    def test_pfp_candidate_collection(self):
        """Should collect candidates from all weight layers."""
        pfp = ProtectedFairnessPruning(
            self.model, self.loader, self.cfg, self.device
        )
        candidates = pfp._get_pruning_candidates()
        assert len(candidates) > 0
        for c in candidates:
            assert "layer" in c
            assert "idx" in c
            assert "magnitude" in c
            assert c["magnitude"] >= 0


class TestATWS:
    """Tests for Adaptive Trilemma Weight Scheduler."""

    def test_phase1_weights(self):
        from src.training.atws import AdaptiveTrilemmaWeightScheduler
        sched = AdaptiveTrilemmaWeightScheduler(max_epochs=100)
        w = sched.step(epoch=1)
        # Phase I: α dominant
        assert w.alpha > w.beta
        assert w.alpha > w.gamma

    def test_weights_sum_to_one(self):
        from src.training.atws import AdaptiveTrilemmaWeightScheduler
        sched = AdaptiveTrilemmaWeightScheduler(max_epochs=100)
        for epoch in [1, 20, 40, 60, 80, 99]:
            w = sched.step(epoch=epoch, val_f1=0.90, val_eod_gender=0.03,
                           val_eod_ethnicity=0.04)
            total = w.alpha + w.beta + w.gamma
            assert abs(total - 1.0) < 1e-6, (
                f"Epoch {epoch}: weights sum = {total:.6f} ≠ 1.0"
            )

    def test_override1_increases_beta(self):
        """When EOD exceeds threshold, β should increase."""
        from src.training.atws import AdaptiveTrilemmaWeightScheduler
        sched = AdaptiveTrilemmaWeightScheduler(max_epochs=100)
        # Normal step
        w_normal = sched.step(epoch=50, val_f1=0.90,
                               val_eod_gender=0.03, val_eod_ethnicity=0.03)
        # Reset for clean state
        sched2 = AdaptiveTrilemmaWeightScheduler(max_epochs=100)
        # EOD violation
        w_violation = sched2.step(epoch=50, val_f1=0.90,
                                   val_eod_gender=0.15, val_eod_ethnicity=0.03)
        assert w_violation.beta > w_normal.beta

    def test_override2_resets_to_phase1(self):
        """When F1 drops below threshold, weights reset to Phase I."""
        from src.training.atws import (
            AdaptiveTrilemmaWeightScheduler, PHASE1_WEIGHTS
        )
        sched = AdaptiveTrilemmaWeightScheduler(max_epochs=100)
        # Low F1 in Phase III
        w = sched.step(epoch=80, val_f1=0.75,
                       val_eod_gender=0.03, val_eod_ethnicity=0.03)
        # Should reset: α should be high like Phase I
        assert abs(w.alpha - PHASE1_WEIGHTS.normalize().alpha) < 0.02

    def test_history_recorded(self):
        from src.training.atws import AdaptiveTrilemmaWeightScheduler
        sched = AdaptiveTrilemmaWeightScheduler(max_epochs=50)
        for epoch in range(10):
            sched.step(epoch=epoch, val_f1=0.88, val_eod_gender=0.04,
                       val_eod_ethnicity=0.05)
        assert len(sched.history) == 10
        assert sched.history[0]["epoch"] == 0
        assert sched.history[-1]["epoch"] == 9


class TestPareto:
    """Tests for Pareto frontier and DFZ analysis."""

    def _make_config(self, cid, f1, eod_g, eod_e, size, latency):
        from src.evaluation.pareto import ConfigResult
        l_acc  = 1 - f1
        l_fair = 0.5 * (eod_g + eod_e)
        l_eff  = 0.5 * (size/10.0) + 0.5 * (latency/300.0)
        return ConfigResult(
            config_id=cid, config_name=cid,
            l_acc=l_acc, l_fair=l_fair, l_eff=l_eff,
            f1=f1, eod_gender=eod_g, eod_ethnicity=eod_e,
            size_mb=size, latency_e3_ms=latency,
        )

    def test_pareto_dominance_detected(self):
        from src.evaluation.pareto import is_pareto_dominated
        # B strictly dominates A on all objectives
        a = self._make_config("A", f1=0.87, eod_g=0.08, eod_e=0.11, size=6.3, latency=187)
        b = self._make_config("B", f1=0.90, eod_g=0.02, eod_e=0.04, size=6.3, latency=187)
        # A3 should be dominated by B1 (paper finding)
        assert is_pareto_dominated(a, [a, b])
        assert not is_pareto_dominated(b, [a, b])

    def test_dfz_qualification(self):
        from src.evaluation.pareto import compute_dfz
        configs = [
            self._make_config("C2", f1=0.934, eod_g=0.020, eod_e=0.037, size=6.3, latency=187),
            self._make_config("A2", f1=0.946, eod_g=0.021, eod_e=0.039, size=27.8, latency=214),
            self._make_config("A3", f1=0.874, eod_g=0.083, eod_e=0.116, size=6.3, latency=187),
        ]
        dfz = compute_dfz(configs)
        dfz_ids = {c.config_id for c in dfz}
        assert "C2" in dfz_ids           # Qualifies
        assert "A2" not in dfz_ids       # Too large
        assert "A3" not in dfz_ids       # EOD_eth violation

    def test_knee_point_is_c2(self):
        """C2 should be knee point (closest to ideal) among DFZ configs."""
        from src.evaluation.pareto import compute_dfz, find_knee_point
        # Reproduce paper Table 3 results
        configs = [
            self._make_config("B1", f1=0.904, eod_g=0.021, eod_e=0.039, size=6.3, latency=187),
            self._make_config("B2", f1=0.919, eod_g=0.024, eod_e=0.042, size=7.1, latency=142),
            self._make_config("B3", f1=0.891, eod_g=0.031, eod_e=0.050, size=4.2, latency=96),
            self._make_config("B4", f1=0.887, eod_g=0.023, eod_e=0.040, size=3.1, latency=91),
            self._make_config("C2", f1=0.934, eod_g=0.020, eod_e=0.037, size=6.3, latency=187),
            self._make_config("C3", f1=0.938, eod_g=0.022, eod_e=0.038, size=7.1, latency=142),
            self._make_config("C4", f1=0.871, eod_g=0.024, eod_e=0.041, size=6.3, latency=187),
        ]
        dfz = compute_dfz(configs)
        knee = find_knee_point(dfz)
        assert knee is not None
        assert knee.config_id == "C2", (
            f"Expected C2 as knee, got {knee.config_id} (‖L‖₂={knee.distance_to_ideal():.3f})"
        )
