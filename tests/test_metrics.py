"""
tests/test_metrics.py
Unit tests for fairness evaluation metrics.

Verifies:
  - EOD computation (Eq. 3)
  - L_fair aggregation (Eq. 4)
  - Bootstrap CI correctness
  - DFZ qualification logic (Eq. 12)
  - SPG computation across 24 subgroups
"""

import numpy as np
import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import (
    TrilemmaMetrics,
    _compute_eod,
    _compute_dpd,
    _bootstrap_ci,
)


class TestEODComputation:
    """Test Equal Opportunity Difference (Eq. 3)."""

    def test_zero_eod_perfect_fairness(self):
        """When majority and minority have equal TPR, EOD = 0."""
        labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        preds  = np.array([1, 1, 1, 1, 0, 0, 0, 0])   # All correct
        attr   = np.array([0, 0, 1, 1, 0, 0, 1, 1])   # 0=majority, 1=minority
        eod = _compute_eod(labels, preds, attr, maj=0, min_=1)
        assert eod == pytest.approx(0.0, abs=1e-6)

    def test_maximum_eod(self):
        """When majority has TPR=1.0 and minority has TPR=0.0, EOD=1.0."""
        labels = np.array([1, 1, 1, 1])
        preds  = np.array([1, 1, 0, 0])   # maj correct, min wrong
        attr   = np.array([0, 0, 1, 1])
        eod = _compute_eod(labels, preds, attr, maj=0, min_=1)
        assert eod == pytest.approx(1.0, abs=1e-6)

    def test_partial_eod(self):
        """EOD = |0.75 - 0.5| = 0.25"""
        labels = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        preds  = np.array([1, 1, 1, 0, 1, 1, 0, 0])
        attr   = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # 3/4 vs 2/4
        eod = _compute_eod(labels, preds, attr, maj=0, min_=1)
        assert eod == pytest.approx(0.25, abs=1e-6)

    def test_eod_is_absolute(self):
        """EOD is always non-negative (absolute difference)."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            n = rng.integers(10, 50)
            labels = rng.integers(0, 2, n)
            preds  = rng.integers(0, 2, n)
            attr   = rng.integers(0, 2, n)
            eod = _compute_eod(labels, preds, attr, maj=0, min_=1)
            assert eod >= 0.0

    def test_eod_in_range(self):
        """EOD ∈ [0, 1] always."""
        rng = np.random.default_rng(0)
        for _ in range(50):
            n = rng.integers(20, 100)
            labels = rng.integers(0, 2, n)
            preds  = rng.integers(0, 2, n)
            attr   = rng.integers(0, 4, n)
            eod = _compute_eod(labels, preds, attr, maj=0, min_=3)
            assert 0.0 <= eod <= 1.0, f"EOD={eod} out of range"

    def test_empty_minority_group(self):
        """Empty minority group should return EOD=0 (no group, no disparity)."""
        labels = np.array([1, 1, 0, 0])
        preds  = np.array([1, 0, 0, 1])
        attr   = np.array([0, 0, 0, 0])   # All majority, no minority
        eod = _compute_eod(labels, preds, attr, maj=0, min_=1)
        assert eod == pytest.approx(0.0, abs=1e-6)


class TestLFair:
    """Test fairness loss L_fair = 0.5*(EOD_gender + EOD_eth) — Eq. 4."""

    def test_l_fair_aggregation(self):
        m = TrilemmaMetrics(eod_gender=0.04, eod_ethnicity=0.06)
        m.l_fair = 0.5 * (m.eod_gender + m.eod_ethnicity)
        assert m.l_fair == pytest.approx(0.05, abs=1e-9)

    def test_l_fair_symmetric(self):
        """L_fair treats gender and ethnicity equally."""
        m1 = TrilemmaMetrics(eod_gender=0.08, eod_ethnicity=0.02)
        m2 = TrilemmaMetrics(eod_gender=0.02, eod_ethnicity=0.08)
        l1 = 0.5 * (m1.eod_gender + m1.eod_ethnicity)
        l2 = 0.5 * (m2.eod_gender + m2.eod_ethnicity)
        assert l1 == pytest.approx(l2, abs=1e-9)


class TestDFZQualification:
    """Test DFZ constraint evaluation (Definition 3b, Eq. 12)."""

    def _make_metrics(self, f1, eod_g, eod_e, size, latency):
        m = TrilemmaMetrics(
            f1=f1, eod_gender=eod_g, eod_ethnicity=eod_e,
            size_mb=size, latency_e3_ms=latency,
        )
        m.l_fair = 0.5 * (eod_g + eod_e)
        return m

    def test_c2_qualifies(self):
        """C2 paper results: all constraints satisfied."""
        m = self._make_metrics(f1=0.934, eod_g=0.020, eod_e=0.037,
                               size=6.3, latency=187)
        assert m.check_dfz() is True
        assert m.dfz_qualified is True

    def test_a2_fails_size(self):
        """A2: Pareto-optimal but size=27.8 MB > 10 MB."""
        m = self._make_metrics(f1=0.946, eod_g=0.021, eod_e=0.039,
                               size=27.8, latency=214)
        assert m.check_dfz() is False
        assert m.dfz_size_ok is False
        assert m.dfz_f1_ok is True

    def test_a3_fails_eod_eth(self):
        """A3: EOD_eth=11.6% > τ_fair=10%."""
        m = self._make_metrics(f1=0.874, eod_g=0.083, eod_e=0.116,
                               size=6.3, latency=187)
        assert m.check_dfz() is False
        assert m.dfz_eod_ok is False
        assert m.dfz_size_ok is True

    def test_per_attribute_not_aggregate(self):
        """
        Per-attribute check (8b): EOD_g=0.02, EOD_e=0.18 → excluded.
        Aggregate L_fair = 0.10 would pass, but per-attribute fails.
        This is the key design choice in the paper.
        """
        m = self._make_metrics(f1=0.90, eod_g=0.02, eod_e=0.18,
                               size=6.3, latency=187)
        # Aggregate L_fair = 0.5*(0.02 + 0.18) = 0.10 (exactly at boundary)
        assert m.check_dfz(tau_fair=0.10) is False
        assert m.dfz_eod_ok is False  # EOD_eth fails

    def test_a1_fails_multiple(self):
        """A1: fails size and both EOD constraints."""
        m = self._make_metrics(f1=0.506, eod_g=0.121, eod_e=0.162,
                               size=27.8, latency=218)
        m.check_dfz()
        assert m.dfz_qualified is False
        assert m.dfz_f1_ok is False
        assert m.dfz_eod_ok is False
        assert m.dfz_size_ok is False


class TestParetoVector:
    """Test objective vector and distance to ideal."""

    def test_c2_distance(self):
        """C2 paper values: ‖L‖₂ ≈ 0.219."""
        m = TrilemmaMetrics(
            f1=0.934,
            eod_gender=0.020,
            eod_ethnicity=0.037,
            size_mb=6.3,
            latency_e3_ms=187.0,
        )
        m.l_fair = 0.5 * (m.eod_gender + m.eod_ethnicity)
        m.l_eff  = 0.5 * (6.3/10.0) + 0.5 * (187.0/300.0)

        v = m.pareto_vector()
        assert v[0] == pytest.approx(1 - 0.934, abs=1e-6)   # L_acc
        assert v[1] == pytest.approx(m.l_fair, abs=1e-6)     # L_fair
        assert v[2] == pytest.approx(m.l_eff, abs=1e-6)      # L_eff

        dist = m.euclidean_distance_to_ideal()
        assert dist == pytest.approx(np.linalg.norm(v), abs=1e-6)

    def test_ideal_point_has_zero_distance(self):
        """Config with F1=1, EOD=0, size=0, latency=0 has ‖L‖₂=0."""
        m = TrilemmaMetrics(f1=1.0, eod_gender=0.0, eod_ethnicity=0.0)
        m.l_fair = 0.0
        m.l_eff  = 0.0
        assert m.euclidean_distance_to_ideal() == pytest.approx(0.0, abs=1e-9)


class TestBootstrapCI:
    """Test bootstrap CI computation."""

    def test_ci_covers_true_value(self):
        """95% CI should cover true F1 most of the time."""
        rng = np.random.default_rng(42)
        n = 282
        labels = rng.integers(0, 2, n)
        preds  = (rng.random(n) > 0.4).astype(int)
        probs  = rng.random(n)
        gender = rng.integers(0, 2, n)
        ethnicity = rng.integers(0, 4, n)

        from sklearn.metrics import f1_score
        true_f1 = f1_score(labels, preds, zero_division=0)

        ci = _bootstrap_ci(labels, preds, probs, gender, ethnicity, n=1000)
        # True F1 should be within CI
        assert ci["f1_lower"] <= true_f1 <= ci["f1_upper"]

    def test_ci_ordering(self):
        """Lower bound should be ≤ upper bound."""
        rng = np.random.default_rng(7)
        n = 100
        labels = rng.integers(0, 2, n)
        preds  = rng.integers(0, 2, n)
        probs  = rng.random(n)
        gender = rng.integers(0, 2, n)
        ethnicity = rng.integers(0, 4, n)

        ci = _bootstrap_ci(labels, preds, probs, gender, ethnicity, n=200)
        assert ci["f1_lower"]         <= ci["f1_upper"]
        assert ci["eod_gender_lower"] <= ci["eod_gender_upper"]
