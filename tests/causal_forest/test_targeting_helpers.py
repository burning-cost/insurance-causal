"""
Tests for TargetingEvaluator internal methods and TargetingResult dataclass.

Covers:
- _toc_curve: shape, values at q=1.0 should be zero (all obs included)
- _compute_rate: autoc and qini branches, invalid method raises
- _bootstrap: returns correct shapes
- _compute_dr_scores: returns array of correct shape
- TargetingResult.plot_toc: returns a figure
- TargetingResult.summary: edge cases (non-significant, negative)

These tests do NOT require econml — they work directly on the private methods
using synthetic numpy arrays.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_causal.causal_forest.targeting import TargetingEvaluator, TargetingResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_dr():
    """Synthetic DR scores and CATE proxy."""
    rng = np.random.default_rng(42)
    n = 200
    # Simulate: higher tau_hat -> higher DR score (positive AUTOC)
    tau_hat = rng.normal(-0.2, 0.1, n)
    dr_scores = tau_hat + rng.normal(0, 0.05, n)  # correlated with tau_hat
    return dr_scores, tau_hat


@pytest.fixture
def evaluator():
    return TargetingEvaluator(n_bootstrap=10, n_toc_points=10, random_state=0)


# ---------------------------------------------------------------------------
# _toc_curve
# ---------------------------------------------------------------------------

class TestTocCurve:
    def test_returns_array_of_correct_length(self, evaluator, synthetic_dr):
        dr_scores, tau_hat = synthetic_dr
        q_grid = np.linspace(0.1, 1.0, 5)
        toc = evaluator._toc_curve(dr_scores, tau_hat, q_grid)
        assert toc.shape == (5,)

    def test_toc_at_q_one_near_zero(self, evaluator, synthetic_dr):
        """At q=1, all observations are included -> TOC(1) = mean - mean = 0."""
        dr_scores, tau_hat = synthetic_dr
        q_grid = np.array([1.0])
        toc = evaluator._toc_curve(dr_scores, tau_hat, q_grid)
        assert abs(toc[0]) < 1e-10, f"TOC(1) should be ~0, got {toc[0]}"

    def test_toc_finite(self, evaluator, synthetic_dr):
        dr_scores, tau_hat = synthetic_dr
        q_grid = np.linspace(0.05, 1.0, 10)
        toc = evaluator._toc_curve(dr_scores, tau_hat, q_grid)
        assert np.all(np.isfinite(toc))

    def test_toc_positive_when_correlated(self, evaluator):
        """When DR scores are perfectly ordered by tau_hat, TOC should be positive for q<1."""
        n = 100
        tau_hat = np.arange(n, dtype=float)  # perfectly ordered
        dr_scores = np.arange(n, dtype=float)  # same ordering
        q_grid = np.array([0.1, 0.5])
        toc = evaluator._toc_curve(dr_scores, tau_hat, q_grid)
        # Top 10% have highest dr_scores -> mean > global mean -> TOC > 0
        assert toc[0] > 0


# ---------------------------------------------------------------------------
# _compute_rate
# ---------------------------------------------------------------------------

class TestComputeRate:
    def test_autoc_returns_float(self, evaluator, synthetic_dr):
        dr_scores, tau_hat = synthetic_dr
        q_grid = evaluator.q_grid
        result = evaluator._compute_rate(dr_scores, tau_hat, q_grid, "autoc")
        assert isinstance(result, float)

    def test_qini_returns_float(self, evaluator, synthetic_dr):
        dr_scores, tau_hat = synthetic_dr
        q_grid = evaluator.q_grid
        result = evaluator._compute_rate(dr_scores, tau_hat, q_grid, "qini")
        assert isinstance(result, float)

    def test_invalid_method_raises(self, evaluator, synthetic_dr):
        dr_scores, tau_hat = synthetic_dr
        with pytest.raises(ValueError, match="Unknown method"):
            evaluator._compute_rate(dr_scores, tau_hat, evaluator.q_grid, "banana")

    def test_autoc_finite(self, evaluator, synthetic_dr):
        dr_scores, tau_hat = synthetic_dr
        result = evaluator._compute_rate(dr_scores, tau_hat, evaluator.q_grid, "autoc")
        assert np.isfinite(result)

    def test_qini_finite(self, evaluator, synthetic_dr):
        dr_scores, tau_hat = synthetic_dr
        result = evaluator._compute_rate(dr_scores, tau_hat, evaluator.q_grid, "qini")
        assert np.isfinite(result)

    def test_autoc_ge_qini_for_front_loaded_signal(self):
        """AUTOC weights early quantiles more heavily. With signal concentrated at top
        quantiles, AUTOC > QINI."""
        ev = TargetingEvaluator(n_bootstrap=5, n_toc_points=10, random_state=0)
        n = 100
        # All signal in top 10%: highest tau_hat rows have very high DR scores
        tau_hat = np.arange(n, dtype=float)
        dr_scores = np.zeros(n)
        dr_scores[-10:] = 10.0  # top 10% by tau_hat have high DR scores
        # AUTOC integrates toc/q; for front-loaded signal, AUTOC amplifies early gains
        autoc = ev._compute_rate(dr_scores, tau_hat, ev.q_grid, "autoc")
        qini = ev._compute_rate(dr_scores, tau_hat, ev.q_grid, "qini")
        assert autoc >= qini, f"AUTOC={autoc:.4f} should >= QINI={qini:.4f} for front-loaded signal"


# ---------------------------------------------------------------------------
# _bootstrap
# ---------------------------------------------------------------------------

class TestBootstrap:
    def test_returns_two_lists(self, evaluator, synthetic_dr):
        dr_scores, tau_hat = synthetic_dr
        rates, toc_curves = evaluator._bootstrap(dr_scores, tau_hat, evaluator.q_grid)
        assert isinstance(rates, list)
        assert isinstance(toc_curves, list)

    def test_boot_rates_length_equals_n_bootstrap(self, evaluator, synthetic_dr):
        dr_scores, tau_hat = synthetic_dr
        rates, _ = evaluator._bootstrap(dr_scores, tau_hat, evaluator.q_grid)
        assert len(rates) == evaluator.n_bootstrap

    def test_toc_curves_length_equals_n_bootstrap(self, evaluator, synthetic_dr):
        dr_scores, tau_hat = synthetic_dr
        _, toc_curves = evaluator._bootstrap(dr_scores, tau_hat, evaluator.q_grid)
        assert len(toc_curves) == evaluator.n_bootstrap

    def test_toc_curves_correct_shape(self, evaluator, synthetic_dr):
        dr_scores, tau_hat = synthetic_dr
        _, toc_curves = evaluator._bootstrap(dr_scores, tau_hat, evaluator.q_grid)
        n_points = len(evaluator.q_grid)
        for tc in toc_curves:
            assert tc.shape == (n_points,)

    def test_boot_rates_finite(self, evaluator, synthetic_dr):
        dr_scores, tau_hat = synthetic_dr
        rates, _ = evaluator._bootstrap(dr_scores, tau_hat, evaluator.q_grid)
        assert all(np.isfinite(r) for r in rates)

    def test_different_seeds_give_different_results(self, synthetic_dr):
        dr_scores, tau_hat = synthetic_dr
        ev1 = TargetingEvaluator(n_bootstrap=10, n_toc_points=5, random_state=0)
        ev2 = TargetingEvaluator(n_bootstrap=10, n_toc_points=5, random_state=99)
        rates1, _ = ev1._bootstrap(dr_scores, tau_hat, ev1.q_grid)
        rates2, _ = ev2._bootstrap(dr_scores, tau_hat, ev2.q_grid)
        assert rates1 != rates2


# ---------------------------------------------------------------------------
# _compute_dr_scores
# ---------------------------------------------------------------------------

class TestComputeDrScores:
    def test_returns_array_of_correct_length(self, evaluator):
        rng = np.random.default_rng(7)
        n = 100
        Y = rng.normal(0, 1, n)
        W = rng.normal(0, 0.1, n)
        X = rng.normal(0, 1, (n, 3))
        tau_hat = rng.normal(-0.2, 0.05, n)
        result = evaluator._compute_dr_scores(Y, W, X, tau_hat)
        assert result.shape == (n,)

    def test_returns_finite_values(self, evaluator):
        rng = np.random.default_rng(0)
        n = 80
        Y = rng.normal(0, 1, n)
        W = rng.normal(0, 0.1, n)
        X = rng.normal(0, 1, (n, 2))
        tau_hat = rng.normal(-0.2, 0.05, n)
        result = evaluator._compute_dr_scores(Y, W, X, tau_hat)
        assert np.all(np.isfinite(result))

    def test_near_zero_treatment_variance_warns(self):
        """If treatment has near-zero variance, DR computation warns and returns tau_hat."""
        ev = TargetingEvaluator(n_bootstrap=5, n_toc_points=5, random_state=0)
        n = 50
        Y = np.random.default_rng(0).normal(0, 1, n)
        W = np.ones(n) * 0.5  # constant treatment -> zero variance
        X = np.random.default_rng(0).normal(0, 1, (n, 2))
        tau_hat = np.random.default_rng(0).normal(-0.2, 0.05, n)
        with pytest.warns(UserWarning, match="near zero"):
            result = ev._compute_dr_scores(Y, W, X, tau_hat)
        # Falls back to tau_hat
        np.testing.assert_array_equal(result, tau_hat)


# ---------------------------------------------------------------------------
# TargetingResult.summary() edge cases
# ---------------------------------------------------------------------------

class TestTargetingResultSummary:
    def _make_result(self, autoc, autoc_se):
        q = np.linspace(0.05, 1.0, 5).tolist()
        toc = [0.0] * 5
        toc_df = pl.DataFrame({
            "q": q,
            "toc": toc,
            "se_lower": toc,
            "se_upper": toc,
        })
        ci_lower = autoc - 1.96 * autoc_se
        ci_upper = autoc + 1.96 * autoc_se
        return TargetingResult(
            autoc=autoc,
            autoc_se=autoc_se,
            autoc_ci_lower=ci_lower,
            autoc_ci_upper=ci_upper,
            qini=autoc * 0.8,
            n_obs=500,
            toc_curve=toc_df,
        )

    def test_summary_significant_positive(self):
        result = self._make_result(autoc=1.0, autoc_se=0.1)  # CI lower > 0
        s = result.summary()
        assert "significant" in s.lower()
        assert "informative" in s.lower()

    def test_summary_positive_not_significant(self):
        result = self._make_result(autoc=0.1, autoc_se=1.0)  # CI lower < 0, autoc > 0
        s = result.summary()
        assert "weak evidence" in s.lower() or "positive but" in s.lower()

    def test_summary_negative(self):
        result = self._make_result(autoc=-0.5, autoc_se=0.1)  # autoc < 0
        s = result.summary()
        assert "not significant" in s.lower() or "no evidence" in s.lower()

    def test_repr_contains_autoc(self):
        result = self._make_result(autoc=0.3, autoc_se=0.05)
        r = repr(result)
        assert "AUTOC" in r


# ---------------------------------------------------------------------------
# TargetingResult.plot_toc
# ---------------------------------------------------------------------------

class TestPlotToc:
    def test_returns_figure(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        q = np.linspace(0.05, 1.0, 5).tolist()
        toc = [0.1, 0.08, 0.05, 0.02, 0.0]
        toc_df = pl.DataFrame({
            "q": q,
            "toc": toc,
            "se_lower": [x - 0.01 for x in toc],
            "se_upper": [x + 0.01 for x in toc],
        })
        result = TargetingResult(
            autoc=0.05, autoc_se=0.02,
            autoc_ci_lower=0.01, autoc_ci_upper=0.09,
            qini=0.04, n_obs=200,
            toc_curve=toc_df,
        )
        fig = result.plot_toc()
        assert isinstance(fig, plt.Figure)
        plt.close("all")
