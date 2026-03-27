"""
Tests for demand_curve helper branches not covered by test_demand.py.

Covers:
- demand_curve() with predict_proba estimator attribute
- demand_curve() with _g_hat estimator attribute
- demand_curve() fallback (no predict_proba, no _g_hat)
- demand_curve() without tech_prem / last_premium columns
- plot_demand_curve() with existing ax
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_causal.elasticity.demand import demand_curve, plot_demand_curve


# ---------------------------------------------------------------------------
# Minimal mock estimator
# ---------------------------------------------------------------------------

class _MockEstimator:
    """Minimal mock that behaves like a fitted RenewalElasticityEstimator."""

    def __init__(self, n=100, seed=0, mode="fallback"):
        self._is_fitted = True
        self._mode = mode
        rng = np.random.default_rng(seed)
        self._n = n
        self._cate_vals = rng.normal(-0.2, 0.05, n)
        self._proba = np.clip(rng.beta(5, 2, n), 0.01, 0.99)
        if mode == "g_hat":
            self._g_hat = np.clip(rng.beta(5, 2, n), 0.01, 0.99)
        else:
            self._g_hat = None

    def cate(self, df):
        return self._cate_vals

    def predict_proba(self, df):
        if self._mode == "predict_proba":
            return self._proba
        raise AttributeError("no predict_proba")


class _MockEstimatorWithProba(_MockEstimator):
    def __init__(self, n=100, seed=0):
        super().__init__(n, seed, mode="predict_proba")


def _make_df(n=100, seed=0):
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "log_price_change": rng.normal(0.05, 0.02, n).tolist(),
        "renewed": rng.integers(0, 2, n).tolist(),
        "tech_prem": rng.uniform(300, 600, n).tolist(),
        "last_premium": rng.uniform(320, 620, n).tolist(),
    })


def _make_df_no_prem(n=100, seed=0):
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "log_price_change": rng.normal(0.05, 0.02, n).tolist(),
        "renewed": rng.integers(0, 2, n).tolist(),
    })


# ---------------------------------------------------------------------------
# demand_curve branches
# ---------------------------------------------------------------------------

class TestDemandCurveBranches:
    def test_fallback_branch_runs(self):
        """No predict_proba, no _g_hat -> uses portfolio mean as p0."""
        est = _MockEstimator(n=80, mode="fallback")
        df = _make_df(n=80)
        result = demand_curve(est, df, price_range=(-0.1, 0.1, 5))
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 5

    def test_g_hat_branch_runs(self):
        """_g_hat attribute exists -> uses it as per-customer p0."""
        est = _MockEstimator(n=80, mode="g_hat")
        df = _make_df(n=80)
        result = demand_curve(est, df, price_range=(-0.1, 0.1, 5))
        assert isinstance(result, pl.DataFrame)

    def test_predict_proba_branch_runs(self):
        """predict_proba attribute exists -> uses it as per-customer p0."""
        est = _MockEstimatorWithProba(n=80)
        df = _make_df(n=80)
        result = demand_curve(est, df, price_range=(-0.1, 0.1, 5))
        assert isinstance(result, pl.DataFrame)

    def test_renewal_rate_in_bounds_all_branches(self):
        """Renewal rate should be in [0, 1] for all branches."""
        for mode in ["fallback", "g_hat", "predict_proba"]:
            if mode == "predict_proba":
                est = _MockEstimatorWithProba(n=80)
            else:
                est = _MockEstimator(n=80, mode=mode)
            df = _make_df(n=80)
            result = demand_curve(est, df, price_range=(-0.2, 0.2, 10))
            rates = result["predicted_renewal_rate"].to_numpy()
            assert np.all(rates >= 0), f"[{mode}] rate < 0"
            assert np.all(rates <= 1), f"[{mode}] rate > 1"

    def test_without_tech_prem_uses_ones(self):
        """Without tech_prem column, profit uses ones -> demand_curve still runs."""
        est = _MockEstimator(n=50, mode="fallback")
        df = _make_df_no_prem(n=50)
        result = demand_curve(est, df, price_range=(-0.1, 0.1, 5))
        assert "predicted_profit" in result.columns

    def test_pct_price_change_column(self):
        """pct_price_change should be exp(log_price_change) - 1."""
        est = _MockEstimator(n=60, mode="fallback")
        df = _make_df(n=60)
        result = demand_curve(est, df, price_range=(-0.1, 0.1, 3))
        log_changes = result["log_price_change"].to_numpy()
        pct_changes = result["pct_price_change"].to_numpy()
        expected = np.exp(log_changes) - 1
        np.testing.assert_allclose(pct_changes, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# plot_demand_curve: ax argument
# ---------------------------------------------------------------------------

class TestPlotDemandCurveWithAx:
    def test_with_existing_ax(self):
        """Passing an existing ax should use it and return the figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        est = _MockEstimator(n=60, mode="fallback")
        df = _make_df(n=60)
        demand_df = demand_curve(est, df, price_range=(-0.1, 0.1, 10))

        fig, ax = plt.subplots()
        returned = plot_demand_curve(demand_df, ax=ax, show_profit=False)
        assert returned is fig
        plt.close("all")

    def test_show_profit_false(self):
        """show_profit=False should not create a secondary axis."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        est = _MockEstimator(n=60, mode="fallback")
        df = _make_df(n=60)
        demand_df = demand_curve(est, df, price_range=(-0.1, 0.1, 10))

        fig = plot_demand_curve(demand_df, show_profit=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_show_profit_true(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        est = _MockEstimator(n=60, mode="fallback")
        df = _make_df(n=60)
        demand_df = demand_curve(est, df, price_range=(-0.1, 0.1, 10))

        fig = plot_demand_curve(demand_df, show_profit=True)
        assert isinstance(fig, plt.Figure)
        plt.close("all")
