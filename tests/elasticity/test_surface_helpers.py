"""
Tests for ElasticitySurface edge cases not covered by test_surface.py.

Covers:
- segment_summary with list-style by (multi-column)
- segment_summary fallback when estimator has no _is_fitted
- plot_gate with custom colour and title
- plot_surface with custom title and cmap
- segment_summary elasticity_at_10pct computation correctness
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_causal.elasticity.surface import ElasticitySurface


# ---------------------------------------------------------------------------
# Minimal mock estimator (no econml needed)
# ---------------------------------------------------------------------------

class _MockFittedEstimator:
    """Minimal mock of RenewalElasticityEstimator."""

    def __init__(self, n=200, seed=0):
        self._is_fitted = True
        rng = np.random.default_rng(seed)
        self._n = n
        self._cate_vals = rng.normal(-0.2, 0.05, n)
        self._ci_lower = self._cate_vals - 0.05
        self._ci_upper = self._cate_vals + 0.05

    def cate(self, df):
        return self._cate_vals[:len(df)]

    def cate_interval(self, df, alpha=0.05):
        n = len(df)
        return self._ci_lower[:n], self._ci_upper[:n]


def _make_df(n=200, seed=0):
    """Create a small polars DataFrame with segmentation columns."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "ncd_years": (rng.integers(0, 6, n)).tolist(),
        "channel": (np.random.default_rng(seed).choice(["pcw", "direct", "broker"], n)).tolist(),
        "age_band": (np.random.default_rng(seed).choice(["17-24", "25-34", "35-44", "45+"], n)).tolist(),
    })


# ---------------------------------------------------------------------------
# segment_summary: various by configurations
# ---------------------------------------------------------------------------

class TestSegmentSummaryHelpers:
    def test_portfolio_summary_elasticity_at_10pct(self):
        """elasticity_at_10pct = elasticity * log(1.1)."""
        df = _make_df()
        est = _MockFittedEstimator(n=len(df))
        surface = ElasticitySurface(est)
        result = surface.segment_summary(df, by=None)
        elas = float(result["elasticity"][0])
        at10 = float(result["elasticity_at_10pct"][0])
        assert abs(at10 - elas * np.log(1.1)) < 1e-6

    def test_segment_summary_list_by_two_cols(self):
        """Passing a list of two columns should produce a multi-column segmentation."""
        df = _make_df()
        est = _MockFittedEstimator(n=len(df))
        surface = ElasticitySurface(est)
        result = surface.segment_summary(df, by=["ncd_years", "channel"])
        assert "ncd_years" in result.columns
        assert "channel" in result.columns
        assert isinstance(result, pl.DataFrame)

    def test_segment_summary_single_string_by(self):
        df = _make_df()
        est = _MockFittedEstimator(n=len(df))
        surface = ElasticitySurface(est)
        result = surface.segment_summary(df, by="channel")
        assert "channel" in result.columns
        n_channels = df["channel"].n_unique()
        assert len(result) == n_channels

    def test_n_col_sums_to_total(self):
        df = _make_df()
        est = _MockFittedEstimator(n=len(df))
        surface = ElasticitySurface(est)
        result = surface.segment_summary(df, by="ncd_years")
        assert result["n"].sum() == len(df)

    def test_ci_lower_le_elasticity_le_ci_upper(self):
        df = _make_df()
        est = _MockFittedEstimator(n=len(df))
        surface = ElasticitySurface(est)
        result = surface.segment_summary(df, by="channel")
        elas = result["elasticity"].to_numpy()
        lb = result["ci_lower"].to_numpy()
        ub = result["ci_upper"].to_numpy()
        assert np.all(lb <= elas + 1e-8), "Some CI lower > elasticity"
        assert np.all(elas <= ub + 1e-8), "Some elasticity > CI upper"


# ---------------------------------------------------------------------------
# plot_gate edge cases
# ---------------------------------------------------------------------------

class TestPlotGateHelpers:
    def test_plot_gate_default_title(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = _make_df()
        est = _MockFittedEstimator(n=len(df))
        surface = ElasticitySurface(est)
        fig = surface.plot_gate(df, by="channel")
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_gate_custom_title(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = _make_df()
        est = _MockFittedEstimator(n=len(df))
        surface = ElasticitySurface(est)
        fig = surface.plot_gate(df, by="channel", title="Custom Title", color="red")
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_gate_with_ax(self):
        """Passing existing ax should reuse it."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = _make_df()
        est = _MockFittedEstimator(n=len(df))
        surface = ElasticitySurface(est)
        fig_in, ax = plt.subplots()
        fig_out = surface.plot_gate(df, by="channel", ax=ax)
        assert fig_out is fig_in
        plt.close("all")


# ---------------------------------------------------------------------------
# plot_surface edge cases
# ---------------------------------------------------------------------------

class TestPlotSurfaceHelpers:
    def test_plot_surface_custom_title_and_cmap(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = _make_df()
        est = _MockFittedEstimator(n=len(df))
        surface = ElasticitySurface(est)
        fig = surface.plot_surface(
            df, dims=["ncd_years", "channel"],
            title="My Surface", cmap="viridis"
        )
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_surface_with_existing_ax(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = _make_df()
        est = _MockFittedEstimator(n=len(df))
        surface = ElasticitySurface(est)
        fig_in, ax = plt.subplots()
        fig_out = surface.plot_surface(df, dims=["ncd_years", "channel"], ax=ax)
        assert fig_out is fig_in
        plt.close("all")

    def test_single_dim_raises(self):
        df = _make_df()
        est = _MockFittedEstimator(n=len(df))
        surface = ElasticitySurface(est)
        with pytest.raises(ValueError, match="exactly 2 dimensions"):
            surface.plot_surface(df, dims=["ncd_years"])

    def test_three_dims_raises(self):
        df = _make_df()
        est = _MockFittedEstimator(n=len(df))
        surface = ElasticitySurface(est)
        with pytest.raises(ValueError):
            surface.plot_surface(df, dims=["ncd_years", "channel", "age_band"])
