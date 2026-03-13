"""
Tests for ElasticitySurface: segment summaries and plotting functions.

Verifies:
    - segment_summary() returns the right columns and row counts
    - elasticity_at_10pct column is correctly computed
    - plot_gate() returns a matplotlib Figure
    - plot_surface() returns a matplotlib Figure
    - Portfolio summary (by=None) returns a single row
"""

import numpy as np
import polars as pl
import pytest

from insurance_causal.elasticity.surface import ElasticitySurface


class TestSegmentSummary:
    def test_portfolio_summary_one_row(self, fitted_estimator, small_df):
        surface = ElasticitySurface(fitted_estimator)
        result = surface.segment_summary(small_df, by=None)
        assert len(result) == 1

    def test_portfolio_summary_columns(self, fitted_estimator, small_df):
        surface = ElasticitySurface(fitted_estimator)
        result = surface.segment_summary(small_df, by=None)
        for col in ["segment", "elasticity", "ci_lower", "ci_upper", "n", "elasticity_at_10pct"]:
            assert col in result.columns

    def test_segment_summary_by_ncd(self, fitted_estimator, small_df):
        surface = ElasticitySurface(fitted_estimator)
        result = surface.segment_summary(small_df, by="ncd_years")
        n_groups = small_df["ncd_years"].n_unique()
        assert len(result) == n_groups

    def test_segment_summary_expected_columns(self, fitted_estimator, small_df):
        surface = ElasticitySurface(fitted_estimator)
        result = surface.segment_summary(small_df, by="ncd_years")
        for col in ["ncd_years", "elasticity", "ci_lower", "ci_upper", "n", "elasticity_at_10pct"]:
            assert col in result.columns

    def test_segment_summary_n_sums_to_total(self, fitted_estimator, small_df):
        surface = ElasticitySurface(fitted_estimator)
        result = surface.segment_summary(small_df, by="channel")
        assert result["n"].sum() == len(small_df)

    def test_segment_summary_elasticity_at_10pct(self, fitted_estimator, small_df):
        """elasticity_at_10pct should be elasticity * log(1.1) ≈ elasticity * 0.0953."""
        surface = ElasticitySurface(fitted_estimator)
        result = surface.segment_summary(small_df, by="ncd_years")
        elas = result["elasticity"].to_numpy()
        at_10 = result["elasticity_at_10pct"].to_numpy()
        expected = elas * np.log(1.1)
        np.testing.assert_allclose(at_10, expected, rtol=1e-3)

    def test_segment_summary_multi_by(self, fitted_estimator, small_df):
        """Passing a list of columns should work (segment by multiple dims)."""
        surface = ElasticitySurface(fitted_estimator)
        result = surface.segment_summary(small_df, by=["ncd_years", "channel"])
        assert isinstance(result, pl.DataFrame)
        assert "ncd_years" in result.columns
        assert "channel" in result.columns

    def test_unfitted_estimator_raises(self, small_df):
        from insurance_causal.elasticity.fit import RenewalElasticityEstimator
        est = RenewalElasticityEstimator(n_estimators=20, catboost_iterations=50)
        surface = ElasticitySurface(est)
        with pytest.raises(RuntimeError, match="not fitted"):
            surface.segment_summary(small_df, by="ncd_years")


class TestPlotGATE:
    def test_returns_figure(self, fitted_estimator, small_df):
        import matplotlib
        matplotlib.use("Agg")
        surface = ElasticitySurface(fitted_estimator)
        fig = surface.plot_gate(small_df, by="ncd_years")
        import matplotlib.pyplot as plt
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_custom_title(self, fitted_estimator, small_df):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        surface = ElasticitySurface(fitted_estimator)
        fig = surface.plot_gate(small_df, by="channel", title="My custom title")
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_by_channel(self, fitted_estimator, small_df):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        surface = ElasticitySurface(fitted_estimator)
        fig = surface.plot_gate(small_df, by="channel")
        assert fig is not None
        plt.close("all")


class TestPlotSurface:
    def test_returns_figure(self, fitted_estimator, small_df):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        surface = ElasticitySurface(fitted_estimator)
        fig = surface.plot_surface(small_df, dims=["ncd_years", "channel"])
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_wrong_dims_count_raises(self, fitted_estimator, small_df):
        surface = ElasticitySurface(fitted_estimator)
        with pytest.raises(ValueError, match="exactly 2 dimensions"):
            surface.plot_surface(small_df, dims=["ncd_years"])

    def test_three_dims_raises(self, fitted_estimator, small_df):
        surface = ElasticitySurface(fitted_estimator)
        with pytest.raises(ValueError):
            surface.plot_surface(small_df, dims=["ncd_years", "channel", "age_band"])
