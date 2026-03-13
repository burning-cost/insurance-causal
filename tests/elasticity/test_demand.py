"""
Tests for demand curve utilities.

Verifies:
    - demand_curve() returns a polars DataFrame with correct columns
    - Renewal rate is monotone decreasing as price increases
    - plot_demand_curve() returns a matplotlib Figure
    - price_range parameter controls output length
"""

import numpy as np
import polars as pl
import pytest

from insurance_causal.elasticity.demand import demand_curve, plot_demand_curve


class TestDemandCurve:
    def test_returns_polars(self, fitted_estimator, small_df):
        result = demand_curve(fitted_estimator, small_df)
        assert isinstance(result, pl.DataFrame)

    def test_expected_columns(self, fitted_estimator, small_df):
        result = demand_curve(fitted_estimator, small_df)
        for col in ["log_price_change", "pct_price_change", "predicted_renewal_rate",
                    "predicted_profit", "predicted_revenue"]:
            assert col in result.columns

    def test_row_count_matches_price_range(self, fitted_estimator, small_df):
        result = demand_curve(fitted_estimator, small_df, price_range=(-0.2, 0.2, 30))
        assert len(result) == 30

    def test_renewal_rate_in_bounds(self, fitted_estimator, small_df):
        result = demand_curve(fitted_estimator, small_df)
        rates = result["predicted_renewal_rate"].to_numpy()
        assert np.all(rates >= 0.0), "Renewal rate below 0"
        assert np.all(rates <= 1.0), "Renewal rate above 1"

    def test_renewal_rate_decreases_with_price(self, fitted_estimator, small_df):
        """Overall renewal rate should fall as price increases."""
        result = demand_curve(
            fitted_estimator, small_df, price_range=(-0.3, 0.3, 20)
        ).sort("log_price_change")
        rates = result["predicted_renewal_rate"].to_numpy()
        # Check that the overall trend is downward
        # Fit a line: slope should be negative
        x = np.arange(len(rates))
        slope = np.polyfit(x, rates, 1)[0]
        assert slope < 0, (
            f"Renewal rate does not decrease with price (slope={slope:.4f}). "
            "Check that true_elasticity is negative in the DGP."
        )

    def test_unfitted_estimator_raises(self, small_df):
        from insurance_causal.elasticity.fit import RenewalElasticityEstimator
        est = RenewalElasticityEstimator(n_estimators=20, catboost_iterations=50)
        with pytest.raises(RuntimeError, match="not fitted"):
            demand_curve(est, small_df)

    def test_missing_log_price_change_raises(self, fitted_estimator, small_df):
        df_no_col = small_df.drop("log_price_change")
        with pytest.raises(ValueError, match="log_price_change"):
            demand_curve(fitted_estimator, df_no_col)


class TestPlotDemandCurve:
    def test_returns_figure(self, fitted_estimator, small_df):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        demand_df = demand_curve(fitted_estimator, small_df)
        fig = plot_demand_curve(demand_df)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_without_profit(self, fitted_estimator, small_df):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        demand_df = demand_curve(fitted_estimator, small_df)
        fig = plot_demand_curve(demand_df, show_profit=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")
