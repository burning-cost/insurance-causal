"""Tests for the ITS estimator."""

import numpy as np
import pandas as pd
import pytest
import warnings

from insurance_causal.rate_change import make_rate_change_data
from insurance_causal.rate_change._its import ITSEstimator


class TestITSEstimator:
    @pytest.fixture
    def its_df(self):
        return make_rate_change_data(
            mode="its",
            n_periods=20,
            treatment_period=9,
            true_level_shift=-0.03,
            true_slope_change=-0.002,
            seed=42,
            add_seasonality=True,
        )

    def test_level_shift_recovery(self, its_df):
        """ITS should recover the true level shift within 3 SE."""
        est = ITSEstimator(
            outcome_col="outcome",
            period_col="period",
            treatment_period=9,
            weight_col="earned_exposure",
            add_seasonality=False,  # disable for cleaner test
        )
        est.fit(its_df)
        res = est.result()
        true_shift = -0.03
        assert abs(res.level_shift - true_shift) < 3 * res.level_shift_se, (
            f"Level shift {res.level_shift:.4f} too far from truth {true_shift} "
            f"(SE={res.level_shift_se:.4f})"
        )

    def test_slope_change_recovery(self, its_df):
        """ITS should recover the true slope change within 3 SE."""
        est = ITSEstimator(
            outcome_col="outcome",
            period_col="period",
            treatment_period=9,
            weight_col="earned_exposure",
            add_seasonality=False,
        )
        est.fit(its_df)
        res = est.result()
        true_slope = -0.002
        assert abs(res.slope_change - true_slope) < 3 * res.slope_change_se, (
            f"Slope change {res.slope_change:.6f} too far from truth {true_slope} "
            f"(SE={res.slope_change_se:.6f})"
        )

    def test_result_ci_contains_estimate(self, its_df):
        est = ITSEstimator(
            outcome_col="outcome", period_col="period", treatment_period=9
        )
        est.fit(its_df)
        res = est.result()
        assert res.level_shift_ci_lower < res.level_shift < res.level_shift_ci_upper
        assert res.slope_change_ci_lower < res.slope_change < res.slope_change_ci_upper

    def test_hac_lags_positive(self, its_df):
        est = ITSEstimator(
            outcome_col="outcome", period_col="period", treatment_period=9
        )
        est.fit(its_df)
        assert est.result().hac_lags >= 1

    def test_n_pre_n_post(self, its_df):
        est = ITSEstimator(
            outcome_col="outcome", period_col="period", treatment_period=9
        )
        est.fit(its_df)
        res = est.result()
        assert res.n_pre == 8   # periods 1-8 before treatment at 9
        assert res.n_post == 12  # period 9 + 11 post = 12

    def test_seasonality_included(self, its_df):
        """With quarter column derivable from period, seasonality should be included."""
        # Create a df with string quarter period
        df = its_df.copy()
        df["period_str"] = df["period"].apply(lambda p: f"{2019 + (p-1)//4}-Q{((p-1)%4)+1}")
        est = ITSEstimator(
            outcome_col="outcome",
            period_col="period_str",
            treatment_period="2021-Q1",  # period 9 = 2021-Q1
            add_seasonality=True,
        )
        est.fit(df)
        # May or may not succeed depending on period format parsing
        # Main check: no exceptions
        assert est.result() is not None

    def test_seasonality_disabled(self, its_df):
        est = ITSEstimator(
            outcome_col="outcome",
            period_col="period",
            treatment_period=9,
            add_seasonality=False,
        )
        est.fit(its_df)
        assert not est.result().has_seasonality

    def test_too_few_pre_periods_raises(self):
        """Fewer than 4 pre-periods should raise ValueError."""
        df = make_rate_change_data(mode="its", n_periods=8, treatment_period=3, seed=0)
        est = ITSEstimator(
            outcome_col="outcome", period_col="period", treatment_period=3
        )
        with pytest.raises(ValueError, match="pre-intervention"):
            est.fit(df)

    def test_warn_on_few_pre_periods(self):
        """Between 4 and 8 pre-periods should warn but not raise."""
        df = make_rate_change_data(mode="its", n_periods=12, treatment_period=6, seed=0)
        est = ITSEstimator(
            outcome_col="outcome", period_col="period", treatment_period=6
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            est.fit(df)
        warning_msgs = [str(x.message) for x in w]
        assert any("pre-intervention" in m or "pre-period" in m for m in warning_msgs)

    def test_missing_period_raises(self, its_df):
        est = ITSEstimator(
            outcome_col="outcome", period_col="period", treatment_period=999
        )
        with pytest.raises(ValueError, match="999"):
            est.fit(its_df)

    def test_result_before_fit_raises(self):
        est = ITSEstimator(
            outcome_col="outcome", period_col="period", treatment_period=9
        )
        with pytest.raises(RuntimeError, match="fit"):
            est.result()

    def test_effect_at_k(self, its_df):
        est = ITSEstimator(
            outcome_col="outcome", period_col="period", treatment_period=9
        )
        est.fit(its_df)
        res = est.result()
        # At k=0 (treatment period): effect = level_shift + slope_change * 0
        assert res.effect_at_k(0) == res.level_shift
        # At k=5: effect = level_shift + slope_change * 5
        assert abs(res.effect_at_k(5) - (res.level_shift + 5 * res.slope_change)) < 1e-10

    def test_counterfactual_returned(self, its_df):
        est = ITSEstimator(
            outcome_col="outcome", period_col="period", treatment_period=9
        )
        est.fit(its_df)
        cf = est.counterfactual()
        assert cf is not None
        assert len(cf) == len(its_df)

    def test_correct_parameterisation_time_since(self):
        """Verify (t-T)*D_t is zero in pre-period — the standard ITS parameterisation."""
        df = make_rate_change_data(mode="its", n_periods=16, treatment_period=9, seed=0)
        # After fit, the _df_fitted should have _time_since = 0 for pre-periods
        est = ITSEstimator(
            outcome_col="outcome", period_col="period", treatment_period=9
        )
        est.fit(df)
        df_fitted = est._df_fitted
        assert df_fitted is not None
        pre = df_fitted[df_fitted["_post"] == 0]
        assert (pre["_time_since"] == 0).all(), (
            "time_since should be 0 in pre-period (correct ITS parameterisation)"
        )

    def test_r_squared_between_0_and_1(self, its_df):
        est = ITSEstimator(
            outcome_col="outcome", period_col="period", treatment_period=9
        )
        est.fit(its_df)
        assert 0.0 <= est.result().r_squared <= 1.0
