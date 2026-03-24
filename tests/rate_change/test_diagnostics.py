"""Tests for DiD diagnostics: parallel trends and staggered adoption."""

import numpy as np
import pandas as pd
import pytest

from insurance_causal.rate_change import make_rate_change_data
from insurance_causal.rate_change._diagnostics import (
    check_parallel_trends,
    check_staggered_adoption,
)


class TestCheckParallelTrends:
    @pytest.fixture
    def parallel_df(self):
        return make_rate_change_data(
            n_segments=30, n_periods=16, treatment_period=9, seed=0
        )

    def test_returns_dict_with_expected_keys(self, parallel_df):
        result = check_parallel_trends(
            df=parallel_df,
            outcome_col="outcome",
            treated_col="treated",
            period_col="period",
            treatment_period=9,
            unit_col="segment",
            weight_col="earned_exposure",
        )
        assert "f_stat" in result
        assert "p_value" in result
        assert "coefs" in result
        assert "ses" in result
        assert "periods" in result
        assert "passed" in result

    def test_passes_under_parallel_trends(self, parallel_df):
        """Under the parallel trends DGP, pre-trend test should pass."""
        result = check_parallel_trends(
            df=parallel_df,
            outcome_col="outcome",
            treated_col="treated",
            period_col="period",
            treatment_period=9,
        )
        # Under correct DGP, should not strongly reject
        # (not guaranteed at 5%, but should pass at 1%)
        if result["p_value"] is not None:
            assert result["p_value"] > 0.01, (
                f"Pre-trend test rejected at 1% under correct DGP: p={result['p_value']}"
            )

    def test_coef_length_matches_periods(self, parallel_df):
        result = check_parallel_trends(
            df=parallel_df,
            outcome_col="outcome",
            treated_col="treated",
            period_col="period",
            treatment_period=9,
            n_pre_periods=4,
        )
        if result["coefs"]:
            assert len(result["coefs"]) == len(result["ses"])
            assert len(result["coefs"]) == len(result["periods"])

    def test_invalid_treatment_period_raises(self, parallel_df):
        with pytest.raises(ValueError, match="999"):
            check_parallel_trends(
                df=parallel_df,
                outcome_col="outcome",
                treated_col="treated",
                period_col="period",
                treatment_period=999,
            )

    def test_pre_periods_are_negative(self, parallel_df):
        """Event study periods should all be negative (pre-treatment)."""
        result = check_parallel_trends(
            df=parallel_df,
            outcome_col="outcome",
            treated_col="treated",
            period_col="period",
            treatment_period=9,
            n_pre_periods=4,
        )
        if result["periods"]:
            assert all(p < 0 for p in result["periods"])


class TestCheckStaggeredAdoption:
    def test_single_cohort_not_staggered(self):
        df = make_rate_change_data(n_segments=20, n_periods=16, seed=0)
        result = check_staggered_adoption(
            df, treated_col="treated", period_col="period", unit_col="segment"
        )
        assert not result["is_staggered"]
        assert result["n_cohorts"] == 1

    def test_two_cohorts_flagged_as_staggered(self):
        df = make_rate_change_data(n_segments=20, n_periods=16, seed=0)
        segs = sorted(df["segment"].unique())
        early_segs = set(segs[:10])
        df = df.copy()
        df["treated"] = 0
        df.loc[df["segment"].isin(early_segs) & (df["period"] >= 7), "treated"] = 1
        df.loc[~df["segment"].isin(early_segs) & (df["period"] >= 11), "treated"] = 1

        result = check_staggered_adoption(
            df, treated_col="treated", period_col="period", unit_col="segment"
        )
        assert result["is_staggered"]
        assert result["n_cohorts"] == 2
        assert "Goodman-Bacon" in result["message"] or "staggered" in result["message"].lower()

    def test_no_unit_col_returns_not_staggered(self):
        df = make_rate_change_data(n_segments=20, seed=0)
        result = check_staggered_adoption(
            df, treated_col="treated", period_col="period", unit_col=None
        )
        assert not result["is_staggered"]

    def test_no_treated_units(self):
        df = make_rate_change_data(n_segments=10, seed=0)
        df["treated"] = 0
        result = check_staggered_adoption(
            df, treated_col="treated", period_col="period", unit_col="segment"
        )
        assert result["n_cohorts"] == 0

    def test_cohort_sizes_sum_to_total_treated(self):
        df = make_rate_change_data(n_segments=20, n_periods=16, seed=0)
        segs = sorted(df["segment"].unique())
        early_segs = set(segs[:8])
        late_segs = set(segs[8:16])
        df = df.copy()
        df["treated"] = 0
        df.loc[df["segment"].isin(early_segs) & (df["period"] >= 5), "treated"] = 1
        df.loc[df["segment"].isin(late_segs) & (df["period"] >= 9), "treated"] = 1

        result = check_staggered_adoption(
            df, treated_col="treated", period_col="period", unit_col="segment"
        )
        total = sum(result["cohort_sizes"].values())
        assert total == 16  # 8 early + 8 late
