"""Tests for the RateChangeEvaluator main class."""

import numpy as np
import pandas as pd
import pytest
import warnings

from insurance_causal.rate_change import (
    RateChangeEvaluator,
    make_rate_change_data,
    RateChangeResult,
    DiDResult,
    ITSResult,
)


class TestRateChangeEvaluatorAutoSelection:
    def test_selects_did_when_control_present(self):
        df = make_rate_change_data(n_segments=20, seed=0)
        evaluator = RateChangeEvaluator(
            outcome_col="outcome",
            treatment_period=9,
        )
        result = evaluator.fit(df)
        assert result._result.method == "DiD"

    def test_selects_its_when_all_treated(self):
        df = make_rate_change_data(mode="its", n_periods=20, seed=0)
        evaluator = RateChangeEvaluator(
            outcome_col="outcome",
            treatment_period=9,
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = evaluator.fit(df)
        assert result._result.method == "ITS"

    def test_its_warning_emitted(self):
        df = make_rate_change_data(mode="its", n_periods=20, seed=0)
        evaluator = RateChangeEvaluator(
            outcome_col="outcome",
            treatment_period=9,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            evaluator.fit(df)
        msgs = [str(x.message) for x in w]
        assert any("ITS" in m or "control group" in m for m in msgs)


class TestRateChangeEvaluatorDiD:
    @pytest.fixture
    def did_evaluator_and_df(self):
        df = make_rate_change_data(n_segments=40, n_periods=16, seed=42)
        evaluator = RateChangeEvaluator(
            outcome_col="outcome",
            treatment_period=9,
            unit_col="segment",
            weight_col="earned_exposure",
        )
        evaluator.fit(df)
        return evaluator, df

    def test_result_type(self, did_evaluator_and_df):
        ev, df = did_evaluator_and_df
        assert isinstance(ev._result, RateChangeResult)
        assert isinstance(ev._result.did, DiDResult)
        assert ev._result.its is None

    def test_summary_contains_att(self, did_evaluator_and_df):
        ev, df = did_evaluator_and_df
        summary = ev.summary()
        assert "ATT" in summary
        assert "DiD" in summary or "Difference-in-Differences" in summary

    def test_parallel_trends_test_method(self, did_evaluator_and_df):
        ev, df = did_evaluator_and_df
        pt = ev.parallel_trends_test()
        assert "f_stat" in pt
        assert "p_value" in pt
        assert "passed" in pt

    def test_method_chaining(self):
        df = make_rate_change_data(n_segments=20, seed=0)
        ev = RateChangeEvaluator(outcome_col="outcome", treatment_period=9)
        result = ev.fit(df)
        # fit() returns self for chaining
        assert result is ev

    def test_plot_its_raises_for_did(self, did_evaluator_and_df):
        ev, df = did_evaluator_and_df
        with pytest.raises(RuntimeError, match="ITS"):
            ev.plot_its(df)

    def test_plot_event_study_for_did(self, did_evaluator_and_df):
        pytest.importorskip("matplotlib")
        ev, df = did_evaluator_and_df
        ax = ev.plot_event_study()
        assert ax is not None

    def test_plot_pre_post_for_did(self, did_evaluator_and_df):
        pytest.importorskip("matplotlib")
        ev, df = did_evaluator_and_df
        ax = ev.plot_pre_post(df)
        assert ax is not None


class TestRateChangeEvaluatorITS:
    @pytest.fixture
    def its_evaluator_and_df(self):
        df = make_rate_change_data(mode="its", n_periods=20, seed=42)
        evaluator = RateChangeEvaluator(
            outcome_col="outcome",
            treatment_period=9,
            weight_col="earned_exposure",
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            evaluator.fit(df)
        return evaluator, df

    def test_result_type(self, its_evaluator_and_df):
        ev, df = its_evaluator_and_df
        assert isinstance(ev._result.its, ITSResult)
        assert ev._result.did is None

    def test_summary_contains_level_shift(self, its_evaluator_and_df):
        ev, df = its_evaluator_and_df
        summary = ev.summary()
        assert "Level shift" in summary
        assert "ITS" in summary or "Interrupted" in summary

    def test_parallel_trends_not_applicable_for_its(self, its_evaluator_and_df):
        ev, df = its_evaluator_and_df
        pt = ev.parallel_trends_test()
        assert pt["passed"] is True
        assert "note" in pt

    def test_plot_event_study_raises_for_its(self, its_evaluator_and_df):
        ev, df = its_evaluator_and_df
        with pytest.raises(RuntimeError, match="DiD"):
            ev.plot_event_study()

    def test_plot_its_for_its(self, its_evaluator_and_df):
        pytest.importorskip("matplotlib")
        ev, df = its_evaluator_and_df
        ax = ev.plot_its(df)
        assert ax is not None


class TestRateChangeEvaluatorValidation:
    def test_missing_outcome_col_raises(self):
        df = make_rate_change_data(seed=0)
        ev = RateChangeEvaluator(outcome_col="nonexistent", treatment_period=9)
        with pytest.raises(ValueError, match="nonexistent"):
            ev.fit(df)

    def test_missing_treatment_period_raises(self):
        df = make_rate_change_data(seed=0)
        ev = RateChangeEvaluator(outcome_col="outcome", treatment_period=999)
        with pytest.raises(ValueError, match="999"):
            ev.fit(df)

    def test_result_before_fit_raises(self):
        ev = RateChangeEvaluator(outcome_col="outcome", treatment_period=9)
        with pytest.raises(RuntimeError, match="fit"):
            ev.summary()

    def test_missing_weight_col_raises(self):
        df = make_rate_change_data(seed=0)
        ev = RateChangeEvaluator(
            outcome_col="outcome",
            treatment_period=9,
            weight_col="nonexistent_col",
        )
        with pytest.raises(ValueError, match="nonexistent_col"):
            ev.fit(df)


class TestShockProximityWarning:
    def test_warns_near_gipp(self):
        """GIPP took effect 2022-Q1; treatment near this should warn."""
        df = make_rate_change_data(n_segments=20, n_periods=12, seed=0)
        # Map periods to YYYY-QN strings near GIPP
        period_map = {p: f"{2021 + (p-1)//4}-Q{((p-1)%4)+1}" for p in range(1, 13)}
        df["period_q"] = df["period"].map(period_map)
        # treatment_period is 2022-Q1 (GIPP period)
        ev = RateChangeEvaluator(
            outcome_col="outcome",
            period_col="period_q",
            treatment_period="2022-Q1",
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ev.fit(df)
        msgs = [str(x.message) for x in w]
        assert any("shock" in m.lower() or "GIPP" in m or "PS21" in m for m in msgs)

    def test_no_shock_warning_for_benign_period(self):
        """Period well away from known shocks should not produce shock warnings."""
        df = make_rate_change_data(n_segments=20, n_periods=12, seed=0)
        period_map = {p: f"{2015 + (p-1)//4}-Q{((p-1)%4)+1}" for p in range(1, 13)}
        df["period_q"] = df["period"].map(period_map)
        ev = RateChangeEvaluator(
            outcome_col="outcome",
            period_col="period_q",
            treatment_period="2015-Q3",
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ev.fit(df)
        msgs = [str(x.message) for x in w]
        shock_msgs = [m for m in msgs if "shock" in m.lower() and "quarters of UK" in m]
        assert len(shock_msgs) == 0


class TestStaggeredAdoptionWarning:
    def test_staggered_warning_emitted(self):
        df = make_rate_change_data(n_segments=20, n_periods=16, seed=0)
        # Create two cohorts: first half treated at period 7, second half at period 9
        segs = sorted(df["segment"].unique())
        early_segs = set(segs[:10])
        df = df.copy()
        # Rebuild treated column with staggered adoption
        df["treated"] = 0
        df.loc[df["segment"].isin(early_segs) & (df["period"] >= 7), "treated"] = 1
        df.loc[~df["segment"].isin(early_segs) & (df["period"] >= 9), "treated"] = 1

        ev = RateChangeEvaluator(
            outcome_col="outcome",
            treatment_period=7,
            unit_col="segment",
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ev.fit(df)
        msgs = [str(x.message) for x in w]
        staggered_msgs = [m for m in msgs if "staggered" in m.lower() or "cohort" in m.lower()]
        assert len(staggered_msgs) > 0, f"Expected staggered warning, got: {msgs}"
