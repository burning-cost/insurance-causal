"""
Unit tests for RateChangeEvaluator that don't require a full fit.

Existing tests in test_evaluator.py cover the full fit path. These tests
target the constructor validation, resolve_method logic, _validate_df,
_check_is_policy_level, and the unfitted guard on summary() and
parallel_trends_test().

None of these require fitting a model — they run in milliseconds locally.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from insurance_causal.rate_change._evaluator import RateChangeEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_evaluator(**kwargs):
    defaults = dict(
        method="auto",
        outcome_col="loss_ratio",
        period_col="period",
        treated_col="treated",
        change_period=7,
        exposure_col="exposure",
        unit_col="segment_id",
    )
    defaults.update(kwargs)
    return RateChangeEvaluator(**defaults)


def _make_minimal_df(n_segs=4, n_periods=8, change_period=5, seed=0):
    """Build a minimal valid DataFrame for DiD testing."""
    rng = np.random.default_rng(seed)
    rows = []
    for seg in range(n_segs):
        treated = int(seg >= n_segs // 2)
        for period in range(1, n_periods + 1):
            rows.append({
                "segment_id": f"seg_{seg}",
                "period": period,
                "treated": treated,
                "loss_ratio": float(rng.uniform(0.4, 0.8)),
                "exposure": float(rng.uniform(50, 200)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestRateChangeEvaluatorInit:
    def test_valid_construction(self):
        ev = _make_evaluator()
        assert ev.method == "auto"
        assert ev.change_period == 7
        assert not ev._fitted

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            _make_evaluator(method="twoway_fe")

    def test_change_period_none_raises(self):
        with pytest.raises(ValueError, match="change_period"):
            _make_evaluator(change_period=None)

    def test_did_method_accepted(self):
        ev = _make_evaluator(method="did")
        assert ev.method == "did"

    def test_its_method_accepted(self):
        ev = _make_evaluator(method="its")
        assert ev.method == "its"

    def test_auto_method_accepted(self):
        ev = _make_evaluator(method="auto")
        assert ev.method == "auto"

    def test_quarter_string_change_period_accepted(self):
        ev = _make_evaluator(change_period="2022Q3")
        assert ev.change_period == "2022Q3"

    def test_alpha_stored(self):
        ev = _make_evaluator()
        assert ev.alpha == 0.05

    def test_cluster_col_defaults_to_unit_col(self):
        """When cluster_col is not specified, it should default to unit_col."""
        ev = _make_evaluator(unit_col="segment_id", cluster_col=None)
        assert ev.cluster_col == "segment_id"

    def test_explicit_cluster_col_stored(self):
        ev = _make_evaluator(cluster_col="region")
        assert ev.cluster_col == "region"

    def test_result_none_before_fit(self):
        ev = _make_evaluator()
        assert ev._result is None


# ---------------------------------------------------------------------------
# summary() and parallel_trends_test() guard before fit
# ---------------------------------------------------------------------------


class TestUnfittedGuards:
    def test_summary_raises_before_fit(self):
        ev = _make_evaluator()
        with pytest.raises(RuntimeError, match="fit"):
            ev.summary()

    def test_parallel_trends_test_raises_before_fit(self):
        ev = _make_evaluator()
        with pytest.raises(RuntimeError, match="fit"):
            ev.parallel_trends_test()

    def test_plot_event_study_raises_before_fit(self):
        ev = _make_evaluator()
        with pytest.raises(RuntimeError):
            ev.plot_event_study()

    def test_plot_pre_post_raises_before_fit(self):
        ev = _make_evaluator()
        with pytest.raises(RuntimeError):
            ev.plot_pre_post()


# ---------------------------------------------------------------------------
# _validate_df
# ---------------------------------------------------------------------------


class TestValidateDf:
    def test_valid_df_passes(self):
        ev = _make_evaluator()
        df = _make_minimal_df()
        ev._validate_df(df)  # should not raise

    def test_missing_outcome_raises(self):
        ev = _make_evaluator(outcome_col="loss_ratio")
        df = _make_minimal_df()
        df = df.drop(columns=["loss_ratio"])
        with pytest.raises(ValueError, match="loss_ratio"):
            ev._validate_df(df)

    def test_missing_period_raises(self):
        ev = _make_evaluator(period_col="period")
        df = _make_minimal_df()
        df = df.drop(columns=["period"])
        with pytest.raises(ValueError, match="period"):
            ev._validate_df(df)

    def test_missing_treated_col_for_did_raises(self):
        ev = _make_evaluator(method="did", treated_col="treated")
        df = _make_minimal_df()
        df = df.drop(columns=["treated"])
        with pytest.raises(ValueError, match="treated"):
            ev._validate_df(df)

    def test_missing_treated_col_for_auto_passes(self):
        """For auto, treated_col absence shouldn't raise in _validate_df."""
        ev = _make_evaluator(method="auto", treated_col="treated")
        df = _make_minimal_df()
        df = df.drop(columns=["treated"])
        # auto method does not strictly require treated_col at _validate_df stage
        ev._validate_df(df)  # may or may not raise — just don't crash


# ---------------------------------------------------------------------------
# _resolve_method
# ---------------------------------------------------------------------------


class TestResolveMethod:
    def test_auto_with_control_group_picks_did(self):
        ev = _make_evaluator(method="auto", treated_col="treated")
        df = _make_minimal_df()
        df["_period_enc_"] = df["period"]  # pre-encode for _resolve_method
        method = ev._resolve_method(df)
        assert method == "did"

    def test_auto_without_control_group_picks_its(self):
        ev = _make_evaluator(method="auto", treated_col="treated")
        df = _make_minimal_df()
        df["treated"] = 1  # all treated, no control
        df["_period_enc_"] = df["period"]
        method = ev._resolve_method(df)
        assert method == "its"

    def test_did_forced_raises_without_control(self):
        ev = _make_evaluator(method="did", treated_col="treated")
        df = _make_minimal_df()
        df["treated"] = 1  # no control group
        df["_period_enc_"] = df["period"]
        with pytest.raises(ValueError, match="No control group"):
            ev._resolve_method(df)

    def test_its_forced_returns_its(self):
        ev = _make_evaluator(method="its")
        df = _make_minimal_df()
        df["_period_enc_"] = df["period"]
        method = ev._resolve_method(df)
        assert method == "its"

    def test_auto_with_no_treated_col_picks_its(self):
        ev = _make_evaluator(method="auto", treated_col=None)
        df = _make_minimal_df()
        df["_period_enc_"] = df["period"]
        method = ev._resolve_method(df)
        assert method == "its"


# ---------------------------------------------------------------------------
# _check_is_policy_level
# ---------------------------------------------------------------------------


class TestCheckIsPolicyLevel:
    def test_segment_period_level_returns_false(self):
        """One row per segment-period: not policy level."""
        ev = _make_evaluator()
        df = _make_minimal_df()
        df["_period_enc_"] = df["period"]
        result = ev._check_is_policy_level(df)
        assert not result

    def test_policy_level_returns_true(self):
        """Multiple policies per segment-period: is policy level."""
        ev = _make_evaluator()
        # Two policies in same segment_id + period
        df = pd.DataFrame({
            "segment_id": ["seg_0", "seg_0", "seg_1"],
            "period": [1, 1, 1],
            "treated": [1, 1, 0],
            "loss_ratio": [0.5, 0.6, 0.4],
            "exposure": [1.0, 1.0, 1.0],
            "_period_enc_": [1, 1, 1],
        })
        result = ev._check_is_policy_level(df)
        assert result

    def test_no_unit_col_returns_false(self):
        ev = _make_evaluator(unit_col=None)
        df = _make_minimal_df()
        df["_period_enc_"] = df["period"]
        result = ev._check_is_policy_level(df)
        assert not result


# ---------------------------------------------------------------------------
# fit() — warning for missing exposure_col
# ---------------------------------------------------------------------------


class TestFitWarnings:
    def test_warns_without_exposure_col(self):
        """RateChangeEvaluator must warn when no exposure_col is provided."""
        ev = _make_evaluator(exposure_col=None)
        df = _make_minimal_df()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                ev.fit(df)
            except Exception:
                pass  # we only care whether the warning was emitted
            exposure_warnings = [x for x in w if "exposure" in str(x.message).lower()]
            assert len(exposure_warnings) >= 1, (
                "Expected a UserWarning about missing exposure_col"
            )

    def test_invalid_method_at_init_raises(self):
        """Invalid method should be caught at __init__ time, not at fit time."""
        with pytest.raises(ValueError, match="method"):
            RateChangeEvaluator(
                method="difference_in_difference",
                outcome_col="loss_ratio",
                period_col="period",
                change_period=5,
            )
