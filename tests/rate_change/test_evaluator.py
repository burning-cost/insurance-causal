"""
Integration tests for RateChangeEvaluator.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from insurance_causal.rate_change import RateChangeEvaluator, make_rate_change_data, make_its_data


def _default_did_ev(**kwargs):
    defaults = dict(
        method="did",
        outcome_col="loss_ratio",
        period_col="period",
        treated_col="treated",
        change_period=7,
        exposure_col="exposure",
        unit_col="segment_id",
    )
    defaults.update(kwargs)
    return RateChangeEvaluator(**defaults)


def _default_its_ev(**kwargs):
    defaults = dict(
        method="its",
        outcome_col="outcome",
        period_col="period",
        change_period=9,
        exposure_col="exposure",
    )
    defaults.update(kwargs)
    return RateChangeEvaluator(**defaults)


def test_auto_selects_did_when_control_present():
    """method='auto' selects DiD when control group (treated==0) exists."""
    df = make_rate_change_data(treated_fraction=0.5, random_state=1)
    ev = RateChangeEvaluator(
        method="auto",
        outcome_col="loss_ratio",
        period_col="period",
        treated_col="treated",
        change_period=7,
        exposure_col="exposure",
        unit_col="segment_id",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ev.fit(df).summary()
    assert result.method == "did"


def test_auto_selects_its_when_no_control():
    """method='auto' falls back to ITS when all segments are treated."""
    df = make_rate_change_data(treated_fraction=1.0, random_state=2)
    ev = RateChangeEvaluator(
        method="auto",
        outcome_col="loss_ratio",
        period_col="period",
        treated_col="treated",
        change_period=7,
        exposure_col="exposure",
        unit_col="segment_id",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ev.fit(df).summary()
    assert result.method == "its"


def test_did_raises_without_control_group():
    """method='did' raises ValueError if no control group."""
    df = make_rate_change_data(treated_fraction=1.0)
    ev = _default_did_ev()
    with pytest.raises(ValueError, match="control group"):
        ev.fit(df)


def test_fit_returns_self():
    """fit() returns the evaluator instance for method chaining."""
    df = make_rate_change_data(n_policies=2000, random_state=3)
    ev = _default_did_ev()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        returned = ev.fit(df)
    assert returned is ev


def test_summary_raises_before_fit():
    """summary() raises RuntimeError if called before fit()."""
    ev = _default_did_ev()
    with pytest.raises(RuntimeError, match="[Ff]it"):
        ev.summary()


def test_summary_returns_rate_change_result():
    from insurance_causal.rate_change import RateChangeResult
    df = make_rate_change_data(n_policies=3000, random_state=4)
    ev = _default_did_ev()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ev.fit(df).summary()
    assert isinstance(result, RateChangeResult)


def test_att_pct_computed_correctly():
    """att_pct = att / pre_mean_treated * 100."""
    df = make_rate_change_data(n_policies=5000, true_att=-0.03, random_state=42)
    ev = _default_did_ev()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ev.fit(df).summary()
    if result.att_pct is not None:
        expected = result.att / result.pre_mean_treated * 100
        assert abs(result.att_pct - expected) < 1e-6


def test_method_chaining():
    """evaluator.fit(df).summary() works in one chain."""
    df = make_rate_change_data(n_policies=3000, random_state=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _default_did_ev().fit(df).summary()
    assert result.att is not None


def test_result_str_does_not_raise():
    """str(result) should not raise an exception."""
    df = make_rate_change_data(n_policies=3000, random_state=6)
    ev = _default_did_ev()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ev.fit(df).summary()
    s = str(result)
    assert "Rate Change Evaluation" in s
    assert "ATT" in s


def test_its_result_str():
    """ITS result str should include level shift info."""
    df = make_its_data(n_periods=20, change_period=11)
    ev = _default_its_ev(change_period=11)
    result = ev.fit(df).summary()
    s = str(result)
    assert "Level shift" in s
    assert "ITS" in s


def test_invalid_method_raises():
    """Invalid method raises ValueError at construction."""
    with pytest.raises(ValueError, match="method"):
        RateChangeEvaluator(method="ols", change_period=5)


def test_missing_change_period_raises():
    """change_period=None raises ValueError at construction."""
    with pytest.raises(ValueError, match="change_period"):
        RateChangeEvaluator(change_period=None)


def test_period_col_required():
    """Raises ValueError if required period_col is missing."""
    df = make_rate_change_data(n_policies=1000)
    ev = _default_did_ev(period_col="nonexistent_period")
    with pytest.raises(ValueError, match="nonexistent_period"):
        ev.fit(df)


def test_did_no_exposure_warns():
    """UserWarning emitted when no exposure_col provided."""
    df = make_rate_change_data(n_policies=2000, n_segments=25, random_state=7)
    ev = _default_did_ev(exposure_col=None)
    with pytest.warns(UserWarning, match="[Ee]xposure"):
        ev.fit(df)


def test_parallel_trends_raises_for_its():
    """parallel_trends_test() raises RuntimeError for ITS model."""
    df = make_its_data(n_periods=20, change_period=11)
    ev = _default_its_ev(change_period=11)
    ev.fit(df)
    with pytest.raises(RuntimeError, match="DiD"):
        ev.parallel_trends_test()


def test_parallel_trends_raises_before_fit():
    """parallel_trends_test() raises RuntimeError if called before fit."""
    ev = _default_did_ev()
    with pytest.raises(RuntimeError, match="[Ff]it"):
        ev.parallel_trends_test()


def test_n_periods_pre_post_correct():
    """n_periods_pre and n_periods_post should sum to total unique periods."""
    df = make_rate_change_data(n_policies=3000, n_periods=12, change_period=7)
    ev = _default_did_ev(change_period=7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ev.fit(df).summary()
    # Periods 1-6 are pre (6 periods), period 7-12 are post (6 periods)
    assert result.n_periods_pre == 6
    assert result.n_periods_post == 6
    assert result.n_periods_pre + result.n_periods_post == 12
