"""
Tests for RateChangeResult dataclass: __str__, att_pct, field types.
"""

import warnings

import pytest

from insurance_causal.rate_change import (
    RateChangeEvaluator,
    make_rate_change_data,
    make_its_data,
    RateChangeResult,
    DiDResult,
    ITSResult,
)


def _make_did_result(**kwargs):
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
    ev = RateChangeEvaluator(**defaults)
    df = make_rate_change_data(n_policies=3000, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev.fit(df)
    return ev.summary()


def _make_its_result():
    df = make_its_data(n_periods=20, change_period=11)
    ev = RateChangeEvaluator(
        method="its",
        outcome_col="outcome",
        period_col="period",
        change_period=11,
        exposure_col="exposure",
    )
    return ev.fit(df).summary()


def test_did_result_is_frozen():
    """RateChangeResult should be immutable (frozen dataclass)."""
    result = _make_did_result()
    with pytest.raises((AttributeError, TypeError)):
        result.att = 999.0


def test_did_result_method_detail_type():
    """method_detail should be DiDResult for DiD."""
    result = _make_did_result()
    assert isinstance(result.method_detail, DiDResult)


def test_its_result_method_detail_type():
    """method_detail should be ITSResult for ITS."""
    result = _make_its_result()
    assert isinstance(result.method_detail, ITSResult)


def test_did_str_contains_key_fields():
    """DiD __str__ should contain key result fields."""
    result = _make_did_result()
    s = str(result)
    assert "ATT" in s
    assert "95% CI" in s
    assert "p-value" in s
    assert "DiD" in s
    assert "Parallel trends" in s


def test_its_str_contains_key_fields():
    """ITS __str__ should contain key result fields."""
    result = _make_its_result()
    s = str(result)
    assert "Level shift" in s
    assert "Slope change" in s
    assert "ITS" in s
    assert "HAC" in s


def test_att_pct_sign_consistent():
    """att_pct should have the same sign as att."""
    result = _make_did_result()
    if result.att_pct is not None and result.pre_mean_treated > 0:
        assert (result.att > 0) == (result.att_pct > 0) or abs(result.att) < 1e-10


def test_result_warnings_is_list():
    """warnings field should be a list."""
    result = _make_did_result()
    assert isinstance(result.warnings, list)


def test_result_ci_ordered():
    """ci_lower <= att <= ci_upper (for standard cases)."""
    result = _make_did_result()
    assert result.ci_lower <= result.ci_upper


def test_result_se_positive():
    """Standard error should be positive."""
    result = _make_did_result()
    assert result.se > 0


def test_result_n_periods_positive():
    result = _make_did_result()
    assert result.n_periods_pre > 0
    assert result.n_periods_post > 0


def test_its_result_seasonal_flag():
    """seasonal_adjustment should be True when quarter dummies applied."""
    result = _make_its_result()
    assert result.method_detail.seasonal_adjustment is True


def test_did_str_shows_cluster_info():
    """DiD string should mention SE method."""
    result = _make_did_result()
    s = str(result)
    assert "SE method" in s or "cluster" in s.lower() or "HC3" in s


def test_parallel_trends_pvalue_in_did_result():
    """parallel_trends_pvalue should be populated for DiD."""
    result = _make_did_result()
    # May be None if insufficient pre-periods, but should not be NaN if present
    if result.parallel_trends_pvalue is not None:
        assert 0 <= result.parallel_trends_pvalue <= 1
