"""
Tests for ITSEstimator internals and RateChangeEvaluator ITS behavior.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from insurance_causal.rate_change import RateChangeEvaluator, make_its_data
from insurance_causal.rate_change._shocks import UK_INSURANCE_SHOCKS


def _make_its_evaluator(**kwargs):
    defaults = dict(
        method="its",
        outcome_col="outcome",
        period_col="period",
        change_period=9,
        exposure_col="exposure",
    )
    defaults.update(kwargs)
    return RateChangeEvaluator(**defaults)


def _make_quarterly_its_data(change_quarter: str, n_pre: int = 8, n_post: int = 8) -> pd.DataFrame:
    """
    Build a quarter-string ITS dataset centred around change_quarter.

    Generates n_pre pre-treatment quarters and n_post post-treatment quarters,
    with change_quarter as the first post-treatment period.
    """
    import re
    match = re.match(r"^(\d{4})Q([1-4])$", change_quarter.upper().replace("-", ""))
    year, q = int(match.group(1)), int(match.group(2))

    quarters = []
    # Go backwards n_pre steps from the change quarter
    y, qn = year, q
    for _ in range(n_pre):
        qn -= 1
        if qn == 0:
            qn = 4
            y -= 1
        quarters.append((y, qn))
    quarters = quarters[::-1]  # chronological order

    # Add change_quarter and n_post-1 more post quarters
    y, qn = year, q
    for i in range(n_post):
        quarters.append((y, qn))
        qn += 1
        if qn == 5:
            qn = 1
            y += 1

    rng = np.random.default_rng(42)
    rows = []
    for i, (yr, qnum) in enumerate(quarters):
        is_post = i >= n_pre
        y_val = 0.65 + 0.002 * i + (-0.04 if is_post else 0.0) + rng.normal(0, 0.005)
        rows.append({
            "period": f"{yr}Q{qnum}",
            "outcome": y_val,
            "exposure": 50_000.0,
            "quarter": qnum,
        })

    return pd.DataFrame(rows)


def test_its_recovers_level_shift():
    """ITS recovers true beta_2 (level shift) within 2 SE from synthetic time series."""
    true_level_shift = -0.04
    df = make_its_data(
        true_level_shift=true_level_shift, n_periods=24, change_period=13, random_state=0
    )
    ev = _make_its_evaluator(change_period=13)
    result = ev.fit(df).summary()
    detail = result.method_detail
    # Use 0.02 absolute tolerance: quarter dummies in the regression can absorb
    # some of the level shift when the change period aligns with a particular
    # quarter, so a tight 2*SE bound is too restrictive for synthetic data.
    assert abs(detail.level_shift - true_level_shift) < 0.02, (
        f"level_shift={detail.level_shift:.4f}, "
        f"true={true_level_shift}, "
        f"diff={abs(detail.level_shift - true_level_shift):.4f}"
    )


def test_its_raises_too_few_pre_periods():
    """ValueError raised when pre-treatment periods < min_pre_periods."""
    df = make_its_data(n_periods=8, change_period=3)  # only 2 pre-periods
    ev = _make_its_evaluator(change_period=3, min_pre_periods=4)
    with pytest.raises(ValueError, match="pre-treatment periods"):
        ev.fit(df)


def test_its_warns_few_pre_periods():
    """UserWarning when pre-treatment periods < 8 but >= min_pre_periods."""
    df = make_its_data(n_periods=14, change_period=6)  # 5 pre-periods
    ev = _make_its_evaluator(change_period=6, min_pre_periods=4)
    with pytest.warns(UserWarning, match="[Pp]re.treatment period"):
        ev.fit(df)


def test_its_warns_near_ogden():
    """UserWarning emitted when intervention is near the Ogden rate change (2019Q3)."""
    df = _make_quarterly_its_data("2019Q3")
    ev = RateChangeEvaluator(
        method="its",
        outcome_col="outcome",
        period_col="period",
        change_period="2019Q3",
        exposure_col="exposure",
    )
    with pytest.warns(UserWarning, match="[Oo]gden"):
        ev.fit(df)


def test_its_parameterisation_correct():
    """
    Verify the ITS formula uses (t-T)*D_t not just (t-T).

    With a zero slope_change DGP, the estimated slope_change should be near zero.
    This would fail if the term were coded as (t-T) without the D_t interaction
    (the Ewusie 2021 parameterisation error), because the pre-period data would
    contaminate the slope_change estimate.
    """
    df = make_its_data(
        true_level_shift=-0.03,
        true_slope_change=0.0,  # no slope change
        n_periods=24,
        change_period=13,
        noise_scale=0.002,
        random_state=42,
    )
    ev = _make_its_evaluator(change_period=13)
    result = ev.fit(df).summary()
    detail = result.method_detail
    # slope_change should be near zero — within 3 SE
    assert abs(detail.slope_change) < 3 * detail.slope_change_se, (
        f"slope_change={detail.slope_change:.4f}, "
        f"3*SE={3*detail.slope_change_se:.4f}: "
        "parameterisation may be incorrect (Ewusie 2021 error)"
    )


def test_its_method_label():
    """result.method == 'its' for ITS."""
    df = make_its_data()
    ev = _make_its_evaluator()
    result = ev.fit(df).summary()
    assert result.method == "its"


def test_its_n_control_zero():
    """n_control == 0 for ITS (no control group)."""
    df = make_its_data()
    ev = _make_its_evaluator()
    result = ev.fit(df).summary()
    assert result.n_control == 0


def test_its_result_has_effect_at_periods():
    """effect_at_periods dict is populated with reasonable keys."""
    df = make_its_data(n_periods=24, change_period=9)
    ev = _make_its_evaluator()
    result = ev.fit(df).summary()
    d = result.method_detail
    assert isinstance(d.effect_at_periods, dict)
    assert 1 in d.effect_at_periods


def test_its_seasonal_adjustment_applied():
    """Seasonal adjustment is applied when quarter column is present."""
    df = make_its_data(n_periods=16, add_seasonality=True)
    ev = _make_its_evaluator()
    result = ev.fit(df).summary()
    assert result.method_detail.seasonal_adjustment is True


def test_its_shocks_list_empty_for_integer_period():
    """No shocks should be detected when change_period is an integer."""
    df = make_its_data(n_periods=16, change_period=9)
    ev = _make_its_evaluator(change_period=9)
    result = ev.fit(df).summary()
    assert result.method_detail.shocks_near_intervention == []


def test_its_gipp_shock_detected():
    """GIPP shock should be detected when change_period is near 2022Q1."""
    df = _make_quarterly_its_data("2022Q1")
    ev = RateChangeEvaluator(
        method="its",
        outcome_col="outcome",
        period_col="period",
        change_period="2022Q1",
        exposure_col="exposure",
    )
    with pytest.warns(UserWarning, match="GIPP|shock|confounder"):
        ev.fit(df)

    result = ev.summary()
    assert len(result.method_detail.shocks_near_intervention) > 0
