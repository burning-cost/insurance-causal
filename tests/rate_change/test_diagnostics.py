"""
Tests for ParallelTrendsDiagnostic and StaggeredAdoptionChecker.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from insurance_causal.rate_change import RateChangeEvaluator, make_rate_change_data
from insurance_causal.rate_change._diagnostics import StaggeredAdoptionChecker


def test_staggered_adoption_checker_no_staggering():
    """Returns is_staggered=False when all treated units start at same period."""
    df = make_rate_change_data(n_policies=2000, n_segments=20, change_period=7)

    # Build a segment-period dataframe manually from make_rate_change_data output
    checker = StaggeredAdoptionChecker()

    # Add encoded period col
    df["_period_enc_"] = df["period"]
    df["_unit_id_enc_"] = df["segment_id"]

    is_staggered, cohorts = checker.check(
        df, treated_col="treated", period_col="_period_enc_", change_period=7
    )
    assert not is_staggered
    assert len(cohorts) == 1


def test_staggered_adoption_checker_detects_staggering():
    """Returns is_staggered=True when treated units have different start periods."""
    # Genuine bug fix: original test set treated=1 for ALL periods for both cohorts,
    # making staggered adoption undetectable. The treated indicator must be
    # period-specific (1 iff period >= cohort_change_period) to encode staggering.
    rng = np.random.default_rng(42)
    rows = []
    for seg in range(20):
        if seg < 8:
            ever_treated = False
            change = 99  # never treated
        elif seg < 14:
            ever_treated = True
            change = 5  # cohort 1: treated from period 5
        else:
            ever_treated = True
            change = 8  # cohort 2: treated from period 8

        for p in range(1, 13):
            # treated is 1 only when this unit is actually being treated this period
            treated_this_period = int(ever_treated and p >= change)
            rows.append({
                "segment_id": seg,
                "_period_enc_": p,
                "_unit_id_enc_": seg,
                "treated": treated_this_period,
                "outcome": 0.6 + rng.normal(0, 0.05),
                "exposure": 100.0,
            })

    df = pd.DataFrame(rows)
    checker = StaggeredAdoptionChecker()
    is_staggered, cohorts = checker.check(df, "treated", "_period_enc_", 5)
    assert is_staggered
    assert len(cohorts) == 2


def test_staggered_adoption_checker_warn():
    """warn_if_staggered emits UserWarning when is_staggered=True."""
    checker = StaggeredAdoptionChecker()
    with pytest.warns(UserWarning, match="[Ss]taggered"):
        msg = checker.warn_if_staggered(True, [5, 8])
    assert msg is not None
    assert "Goodman-Bacon" in msg


def test_staggered_adoption_checker_no_warn_when_clean():
    """warn_if_staggered does NOT emit a warning when is_staggered=False."""
    checker = StaggeredAdoptionChecker()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        msg = checker.warn_if_staggered(False, [7])
    assert msg is None


def test_parallel_trends_result_passes_attribute():
    """passes attribute is True when joint pvalue > 0.05."""
    df = make_rate_change_data(n_policies=10_000, true_att=-0.03, random_state=99)
    ev = RateChangeEvaluator(
        method="did",
        outcome_col="loss_ratio",
        period_col="period",
        treated_col="treated",
        change_period=7,
        exposure_col="exposure",
        unit_col="segment_id",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev.fit(df)
    pt = ev.parallel_trends_test()
    expected_passes = pt.joint_pt_pvalue > 0.05
    assert pt.passes == expected_passes


def test_parallel_trends_event_study_df_columns():
    """event_study_df has the expected columns."""
    df = make_rate_change_data(n_policies=5000, random_state=10)
    ev = RateChangeEvaluator(
        method="did",
        outcome_col="loss_ratio",
        period_col="period",
        treated_col="treated",
        change_period=7,
        exposure_col="exposure",
        unit_col="segment_id",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev.fit(df)
    pt = ev.parallel_trends_test()
    required_cols = {"event_time", "att_e", "se_e", "ci_lower_e", "ci_upper_e"}
    assert required_cols.issubset(pt.event_study_df.columns)
