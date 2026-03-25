"""
Tests for DiDEstimator internals and RateChangeEvaluator DiD behavior.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from insurance_causal.rate_change import RateChangeEvaluator, make_rate_change_data


def _make_basic_evaluator(**kwargs):
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


def test_did_recovers_true_att():
    """DiD recovers DGP ATT within 2 SE on a large synthetic sample."""
    true_att = -0.05
    df = make_rate_change_data(n_policies=20_000, true_att=true_att, random_state=0)
    ev = _make_basic_evaluator()
    result = ev.fit(df).summary()
    assert result.method == "did"
    assert abs(result.att - true_att) < 2 * result.se, (
        f"ATT={result.att:.4f} not within 2SE ({result.se:.4f}) of true_att={true_att}"
    )


def test_did_parallel_trends_passes_when_held_by_dgp():
    """Pre-treatment event study passes when DGP satisfies PT by construction."""
    df = make_rate_change_data(n_policies=20_000, true_att=-0.03, random_state=1)
    ev = _make_basic_evaluator()
    ev.fit(df)
    pt = ev.parallel_trends_test()
    # DGP satisfies PT; joint test should pass at 5% level
    assert pt.joint_pt_pvalue > 0.05, (
        f"Parallel trends test failed with p={pt.joint_pt_pvalue:.4f} "
        "despite DGP satisfying PT by construction"
    )


def test_did_raises_without_control_group():
    """DiD raises ValueError when no control group exists."""
    df = make_rate_change_data(n_policies=2000, treated_fraction=1.0)
    ev = _make_basic_evaluator()
    with pytest.raises(ValueError, match="control group"):
        ev.fit(df)


def test_did_warns_staggered_adoption():
    """UserWarning is emitted when multiple treatment cohorts are detected."""
    # Build a staggered dataset: some segments have first treatment at period 5,
    # others at period 7. We construct this manually.
    rng = np.random.default_rng(999)
    n_segs = 20
    n_periods = 12

    rows = []
    for seg in range(n_segs):
        if seg < 5:
            ever_treated = False
            change = 99  # never
        elif seg < 12:
            ever_treated = True
            change = 5  # cohort A: treated from period 5
        else:
            ever_treated = True
            change = 7  # cohort B: treated from period 7

        for period in range(1, n_periods + 1):
            # treated indicator is 1 only from the segment's actual change period
            is_treated = int(ever_treated and period >= change)
            effect = -0.03 if is_treated else 0.0
            rows.append({
                "segment_id": seg,
                "period": period,
                "treated": is_treated,
                "loss_ratio": 0.65 + effect + rng.normal(0, 0.05),
                "exposure": rng.lognormal(0, 0.3) * 100,
            })

    df_staggered = pd.DataFrame(rows)

    ev = _make_basic_evaluator()
    with pytest.warns(UserWarning, match="[Ss]taggered"):
        ev.fit(df_staggered)

    result = ev.summary()
    assert result.staggered_adoption_detected


def test_cluster_fallback_to_hc3():
    """HC3 SE used and UserWarning emitted with < 20 clusters (segments)."""
    df = make_rate_change_data(n_policies=2000, n_segments=10, random_state=42)
    ev = _make_basic_evaluator()
    with pytest.warns(UserWarning, match="[Cc]luster"):
        result = ev.fit(df).summary()
    assert not result.cluster_se_used
    assert result.n_clusters == 10


def test_did_exposure_weighting_matters():
    """
    Unweighted DiD differs from exposure-weighted DiD with heterogeneous exposure.
    """
    rng = np.random.default_rng(0)
    df = make_rate_change_data(n_policies=5000, true_att=-0.04, random_state=7)
    # Assign highly heterogeneous exposures so weighting matters
    df["exposure"] = rng.lognormal(0, 2, size=len(df))

    ev_weighted = _make_basic_evaluator(exposure_col="exposure")
    ev_unweighted = _make_basic_evaluator(exposure_col=None)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_w = ev_weighted.fit(df).summary()
        r_u = ev_unweighted.fit(df).summary()

    # They should differ (exposure weighting is doing something non-trivial)
    assert abs(r_w.att - r_u.att) > 0.0005, (
        f"Weighted ATT={r_w.att:.5f}, unweighted ATT={r_u.att:.5f}: "
        "expected them to differ with lognormal(0,2) exposures"
    )


def test_did_att_pct_formula():
    """att_pct = ATT / pre_mean_treated * 100."""
    df = make_rate_change_data(n_policies=5000, true_att=-0.03, random_state=42)
    ev = _make_basic_evaluator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ev.fit(df).summary()
    if result.att_pct is not None:
        expected = result.att / result.pre_mean_treated * 100
        assert abs(result.att_pct - expected) < 1e-6


def test_did_result_method_label():
    """result.method should be 'did' when DiD is used."""
    df = make_rate_change_data(n_policies=3000, random_state=5)
    ev = _make_basic_evaluator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ev.fit(df).summary()
    assert result.method == "did"


def test_did_n_control_gt_zero():
    """n_control should be positive when a control group exists."""
    df = make_rate_change_data(n_policies=3000, treated_fraction=0.4)
    ev = _make_basic_evaluator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ev.fit(df).summary()
    assert result.n_control > 0


def test_did_event_study_reference_period():
    """Event study should include e=-1 with zero coefficient (reference)."""
    df = make_rate_change_data(n_policies=5000, random_state=2)
    ev = _make_basic_evaluator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev.fit(df)
    pt = ev.parallel_trends_test()
    ref_row = pt.event_study_df[pt.event_study_df["event_time"] == -1]
    assert len(ref_row) == 1
    assert ref_row["att_e"].iloc[0] == 0.0
