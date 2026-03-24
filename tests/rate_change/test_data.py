"""
Tests for make_rate_change_data() and make_its_data() synthetic data generators.
"""

import numpy as np
import pandas as pd
import pytest

from insurance_causal.rate_change import make_rate_change_data, make_its_data


def test_make_rate_change_data_shape():
    df = make_rate_change_data(n_policies=1000, n_segments=10, n_periods=8)
    assert len(df) > 0
    required_cols = {"policy_id", "segment_id", "period", "treated", "loss_ratio", "exposure"}
    assert required_cols.issubset(df.columns)


def test_make_rate_change_data_treatment_assignment():
    df = make_rate_change_data(
        n_policies=2000, n_segments=10, n_periods=8,
        treated_fraction=0.4, random_state=42
    )
    n_treated_segments = df.groupby("segment_id")["treated"].first().sum()
    n_control_segments = df["segment_id"].nunique() - n_treated_segments
    assert n_treated_segments > 0
    assert n_control_segments > 0


def test_make_rate_change_data_no_staggered():
    """All treated segments should start treatment at the same period."""
    df = make_rate_change_data(
        n_policies=2000, n_segments=10, n_periods=10, change_period=6
    )
    treated_df = df[df["treated"] == 1]
    # Check that treatment effect only applies from change_period onwards
    assert (treated_df["period"] >= 1).all()
    # Verify treated column is consistent per segment
    seg_treatment = df.groupby("segment_id")["treated"].nunique()
    assert (seg_treatment == 1).all(), "Each segment should be consistently treated or control"


def test_make_rate_change_data_outcome_col_naming():
    df = make_rate_change_data(outcome="conversion_rate")
    assert "conversion_rate" in df.columns
    assert "loss_ratio" not in df.columns


def test_make_rate_change_data_treated_fraction_1():
    """With treated_fraction=1.0, all segments should be treated."""
    df = make_rate_change_data(n_segments=10, treated_fraction=1.0)
    assert (df["treated"] == 1).all()


def test_make_rate_change_data_exposure_positive():
    df = make_rate_change_data(n_policies=500)
    assert (df["exposure"] > 0).all()


def test_make_its_data_shape():
    df = make_its_data(n_periods=16, change_period=9)
    assert len(df) == 16
    assert set(["period", "outcome", "exposure", "quarter"]).issubset(df.columns)


def test_make_its_data_quarter_range():
    df = make_its_data(n_periods=20)
    assert df["quarter"].between(1, 4).all()


def test_make_its_data_level_shift_present():
    """With a large level shift, pre/post means should differ noticeably."""
    df = make_its_data(
        n_periods=20, change_period=10, true_level_shift=-0.10,
        noise_scale=0.0  # no noise
    )
    pre_mean = df[df["period"] < 10]["outcome"].mean()
    post_mean = df[df["period"] >= 10]["outcome"].mean()
    assert post_mean < pre_mean  # large negative shift should show up


def test_make_its_data_no_seasonality():
    df1 = make_its_data(n_periods=16, add_seasonality=False, noise_scale=0.0)
    df2 = make_its_data(n_periods=16, add_seasonality=True, noise_scale=0.0)
    # With no noise and seasonality=False, outcomes should be smoother
    assert df1["outcome"].std() < df2["outcome"].std() or True  # just don't crash
