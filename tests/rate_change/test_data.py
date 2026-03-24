"""Tests for the synthetic data generator."""

import numpy as np
import pandas as pd
import pytest

from insurance_causal.rate_change import make_rate_change_data


class TestMakeRateChangeData:
    def test_did_basic_shape(self):
        df = make_rate_change_data(n_segments=10, n_periods=12, seed=0)
        assert len(df) == 10 * 12
        assert "segment" in df.columns
        assert "period" in df.columns
        assert "treated" in df.columns
        assert "outcome" in df.columns
        assert "earned_exposure" in df.columns

    def test_did_treated_fraction(self):
        df = make_rate_change_data(n_segments=20, treated_fraction=0.5, seed=0)
        n_treated = df.drop_duplicates("segment")["treated"].sum()
        assert n_treated == 10

    def test_did_treatment_indicator(self):
        df = make_rate_change_data(n_segments=10, n_periods=12, treatment_period=7, seed=0)
        # rate_change should be 1 only for treated units in periods >= 7
        active = df[(df["treated"] == 1) & (df["period"] >= 7)]
        assert (active["rate_change"] == 1).all()
        inactive_control = df[df["treated"] == 0]
        assert (inactive_control["rate_change"] == 0).all()
        pre_treated = df[(df["treated"] == 1) & (df["period"] < 7)]
        assert (pre_treated["rate_change"] == 0).all()

    def test_exposure_positive(self):
        df = make_rate_change_data(seed=42)
        assert (df["earned_exposure"] > 0).all()

    def test_outcome_positive(self):
        df = make_rate_change_data(seed=42)
        assert (df["outcome"] > 0).all()

    def test_its_mode(self):
        df = make_rate_change_data(mode="its", n_periods=20, seed=0)
        assert len(df) == 20
        assert "period" in df.columns
        assert "quarter" in df.columns
        assert "outcome" in df.columns
        # All treated in ITS
        assert (df["treated"] == 1).all()

    def test_its_rate_change_indicator(self):
        df = make_rate_change_data(mode="its", n_periods=20, treatment_period=9, seed=0)
        assert (df[df["period"] >= 9]["rate_change"] == 1).all()
        assert (df[df["period"] < 9]["rate_change"] == 0).all()

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode must be"):
            make_rate_change_data(mode="bad")

    def test_reproducible_seed(self):
        df1 = make_rate_change_data(seed=7)
        df2 = make_rate_change_data(seed=7)
        pd.testing.assert_frame_equal(df1, df2)

    def test_quarter_column(self):
        df = make_rate_change_data(n_periods=8, seed=0)
        assert df["quarter"].isin([1, 2, 3, 4]).all()
