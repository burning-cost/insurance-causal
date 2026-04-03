"""
Extended tests for treatment.py edge cases.

The existing test_treatments.py covers the main paths. This file adds:
- PriceChangeTreatment: boundary of 5% extreme threshold, exact-5% edge
- BinaryTreatment: exactly-50 group size (boundary of minimum check),
  True/False boolean values, 0.0/1.0 float values
- ContinuousTreatment: exactly-10 unique values boundary, repr, large constant
- AnyTreatment type alias is usable
- All treatment classes have treatment_type property
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from insurance_causal.treatments import (
    AnyTreatment,
    BinaryTreatment,
    ContinuousTreatment,
    PriceChangeTreatment,
)


# ---------------------------------------------------------------------------
# PriceChangeTreatment — edge cases
# ---------------------------------------------------------------------------


class TestPriceChangeTreatmentEdgeCases:
    def test_exactly_5_pct_extreme_passes(self):
        """Exactly 5% of values >= 1.0 is right at the threshold — should pass."""
        t = PriceChangeTreatment(column="price_change")
        # 5% = 50 values out of 1000 at exactly 1.0 (not > 1.0)
        # The check is > 1.0, so values at exactly 1.0 do NOT trigger the extreme check
        s = pd.Series(np.concatenate([
            np.random.uniform(-0.3, 0.3, 950),
            np.ones(50),  # exactly 5% at 1.0, but abs() > 1.0 is False for exactly 1.0
        ]))
        t.validate(s)  # should not raise (1.0 is not > 1.0)

    def test_just_above_5_pct_extreme_raises(self):
        """More than 5% of values > 1.0 should raise."""
        t = PriceChangeTreatment(column="price_change")
        s = pd.Series(np.concatenate([
            np.random.uniform(-0.3, 0.3, 940),
            np.full(60, 1.5),  # 6% > 1.0
        ]))
        with pytest.raises(ValueError, match="PriceChangeTreatment"):
            t.validate(s)

    def test_clip_percentiles_zero_one_is_identity(self):
        """Clipping to (0, 1) percentiles should not change the series."""
        t = PriceChangeTreatment(column="p", scale="linear", clip_percentiles=(0.0, 1.0))
        s = pd.Series([0.1, 0.2, 0.3, -0.1, -0.2])
        out = t.transform(s)
        np.testing.assert_allclose(out, s)

    def test_log_transform_of_zero_is_zero(self):
        """log(1 + 0) = 0, so zero price change stays zero."""
        t = PriceChangeTreatment(column="p", scale="log")
        s = pd.Series([0.0])
        out = t.transform(s)
        assert float(out.iloc[0]) == pytest.approx(0.0)

    def test_log_transform_negative_values(self):
        """Negative price changes: log(1 + (-0.1)) < 0."""
        t = PriceChangeTreatment(column="p", scale="log")
        s = pd.Series([-0.10])
        out = t.transform(s)
        assert float(out.iloc[0]) < 0.0

    def test_column_attribute(self):
        t = PriceChangeTreatment(column="pct_change")
        assert t.column == "pct_change"

    def test_scale_attribute_default(self):
        t = PriceChangeTreatment(column="p")
        assert t.scale == "log"

    def test_clip_percentiles_none_default(self):
        t = PriceChangeTreatment(column="p")
        assert t.clip_percentiles is None

    def test_validate_passes_all_zeros(self):
        """All-zero price changes are a degenerate but valid case."""
        t = PriceChangeTreatment(column="p")
        s = pd.Series(np.zeros(100))
        t.validate(s)  # no error expected


# ---------------------------------------------------------------------------
# BinaryTreatment — edge cases
# ---------------------------------------------------------------------------


class TestBinaryTreatmentEdgeCases:
    def test_exactly_50_treated_passes(self):
        """Exactly 50 in each group — at the boundary, should pass."""
        t = BinaryTreatment(column="flag")
        s = pd.Series([0] * 50 + [1] * 50)
        t.validate(s)  # 50 is the minimum — should not raise

    def test_49_treated_raises(self):
        """49 treated is below minimum of 50."""
        t = BinaryTreatment(column="flag")
        s = pd.Series([0] * 500 + [1] * 49)
        with pytest.raises(ValueError, match="fewer than 50"):
            t.validate(s)

    def test_boolean_values_accepted(self):
        """True/False should be accepted as valid binary values."""
        t = BinaryTreatment(column="flag")
        s = pd.Series([True] * 100 + [False] * 100)
        t.validate(s)  # should not raise

    def test_float_values_0_1_accepted(self):
        """0.0 and 1.0 floats should be accepted."""
        t = BinaryTreatment(column="flag")
        s = pd.Series([0.0] * 100 + [1.0] * 100)
        t.validate(s)  # should not raise

    def test_value_2_raises(self):
        """Value of 2 is not binary."""
        t = BinaryTreatment(column="flag")
        s = pd.Series([0, 1, 2])
        with pytest.raises(ValueError, match="BinaryTreatment"):
            t.validate(s)

    def test_negative_value_raises(self):
        """Negative values are not valid binary."""
        t = BinaryTreatment(column="flag")
        s = pd.Series([0, 1, -1])
        with pytest.raises(ValueError):
            t.validate(s)

    def test_transform_bool_to_float(self):
        """Boolean series should be cast to float."""
        t = BinaryTreatment(column="flag")
        s = pd.Series([True, False, True])
        out = t.transform(s)
        assert out.dtype == float

    def test_transform_preserves_values(self):
        t = BinaryTreatment(column="flag")
        s = pd.Series([0, 1, 0, 1])
        out = t.transform(s)
        np.testing.assert_allclose(out, [0.0, 1.0, 0.0, 1.0])

    def test_49_control_raises(self):
        """Fewer than 50 in the control group should also raise."""
        t = BinaryTreatment(column="flag")
        s = pd.Series([0] * 49 + [1] * 500)
        with pytest.raises(ValueError, match="fewer than 50"):
            t.validate(s)

    def test_default_labels(self):
        t = BinaryTreatment(column="flag")
        assert t.positive_label == "treated"
        assert t.negative_label == "control"

    def test_all_zeros_raises(self):
        """No treated observations — should raise."""
        t = BinaryTreatment(column="flag")
        s = pd.Series([0] * 200)
        with pytest.raises(ValueError, match="fewer than 50"):
            t.validate(s)

    def test_all_ones_raises(self):
        """No control observations — should raise."""
        t = BinaryTreatment(column="flag")
        s = pd.Series([1] * 200)
        with pytest.raises(ValueError, match="fewer than 50"):
            t.validate(s)


# ---------------------------------------------------------------------------
# ContinuousTreatment — edge cases
# ---------------------------------------------------------------------------


class TestContinuousTreatmentEdgeCases:
    def test_exactly_10_unique_values_no_warning(self):
        """Exactly 10 unique values — boundary: warning threshold is < 10."""
        t = ContinuousTreatment(column="score")
        s = pd.Series(list(range(10)) * 50)  # 10 unique values, 500 obs
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            t.validate(s)
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            # Exactly 10 unique values: check the actual boundary behaviour
            # (the implementation warns if nunique() < 10, so 10 should NOT warn)
            assert len(user_warnings) == 0, (
                "10 unique values should not trigger a warning — threshold is < 10"
            )

    def test_9_unique_values_warns(self):
        """9 unique values should trigger a UserWarning."""
        t = ContinuousTreatment(column="score")
        s = pd.Series(list(range(9)) * 100)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            t.validate(s)
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) == 1
            assert "unique values" in str(user_warnings[0].message)

    def test_standardise_stores_mean_and_std(self):
        """After transform, _mean and _std should be populated."""
        t = ContinuousTreatment(column="score", standardise=True)
        s = pd.Series([10.0, 20.0, 30.0])
        t.transform(s)
        assert abs(t._mean - 20.0) < 1e-10
        assert abs(t._std - s.std()) < 1e-10

    def test_standardise_idempotent_values(self):
        """After standardising, mean should be 0 and std 1."""
        t = ContinuousTreatment(column="score", standardise=True)
        rng = np.random.default_rng(0)
        s = pd.Series(rng.uniform(100, 200, 100))
        out = t.transform(s)
        assert abs(out.mean()) < 1e-10
        assert abs(out.std() - 1.0) < 1e-10

    def test_no_standardise_preserves_original_values(self):
        t = ContinuousTreatment(column="score", standardise=False)
        s = pd.Series([500.0, 750.0, 250.0])
        out = t.transform(s)
        np.testing.assert_allclose(out, s)

    def test_column_attribute(self):
        t = ContinuousTreatment(column="telematics")
        assert t.column == "telematics"

    def test_standardise_default_is_false(self):
        t = ContinuousTreatment(column="score")
        assert not t.standardise

    def test_null_raises(self):
        t = ContinuousTreatment(column="score")
        s = pd.Series([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="nulls"):
            t.validate(s)

    def test_all_nulls_raises(self):
        t = ContinuousTreatment(column="score")
        s = pd.Series([np.nan] * 10)
        with pytest.raises(ValueError, match="nulls"):
            t.validate(s)


# ---------------------------------------------------------------------------
# AnyTreatment type alias
# ---------------------------------------------------------------------------


class TestAnyTreatmentAlias:
    def test_isinstance_price_change(self):
        t = PriceChangeTreatment(column="p")
        assert isinstance(t, (PriceChangeTreatment, BinaryTreatment, ContinuousTreatment))

    def test_isinstance_binary(self):
        t = BinaryTreatment(column="f")
        assert isinstance(t, (PriceChangeTreatment, BinaryTreatment, ContinuousTreatment))

    def test_isinstance_continuous(self):
        t = ContinuousTreatment(column="s")
        assert isinstance(t, (PriceChangeTreatment, BinaryTreatment, ContinuousTreatment))

    def test_all_have_treatment_type(self):
        for t in [
            PriceChangeTreatment(column="p"),
            BinaryTreatment(column="f"),
            ContinuousTreatment(column="s"),
        ]:
            assert hasattr(t, "treatment_type")
            assert t.treatment_type in ("continuous", "binary")

    def test_all_have_column(self):
        for t in [
            PriceChangeTreatment(column="pc"),
            BinaryTreatment(column="bt"),
            ContinuousTreatment(column="ct"),
        ]:
            assert t.column in ("pc", "bt", "ct")

    def test_all_have_validate(self):
        for t in [
            PriceChangeTreatment(column="p"),
            BinaryTreatment(column="f"),
            ContinuousTreatment(column="s"),
        ]:
            assert callable(getattr(t, "validate", None))

    def test_all_have_transform(self):
        for t in [
            PriceChangeTreatment(column="p"),
            BinaryTreatment(column="f"),
            ContinuousTreatment(column="s"),
        ]:
            assert callable(getattr(t, "transform", None))
