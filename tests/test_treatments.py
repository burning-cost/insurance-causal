"""
Tests for treatment specification classes.

These tests run locally — no heavy computation, just validation and
transformation logic.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from insurance_causal.treatments import (
    PriceChangeTreatment,
    BinaryTreatment,
    ContinuousTreatment,
)


class TestPriceChangeTreatment:
    def test_log_transform(self):
        t = PriceChangeTreatment(column="price_change", scale="log")
        s = pd.Series([0.0, 0.05, -0.10, 0.20])
        out = t.transform(s)
        expected = np.log1p(s)
        np.testing.assert_allclose(out, expected)

    def test_linear_transform_identity(self):
        t = PriceChangeTreatment(column="price_change", scale="linear")
        s = pd.Series([0.0, 0.05, -0.10])
        out = t.transform(s)
        np.testing.assert_allclose(out, s)

    def test_clip_percentiles(self):
        t = PriceChangeTreatment(
            column="price_change", scale="linear",
            clip_percentiles=(0.05, 0.95)
        )
        s = pd.Series(np.linspace(-0.5, 0.5, 100))
        out = t.transform(s)
        # After clipping to p5/p95, extremes should be clipped
        assert out.min() >= s.quantile(0.05) - 1e-10
        assert out.max() <= s.quantile(0.95) + 1e-10

    def test_validate_passes_normal_data(self):
        t = PriceChangeTreatment(column="price_change")
        s = pd.Series(np.random.uniform(-0.3, 0.3, 500))
        t.validate(s)  # Should not raise

    def test_validate_rejects_nulls(self):
        t = PriceChangeTreatment(column="price_change")
        s = pd.Series([0.05, np.nan, -0.10])
        with pytest.raises(ValueError, match="nulls"):
            t.validate(s)

    def test_validate_rejects_extreme_values(self):
        t = PriceChangeTreatment(column="price_change")
        # > 5% of values exceed ±100% (i.e., column is in percentage points, not proportions)
        s = pd.Series(np.concatenate([
            np.random.uniform(-0.3, 0.3, 900),
            np.repeat(150.0, 100),  # 10% extreme — above 5% threshold
        ]))
        with pytest.raises(ValueError, match="PriceChangeTreatment"):
            t.validate(s)

    def test_treatment_type(self):
        t = PriceChangeTreatment(column="price_change")
        assert t.treatment_type == "continuous"


class TestBinaryTreatment:
    def test_transform_to_float(self):
        t = BinaryTreatment(column="is_aggregator")
        s = pd.Series([0, 1, 0, 1])
        out = t.transform(s)
        assert out.dtype == float

    def test_validate_valid_binary(self):
        t = BinaryTreatment(column="is_aggregator")
        s = pd.Series([0] * 200 + [1] * 200)
        t.validate(s)  # Should not raise

    def test_validate_rejects_non_binary(self):
        t = BinaryTreatment(column="channel")
        s = pd.Series([0, 1, 2])  # 2 is not binary
        with pytest.raises(ValueError, match="BinaryTreatment"):
            t.validate(s)

    def test_validate_rejects_nulls(self):
        t = BinaryTreatment(column="is_discount")
        s = pd.Series([0.0, 1.0, np.nan])
        with pytest.raises(ValueError, match="nulls"):
            t.validate(s)

    def test_validate_rejects_too_few_treated(self):
        t = BinaryTreatment(column="flag")
        # Only 10 treated — below the 50 minimum
        s = pd.Series([0] * 500 + [1] * 10)
        with pytest.raises(ValueError, match="fewer than 50"):
            t.validate(s)

    def test_treatment_type(self):
        t = BinaryTreatment(column="flag")
        assert t.treatment_type == "binary"

    def test_labels(self):
        t = BinaryTreatment(column="channel", positive_label="aggregator", negative_label="direct")
        assert t.positive_label == "aggregator"
        assert t.negative_label == "direct"


class TestContinuousTreatment:
    def test_no_standardise(self):
        t = ContinuousTreatment(column="telematics_score")
        s = pd.Series([100.0, 200.0, 150.0])
        out = t.transform(s)
        np.testing.assert_allclose(out, s.astype(float))

    def test_standardise(self):
        t = ContinuousTreatment(column="telematics_score", standardise=True)
        s = pd.Series([0.0, 10.0, 20.0, 30.0, 40.0])
        out = t.transform(s)
        np.testing.assert_allclose(out.mean(), 0.0, atol=1e-10)
        np.testing.assert_allclose(out.std(), 1.0, atol=1e-10)

    def test_standardise_zero_variance_raises(self):
        t = ContinuousTreatment(column="constant", standardise=True)
        s = pd.Series([5.0] * 100)
        with pytest.raises(ValueError, match="zero variance"):
            t.transform(s)

    def test_validate_warns_few_unique_values(self):
        t = ContinuousTreatment(column="ordinal")
        s = pd.Series([1, 2, 3, 1, 2, 3, 1, 2, 3])  # only 3 unique values
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            t.validate(s)
            assert len(w) == 1
            assert "unique values" in str(w[0].message)

    def test_validate_nulls(self):
        t = ContinuousTreatment(column="score")
        s = pd.Series([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="nulls"):
            t.validate(s)

    def test_treatment_type(self):
        t = ContinuousTreatment(column="score")
        assert t.treatment_type == "continuous"
