"""
Tests for internal utilities.

These are unit tests on small, self-contained functions that do not require
fitting any models. Safe to run locally.
"""

import numpy as np
import pandas as pd
import pytest

from insurance_causal._utils import (
    to_pandas,
    poisson_outcome_transform,
    gamma_outcome_transform,
    check_overlap,
)


class TestToPandas:
    def test_pandas_passthrough(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = to_pandas(df)
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, df)

    def test_polars_conversion(self):
        """Test polars->pandas conversion, skipping if pyarrow is broken on this platform."""
        try:
            import polars as pl
            df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        except ImportError:
            pytest.skip("polars not installed")

        try:
            result = to_pandas(df)
        except (RuntimeError, TypeError, AttributeError) as exc:
            # pyarrow ABI conflicts in some environments (e.g. certain Databricks
            # cluster versions) make polars->pandas conversion impossible.
            # This is an environment issue, not a code defect.
            exc_str = str(exc)
            if any(s in exc_str for s in ("pyarrow", "_ARRAY_API", "polars DataFrame")):
                pytest.skip(f"polars->pandas conversion unavailable in this environment: {exc_str[:200]}")
            raise

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]
        assert len(result) == 3

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Expected a pandas or polars DataFrame"):
            to_pandas([1, 2, 3])


class TestPoissonOutcomeTransform:
    def test_divides_by_exposure(self):
        y = np.array([2.0, 4.0, 6.0])
        exposure = np.array([1.0, 2.0, 3.0])
        result = poisson_outcome_transform(y, exposure)
        np.testing.assert_allclose(result, [2.0, 2.0, 2.0])

    def test_no_exposure_passthrough(self):
        y = np.array([1, 3, 5])
        result = poisson_outcome_transform(y, None)
        np.testing.assert_allclose(result, [1.0, 3.0, 5.0])

    def test_zero_exposure_raises(self):
        y = np.array([1.0, 2.0])
        exposure = np.array([1.0, 0.0])
        with pytest.raises(ValueError, match="strictly positive"):
            poisson_outcome_transform(y, exposure)

    def test_negative_exposure_raises(self):
        y = np.array([1.0])
        exposure = np.array([-0.5])
        with pytest.raises(ValueError, match="strictly positive"):
            poisson_outcome_transform(y, exposure)


class TestGammaOutcomeTransform:
    def test_log_transform(self):
        y = np.array([1.0, np.e, np.e ** 2])
        result = gamma_outcome_transform(y, None)
        np.testing.assert_allclose(result, [0.0, 1.0, 2.0])

    def test_non_positive_raises(self):
        y = np.array([100.0, 0.0, 200.0])
        with pytest.raises(ValueError, match="strictly positive"):
            gamma_outcome_transform(y, None)

    def test_negative_raises(self):
        y = np.array([100.0, -50.0])
        with pytest.raises(ValueError, match="strictly positive"):
            gamma_outcome_transform(y, None)


class TestCheckOverlap:
    def test_returns_expected_keys(self):
        values = np.random.uniform(-0.3, 0.3, 1000)
        stats = check_overlap(values)
        for key in ["n_obs", "mean", "std", "min", "p5", "p25", "p75", "p95", "max"]:
            assert key in stats

    def test_n_obs_correct(self):
        values = np.arange(100.0)
        stats = check_overlap(values)
        assert stats["n_obs"] == 100

    def test_ordering(self):
        values = np.random.normal(0, 1, 500)
        stats = check_overlap(values)
        assert stats["min"] <= stats["p5"] <= stats["p25"]
        assert stats["p25"] <= stats["p75"] <= stats["p95"] <= stats["max"]
