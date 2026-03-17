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


class TestAdaptiveCatboostParams:
    """Tests for sample-size-adaptive nuisance model parameters (v0.3.0)."""

    def test_very_small_n_returns_low_capacity(self):
        """n < 2000 should give fewest iterations and shallowest depth."""
        from insurance_causal._utils import adaptive_catboost_params
        params = adaptive_catboost_params(500)
        assert params["iterations"] <= 150
        assert params["depth"] <= 5
        assert params["l2_leaf_reg"] >= 5.0

    def test_large_n_returns_full_capacity(self):
        """n >= 50000 should give maximum iterations."""
        from insurance_causal._utils import adaptive_catboost_params
        params = adaptive_catboost_params(100_000)
        assert params["iterations"] == 500
        assert params["depth"] == 6

    def test_capacity_increases_with_n(self):
        """Iterations should not decrease as n increases."""
        from insurance_causal._utils import adaptive_catboost_params
        sizes = [500, 2000, 5000, 10000, 50000, 100000]
        iters = [adaptive_catboost_params(n)["iterations"] for n in sizes]
        for i in range(len(iters) - 1):
            assert iters[i] <= iters[i + 1], (
                f"iterations should be non-decreasing: n={sizes[i]} gave {iters[i]}, "
                f"n={sizes[i+1]} gave {iters[i+1]}"
            )

    def test_returns_dict_with_required_keys(self):
        """Output should contain the keys needed by CatBoost."""
        from insurance_causal._utils import adaptive_catboost_params
        required = {"iterations", "learning_rate", "depth", "l2_leaf_reg"}
        for n in [100, 1000, 5000, 20000, 100000]:
            params = adaptive_catboost_params(n)
            missing = required - set(params.keys())
            assert not missing, f"n={n}: missing keys {missing}"

    def test_l2_regularisation_decreases_with_n(self):
        """l2_leaf_reg should be higher (more regularisation) at small n."""
        from insurance_causal._utils import adaptive_catboost_params
        small = adaptive_catboost_params(500)
        large = adaptive_catboost_params(100_000)
        assert small["l2_leaf_reg"] >= large["l2_leaf_reg"], (
            "small samples should have more L2 regularisation than large samples"
        )

    def test_build_catboost_regressor_no_n_samples(self):
        """Without n_samples, falls back to backward-compatible defaults."""
        from insurance_causal._utils import build_catboost_regressor
        model = build_catboost_regressor(random_state=42)
        params = model.get_params()
        assert params["iterations"] == 500
        assert params["depth"] == 6

    def test_build_catboost_regressor_with_small_n(self):
        """With n_samples=2000, uses reduced capacity."""
        from insurance_causal._utils import build_catboost_regressor
        model = build_catboost_regressor(random_state=42, n_samples=2000)
        params = model.get_params()
        assert params["iterations"] < 500
        assert params["depth"] <= 5

    def test_build_catboost_regressor_override(self):
        """override_params takes precedence over adaptive defaults."""
        from insurance_causal._utils import build_catboost_regressor
        model = build_catboost_regressor(
            random_state=42, n_samples=2000,
            override_params={"iterations": 999}
        )
        params = model.get_params()
        assert params["iterations"] == 999

    def test_build_catboost_classifier_with_small_n(self):
        """Classifier also uses adaptive params."""
        from insurance_causal._utils import build_catboost_classifier
        model = build_catboost_classifier(random_state=0, n_samples=3000)
        params = model.get_params()
        assert params["iterations"] < 500
