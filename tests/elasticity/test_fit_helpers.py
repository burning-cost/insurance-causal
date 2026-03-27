"""
Tests for elasticity/fit.py module-level helpers and edge cases.

Covers:
- _to_pandas / _to_polars helpers
- _extract_arrays: categorical encoding, NaN filling
- RenewalElasticityEstimator._build_estimator error paths
- dr_learner with binary_outcome=False raises ValueError
- cate_interval fallback for estimators without effect_interval
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest

from insurance_causal.elasticity.fit import (
    RenewalElasticityEstimator,
    _to_pandas,
    _to_polars,
    _extract_arrays,
)


# ---------------------------------------------------------------------------
# _to_pandas
# ---------------------------------------------------------------------------

class TestToPandasElasticity:
    def test_polars_to_pandas(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        result = _to_pandas(df)
        assert isinstance(result, pd.DataFrame)

    def test_pandas_passthrough(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = _to_pandas(df)
        assert result is df

    def test_values_correct(self):
        df = pl.DataFrame({"x": [10, 20, 30]})
        result = _to_pandas(df)
        np.testing.assert_array_equal(result["x"].values, [10, 20, 30])


# ---------------------------------------------------------------------------
# _to_polars
# ---------------------------------------------------------------------------

class TestToPolarsElasticity:
    def test_pandas_to_polars(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = _to_polars(df)
        assert isinstance(result, pl.DataFrame)

    def test_polars_passthrough(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        result = _to_polars(df)
        assert result is df


# ---------------------------------------------------------------------------
# _extract_arrays
# ---------------------------------------------------------------------------

class TestExtractArraysElasticity:
    def _make_df(self, n=40, with_cat=False):
        rng = np.random.default_rng(1)
        data = {
            "outcome": rng.integers(0, 2, n).astype(float),
            "treatment": rng.normal(0.05, 0.02, n),
            "age": rng.integers(20, 70, n).astype(float),
            "ncd": rng.integers(0, 6, n).astype(float),
        }
        if with_cat:
            data["channel"] = np.random.default_rng(1).choice(["pcw", "direct", "broker"], n)
        return pd.DataFrame(data)

    def test_returns_four_elements(self):
        df = self._make_df()
        result = _extract_arrays(df, "outcome", "treatment", ["age"])
        assert len(result) == 4

    def test_y_shape(self):
        df = self._make_df(n=30)
        Y, D, X, names = _extract_arrays(df, "outcome", "treatment", ["age"])
        assert Y.shape == (30,)

    def test_x_shape_with_numeric(self):
        df = self._make_df(n=30)
        Y, D, X, names = _extract_arrays(df, "outcome", "treatment", ["age", "ncd"])
        assert X.shape == (30, 2)

    def test_categorical_encoding(self):
        df = self._make_df(n=60, with_cat=True)
        Y, D, X, names = _extract_arrays(df, "outcome", "treatment", ["age", "channel"])
        # age + 2 channel dummies (drop_first=True, 3 levels)
        assert X.shape[1] == 3

    def test_nan_filled(self):
        df = self._make_df(n=20)
        df.loc[0, "age"] = np.nan
        Y, D, X, names = _extract_arrays(df, "outcome", "treatment", ["age"])
        assert not np.any(np.isnan(X))

    def test_all_float(self):
        df = self._make_df(n=30)
        Y, D, X, names = _extract_arrays(df, "outcome", "treatment", ["age"])
        assert Y.dtype == float
        assert D.dtype == float
        assert X.dtype == float


# ---------------------------------------------------------------------------
# RenewalElasticityEstimator._build_estimator error paths
# ---------------------------------------------------------------------------

class TestBuildEstimatorErrorPaths:
    def test_unknown_cate_model_raises(self):
        pytest.importorskip("econml", reason="econml required")
        est = RenewalElasticityEstimator(cate_model="rocket")
        model_y = est._build_outcome_model()
        model_t = est._build_treatment_model()
        with pytest.raises(ValueError, match="Unknown cate_model"):
            est._build_estimator(model_y, model_t)

    def test_dr_learner_continuous_raises(self):
        """DRLearner requires binary outcome."""
        pytest.importorskip("econml", reason="econml required")
        est = RenewalElasticityEstimator(cate_model="dr_learner", binary_outcome=False)
        model_y = est._build_outcome_model()
        model_t = est._build_treatment_model()
        with pytest.raises(ValueError, match="binary outcome"):
            est._build_estimator(model_y, model_t)

    def test_econml_missing_raises(self, monkeypatch):
        """If econml is not available, _build_estimator raises ImportError."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "econml" in name:
                raise ImportError("mocked econml missing")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        est = RenewalElasticityEstimator()
        with pytest.raises(ImportError, match="EconML"):
            est._build_estimator(object(), object())

    def test_causal_forest_rounding_warns(self):
        """n_estimators=5 with n_folds=2 (divisor=4) -> rounds to 8."""
        pytest.importorskip("econml", reason="econml required")
        est = RenewalElasticityEstimator(
            cate_model="causal_forest", n_folds=2, n_estimators=5
        )
        model_y = est._build_outcome_model()
        model_t = est._build_treatment_model()
        with pytest.warns(UserWarning, match="Rounding up"):
            estimator = est._build_estimator(model_y, model_t)
        assert estimator.n_estimators == 8

    def test_causal_forest_no_rounding_no_warn(self):
        """n_estimators=8 with n_folds=2 (divisor=4) -> no warning."""
        pytest.importorskip("econml", reason="econml required")
        est = RenewalElasticityEstimator(
            cate_model="causal_forest", n_folds=2, n_estimators=8
        )
        model_y = est._build_outcome_model()
        model_t = est._build_treatment_model()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            est._build_estimator(model_y, model_t)
        round_up = [x for x in w if "Rounding up" in str(x.message)]
        assert len(round_up) == 0


# ---------------------------------------------------------------------------
# _build_outcome_model / _build_treatment_model
# ---------------------------------------------------------------------------

class TestBuildModelsElasticity:
    def test_binary_outcome_catboost_returns_classifier(self):
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            pytest.skip("CatBoost not installed")
        est = RenewalElasticityEstimator(binary_outcome=True, catboost_iterations=10)
        model = est._build_outcome_model()
        assert isinstance(model, CatBoostClassifier)

    def test_continuous_outcome_catboost_returns_regressor(self):
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            pytest.skip("CatBoost not installed")
        est = RenewalElasticityEstimator(binary_outcome=False, catboost_iterations=10)
        model = est._build_outcome_model()
        assert isinstance(model, CatBoostRegressor)

    def test_custom_outcome_model_passthrough(self):
        """Passing a custom model object bypasses CatBoost."""
        from sklearn.linear_model import LinearRegression
        custom = LinearRegression()
        est = RenewalElasticityEstimator(outcome_model=custom, catboost_iterations=10)
        model = est._build_outcome_model()
        assert model is custom

    def test_custom_treatment_model_passthrough(self):
        from sklearn.linear_model import Ridge
        custom = Ridge()
        est = RenewalElasticityEstimator(treatment_model=custom, catboost_iterations=10)
        model = est._build_treatment_model()
        assert model is custom

    def test_treatment_model_catboost_returns_regressor(self):
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            pytest.skip("CatBoost not installed")
        est = RenewalElasticityEstimator(catboost_iterations=10)
        model = est._build_treatment_model()
        assert isinstance(model, CatBoostRegressor)


# ---------------------------------------------------------------------------
# cate_interval fallback (estimator without effect_interval)
# ---------------------------------------------------------------------------

class TestCateIntervalFallback:
    """Verify cate_interval falls back to ATE bounds when effect_interval missing."""

    def test_fallback_warns_and_returns_correct_shape(self):
        pytest.importorskip("econml", reason="econml required")

        # Build a minimal fitted estimator and replace its internal _estimator
        # with one that lacks effect_interval
        from insurance_causal.elasticity.data import make_renewal_data

        df = make_renewal_data(n=200, seed=0)
        confounders = ["age", "ncd_years", "vehicle_group", "channel"]

        est = RenewalElasticityEstimator(
            cate_model="linear_dml",
            binary_outcome=False,
            n_estimators=8,
            n_folds=2,
            catboost_iterations=30,
        )
        est.fit(df, confounders=confounders)

        # Monkey-patch: remove effect_interval
        # Capture the original estimator before replacing it to avoid infinite
        # recursion — after the assignment est._estimator IS the new object.
        _original_estimator = est._estimator

        class _NoIntervalEstimator:
            def effect(self, X):
                return _original_estimator.effect(X)
            def ate_interval(self, X=None, alpha=0.05):
                return _original_estimator.ate_interval(X=X, alpha=alpha)
            # no effect_interval

        est._estimator = _NoIntervalEstimator()

        with pytest.warns(UserWarning, match="confidence intervals"):
            lb, ub = est.cate_interval(df)

        assert lb.shape == (len(df),)
        assert ub.shape == (len(df),)
        assert np.all(lb <= ub)
