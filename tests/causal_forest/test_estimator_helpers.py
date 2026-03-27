"""
Tests for HeterogeneousElasticityEstimator internal helpers and edge cases.

Covers the module-level helper functions (_to_pandas, _to_polars,
_extract_arrays, _extract_features) and estimator-internal logic
(_build_outcome_model, _build_treatment_model, _build_estimator,
gate small-group warning) that are not exercised by the smoke tests
in test_estimator.py.

These tests do NOT require econml — they test pure data-wrangling code.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest

from insurance_causal.causal_forest.estimator import (
    HeterogeneousElasticityEstimator,
    _to_pandas,
    _to_polars,
    _extract_arrays,
    _extract_features,
)


# ---------------------------------------------------------------------------
# _to_pandas
# ---------------------------------------------------------------------------

class TestToPandas:
    def test_polars_converted_to_pandas(self):
        df_pl = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        result = _to_pandas(df_pl)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]
        assert len(result) == 3

    def test_pandas_passthrough(self):
        df_pd = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = _to_pandas(df_pd)
        assert result is df_pd

    def test_values_preserved(self):
        df_pl = pl.DataFrame({"x": [10, 20, 30]})
        result = _to_pandas(df_pl)
        np.testing.assert_array_equal(result["x"].values, [10, 20, 30])


# ---------------------------------------------------------------------------
# _to_polars
# ---------------------------------------------------------------------------

class TestToPolars:
    def test_pandas_converted_to_polars(self):
        df_pd = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        result = _to_polars(df_pd)
        assert isinstance(result, pl.DataFrame)
        assert "a" in result.columns

    def test_polars_passthrough(self):
        df_pl = pl.DataFrame({"a": [1, 2, 3]})
        result = _to_polars(df_pl)
        assert result is df_pl

    def test_values_preserved(self):
        df_pd = pd.DataFrame({"y": [7, 8, 9]})
        result = _to_polars(df_pd)
        assert result["y"].to_list() == [7, 8, 9]


# ---------------------------------------------------------------------------
# _extract_arrays
# ---------------------------------------------------------------------------

class TestExtractArrays:
    def _make_df(self, n=50, with_cat=False):
        rng = np.random.default_rng(0)
        data = {
            "outcome": rng.integers(0, 2, n).astype(float),
            "treatment": rng.normal(0, 0.1, n),
            "age": rng.integers(20, 70, n).astype(float),
            "ncd": rng.integers(0, 6, n).astype(float),
        }
        if with_cat:
            data["channel"] = np.random.choice(["pcw", "direct", "broker"], n)
        return pd.DataFrame(data)

    def test_returns_five_elements(self):
        df = self._make_df()
        result = _extract_arrays(df, "outcome", "treatment", ["age", "ncd"])
        assert len(result) == 5

    def test_y_shape(self):
        df = self._make_df(n=40)
        Y, D, X, names, sw = _extract_arrays(df, "outcome", "treatment", ["age"])
        assert Y.shape == (40,)

    def test_d_shape(self):
        df = self._make_df(n=40)
        Y, D, X, names, sw = _extract_arrays(df, "outcome", "treatment", ["age"])
        assert D.shape == (40,)

    def test_x_has_correct_columns(self):
        df = self._make_df(n=40)
        Y, D, X, names, sw = _extract_arrays(df, "outcome", "treatment", ["age", "ncd"])
        assert X.shape[1] == 2
        assert "age" in names and "ncd" in names

    def test_no_exposure_returns_none_weight(self):
        df = self._make_df(n=30)
        Y, D, X, names, sw = _extract_arrays(df, "outcome", "treatment", ["age"])
        assert sw is None

    def test_exposure_divides_y(self):
        df = self._make_df(n=20)
        df["exposure"] = 2.0
        Y, D, X, names, sw = _extract_arrays(
            df, "outcome", "treatment", ["age"], exposure_col="exposure"
        )
        # Y should be original / 2
        expected = df["outcome"].values / 2.0
        np.testing.assert_allclose(Y, expected)

    def test_exposure_returns_sample_weight(self):
        df = self._make_df(n=20)
        df["exposure"] = 3.0
        Y, D, X, names, sw = _extract_arrays(
            df, "outcome", "treatment", ["age"], exposure_col="exposure"
        )
        assert sw is not None
        np.testing.assert_allclose(sw, 3.0)

    def test_negative_exposure_raises(self):
        df = self._make_df(n=10)
        df["exp"] = 1.0
        df.loc[0, "exp"] = -0.5
        with pytest.raises(ValueError, match="strictly positive"):
            _extract_arrays(df, "outcome", "treatment", ["age"], exposure_col="exp")

    def test_zero_exposure_raises(self):
        df = self._make_df(n=10)
        df["exp"] = 1.0
        df.loc[2, "exp"] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            _extract_arrays(df, "outcome", "treatment", ["age"], exposure_col="exp")

    def test_categorical_columns_one_hot_encoded(self):
        df = self._make_df(n=60, with_cat=True)
        Y, D, X, names, sw = _extract_arrays(
            df, "outcome", "treatment", ["age", "channel"]
        )
        # channel has 3 levels, drop_first gives 2 dummies
        assert X.shape[1] == 3  # age + 2 channel dummies

    def test_nan_filled_with_mean(self):
        df = self._make_df(n=20)
        df.loc[0, "age"] = np.nan
        Y, D, X, names, sw = _extract_arrays(df, "outcome", "treatment", ["age"])
        assert not np.any(np.isnan(X))

    def test_all_float_output(self):
        df = self._make_df(n=30)
        Y, D, X, names, sw = _extract_arrays(df, "outcome", "treatment", ["age", "ncd"])
        assert Y.dtype == float
        assert D.dtype == float
        assert X.dtype == float


# ---------------------------------------------------------------------------
# _extract_features
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    def test_returns_x_and_names(self):
        df = pd.DataFrame({
            "age": [30.0, 40.0, 50.0],
            "ncd": [0.0, 3.0, 5.0],
        })
        X, names = _extract_features(df, ["age", "ncd"])
        assert X.shape == (3, 2)
        assert "age" in names

    def test_categorical_one_hot_encoded(self):
        df = pd.DataFrame({
            "age": [30.0, 40.0, 50.0],
            "channel": ["pcw", "direct", "broker"],
        })
        X, names = _extract_features(df, ["age", "channel"])
        # 1 numeric + 2 dummies (drop_first=True from 3 levels)
        assert X.shape[1] == 3

    def test_nan_filled(self):
        df = pd.DataFrame({"age": [30.0, np.nan, 50.0]})
        X, names = _extract_features(df, ["age"])
        assert not np.any(np.isnan(X))

    def test_all_float(self):
        df = pd.DataFrame({"age": [30.0, 40.0], "ncd": [2.0, 4.0]})
        X, names = _extract_features(df, ["age", "ncd"])
        assert X.dtype == float


# ---------------------------------------------------------------------------
# HeterogeneousElasticityEstimator: _check_fitted
# ---------------------------------------------------------------------------

class TestCheckFitted:
    def test_unfitted_cate_raises(self):
        est = HeterogeneousElasticityEstimator()
        df = pl.DataFrame({"age": [30.0, 40.0]})
        with pytest.raises(RuntimeError, match="not fitted"):
            est.cate(df)

    def test_unfitted_cate_interval_raises(self):
        est = HeterogeneousElasticityEstimator()
        df = pl.DataFrame({"age": [30.0, 40.0]})
        with pytest.raises(RuntimeError, match="not fitted"):
            est.cate_interval(df)

    def test_unfitted_gate_raises(self):
        est = HeterogeneousElasticityEstimator()
        df = pl.DataFrame({"age": [30.0, 40.0], "group": ["a", "b"]})
        with pytest.raises(RuntimeError, match="not fitted"):
            est.gate(df, by="group")


# ---------------------------------------------------------------------------
# _build_outcome_model / _build_treatment_model (fallback to GBM)
# ---------------------------------------------------------------------------

class TestBuildModels:
    """Test model building with and without CatBoost installed."""

    def test_build_outcome_model_binary(self):
        """With catboost installed, should return a CatBoostClassifier."""
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            pytest.skip("CatBoost not installed")
        est = HeterogeneousElasticityEstimator(binary_outcome=True, catboost_iterations=10)
        model = est._build_outcome_model()
        assert isinstance(model, CatBoostClassifier)

    def test_build_outcome_model_continuous(self):
        """With catboost installed, continuous outcome -> CatBoostRegressor."""
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            pytest.skip("CatBoost not installed")
        est = HeterogeneousElasticityEstimator(binary_outcome=False, catboost_iterations=10)
        model = est._build_outcome_model()
        assert isinstance(model, CatBoostRegressor)

    def test_build_treatment_model_returns_regressor(self):
        """Treatment model is always a regressor (CatBoost or GBM fallback)."""
        try:
            from catboost import CatBoostRegressor
            expected_cls = CatBoostRegressor
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            expected_cls = GradientBoostingRegressor
        est = HeterogeneousElasticityEstimator(catboost_iterations=10)
        model = est._build_treatment_model()
        assert isinstance(model, expected_cls)

    def test_build_estimator_raises_without_econml(self, monkeypatch):
        """_build_estimator raises ImportError if econml not available."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "econml.dml":
                raise ImportError("mocked econml missing")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        est = HeterogeneousElasticityEstimator()
        model_y = object()
        model_t = object()
        with pytest.raises(ImportError, match="EconML"):
            est._build_estimator(model_y, model_t)


# ---------------------------------------------------------------------------
# n_estimators auto-rounding (without fitting)
# ---------------------------------------------------------------------------

class TestNEstimatorsRounding:
    def test_divisible_no_warning(self):
        """n_estimators=40 with n_folds=2 (divisor=4) should not warn."""
        try:
            from econml.dml import CausalForestDML
        except ImportError:
            pytest.skip("econml not installed")

        est = HeterogeneousElasticityEstimator(n_folds=2, n_estimators=40)
        model_y = est._build_outcome_model()
        model_t = est._build_treatment_model()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            est._build_estimator(model_y, model_t)
        round_up_warnings = [x for x in w if "Rounding up" in str(x.message)]
        assert len(round_up_warnings) == 0

    def test_not_divisible_warns_and_rounds_up(self):
        """n_estimators=41 with n_folds=2 -> rounds to 44."""
        try:
            from econml.dml import CausalForestDML
        except ImportError:
            pytest.skip("econml not installed")

        est = HeterogeneousElasticityEstimator(n_folds=2, n_estimators=41)
        model_y = est._build_outcome_model()
        model_t = est._build_treatment_model()
        with pytest.warns(UserWarning, match="Rounding up"):
            estimator = est._build_estimator(model_y, model_t)
        # 44 is the next multiple of 4 after 41
        assert estimator.n_estimators == 44


# ---------------------------------------------------------------------------
# gate() small-group warning (without full fit — inject a mock estimator)
# ---------------------------------------------------------------------------

class TestGateSmallGroupWarning:
    """Verify gate() emits a warning when any group has < 500 observations."""

    def test_small_group_triggers_warning(self):
        """If a group has < 500 rows, gate() should warn."""
        pytest.importorskip("econml", reason="econml required for this test")

        from insurance_causal.causal_forest.data import make_hte_renewal_data

        df = make_hte_renewal_data(n=300, seed=0)  # very small: all groups will be small

        # Build a minimal fitted estimator using the conftest fixtures logic
        confounders = ["age", "ncd_years", "vehicle_group", "channel"]
        est = HeterogeneousElasticityEstimator(
            n_folds=2, n_estimators=4, min_samples_leaf=5,
            catboost_iterations=10, random_state=0,
        )
        est.fit(df, confounders=confounders)

        with pytest.warns(UserWarning, match="< 500"):
            est.gate(df, by="channel")
