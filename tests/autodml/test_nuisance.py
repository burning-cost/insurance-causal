"""Tests for _nuisance.py — nuisance outcome models."""
import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from insurance_causal.autodml._nuisance import (
    SklearnNuisance,
    build_nuisance_model,
    _build_DX,
)
from insurance_causal.autodml._types import OutcomeFamily


def make_regression_data(n=200, p=4, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    D = rng.uniform(200, 600, n)
    Y = -0.002 * D + 0.5 * X[:, 0] + rng.randn(n) * 0.1
    return X, D, Y


class TestBuildDX:
    def test_stacks_correctly(self):
        D = np.array([1.0, 2.0, 3.0])
        X = np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
        DX = _build_DX(D, X)
        assert DX.shape == (3, 3)
        np.testing.assert_array_equal(DX[:, 0], D)
        np.testing.assert_array_equal(DX[:, 1:], X)

    def test_1d_D(self):
        D = np.ones(5)
        X = np.ones((5, 3))
        DX = _build_DX(D, X)
        assert DX.shape == (5, 4)


class TestSklearnNuisance:
    def test_fit_predict_shapes(self):
        X, D, Y = make_regression_data()
        m = SklearnNuisance()
        m.fit(D, X, Y)
        pred = m.predict(D, X)
        assert pred.shape == (200,)

    def test_raises_before_fit(self):
        m = SklearnNuisance()
        X, D, _ = make_regression_data(n=10)
        with pytest.raises(RuntimeError):
            m.predict(D, X)

    def test_callable_interface(self):
        X, D, Y = make_regression_data()
        m = SklearnNuisance()
        m.fit(D, X, Y)
        pred = m(D, X)
        assert pred.shape == (200,)

    def test_with_sample_weight(self):
        X, D, Y = make_regression_data()
        w = np.ones(200) * 2.0
        m = SklearnNuisance()
        m.fit(D, X, Y, sample_weight=w)
        pred = m.predict(D, X)
        assert pred.shape == (200,)

    def test_poisson_clips_prediction(self):
        X, D, Y = make_regression_data()
        Y = np.abs(Y) + 0.01
        m = SklearnNuisance(outcome_family=OutcomeFamily.POISSON)
        m.fit(D, X, Y)
        pred = m.predict(D, X)
        assert np.all(pred > 0)

    def test_gamma_clips_prediction(self):
        X, D, Y = make_regression_data()
        Y = np.abs(Y) + 0.01
        m = SklearnNuisance(outcome_family=OutcomeFamily.GAMMA)
        m.fit(D, X, Y)
        pred = m.predict(D, X)
        assert np.all(pred > 0)

    def test_custom_estimator(self):
        X, D, Y = make_regression_data()
        m = SklearnNuisance(estimator=Ridge(alpha=1.0))
        m.fit(D, X, Y)
        pred = m.predict(D, X)
        assert pred.shape == (200,)

    def test_predict_at_new_D(self):
        X, D, Y = make_regression_data()
        m = SklearnNuisance()
        m.fit(D, X, Y)
        D_new = np.full(200, 400.0)
        pred = m.predict(D_new, X)
        assert pred.shape == (200,)

    def test_no_nan_predictions(self):
        X, D, Y = make_regression_data()
        m = SklearnNuisance()
        m.fit(D, X, Y)
        pred = m.predict(D, X)
        assert not np.any(np.isnan(pred))


class TestBuildNuisanceModel:
    def test_sklearn_backend(self):
        m = build_nuisance_model(outcome_family=OutcomeFamily.GAUSSIAN, backend="sklearn")
        assert isinstance(m, SklearnNuisance)

    def test_linear_backend(self):
        from insurance_causal.autodml._nuisance import SklearnNuisance
        m = build_nuisance_model(outcome_family=OutcomeFamily.GAUSSIAN, backend="linear")
        assert isinstance(m, SklearnNuisance)

    def test_unknown_backend_defaults_sklearn(self):
        m = build_nuisance_model(outcome_family=OutcomeFamily.GAUSSIAN, backend="unknown")
        assert isinstance(m, SklearnNuisance)

    def test_poisson_family(self):
        m = build_nuisance_model(outcome_family=OutcomeFamily.POISSON)
        assert isinstance(m, SklearnNuisance)

    def test_catboost_unavailable_raises(self):
        """CatBoost backend should raise ImportError if catboost not installed."""
        try:
            import catboost  # noqa: F401
            pytest.skip("catboost is installed")
        except ImportError:
            from insurance_causal.autodml._nuisance import CatBoostNuisance
            m = CatBoostNuisance()
            X, D, Y = make_regression_data(n=10)
            with pytest.raises(ImportError, match="CatBoost"):
                m.fit(D, X, Y)
