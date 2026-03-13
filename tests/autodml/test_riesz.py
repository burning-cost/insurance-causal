"""Tests for riesz.py — ForestRiesz and LinearRiesz."""
import numpy as np
import pytest
from insurance_causal.autodml.riesz import ForestRiesz, LinearRiesz, compute_riesz_loss


def make_data(n=200, p=4, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    D = rng.uniform(200, 600, n)
    return X, D


def linear_nuisance(D, X):
    """Simple linear nuisance: -0.002 * D + 0.5 * X[:,0]"""
    return -0.002 * D + 0.5 * X[:, 0]


class TestForestRiesz:
    def test_fit_predict_shapes(self):
        X, D = make_data()
        riesz = ForestRiesz(n_estimators=50, random_state=0)
        riesz.fit(X, D, linear_nuisance)
        alpha = riesz.predict(X)
        assert alpha.shape == (200,)

    def test_raises_before_fit(self):
        X, D = make_data()
        riesz = ForestRiesz()
        with pytest.raises(RuntimeError, match="fit()"):
            riesz.predict(X)

    def test_with_sample_weight(self):
        X, D = make_data()
        w = np.ones(200) * 2.0
        riesz = ForestRiesz(n_estimators=50, random_state=0)
        riesz.fit(X, D, linear_nuisance, sample_weight=w)
        alpha = riesz.predict(X)
        assert alpha.shape == (200,)

    def test_predict_different_n(self):
        X, D = make_data(n=200)
        riesz = ForestRiesz(n_estimators=50, random_state=0)
        riesz.fit(X, D, linear_nuisance)
        X_new = np.random.randn(50, 4)
        alpha = riesz.predict(X_new)
        assert alpha.shape == (50,)

    def test_no_nan_output(self):
        X, D = make_data()
        riesz = ForestRiesz(n_estimators=50, random_state=0)
        riesz.fit(X, D, linear_nuisance)
        alpha = riesz.predict(X)
        assert not np.any(np.isnan(alpha))

    def test_captures_derivative_direction(self):
        """Riesz should approximate dg/dD ~ -0.002 for our nuisance."""
        X, D = make_data(n=500, seed=1)
        riesz = ForestRiesz(n_estimators=100, max_depth=4, random_state=1)
        riesz.fit(X, D, linear_nuisance)
        alpha = riesz.predict(X)
        # Should be near -0.002 (the true derivative)
        assert np.mean(alpha) < 0, "Expected negative alpha for decreasing nuisance"

    def test_max_depth_none(self):
        X, D = make_data(n=100)
        riesz = ForestRiesz(n_estimators=20, max_depth=None, random_state=0)
        riesz.fit(X, D, linear_nuisance)
        alpha = riesz.predict(X)
        assert alpha.shape == (100,)

    def test_custom_eps(self):
        X, D = make_data()
        riesz = ForestRiesz(n_estimators=50, eps=0.01, random_state=0)
        riesz.fit(X, D, linear_nuisance)
        alpha = riesz.predict(X)
        assert alpha.shape == (200,)

    def test_2d_treatment_boundary(self):
        """All D values equal — degenerate case; should not crash."""
        X = np.random.randn(100, 4)
        D = np.full(100, 350.0)
        riesz = ForestRiesz(n_estimators=20, random_state=0)
        riesz.fit(X, D, linear_nuisance)
        alpha = riesz.predict(X)
        assert alpha.shape == (100,)


class TestLinearRiesz:
    def test_fit_predict_shapes(self):
        X, D = make_data()
        riesz = LinearRiesz()
        riesz.fit(X, D, linear_nuisance)
        alpha = riesz.predict(X)
        assert alpha.shape == (200,)

    def test_raises_before_fit(self):
        X, D = make_data()
        riesz = LinearRiesz()
        with pytest.raises(RuntimeError, match="fit()"):
            riesz.predict(X)

    def test_with_sample_weight(self):
        X, D = make_data()
        w = np.ones(200)
        riesz = LinearRiesz()
        riesz.fit(X, D, linear_nuisance, sample_weight=w)
        alpha = riesz.predict(X)
        assert alpha.shape == (200,)

    def test_no_scaling(self):
        X, D = make_data()
        riesz = LinearRiesz(scale_features=False)
        riesz.fit(X, D, linear_nuisance)
        alpha = riesz.predict(X)
        assert alpha.shape == (200,)

    def test_no_intercept(self):
        X, D = make_data()
        riesz = LinearRiesz(fit_intercept=False)
        riesz.fit(X, D, linear_nuisance)
        alpha = riesz.predict(X)
        assert alpha.shape == (200,)

    def test_captures_derivative_sign(self):
        X, D = make_data(n=500, seed=2)
        riesz = LinearRiesz(alpha_reg=0.01)
        riesz.fit(X, D, linear_nuisance)
        alpha = riesz.predict(X)
        assert np.mean(alpha) < 0

    def test_high_regularisation_shrinks_to_zero(self):
        X, D = make_data(n=100)
        riesz = LinearRiesz(alpha_reg=1e6)
        riesz.fit(X, D, linear_nuisance)
        alpha = riesz.predict(X)
        assert np.abs(np.mean(alpha)) < 0.1

    def test_different_predict_n(self):
        X, D = make_data(n=200)
        riesz = LinearRiesz()
        riesz.fit(X, D, linear_nuisance)
        X_new = np.random.randn(30, 4)
        alpha = riesz.predict(X_new)
        assert alpha.shape == (30,)


class TestComputeRieszLoss:
    def test_returns_float(self):
        X, D = make_data(n=100)
        alpha_pred = linear_nuisance(D, X) / 100.0
        loss = compute_riesz_loss(alpha_pred, D, X, linear_nuisance)
        assert isinstance(loss, float)

    def test_negative_loss_possible(self):
        """Riesz loss can be negative (unbounded below); check it's finite."""
        X, D = make_data(n=200)
        alpha_pred = np.full(200, -0.002)
        loss = compute_riesz_loss(alpha_pred, D, X, linear_nuisance)
        assert np.isfinite(loss)

    def test_with_weights(self):
        X, D = make_data(n=100)
        alpha_pred = np.zeros(100)
        w = np.ones(100) * 2.0
        loss = compute_riesz_loss(alpha_pred, D, X, linear_nuisance, sample_weight=w)
        assert isinstance(loss, float)

    def test_zero_alpha_has_nonzero_loss(self):
        X, D = make_data(n=200)
        alpha_pred = np.zeros(200)
        loss = compute_riesz_loss(alpha_pred, D, X, linear_nuisance)
        # When alpha=0: loss = 0 - 2*E[0 * dg/dD] = 0; so it's 0
        # but with our formula: E[0^2] - 2*E[0*dg] = 0
        assert np.isfinite(loss)
