"""Tests for _crossfit.py — cross-fitting infrastructure."""
import numpy as np
import pytest
from insurance_causal.autodml._crossfit import cross_fit_nuisance, compute_ame_scores
from insurance_causal.autodml._types import OutcomeFamily
from insurance_causal.autodml.riesz import ForestRiesz, LinearRiesz


def make_data(n=300, p=4, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    D = rng.uniform(200, 600, n)
    Y = -0.002 * D + 0.5 * X[:, 0] + rng.randn(n) * 0.1
    return X, D, Y


class TestCrossFitNuisance:
    def test_output_shapes(self):
        X, D, Y = make_data(n=200)
        g_hat, alpha_hat, folds, models = cross_fit_nuisance(
            X, D, Y, n_folds=3, riesz_class=ForestRiesz,
            riesz_kwargs={"n_estimators": 20, "random_state": 0}
        )
        assert g_hat.shape == (200,)
        assert alpha_hat.shape == (200,)
        assert len(folds) == 3
        assert len(models) == 3

    def test_no_nan_g_hat(self):
        X, D, Y = make_data()
        g_hat, alpha_hat, _, _ = cross_fit_nuisance(
            X, D, Y, n_folds=3, riesz_kwargs={"n_estimators": 20}
        )
        assert not np.any(np.isnan(g_hat))

    def test_fold_indices_cover_all(self):
        X, D, Y = make_data(n=200)
        _, _, folds, _ = cross_fit_nuisance(
            X, D, Y, n_folds=4, riesz_kwargs={"n_estimators": 20}
        )
        all_eval = np.concatenate([ev for _, ev in folds])
        assert len(np.unique(all_eval)) == 200

    def test_with_linear_riesz(self):
        X, D, Y = make_data()
        g_hat, alpha_hat, _, _ = cross_fit_nuisance(
            X, D, Y, n_folds=3, riesz_class=LinearRiesz
        )
        assert not np.any(np.isnan(alpha_hat))

    def test_with_exposure(self):
        rng = np.random.RandomState(0)
        X, D, Y = make_data(n=200)
        exposure = rng.exponential(1.0, size=200)
        Y_count = np.abs(Y) * exposure
        g_hat, alpha_hat, _, _ = cross_fit_nuisance(
            X, D, Y_count,
            outcome_family=OutcomeFamily.POISSON,
            n_folds=3,
            exposure=exposure,
            riesz_kwargs={"n_estimators": 20},
        )
        assert g_hat.shape == (200,)

    def test_with_sample_weight(self):
        X, D, Y = make_data()
        w = np.ones(300) * 2.0
        g_hat, _, _, _ = cross_fit_nuisance(
            X, D, Y, n_folds=3, sample_weight=w,
            riesz_kwargs={"n_estimators": 20},
        )
        assert g_hat.shape == (300,)

    def test_linear_backend(self):
        X, D, Y = make_data()
        g_hat, alpha_hat, _, _ = cross_fit_nuisance(
            X, D, Y, n_folds=3, nuisance_backend="linear"
        )
        assert not np.any(np.isnan(g_hat))

    def test_reproducible_with_seed(self):
        X, D, Y = make_data()
        kwargs = {"n_folds": 3, "riesz_kwargs": {"n_estimators": 20, "random_state": 7}, "random_state": 42}
        g1, a1, _, _ = cross_fit_nuisance(X, D, Y, **kwargs)
        g2, a2, _, _ = cross_fit_nuisance(X, D, Y, **kwargs)
        np.testing.assert_array_almost_equal(g1, g2)


class TestComputeAmeScores:
    def test_returns_float_and_array(self):
        n = 100
        Y = np.random.randn(n)
        g_hat = np.random.randn(n) * 0.1
        alpha_hat = np.full(n, -0.002)
        ame, psi = compute_ame_scores(Y, g_hat, alpha_hat)
        assert isinstance(ame, float)
        assert psi.shape == (n,)

    def test_ame_is_mean_of_psi(self):
        n = 1000
        Y = np.zeros(n)
        g_hat = np.zeros(n)
        alpha_hat = np.full(n, 0.1)
        ame, psi = compute_ame_scores(Y, g_hat, alpha_hat)
        assert abs(ame - np.mean(psi)) < 1e-10

    def test_with_sample_weight(self):
        n = 100
        Y = np.random.randn(n)
        g_hat = np.random.randn(n)
        alpha_hat = np.random.randn(n) * 0.01
        w = np.ones(n) * 3.0
        ame, psi = compute_ame_scores(Y, g_hat, alpha_hat, sample_weight=w)
        assert isinstance(ame, float)
        assert np.isfinite(ame)

    def test_psi_length_matches_n(self):
        n = 500
        ame, psi = compute_ame_scores(
            np.ones(n), np.zeros(n), np.full(n, -0.001)
        )
        assert len(psi) == n
