"""
Tests for exposure weighting helpers.
"""

import numpy as np
import pytest

from insurance_causal.causal_forest.exposure import (
    build_exposure_weighted_nuisances,
    prepare_rate_outcome,
)


class TestPrepareRateOutcome:
    def test_basic_division(self):
        Y = np.array([2.0, 4.0, 6.0])
        exposure = np.array([2.0, 2.0, 2.0])
        Y_rate, exp_out = prepare_rate_outcome(Y, exposure)
        np.testing.assert_allclose(Y_rate, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(exp_out, [2.0, 2.0, 2.0])

    def test_zero_exposure_raises(self):
        Y = np.array([1.0, 2.0])
        exposure = np.array([1.0, 0.0])
        with pytest.raises(ValueError, match="strictly positive"):
            prepare_rate_outcome(Y, exposure)

    def test_negative_exposure_raises(self):
        Y = np.array([1.0, 2.0])
        exposure = np.array([1.0, -0.5])
        with pytest.raises(ValueError, match="strictly positive"):
            prepare_rate_outcome(Y, exposure)

    def test_negative_outcome_raises(self):
        Y = np.array([1.0, -1.0])
        exposure = np.array([1.0, 1.0])
        with pytest.raises(ValueError, match="non-negative"):
            prepare_rate_outcome(Y, exposure)

    def test_tiny_exposure_warns(self):
        Y = np.array([1.0, 0.0, 1.0])
        exposure = np.array([1.0, 0.005, 1.0])  # 0.005 < 0.01 threshold
        with pytest.warns(UserWarning, match="exposure < 0.01"):
            prepare_rate_outcome(Y, exposure)


class TestBuildExposureWeightedNuisances:
    def test_returns_two_models(self):
        model_y, model_t = build_exposure_weighted_nuisances(
            binary_outcome=False,
            catboost_iterations=10,
        )
        assert model_y is not None
        assert model_t is not None

    def test_binary_outcome_returns_classifier(self):
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError:
            pytest.skip("CatBoost not installed")
        model_y, model_t = build_exposure_weighted_nuisances(
            binary_outcome=True,
            catboost_iterations=10,
        )
        assert isinstance(model_y, CatBoostClassifier)
        assert isinstance(model_t, CatBoostRegressor)

    def test_continuous_outcome_returns_regressor(self):
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            pytest.skip("CatBoost not installed")
        model_y, model_t = build_exposure_weighted_nuisances(
            binary_outcome=False,
            catboost_iterations=10,
        )
        assert isinstance(model_y, CatBoostRegressor)
        assert isinstance(model_t, CatBoostRegressor)
