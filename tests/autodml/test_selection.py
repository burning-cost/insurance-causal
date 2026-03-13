"""Tests for SelectionCorrectedElasticity."""
import numpy as np
import pandas as pd
import pytest
from insurance_causal.autodml.selection import SelectionCorrectedElasticity
from insurance_causal.autodml.dgp import SyntheticContinuousDGP
from insurance_causal.autodml._types import EstimationResult


def make_selection_data(n=500, seed=0):
    dgp = SyntheticContinuousDGP(
        n=n, n_features=4, outcome_family="gaussian",
        selection_strength=1.5, random_state=seed
    )
    X, D, Y, S = dgp.generate(include_selection=True)
    # Replace NaN Y with 0 for non-renewers (the estimator should ignore them)
    Y_obs = np.where(np.isnan(Y), 0.0, Y)
    return X, D, Y_obs, S


class TestSelectionCorrectedElasticityBasic:
    def test_fit_returns_self(self):
        X, D, Y, S = make_selection_data()
        model = SelectionCorrectedElasticity(n_folds=2, random_state=0)
        assert model.fit(X, D, Y, S) is model

    def test_is_fitted_after_fit(self):
        X, D, Y, S = make_selection_data()
        model = SelectionCorrectedElasticity(n_folds=2, random_state=0)
        model.fit(X, D, Y, S)
        assert model._is_fitted

    def test_estimate_raises_before_fit(self):
        model = SelectionCorrectedElasticity()
        with pytest.raises(RuntimeError):
            model.estimate()

    def test_estimate_returns_result(self):
        X, D, Y, S = make_selection_data()
        model = SelectionCorrectedElasticity(n_folds=2, random_state=0)
        model.fit(X, D, Y, S)
        result = model.estimate()
        assert isinstance(result, EstimationResult)

    def test_pi_hat_in_range(self):
        X, D, Y, S = make_selection_data()
        model = SelectionCorrectedElasticity(n_folds=2, random_state=0)
        model.fit(X, D, Y, S)
        assert np.all(model.pi_hat_ >= 0.05)
        assert np.all(model.pi_hat_ <= 0.95)

    def test_ci_ordering(self):
        X, D, Y, S = make_selection_data()
        model = SelectionCorrectedElasticity(n_folds=2, random_state=0)
        model.fit(X, D, Y, S)
        result = model.estimate()
        assert result.ci_low < result.ci_high

    def test_se_positive(self):
        X, D, Y, S = make_selection_data()
        model = SelectionCorrectedElasticity(n_folds=2, random_state=0)
        model.fit(X, D, Y, S)
        result = model.estimate()
        assert result.se > 0

    def test_n_obs_correct(self):
        X, D, Y, S = make_selection_data(n=400)
        model = SelectionCorrectedElasticity(n_folds=2, random_state=0)
        model.fit(X, D, Y, S)
        result = model.estimate()
        assert result.n_obs == 400

    def test_notes_contain_selection_rate(self):
        X, D, Y, S = make_selection_data()
        model = SelectionCorrectedElasticity(n_folds=2, random_state=0)
        model.fit(X, D, Y, S)
        result = model.estimate()
        assert "selection_rate" in result.notes

    def test_g_hat_alpha_hat_shapes(self):
        X, D, Y, S = make_selection_data(n=400)
        model = SelectionCorrectedElasticity(n_folds=2, random_state=0)
        model.fit(X, D, Y, S)
        assert model.g_hat_.shape == (400,)
        assert model.alpha_hat_.shape == (400,)


class TestSelectionCorrectedElasticityVariants:
    def test_accepts_dataframe(self):
        X, D, Y, S = make_selection_data(n=300)
        df = pd.DataFrame(X)
        model = SelectionCorrectedElasticity(n_folds=2, random_state=0)
        model.fit(df, D, Y, S)
        assert model._is_fitted

    def test_bootstrap_inference(self):
        X, D, Y, S = make_selection_data(n=300)
        model = SelectionCorrectedElasticity(inference="bootstrap", n_folds=2, random_state=0)
        model.fit(X, D, Y, S)
        result = model.estimate()
        assert isinstance(result, EstimationResult)

    def test_with_sample_weight(self):
        X, D, Y, S = make_selection_data(n=300)
        w = np.ones(300)
        model = SelectionCorrectedElasticity(n_folds=2, random_state=0)
        model.fit(X, D, Y, S, sample_weight=w)
        result = model.estimate()
        assert isinstance(result, EstimationResult)


class TestSensitivityBounds:
    def test_sensitivity_bounds_requires_fit(self):
        model = SelectionCorrectedElasticity()
        with pytest.raises(RuntimeError):
            model.sensitivity_bounds()

    def test_sensitivity_bounds_returns_dict(self):
        X, D, Y, S = make_selection_data(n=400)
        model = SelectionCorrectedElasticity(n_folds=2, random_state=0)
        model.fit(X, D, Y, S)
        bounds = model.sensitivity_bounds()
        assert isinstance(bounds, dict)

    def test_gamma_1_is_point_identified(self):
        X, D, Y, S = make_selection_data(n=400)
        model = SelectionCorrectedElasticity(n_folds=2, random_state=0)
        model.fit(X, D, Y, S)
        bounds = model.sensitivity_bounds(gamma_grid=np.array([1.0]))
        b = bounds[1.0]
        # With Gamma=1, lower and upper should be equal (point identified)
        assert abs(b["lower"] - b["upper"]) < 0.01

    def test_larger_gamma_gives_wider_bounds(self):
        X, D, Y, S = make_selection_data(n=400)
        model = SelectionCorrectedElasticity(n_folds=2, random_state=0)
        model.fit(X, D, Y, S)
        bounds = model.sensitivity_bounds(gamma_grid=np.array([1.0, 2.0]))
        # Width is measured as absolute distance between bounds
        width_1 = abs(bounds[1.0]["upper"] - bounds[1.0]["lower"])
        width_2 = abs(bounds[2.0]["upper"] - bounds[2.0]["lower"])
        assert width_2 >= width_1

    def test_custom_gamma_grid(self):
        X, D, Y, S = make_selection_data(n=400)
        model = SelectionCorrectedElasticity(n_folds=2, random_state=0)
        model.fit(X, D, Y, S)
        bounds = model.sensitivity_bounds(gamma_grid=np.array([1.0, 1.5, 3.0]))
        assert len(bounds) == 3
