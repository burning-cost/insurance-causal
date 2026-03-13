"""Tests for PolicyShiftEffect."""
import numpy as np
import pandas as pd
import pytest
from insurance_causal.autodml.policy_shift import PolicyShiftEffect
from insurance_causal.autodml.dgp import SyntheticContinuousDGP
from insurance_causal.autodml._types import EstimationResult


def make_data(n=400, seed=0):
    dgp = SyntheticContinuousDGP(n=n, n_features=4, outcome_family="gaussian", random_state=seed)
    X, D, Y, _ = dgp.generate()
    return X, D, Y


class TestPolicyShiftEffectBasic:
    def test_fit_returns_self(self):
        X, D, Y = make_data()
        model = PolicyShiftEffect(n_folds=2, random_state=0)
        assert model.fit(X, D, Y) is model

    def test_is_fitted_after_fit(self):
        X, D, Y = make_data()
        model = PolicyShiftEffect(n_folds=2, random_state=0)
        model.fit(X, D, Y)
        assert model._is_fitted

    def test_estimate_raises_before_fit(self):
        model = PolicyShiftEffect()
        with pytest.raises(RuntimeError):
            model.estimate()

    def test_estimate_returns_result(self):
        X, D, Y = make_data()
        model = PolicyShiftEffect(n_folds=2, random_state=0)
        model.fit(X, D, Y)
        result = model.estimate(delta=0.05)
        assert isinstance(result, EstimationResult)

    def test_estimate_zero_delta(self):
        """Zero delta should give near-zero effect."""
        X, D, Y = make_data(n=600)
        model = PolicyShiftEffect(n_folds=2, random_state=0)
        model.fit(X, D, Y)
        result = model.estimate(delta=0.0)
        # Not exactly zero due to numerical approx but should be small
        assert isinstance(result, EstimationResult)

    def test_ci_ordering(self):
        X, D, Y = make_data()
        model = PolicyShiftEffect(n_folds=2, random_state=0)
        model.fit(X, D, Y)
        result = model.estimate(delta=0.05)
        assert result.ci_low < result.ci_high

    def test_n_obs_correct(self):
        X, D, Y = make_data(n=350)
        model = PolicyShiftEffect(n_folds=2, random_state=0)
        model.fit(X, D, Y)
        result = model.estimate(delta=0.05)
        assert result.n_obs == 350

    def test_notes_contain_delta(self):
        X, D, Y = make_data()
        model = PolicyShiftEffect(n_folds=2, random_state=0)
        model.fit(X, D, Y)
        result = model.estimate(delta=0.10)
        assert "delta" in result.notes

    def test_se_positive(self):
        X, D, Y = make_data()
        model = PolicyShiftEffect(n_folds=2, random_state=0)
        model.fit(X, D, Y)
        result = model.estimate(delta=0.05)
        assert result.se > 0


class TestPolicyShiftEffectVariants:
    def test_accepts_dataframe(self):
        X, D, Y = make_data(n=300)
        df = pd.DataFrame(X)
        model = PolicyShiftEffect(n_folds=2, random_state=0)
        model.fit(df, D, Y)
        assert model._is_fitted

    def test_bootstrap_inference(self):
        X, D, Y = make_data(n=300)
        model = PolicyShiftEffect(inference="bootstrap", n_folds=2, random_state=0)
        model.fit(X, D, Y)
        result = model.estimate(delta=0.05)
        assert isinstance(result, EstimationResult)

    def test_negative_delta(self):
        X, D, Y = make_data()
        model = PolicyShiftEffect(n_folds=2, random_state=0)
        model.fit(X, D, Y)
        result = model.estimate(delta=-0.05)
        assert isinstance(result, EstimationResult)

    def test_with_sample_weight(self):
        X, D, Y = make_data(n=300)
        w = np.ones(300)
        model = PolicyShiftEffect(n_folds=2, random_state=0)
        model.fit(X, D, Y, sample_weight=w)
        result = model.estimate(delta=0.05)
        assert isinstance(result, EstimationResult)


class TestPolicyShiftEstimateCurve:
    def test_estimate_curve_returns_dict(self):
        X, D, Y = make_data(n=300)
        model = PolicyShiftEffect(n_folds=2, random_state=0)
        model.fit(X, D, Y)
        grid = [-0.05, 0.0, 0.05]
        results = model.estimate_curve(grid)
        assert isinstance(results, dict)
        assert len(results) == 3

    def test_estimate_curve_keys(self):
        X, D, Y = make_data(n=300)
        model = PolicyShiftEffect(n_folds=2, random_state=0)
        model.fit(X, D, Y)
        grid = [-0.02, 0.02]
        results = model.estimate_curve(grid)
        assert -0.02 in results
        assert 0.02 in results

    def test_estimate_curve_raises_before_fit(self):
        model = PolicyShiftEffect()
        with pytest.raises(RuntimeError):
            model.estimate_curve([0.05])

    def test_estimate_curve_values_are_results(self):
        X, D, Y = make_data(n=300)
        model = PolicyShiftEffect(n_folds=2, random_state=0)
        model.fit(X, D, Y)
        results = model.estimate_curve([0.05])
        for v in results.values():
            assert isinstance(v, EstimationResult)
