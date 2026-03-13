"""Tests for PremiumElasticity."""
import numpy as np
import pandas as pd
import pytest
from insurance_causal.autodml.elasticity import PremiumElasticity
from insurance_causal.autodml.dgp import SyntheticContinuousDGP
from insurance_causal.autodml._types import EstimationResult, OutcomeFamily


def make_small_data(n=400, seed=0):
    dgp = SyntheticContinuousDGP(n=n, n_features=4, outcome_family="gaussian", random_state=seed)
    X, D, Y, _ = dgp.generate()
    return X, D, Y, dgp.true_ame_


class TestPremiumElasticityBasic:
    def test_fit_returns_self(self):
        X, D, Y, _ = make_small_data()
        model = PremiumElasticity(n_folds=2, random_state=0)
        result = model.fit(X, D, Y)
        assert result is model

    def test_is_fitted_after_fit(self):
        X, D, Y, _ = make_small_data()
        model = PremiumElasticity(n_folds=2, random_state=0)
        model.fit(X, D, Y)
        assert model._is_fitted

    def test_g_hat_alpha_hat_shapes(self):
        X, D, Y, _ = make_small_data(n=300)
        model = PremiumElasticity(n_folds=2, random_state=0)
        model.fit(X, D, Y)
        assert model.g_hat_.shape == (300,)
        assert model.alpha_hat_.shape == (300,)

    def test_no_nan_g_hat(self):
        X, D, Y, _ = make_small_data()
        model = PremiumElasticity(n_folds=2, random_state=0)
        model.fit(X, D, Y)
        assert not np.any(np.isnan(model.g_hat_))

    def test_estimate_raises_before_fit(self):
        model = PremiumElasticity()
        with pytest.raises(RuntimeError, match="fit()"):
            model.estimate()

    def test_estimate_returns_result(self):
        X, D, Y, _ = make_small_data()
        model = PremiumElasticity(n_folds=2, random_state=0)
        model.fit(X, D, Y)
        result = model.estimate()
        assert isinstance(result, EstimationResult)

    def test_result_stored(self):
        X, D, Y, _ = make_small_data()
        model = PremiumElasticity(n_folds=2, random_state=0)
        model.fit(X, D, Y)
        model.estimate()
        assert model.result_ is not None

    def test_ci_ordering(self):
        X, D, Y, _ = make_small_data()
        model = PremiumElasticity(n_folds=2, random_state=0)
        model.fit(X, D, Y)
        result = model.estimate()
        assert result.ci_low < result.estimate < result.ci_high

    def test_n_obs_correct(self):
        X, D, Y, _ = make_small_data(n=300)
        model = PremiumElasticity(n_folds=2, random_state=0)
        model.fit(X, D, Y)
        result = model.estimate()
        assert result.n_obs == 300

    def test_se_positive(self):
        X, D, Y, _ = make_small_data()
        model = PremiumElasticity(n_folds=2, random_state=0)
        model.fit(X, D, Y)
        result = model.estimate()
        assert result.se > 0

    def test_psi_shape(self):
        X, D, Y, _ = make_small_data(n=400)
        model = PremiumElasticity(n_folds=2, random_state=0)
        model.fit(X, D, Y)
        result = model.estimate()
        assert result.psi.shape == (400,)


class TestPremiumElasticityInputVariants:
    def test_accepts_dataframe(self):
        X, D, Y, _ = make_small_data(n=300)
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        model = PremiumElasticity(n_folds=2, random_state=0)
        model.fit(df, D, Y)
        assert model._is_fitted

    def test_accepts_pandas_series(self):
        X, D, Y, _ = make_small_data(n=300)
        model = PremiumElasticity(n_folds=2, random_state=0)
        model.fit(X, pd.Series(D), pd.Series(Y))
        assert model._is_fitted

    def test_with_sample_weight(self):
        X, D, Y, _ = make_small_data(n=300)
        w = np.ones(300) * 2.0
        model = PremiumElasticity(n_folds=2, random_state=0)
        model.fit(X, D, Y, sample_weight=w)
        result = model.estimate()
        assert isinstance(result, EstimationResult)

    def test_with_exposure(self):
        dgp = SyntheticContinuousDGP(n=300, outcome_family="poisson", random_state=0)
        X, D, Y, _, exposure = dgp.generate()
        Y_obs = np.where(np.isnan(Y), 0.0, Y)
        model = PremiumElasticity(outcome_family="poisson", n_folds=2, random_state=0)
        model.fit(X, D, Y_obs, exposure=exposure)
        result = model.estimate()
        assert isinstance(result, EstimationResult)

    def test_linear_riesz_type(self):
        X, D, Y, _ = make_small_data(n=300)
        model = PremiumElasticity(riesz_type="linear", n_folds=2, random_state=0)
        model.fit(X, D, Y)
        result = model.estimate()
        assert isinstance(result, EstimationResult)

    def test_string_outcome_family(self):
        X, D, Y, _ = make_small_data(n=300)
        model = PremiumElasticity(outcome_family="gaussian", n_folds=2, random_state=0)
        model.fit(X, D, Y)
        assert model.outcome_family == OutcomeFamily.GAUSSIAN

    def test_bootstrap_inference(self):
        X, D, Y, _ = make_small_data(n=300)
        model = PremiumElasticity(inference="bootstrap", n_folds=2, random_state=0)
        model.fit(X, D, Y)
        result = model.estimate()
        assert isinstance(result, EstimationResult)
        assert result.se > 0


class TestPremiumElasticitySegments:
    def test_effect_by_segment_requires_fit(self):
        model = PremiumElasticity()
        with pytest.raises(RuntimeError):
            model.effect_by_segment(np.array(["a", "b"]))

    def test_effect_by_segment_returns_list(self):
        X, D, Y, _ = make_small_data(n=400)
        model = PremiumElasticity(n_folds=2, random_state=0)
        model.fit(X, D, Y)
        segments = np.where(X[:, 0] > np.median(X[:, 0]), "high_age", "low_age")
        results = model.effect_by_segment(segments)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_segment_names_correct(self):
        X, D, Y, _ = make_small_data(n=400)
        model = PremiumElasticity(n_folds=2, random_state=0)
        model.fit(X, D, Y)
        segments = np.where(X[:, 0] > np.median(X[:, 0]), "high", "low")
        results = model.effect_by_segment(segments)
        names = {sr.segment_name for sr in results}
        assert names == {"high", "low"}

    def test_segment_n_obs_sums(self):
        X, D, Y, _ = make_small_data(n=400)
        model = PremiumElasticity(n_folds=2, random_state=0)
        model.fit(X, D, Y)
        segments = np.where(X[:, 0] > 0, "A", "B")
        results = model.effect_by_segment(segments)
        total = sum(sr.n_obs for sr in results)
        assert total == 400

    def test_effect_by_segment_dataframe_input(self):
        X, D, Y, _ = make_small_data(n=400)
        model = PremiumElasticity(n_folds=2, random_state=0)
        model.fit(X, D, Y)
        df_seg = pd.DataFrame({
            "age_band": np.where(X[:, 0] > 0, "old", "young"),
        })
        results = model.effect_by_segment(df_seg)
        assert len(results) >= 1


class TestPremiumElasticityRieszLoss:
    def test_riesz_loss_requires_fit(self):
        model = PremiumElasticity()
        with pytest.raises(RuntimeError):
            model.riesz_loss()

    def test_riesz_loss_returns_float(self):
        X, D, Y, _ = make_small_data(n=300)
        model = PremiumElasticity(n_folds=2, random_state=0)
        model.fit(X, D, Y)
        loss = model.riesz_loss()
        assert isinstance(loss, float)
        assert np.isfinite(loss)


class TestPremiumElasticityEstimateDirection:
    def test_negative_beta_gives_negative_ame(self):
        """With beta_D=-0.002, AME should be negative."""
        dgp = SyntheticContinuousDGP(
            n=800, n_features=4, beta_D=-0.002, confounding_strength=0.3,
            outcome_family="gaussian", random_state=42
        )
        X, D, Y, true_ame = dgp.generate()
        model = PremiumElasticity(n_folds=3, random_state=0)
        model.fit(X, D, Y)
        result = model.estimate()
        # The estimate should be negative (same sign as true AME)
        assert result.estimate < 0
