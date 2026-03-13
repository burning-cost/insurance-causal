"""Tests for SyntheticContinuousDGP."""
import numpy as np
import pandas as pd
import pytest
from insurance_causal.autodml.dgp import SyntheticContinuousDGP


class TestSyntheticContinuousDGP:
    def test_gaussian_generate_shapes(self):
        dgp = SyntheticContinuousDGP(n=500, n_features=6, outcome_family="gaussian", random_state=0)
        X, D, Y, S = dgp.generate()
        assert X.shape == (500, 6)
        assert D.shape == (500,)
        assert Y.shape == (500,)
        assert S is None

    def test_poisson_generate_shapes(self):
        dgp = SyntheticContinuousDGP(n=300, outcome_family="poisson", random_state=1)
        X, D, Y, S, exposure = dgp.generate()
        assert X.shape == (300, 8)
        assert exposure.shape == (300,)
        assert np.all(exposure > 0)

    def test_gamma_generate_shapes(self):
        dgp = SyntheticContinuousDGP(n=200, outcome_family="gamma", random_state=2)
        X, D, Y, S = dgp.generate()
        assert Y.shape == (200,)
        assert np.all(Y > 0)

    def test_premium_range(self):
        dgp = SyntheticContinuousDGP(n=1000, random_state=0)
        X, D, Y, S = dgp.generate()
        assert D.min() >= 50.0
        assert D.max() <= 2000.0

    def test_true_ame_set_after_generate(self):
        dgp = SyntheticContinuousDGP(n=500, beta_D=-0.003, outcome_family="gaussian", random_state=0)
        dgp.generate()
        assert dgp.true_ame_ == pytest.approx(-0.003, rel=1e-6)

    def test_true_ame_poisson(self):
        dgp = SyntheticContinuousDGP(n=1000, beta_D=-0.002, outcome_family="poisson", random_state=0)
        dgp.generate()
        # AME = beta_D * E[exp(log_risk + beta_D * D)] — should be negative
        assert dgp.true_ame_ < 0

    def test_true_dose_response_callable(self):
        dgp = SyntheticContinuousDGP(n=500, outcome_family="gaussian", random_state=0)
        dgp.generate()
        assert callable(dgp.true_dose_response_)
        val = dgp.true_dose_response_(350.0)
        assert np.isfinite(val)

    def test_selection_included(self):
        dgp = SyntheticContinuousDGP(n=500, selection_strength=2.0, random_state=0)
        X, D, Y, S = dgp.generate(include_selection=True)
        assert S is not None
        assert S.shape == (500,)
        assert set(S).issubset({0.0, 1.0})
        # Y should be NaN for non-renewers
        assert np.any(np.isnan(Y))

    def test_selection_rate_plausible(self):
        dgp = SyntheticContinuousDGP(n=2000, selection_strength=1.5, random_state=0)
        X, D, Y, S = dgp.generate(include_selection=True)
        # Renewal rate should be between 30% and 90%
        assert 0.3 < S.mean() < 0.9

    def test_no_selection_all_observed(self):
        dgp = SyntheticContinuousDGP(n=500, selection_strength=0.0, random_state=0)
        X, D, Y, S = dgp.generate(include_selection=True)
        assert not np.any(np.isnan(Y))

    def test_confounding_affects_treatment_correlation(self):
        dgp_strong = SyntheticContinuousDGP(n=2000, confounding_strength=1.0, random_state=0)
        dgp_weak = SyntheticContinuousDGP(n=2000, confounding_strength=0.0, random_state=0)
        X_s, D_s, Y_s, _ = dgp_strong.generate()
        X_w, D_w, Y_w, _ = dgp_weak.generate()
        # With strong confounding, D and Y should be more correlated
        corr_strong = abs(np.corrcoef(D_s, Y_s)[0, 1])
        corr_weak = abs(np.corrcoef(D_w, Y_w)[0, 1])
        assert corr_strong > corr_weak

    def test_as_dataframe_columns(self):
        dgp = SyntheticContinuousDGP(n=100, n_features=6, random_state=0)
        df = dgp.as_dataframe()
        assert "premium" in df.columns
        assert "outcome" in df.columns
        assert "age_norm" in df.columns
        assert "ncb_norm" in df.columns
        assert df.shape[0] == 100

    def test_as_dataframe_with_selection(self):
        dgp = SyntheticContinuousDGP(n=100, random_state=0)
        df = dgp.as_dataframe(include_selection=True)
        assert "renewed" in df.columns
        assert set(df["renewed"].dropna()).issubset({0, 1})

    def test_as_dataframe_poisson_has_exposure(self):
        dgp = SyntheticContinuousDGP(n=100, outcome_family="poisson", random_state=0)
        df = dgp.as_dataframe()
        assert "exposure" in df.columns

    def test_invalid_outcome_family_raises(self):
        dgp = SyntheticContinuousDGP(n=100, outcome_family="invalid")
        with pytest.raises(ValueError):
            dgp.generate()

    def test_reproducibility(self):
        dgp1 = SyntheticContinuousDGP(n=200, random_state=42)
        dgp2 = SyntheticContinuousDGP(n=200, random_state=42)
        _, D1, Y1, _ = dgp1.generate()
        _, D2, Y2, _ = dgp2.generate()
        np.testing.assert_array_equal(D1, D2)
        np.testing.assert_array_equal(Y1, Y2)

    def test_different_seeds_differ(self):
        dgp1 = SyntheticContinuousDGP(n=200, random_state=0)
        dgp2 = SyntheticContinuousDGP(n=200, random_state=1)
        _, D1, _, _ = dgp1.generate()
        _, D2, _, _ = dgp2.generate()
        assert not np.allclose(D1, D2)

    def test_n_features_minimum(self):
        dgp = SyntheticContinuousDGP(n=100, n_features=4, random_state=0)
        X, D, Y, S = dgp.generate()
        assert X.shape == (100, 4)
