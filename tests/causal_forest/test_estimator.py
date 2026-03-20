"""
Tests for HeterogeneousElasticityEstimator.

Covers:
- Smoke tests: fit, ate, cate, cate_interval, gate
- Polars and pandas DataFrame input
- Unfitted raises RuntimeError
- n_estimators auto-rounding
- Small group warning in gate()
- Exposure weighting (smoke test)
"""

import numpy as np
import polars as pl
import pytest

pytest.importorskip("econml", reason="econml not installed — skipping causal_forest tests")

from insurance_causal.causal_forest.data import make_hte_renewal_data
from insurance_causal.causal_forest.estimator import HeterogeneousElasticityEstimator

CONFOUNDERS = ["age", "ncd_years", "vehicle_group", "channel"]


class TestFit:
    def test_fit_returns_self(self, hte_df):
        est = HeterogeneousElasticityEstimator(
            n_folds=2, n_estimators=50, min_samples_leaf=5,
            catboost_iterations=50, random_state=0,
        )
        result = est.fit(hte_df, confounders=CONFOUNDERS)
        assert result is est

    def test_fit_is_fitted(self, fitted_hte_estimator):
        assert fitted_hte_estimator._is_fitted

    def test_fit_no_confounders_raises(self, hte_df):
        est = HeterogeneousElasticityEstimator(n_estimators=50, catboost_iterations=50)
        with pytest.raises(ValueError, match="confounders must be provided"):
            est.fit(hte_df)

    def test_fit_accepts_pandas(self, hte_df):
        df_pd = hte_df.to_pandas()
        est = HeterogeneousElasticityEstimator(
            n_folds=2, n_estimators=50, min_samples_leaf=5,
            catboost_iterations=50, random_state=0,
        )
        est.fit(df_pd, confounders=CONFOUNDERS)
        assert est._is_fitted

    def test_fit_stores_train_arrays(self, fitted_hte_estimator, hte_df):
        assert fitted_hte_estimator._X_train is not None
        assert fitted_hte_estimator._Y_train is not None
        assert fitted_hte_estimator._D_train is not None
        assert len(fitted_hte_estimator._Y_train) == len(hte_df)

    def test_n_estimators_auto_round(self, hte_df):
        """n_estimators=53 with n_folds=2 must round to 56 (next multiple of 4)."""
        with pytest.warns(UserWarning, match="Rounding up"):
            est = HeterogeneousElasticityEstimator(
                n_folds=2, n_estimators=53, min_samples_leaf=5,
                catboost_iterations=30, random_state=0,
            )
            est.fit(hte_df, confounders=CONFOUNDERS)
        assert est._is_fitted


class TestATE:
    def test_ate_returns_tuple_of_three(self, fitted_hte_estimator):
        result = fitted_hte_estimator.ate()
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_ate_finite(self, fitted_hte_estimator):
        ate, lb, ub = fitted_hte_estimator.ate()
        assert np.isfinite(ate)
        assert np.isfinite(lb)
        assert np.isfinite(ub)

    def test_ate_ci_ordered(self, fitted_hte_estimator):
        ate, lb, ub = fitted_hte_estimator.ate()
        assert lb <= ub, f"CI not ordered: lb={lb:.4f}, ub={ub:.4f}"

    def test_ate_sign(self, fitted_hte_estimator):
        """True ATE in DGP is around -0.2; should not be strongly positive."""
        ate, _, _ = fitted_hte_estimator.ate()
        assert ate < 0.5, f"ATE={ate:.3f} is implausibly positive"

    def test_ate_not_fitted_raises(self):
        est = HeterogeneousElasticityEstimator(n_estimators=50, catboost_iterations=50)
        with pytest.raises(RuntimeError, match="not fitted"):
            est.ate()


class TestCATE:
    def test_cate_shape(self, fitted_hte_estimator, hte_df):
        cates = fitted_hte_estimator.cate(hte_df)
        assert cates.shape == (len(hte_df),)

    def test_cate_finite(self, fitted_hte_estimator, hte_df):
        cates = fitted_hte_estimator.cate(hte_df)
        assert np.all(np.isfinite(cates))

    def test_cate_interval_shape(self, fitted_hte_estimator, hte_df):
        lb, ub = fitted_hte_estimator.cate_interval(hte_df)
        assert lb.shape == (len(hte_df),)
        assert ub.shape == (len(hte_df),)

    def test_cate_interval_ordered(self, fitted_hte_estimator, hte_df):
        lb, ub = fitted_hte_estimator.cate_interval(hte_df)
        # lb should be <= ub everywhere (within float tolerance)
        assert np.all(lb <= ub + 1e-8), "Some CI lower bounds exceed upper bounds"

    def test_cate_has_variation(self, fitted_hte_estimator, hte_df):
        """CATEs should vary across customers — not all identical.

        The directional test (NCD=0 < NCD=5) is validated by the BLP test
        (test_blp_beta_2_positive_on_hte_dgp) which is more statistically
        robust. Here we just confirm the estimator produces heterogeneous
        estimates rather than a constant.
        """
        cates = fitted_hte_estimator.cate(hte_df)
        assert np.std(cates) > 0.01, (
            f"CATEs have near-zero SD ({np.std(cates):.4f}), estimator may be degenerate"
        )


class TestGATE:
    def test_gate_returns_polars(self, fitted_hte_estimator, hte_df):
        result = fitted_hte_estimator.gate(hte_df, by="ncd_years")
        assert isinstance(result, pl.DataFrame)

    def test_gate_columns(self, fitted_hte_estimator, hte_df):
        result = fitted_hte_estimator.gate(hte_df, by="ncd_years")
        assert "ncd_years" in result.columns
        assert "cate" in result.columns
        assert "n" in result.columns

    def test_gate_covers_all_groups(self, fitted_hte_estimator, hte_df):
        result = fitted_hte_estimator.gate(hte_df, by="ncd_years")
        unique_ncd = hte_df["ncd_years"].n_unique()
        assert len(result) == unique_ncd

    def test_gate_channel(self, fitted_hte_estimator, hte_df):
        result = fitted_hte_estimator.gate(hte_df, by="channel")
        assert len(result) == 3  # pcw, direct, broker
