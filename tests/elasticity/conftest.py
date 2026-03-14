"""
Shared pytest fixtures for insurance-elasticity tests.

Uses small synthetic datasets and low-iteration models to keep test runtime
on Databricks within 10–15 minutes total.
"""

import numpy as np
import polars as pl
import pytest

econml = pytest.importorskip("econml", reason="econml not installed — skipping elasticity tests")

from insurance_causal.elasticity.data import make_renewal_data


CONFOUNDERS = ["age", "ncd_years", "vehicle_group", "channel"]


@pytest.fixture(scope="session")
def small_df():
    """2000-row synthetic renewal dataset (fast; used across all tests)."""
    return make_renewal_data(n=2000, seed=42)


@pytest.fixture(scope="session")
def fitted_estimator(small_df):
    """Fitted RenewalElasticityEstimator on the small dataset.

    n_estimators must be divisible by n_folds * 2 in CausalForestDML (econml
    uses honest splitting which splits the subsample in two). With n_folds=2,
    n_estimators must be divisible by 4. We use 40.
    """
    from insurance_causal.elasticity.fit import RenewalElasticityEstimator
    est = RenewalElasticityEstimator(
        cate_model="causal_forest",
        n_estimators=40,
        catboost_iterations=80,
        n_folds=2,
        random_state=42,
    )
    est.fit(small_df, confounders=CONFOUNDERS)
    return est
