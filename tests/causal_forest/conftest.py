"""
Shared fixtures for causal_forest tests.

Uses n=2000 and very small n_estimators/catboost_iterations to keep
test runtime fast on Databricks. These are smoke tests — correctness
properties are verified with known-heterogeneity DGP, not with
production-scale parameters.
"""

import numpy as np
import polars as pl
import pytest

econml = pytest.importorskip("econml", reason="econml not installed — skipping causal_forest tests")

from insurance_causal.causal_forest.data import make_hte_renewal_data
from insurance_causal.causal_forest.estimator import HeterogeneousElasticityEstimator


CONFOUNDERS = ["age", "ncd_years", "vehicle_group", "channel"]
N_SMALL = 2000
N_ESTIMATORS = 50       # divisible by n_folds*2=4 with n_folds=2
CATBOOST_ITER = 50


@pytest.fixture(scope="session")
def hte_df():
    """2000-row HTE dataset with known elasticity heterogeneity by NCD."""
    return make_hte_renewal_data(n=N_SMALL, seed=42)


@pytest.fixture(scope="session")
def fitted_hte_estimator(hte_df):
    """Fitted HeterogeneousElasticityEstimator on the small HTE dataset."""
    est = HeterogeneousElasticityEstimator(
        binary_outcome=True,
        n_folds=2,
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=10,
        catboost_iterations=CATBOOST_ITER,
        random_state=42,
    )
    est.fit(hte_df, outcome="renewed", treatment="log_price_change",
            confounders=CONFOUNDERS)
    return est


@pytest.fixture(scope="session")
def cates(fitted_hte_estimator, hte_df):
    """Per-row CATE estimates from the fitted estimator."""
    return fitted_hte_estimator.cate(hte_df)
