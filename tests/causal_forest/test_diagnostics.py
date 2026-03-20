"""
Tests for CausalForestDiagnostics.

Key tests:
- Diagnostics run on well-specified data (smoke)
- Degenerate propensity is flagged (all treatment values identical)
- Residual variation below threshold is flagged
- DiagnosticsReport.summary() runs without error
"""

import numpy as np
import polars as pl
import pytest

pytest.importorskip("econml", reason="econml not installed — skipping causal_forest tests")

from insurance_causal.causal_forest.diagnostics import (
    CausalForestDiagnostics,
    DiagnosticsReport,
)
from insurance_causal.causal_forest.data import make_hte_renewal_data
from insurance_causal.causal_forest.estimator import HeterogeneousElasticityEstimator

CONFOUNDERS = ["age", "ncd_years", "vehicle_group", "channel"]


@pytest.fixture(scope="module")
def diagnostics_report(hte_df, fitted_hte_estimator, cates):
    diag = CausalForestDiagnostics(n_splits=5, random_state=42)
    return diag.check(hte_df, estimator=fitted_hte_estimator, cates=cates)


class TestDiagnosticsReport:
    def test_returns_diagnostics_report(self, diagnostics_report):
        assert isinstance(diagnostics_report, DiagnosticsReport)

    def test_n_obs_correct(self, diagnostics_report, hte_df):
        assert diagnostics_report.n_obs == len(hte_df)

    def test_residual_variation_fraction_positive(self, diagnostics_report):
        assert diagnostics_report.residual_variation_fraction >= 0.0

    def test_propensity_std_positive(self, diagnostics_report):
        assert diagnostics_report.propensity_std >= 0.0

    def test_overlap_ok_on_good_data(self, diagnostics_report):
        """Data has genuine price variation — overlap should be OK."""
        assert diagnostics_report.overlap_ok

    def test_residual_variation_ok_on_good_data(self, diagnostics_report):
        """DGP uses price_sd=0.10 — residual variation should exceed 10% threshold."""
        assert diagnostics_report.residual_variation_ok, (
            f"Residual variation = {diagnostics_report.residual_variation_fraction:.4f} "
            "expected to be >= 0.10 with price_sd=0.10 DGP"
        )

    def test_summary_runs(self, diagnostics_report):
        s = diagnostics_report.summary()
        assert isinstance(s, str)
        assert "Residual variation" in s

    def test_all_ok_type(self, diagnostics_report):
        result = diagnostics_report.all_ok()
        assert isinstance(result, bool)


class TestDegeneratePropensity:
    def test_flags_degenerate_propensity(self):
        """A dataset with near-zero treatment variation should flag degenerate propensity."""
        diag = CausalForestDiagnostics(n_splits=3, random_state=42)

        # Create dataset where treatment = constant (degenerate)
        df_degen = make_hte_renewal_data(n=500, seed=0, price_sd=0.001)

        # Fit estimator
        est = HeterogeneousElasticityEstimator(
            n_folds=2, n_estimators=20, min_samples_leaf=5,
            catboost_iterations=20, random_state=0,
        )
        est.fit(df_degen, confounders=CONFOUNDERS)
        cates_degen = est.cate(df_degen)

        report = diag.check(df_degen, estimator=est, cates=cates_degen)

        # With price_sd=0.001, treatment is nearly deterministic.
        # Either overlap_ok=False or residual_variation_ok=False.
        at_least_one_flag = (
            not report.overlap_ok
            or not report.residual_variation_ok
        )
        assert at_least_one_flag, (
            "Diagnostics did not flag near-deterministic treatment. "
            f"overlap_ok={report.overlap_ok}, "
            f"residual_variation_ok={report.residual_variation_ok}, "
            f"residual_variation_fraction={report.residual_variation_fraction:.4f}"
        )

    def test_degenerate_propensity_method(self):
        """degenerate_propensity_test() returns True for constant treatment."""
        diag = CausalForestDiagnostics()
        n = 200
        D = np.full(n, 0.05)  # constant treatment
        X = np.random.default_rng(0).normal(size=(n, 3))
        result = diag.degenerate_propensity_test(D, X)
        assert result is True

    def test_non_degenerate_propensity_method(self):
        """degenerate_propensity_test() returns False for varied treatment."""
        diag = CausalForestDiagnostics()
        rng = np.random.default_rng(0)
        n = 200
        X = rng.normal(size=(n, 3))
        D = 0.05 + 0.10 * X[:, 0] + rng.normal(0, 0.05, n)  # correlated with X
        result = diag.degenerate_propensity_test(D, X)
        assert result is False


class TestWarnings:
    def test_small_dataset_warning_in_report(self):
        """Small dataset (n<5000) should produce a warning in the report."""
        df_small = make_hte_renewal_data(n=500, seed=1)
        est = HeterogeneousElasticityEstimator(
            n_folds=2, n_estimators=20, min_samples_leaf=5,
            catboost_iterations=20, random_state=1,
        )
        est.fit(df_small, confounders=CONFOUNDERS)
        cates = est.cate(df_small)
        diag = CausalForestDiagnostics(n_splits=3, random_state=1)
        report = diag.check(df_small, estimator=est, cates=cates)
        small_warns = [w for w in report.warnings if "observations" in w.lower()]
        assert len(small_warns) > 0, "Expected small dataset warning"
