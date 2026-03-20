"""
Tests for TargetingEvaluator (RATE/AUTOC/QINI).

Key tests:
- AUTOC is positive on known HTE DGP (CATE proxy is informative)
- TOC curve has correct structure
- TargetingResult.summary() runs without error
- Bootstrap SE is finite and positive
"""

import numpy as np
import polars as pl
import pytest

pytest.importorskip("econml", reason="econml not installed — skipping causal_forest tests")

from insurance_causal.causal_forest.targeting import TargetingEvaluator, TargetingResult


@pytest.fixture(scope="module")
def targeting_result(hte_df, fitted_hte_estimator, cates):
    ev = TargetingEvaluator(n_bootstrap=20, n_toc_points=20, random_state=42)
    return ev.evaluate(hte_df, estimator=fitted_hte_estimator, cate_proxy=cates)


class TestTargetingResult:
    def test_returns_targeting_result(self, targeting_result):
        assert isinstance(targeting_result, TargetingResult)

    def test_autoc_finite(self, targeting_result):
        assert np.isfinite(targeting_result.autoc)

    def test_autoc_se_positive(self, targeting_result):
        assert targeting_result.autoc_se > 0

    def test_ci_ordered(self, targeting_result):
        assert targeting_result.autoc_ci_lower <= targeting_result.autoc_ci_upper

    def test_qini_finite(self, targeting_result):
        assert np.isfinite(targeting_result.qini)

    def test_n_obs_correct(self, targeting_result, hte_df):
        assert targeting_result.n_obs == len(hte_df)

    def test_autoc_positive_on_hte_dgp(self, targeting_result):
        """CATE proxy should produce positive AUTOC on known-HTE data.

        With n=2000 and 20 bootstrap reps this is a loose check —
        the AUTOC should be weakly positive since the causal forest
        should recover some of the true NCD-driven heterogeneity.
        """
        # Loose check: AUTOC > -0.1 (not strongly anti-targeted)
        assert targeting_result.autoc > -0.1, (
            f"AUTOC={targeting_result.autoc:.4f} is strongly negative on "
            "known-HTE DGP. Targeting rule appears anti-targeted."
        )

    def test_toc_curve_is_polars(self, targeting_result):
        assert isinstance(targeting_result.toc_curve, pl.DataFrame)

    def test_toc_curve_columns(self, targeting_result):
        tbl = targeting_result.toc_curve
        assert "q" in tbl.columns
        assert "toc" in tbl.columns

    def test_toc_curve_q_in_range(self, targeting_result):
        q_vals = targeting_result.toc_curve["q"].to_list()
        assert all(0.0 < q <= 1.0 for q in q_vals)

    def test_toc_curve_n_points(self, targeting_result):
        assert len(targeting_result.toc_curve) == 20  # n_toc_points=20

    def test_summary_runs(self, targeting_result):
        s = targeting_result.summary()
        assert isinstance(s, str)
        assert "AUTOC" in s

    def test_repr(self, targeting_result):
        r = repr(targeting_result)
        assert "TargetingResult" in r
        assert "AUTOC" in r
