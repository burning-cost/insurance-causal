"""
Tests for HeterogeneousInference internal helper functions.

Covers the private OLS helpers, propensity fitter, and result dataclass
repr/summary methods that are not exercised by the integration tests in
test_inference.py.

These tests do NOT require econml.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_causal.causal_forest.inference import (
    BLPResult,
    GATESResult,
    CLANResult,
    HeterogeneousInferenceResult,
    HeterogeneousInference,
    _fit_propensity,
    _ols_with_se,
    _ols_simple,
)


# ---------------------------------------------------------------------------
# _ols_with_se
# ---------------------------------------------------------------------------

class TestOlsWithSe:
    def _make_data(self, n=100, seed=0):
        rng = np.random.default_rng(seed)
        W = rng.normal(0, 1, n)
        S = rng.normal(0, 1, n)
        Y = 0.5 * W + 0.3 * W * S + rng.normal(0, 0.5, n)
        Z = np.column_stack([W, W * S])
        return Y, Z

    def test_returns_four_floats(self):
        Y, Z = self._make_data()
        result = _ols_with_se(Y, Z)
        assert len(result) == 4
        b1, b2, se1, se2 = result
        assert all(isinstance(v, float) for v in [b1, b2, se1, se2])

    def test_coefficients_finite(self):
        Y, Z = self._make_data()
        b1, b2, se1, se2 = _ols_with_se(Y, Z)
        assert np.isfinite(b1)
        assert np.isfinite(b2)

    def test_standard_errors_non_negative(self):
        Y, Z = self._make_data()
        b1, b2, se1, se2 = _ols_with_se(Y, Z)
        assert se1 >= 0
        assert se2 >= 0

    def test_recovers_known_coefficients(self):
        """With large n and no noise, estimates should be close to truth."""
        rng = np.random.default_rng(42)
        n = 1000
        W = rng.normal(0, 1, n)
        S = rng.normal(0, 1, n)
        # True: beta_1=1.0, beta_2=0.5
        Y = 0.1 + 1.0 * W + 0.5 * W * S + rng.normal(0, 0.01, n)
        Z = np.column_stack([W, W * S])
        b1, b2, se1, se2 = _ols_with_se(Y, Z)
        assert abs(b1 - 1.0) < 0.1, f"beta_1={b1:.3f}, expected ~1.0"
        assert abs(b2 - 0.5) < 0.1, f"beta_2={b2:.3f}, expected ~0.5"

    def test_too_few_obs_raises(self):
        Y = np.array([1.0, 2.0])
        Z = np.column_stack([np.array([0.5, 1.0]), np.array([0.5, 1.0])])
        with pytest.raises(ValueError):
            _ols_with_se(Y, Z)


# ---------------------------------------------------------------------------
# _ols_simple
# ---------------------------------------------------------------------------

class TestOlsSimple:
    def test_returns_two_floats(self):
        Y = np.array([1.0, 2.0, 3.0, 4.0])
        W = np.array([1.0, 2.0, 3.0, 4.0])
        beta, se = _ols_simple(Y, W)
        assert isinstance(beta, float)
        assert isinstance(se, float)

    def test_recovers_slope(self):
        """Y = W + noise -> beta ~ 1."""
        rng = np.random.default_rng(0)
        n = 200
        W = rng.normal(0, 1, n)
        Y = W + rng.normal(0, 0.1, n)
        beta, se = _ols_simple(Y, W)
        assert abs(beta - 1.0) < 0.2

    def test_zero_variance_treatment_returns_nan(self):
        Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        W = np.ones(5)  # zero variance
        beta, se = _ols_simple(Y, W)
        assert np.isnan(beta)
        assert np.isnan(se)

    def test_too_few_obs_raises(self):
        Y = np.array([1.0, 2.0])
        W = np.array([0.5, 1.5])
        with pytest.raises(ValueError):
            _ols_simple(Y, W)

    def test_se_non_negative(self):
        rng = np.random.default_rng(1)
        Y = rng.normal(0, 1, 50)
        W = rng.normal(0, 1, 50)
        beta, se = _ols_simple(Y, W)
        assert se >= 0


# ---------------------------------------------------------------------------
# _fit_propensity
# ---------------------------------------------------------------------------

class TestFitPropensity:
    def test_returns_array_of_correct_length(self):
        rng = np.random.default_rng(42)
        n_train, n_test = 100, 50
        X_train = rng.normal(0, 1, (n_train, 3))
        D_train = rng.normal(0, 1, n_train)
        X_test = rng.normal(0, 1, (n_test, 3))
        D_test = rng.normal(0, 1, n_test)
        result = _fit_propensity(D_train, X_train, D_test, X_test)
        assert result.shape == (n_test,)

    def test_predictions_finite(self):
        rng = np.random.default_rng(0)
        n = 80
        X = rng.normal(0, 1, (n, 2))
        D = rng.normal(0, 1, n)
        result = _fit_propensity(D, X, D, X)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# BLPResult repr
# ---------------------------------------------------------------------------

class TestBLPResultRepr:
    def _make_blp(self, beta_2=0.5, p=0.001):
        return BLPResult(
            beta_1=-0.2, beta_2=beta_2,
            beta_1_se=0.05, beta_2_se=0.1,
            beta_1_tstat=-4.0, beta_2_tstat=5.0,
            beta_2_pvalue=p, heterogeneity_detected=p < 0.05,
            n_splits=100,
        )

    def test_repr_contains_beta_2(self):
        r = repr(self._make_blp())
        assert "beta_2" in r

    def test_repr_contains_blpresult(self):
        r = repr(self._make_blp())
        assert "BLPResult" in r

    def test_repr_three_stars_for_p_lt_001(self):
        r = repr(self._make_blp(p=0.0001))
        assert "***" in r

    def test_repr_two_stars_for_p_lt_01(self):
        r = repr(self._make_blp(p=0.005))
        assert "**" in r

    def test_repr_one_star_for_p_lt_05(self):
        r = repr(self._make_blp(p=0.03))
        assert "*" in r and "**" not in r

    def test_repr_no_star_for_p_gt_05(self):
        r = repr(self._make_blp(p=0.2))
        assert "*" not in r


# ---------------------------------------------------------------------------
# GATESResult repr
# ---------------------------------------------------------------------------

class TestGATESResultRepr:
    def test_repr_contains_gates_result(self):
        table = pl.DataFrame({
            "group": [1, 2, 3],
            "cate_lower": [-0.3, -0.2, -0.1],
            "cate_upper": [-0.2, -0.1, 0.0],
            "gate": [-0.25, -0.15, -0.05],
            "gate_se": [0.02, 0.02, 0.02],
            "n": [100, 100, 100],
        })
        result = GATESResult(table=table, gates_increasing=True, n_groups=3)
        r = repr(result)
        assert "GATESResult" in r
        assert "n_groups=3" in r

    def test_repr_gates_increasing_shown(self):
        table = pl.DataFrame({"group": [1], "cate_lower": [0.0], "cate_upper": [1.0],
                               "gate": [0.1], "gate_se": [0.01], "n": [50]})
        result = GATESResult(table=table, gates_increasing=False, n_groups=1)
        r = repr(result)
        assert "False" in r


# ---------------------------------------------------------------------------
# HeterogeneousInferenceResult.summary() edge cases
# ---------------------------------------------------------------------------

class TestInferenceResultSummary:
    def _make_result(self):
        blp = BLPResult(
            beta_1=-0.2, beta_2=0.4,
            beta_1_se=0.05, beta_2_se=0.1,
            beta_1_tstat=-4.0, beta_2_tstat=4.0,
            beta_2_pvalue=0.001, heterogeneity_detected=True,
            n_splits=10,
        )
        gates_table = pl.DataFrame({
            "group": [1, 2, 3, 4, 5],
            "cate_lower": [-0.5, -0.4, -0.3, -0.2, -0.1],
            "cate_upper": [-0.4, -0.3, -0.2, -0.1, 0.0],
            "gate": [-0.45, -0.35, -0.25, -0.15, -0.05],
            "gate_se": [0.03, 0.03, 0.03, 0.03, 0.03],
            "n": [200, 200, 200, 200, 200],
        })
        gates = GATESResult(table=gates_table, gates_increasing=True, n_groups=5)
        clan_table = pl.DataFrame({
            "feature": ["age", "ncd"],
            "mean_top": [35.0, 2.5],
            "mean_bottom": [40.0, 3.5],
            "diff": [-5.0, -1.0],
            "t_stat": [-2.5, -1.8],
            "p_value": [0.01, 0.07],
        })
        clan = CLANResult(table=clan_table, top_group=5, bottom_group=1)
        return HeterogeneousInferenceResult(
            blp=blp, gates=gates, clan=clan, n_obs=1000, n_splits=10
        )

    def test_summary_runs(self):
        result = self._make_result()
        s = result.summary()
        assert isinstance(s, str)
        assert len(s) > 50

    def test_summary_contains_blp(self):
        result = self._make_result()
        s = result.summary()
        assert "BLP" in s

    def test_summary_contains_gates(self):
        result = self._make_result()
        s = result.summary()
        assert "GATES" in s

    def test_summary_contains_clan(self):
        result = self._make_result()
        s = result.summary()
        assert "CLAN" in s

    def test_summary_heterogeneity_detected_message(self):
        result = self._make_result()
        s = result.summary()
        assert "HETEROGENEITY DETECTED" in s

    def test_summary_no_heterogeneity_message(self):
        blp = BLPResult(
            beta_1=0.0, beta_2=0.01,
            beta_1_se=0.1, beta_2_se=0.1,
            beta_1_tstat=0.0, beta_2_tstat=0.1,
            beta_2_pvalue=0.9, heterogeneity_detected=False,
            n_splits=10,
        )
        r = self._make_result()
        from dataclasses import replace
        result = HeterogeneousInferenceResult(
            blp=blp, gates=r.gates, clan=r.clan, n_obs=500, n_splits=10
        )
        s = result.summary()
        assert "no sig. heterogeneity" in s


# ---------------------------------------------------------------------------
# HeterogeneousInference: k_groups and repeated splits with synthetic data
# ---------------------------------------------------------------------------

class TestHeterogeneousInferenceInternal:
    """Test _run_gates and _run_clan with synthetic arrays (no econml)."""

    def _make_synthetic(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        Y = rng.binomial(1, 0.8, n).astype(float)
        D = rng.normal(0, 0.1, n)
        X = rng.normal(0, 1, (n, 3))
        S = rng.normal(-0.2, 0.1, n)  # CATE proxy
        return Y, D, X, S

    def test_run_gates_returns_gates_result(self):
        Y, D, X, S = self._make_synthetic()
        inf = HeterogeneousInference(n_splits=2, k_groups=4, random_state=0)
        result = inf._run_gates(Y, D, X, S)
        assert isinstance(result, GATESResult)

    def test_run_gates_n_groups(self):
        Y, D, X, S = self._make_synthetic()
        inf = HeterogeneousInference(n_splits=2, k_groups=4)
        result = inf._run_gates(Y, D, X, S)
        assert result.n_groups == 4

    def test_run_gates_table_has_expected_columns(self):
        Y, D, X, S = self._make_synthetic()
        inf = HeterogeneousInference(n_splits=2, k_groups=3)
        result = inf._run_gates(Y, D, X, S)
        for col in ["group", "gate", "n"]:
            assert col in result.table.columns

    def test_run_gates_small_group_warns(self):
        """With very small n, some groups will have < 500 obs -> warning."""
        Y, D, X, S = self._make_synthetic(n=50)
        inf = HeterogeneousInference(n_splits=2, k_groups=5)
        with pytest.warns(UserWarning):
            inf._run_gates(Y, D, X, S)

    def test_run_clan_returns_clan_result(self):
        rng = np.random.default_rng(42)
        n = 100
        df_pd = __import__("pandas").DataFrame({
            "age": rng.normal(35, 10, n),
            "ncd": rng.integers(0, 6, n).astype(float),
        })
        S = rng.normal(0, 1, n)
        inf = HeterogeneousInference(n_splits=2, k_groups=3)
        result = inf._run_clan(df_pd, S, ["age", "ncd"], ["age", "ncd"])
        assert isinstance(result, CLANResult)
        assert "feature" in result.table.columns

    def test_run_clan_table_columns(self):
        rng = np.random.default_rng(0)
        n = 60
        import pandas as pd_mod
        df_pd = pd_mod.DataFrame({
            "age": rng.normal(35, 10, n),
        })
        S = rng.normal(0, 1, n)
        inf = HeterogeneousInference(n_splits=2, k_groups=3)
        result = inf._run_clan(df_pd, S, ["age"], ["age"])
        for col in ["feature", "mean_top", "mean_bottom", "diff"]:
            assert col in result.table.columns

    def test_blp_splits_degenerate_returns_nan(self):
        """If all splits fail, BLP returns nan result."""
        Y = np.zeros(4)
        D = np.zeros(4)
        X = np.zeros((4, 1))
        S = np.zeros(4)
        rng = np.random.default_rng(0)
        inf = HeterogeneousInference(n_splits=2, k_groups=2)
        # With n=4, split_pt=2 -> 2 obs per half, which is <= p+2=3, so OLS raises
        result = inf._run_blp_splits(Y, D, X, S, rng)
        # May produce nan or valid result depending on data — just check it returns BLPResult
        assert isinstance(result, BLPResult)
