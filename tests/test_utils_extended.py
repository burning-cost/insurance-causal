"""
Extended tests for _utils.py.

The existing test_utils.py covers the main happy paths for:
- to_pandas, poisson/gamma transforms, check_overlap, adaptive_catboost_params,
  build_catboost_regressor/classifier

This file adds coverage for:
- adaptive_dml_catboost_params: all tier boundaries, monotonicity, key presence
- build_catboost_regressor/classifier: nuisance_params alias (backward compat),
  override takes precedence, no n_samples fallback
- make_doubleml_data: construction sanity
- check_overlap: single value, all-same values, binary treatment
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_causal._utils import (
    adaptive_dml_catboost_params,
    build_catboost_regressor,
    build_catboost_classifier,
    check_overlap,
    make_doubleml_data,
    to_pandas,
)


# ---------------------------------------------------------------------------
# adaptive_dml_catboost_params — tier boundaries
# ---------------------------------------------------------------------------


class TestAdaptiveDMLCatboostParams:
    """
    The DML-specific adaptive params use stricter regularisation than the
    general adaptive_catboost_params. These tests verify:
    1. Each tier returns the expected depth/regularisation direction.
    2. Monotonicity: iterations don't decrease as n increases.
    3. Required keys present for all tiers.
    4. The very-small-sample tier (n < 2000) uses depth <= 3.
    """

    REQUIRED_KEYS = {"iterations", "learning_rate", "depth", "l2_leaf_reg"}

    def test_required_keys_all_tiers(self):
        for n in [100, 500, 1_500, 2_000, 5_000, 9_999, 10_000, 50_000]:
            params = adaptive_dml_catboost_params(n)
            missing = self.REQUIRED_KEYS - set(params.keys())
            assert not missing, f"n={n}: missing keys {missing}"

    def test_very_small_tier_has_shallow_depth(self):
        """n < 2000: depth should be <= 3 (DML-specific aggressive regularisation)."""
        params = adaptive_dml_catboost_params(100)
        assert params["depth"] <= 3, (
            f"Expected depth <= 3 for very small n, got {params['depth']}"
        )

    def test_very_small_tier_has_high_l2(self):
        """n < 2000: l2_leaf_reg should be >= 20 (strict regularisation)."""
        params = adaptive_dml_catboost_params(500)
        assert params["l2_leaf_reg"] >= 20, (
            f"Expected l2_leaf_reg >= 20 for very small n, got {params['l2_leaf_reg']}"
        )

    def test_small_tier_2000_to_10000(self):
        """2000 <= n < 10000 gets moderate regularisation."""
        params = adaptive_dml_catboost_params(5_000)
        assert params["depth"] <= 5

    def test_large_tier_at_boundary(self):
        """n >= 10000 should get larger capacity."""
        params = adaptive_dml_catboost_params(10_000)
        assert params["iterations"] >= 200

    def test_very_large_n(self):
        """n = 100k should use maximum iterations."""
        params = adaptive_dml_catboost_params(100_000)
        assert params["iterations"] >= 300

    def test_monotone_iterations(self):
        """Iterations should not decrease as n increases (or at least stay flat)."""
        sizes = [100, 500, 1_999, 2_000, 9_999, 10_000, 100_000]
        iters = [adaptive_dml_catboost_params(n)["iterations"] for n in sizes]
        for i in range(len(iters) - 1):
            assert iters[i] <= iters[i + 1], (
                f"iterations decreased: n={sizes[i]} gave {iters[i]}, "
                f"n={sizes[i+1]} gave {iters[i+1]}"
            )

    def test_l2_monotone_non_increasing(self):
        """l2_leaf_reg should not increase as n increases (more data = less regularisation)."""
        sizes = [100, 1_999, 2_000, 9_999, 10_000]
        l2s = [adaptive_dml_catboost_params(n)["l2_leaf_reg"] for n in sizes]
        for i in range(len(l2s) - 1):
            assert l2s[i] >= l2s[i + 1], (
                f"l2_leaf_reg should be non-increasing: n={sizes[i]} gave {l2s[i]}, "
                f"n={sizes[i+1]} gave {l2s[i+1]}"
            )

    def test_boundary_n_exactly_2000(self):
        """At exactly n=2000, we cross from very-small into small tier."""
        p_below = adaptive_dml_catboost_params(1_999)
        p_at = adaptive_dml_catboost_params(2_000)
        # The at-boundary params should have >= iterations (or equal)
        assert p_at["iterations"] >= p_below["iterations"] or p_at["depth"] >= p_below["depth"]

    def test_boundary_n_exactly_10000(self):
        """At exactly n=10000, we cross into large tier."""
        p_below = adaptive_dml_catboost_params(9_999)
        p_at = adaptive_dml_catboost_params(10_000)
        assert p_at["iterations"] >= p_below["iterations"]


# ---------------------------------------------------------------------------
# build_catboost_regressor — extended cases
# ---------------------------------------------------------------------------


class TestBuildCatboostRegressorExtended:
    def test_nuisance_params_alias_works(self):
        """nuisance_params is an alias for override_params — both should work."""
        m1 = build_catboost_regressor(
            random_state=0, n_samples=1000,
            override_params={"iterations": 77}
        )
        m2 = build_catboost_regressor(
            random_state=0, n_samples=1000,
            nuisance_params={"iterations": 77}
        )
        assert m1.get_params()["iterations"] == 77
        assert m2.get_params()["iterations"] == 77

    def test_override_params_wins_over_adaptive(self):
        """Explicit override should take precedence over adaptive defaults."""
        m = build_catboost_regressor(
            random_state=0, n_samples=500,
            override_params={"iterations": 999, "depth": 10}
        )
        params = m.get_params()
        assert params["iterations"] == 999
        assert params["depth"] == 10

    def test_random_seed_set(self):
        m = build_catboost_regressor(random_state=123)
        assert m.get_params()["random_seed"] == 123

    def test_verbose_is_zero(self):
        """Model should not print training output during nuisance fitting."""
        m = build_catboost_regressor()
        assert m.get_params()["verbose"] == 0

    def test_allow_writing_files_false(self):
        """CatBoost should not write catboost_info/ directories."""
        m = build_catboost_regressor()
        assert m.get_params()["allow_writing_files"] is False

    def test_loss_function_rmse(self):
        m = build_catboost_regressor()
        assert m.get_params()["loss_function"] == "RMSE"

    def test_large_n_samples_gives_high_capacity(self):
        m = build_catboost_regressor(n_samples=100_000)
        assert m.get_params()["iterations"] >= 300


# ---------------------------------------------------------------------------
# build_catboost_classifier — extended cases
# ---------------------------------------------------------------------------


class TestBuildCatboostClassifierExtended:
    def test_loss_function_logloss(self):
        m = build_catboost_classifier()
        assert m.get_params()["loss_function"] == "Logloss"

    def test_nuisance_params_alias_works(self):
        m = build_catboost_classifier(
            random_state=0, n_samples=1000,
            nuisance_params={"iterations": 55}
        )
        assert m.get_params()["iterations"] == 55

    def test_small_n_reduces_capacity(self):
        m_small = build_catboost_classifier(n_samples=500)
        m_large = build_catboost_classifier(n_samples=100_000)
        assert m_small.get_params()["iterations"] < m_large.get_params()["iterations"]

    def test_verbose_is_zero(self):
        m = build_catboost_classifier()
        assert m.get_params()["verbose"] == 0


# ---------------------------------------------------------------------------
# make_doubleml_data
# ---------------------------------------------------------------------------


class TestMakeDoubleMLData:
    def test_returns_doubleml_data_object(self):
        """make_doubleml_data should return a DoubleMLData object."""
        import doubleml as dml

        df = pd.DataFrame({
            "y": [0.5, 0.8, 0.4, 0.9],
            "d": [0.05, 0.10, -0.05, 0.02],
            "age": [30, 40, 25, 50],
            "region": [1, 2, 1, 3],
        })
        obj = make_doubleml_data(df, "y", "d", ["age", "region"])
        assert isinstance(obj, dml.DoubleMLData)

    def test_correct_outcome_col(self):
        df = pd.DataFrame({
            "y": [1.0, 2.0],
            "d": [0.1, -0.1],
            "x1": [10, 20],
        })
        obj = make_doubleml_data(df, "y", "d", ["x1"])
        # DoubleML stores as .y_col
        assert obj.y_col == "y"

    def test_correct_treatment_col(self):
        df = pd.DataFrame({
            "y": [1.0, 2.0],
            "treatment": [0.1, -0.1],
            "x1": [10, 20],
        })
        obj = make_doubleml_data(df, "y", "treatment", ["x1"])
        assert obj.d_cols == "treatment" or "treatment" in str(obj.d_cols)

    def test_correct_feature_cols(self):
        df = pd.DataFrame({
            "outcome": [0.5, 0.6],
            "treat": [0.0, 0.1],
            "age": [30, 40],
            "ncd": [3, 5],
        })
        obj = make_doubleml_data(df, "outcome", "treat", ["age", "ncd"])
        assert "age" in obj.x_cols
        assert "ncd" in obj.x_cols


# ---------------------------------------------------------------------------
# check_overlap — edge cases
# ---------------------------------------------------------------------------


class TestCheckOverlapEdgeCases:
    def test_single_value(self):
        """Single observation — should not crash."""
        stats = check_overlap(np.array([0.05]))
        assert stats["n_obs"] == 1
        assert stats["mean"] == 0.05
        assert stats["min"] == stats["max"] == 0.05

    def test_all_same_values(self):
        """Constant treatment — std=0, all percentiles equal."""
        values = np.full(100, 0.10)
        stats = check_overlap(values)
        assert stats["std"] == 0.0
        assert stats["min"] == stats["max"] == 0.10
        assert stats["p5"] == stats["p95"] == 0.10

    def test_binary_treatment(self):
        """Binary 0/1 treatment is a valid input."""
        values = np.array([0] * 300 + [1] * 200, dtype=float)
        stats = check_overlap(values)
        assert stats["min"] == 0.0
        assert stats["max"] == 1.0
        assert 0.3 < stats["mean"] < 0.5

    def test_negative_values_accepted(self):
        """Price decreases are negative — must be accepted."""
        values = np.random.uniform(-0.3, 0.3, 500)
        stats = check_overlap(values)
        assert stats["min"] < 0.0
        assert stats["max"] > 0.0

    def test_output_types_are_float(self):
        """All numeric stats should be Python floats."""
        values = np.arange(10.0)
        stats = check_overlap(values)
        for key in ["mean", "std", "min", "p5", "p25", "p75", "p95", "max"]:
            assert isinstance(stats[key], float), f"{key} should be float"

    def test_n_obs_is_int(self):
        values = np.ones(42)
        stats = check_overlap(values)
        assert isinstance(stats["n_obs"], int)
        assert stats["n_obs"] == 42

    def test_percentile_ordering(self):
        """Percentiles must be non-decreasing."""
        rng = np.random.default_rng(42)
        values = rng.normal(0, 0.1, 1000)
        stats = check_overlap(values)
        assert stats["min"] <= stats["p5"]
        assert stats["p5"] <= stats["p25"]
        assert stats["p25"] <= stats["p75"]
        assert stats["p75"] <= stats["p95"]
        assert stats["p95"] <= stats["max"]


# ---------------------------------------------------------------------------
# to_pandas — edge cases
# ---------------------------------------------------------------------------


class TestToPandasEdgeCases:
    def test_single_column_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = to_pandas(df)
        assert list(result.columns) == ["a"]

    def test_empty_dataframe_passthrough(self):
        df = pd.DataFrame()
        result = to_pandas(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_none_raises_type_error(self):
        with pytest.raises(TypeError, match="Expected a pandas or polars DataFrame"):
            to_pandas(None)

    def test_dict_raises_type_error(self):
        with pytest.raises(TypeError):
            to_pandas({"a": [1, 2, 3]})

    def test_numpy_array_raises_type_error(self):
        with pytest.raises(TypeError):
            to_pandas(np.array([[1, 2], [3, 4]]))

    def test_preserves_column_names(self):
        df = pd.DataFrame({"age": [30], "region": ["North"], "ncb": [5]})
        result = to_pandas(df)
        assert list(result.columns) == ["age", "region", "ncb"]

    def test_preserves_dtypes(self):
        df = pd.DataFrame({
            "int_col": pd.array([1, 2], dtype="int64"),
            "float_col": pd.array([1.0, 2.0], dtype="float64"),
        })
        result = to_pandas(df)
        assert result["int_col"].dtype == np.int64
        assert result["float_col"].dtype == np.float64
