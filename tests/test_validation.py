"""
Tests for _validation.py — the input validation helpers.

These functions are called at the top of every public .fit() / __init__()
method, so any bug here surfaces as a confusing error message or, worse,
a silent wrong result. We test every function in the module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_causal._validation import (
    check_dataframe,
    check_not_empty,
    check_columns_exist,
    check_column_exists,
    check_column_numeric,
    check_n_splits,
    check_n_estimators,
    check_confounders,
    check_alpha,
    check_outcome_type,
)


# ---------------------------------------------------------------------------
# check_dataframe
# ---------------------------------------------------------------------------


class TestCheckDataframe:
    def test_pandas_passes(self):
        check_dataframe(pd.DataFrame({"a": [1]}), "df")

    def test_polars_passes(self):
        try:
            import polars as pl
            check_dataframe(pl.DataFrame({"a": [1]}), "df")
        except ImportError:
            pytest.skip("polars not installed")

    def test_dict_raises(self):
        with pytest.raises(TypeError, match="insurance-causal"):
            check_dataframe({"a": [1]}, "df")

    def test_list_raises(self):
        with pytest.raises(TypeError, match="pandas or polars DataFrame"):
            check_dataframe([1, 2, 3], "df")

    def test_none_raises(self):
        with pytest.raises(TypeError, match="None"):
            check_dataframe(None, "df")

    def test_numpy_array_raises(self):
        with pytest.raises(TypeError, match="numpy"):
            check_dataframe(np.array([[1, 2], [3, 4]]), "df")

    def test_param_name_in_message(self):
        """Error message must include the parameter name so the user knows what to fix."""
        with pytest.raises(TypeError) as exc_info:
            check_dataframe([1, 2], "my_df_param")
        assert "my_df_param" in str(exc_info.value)


# ---------------------------------------------------------------------------
# check_not_empty
# ---------------------------------------------------------------------------


class TestCheckNotEmpty:
    def test_nonempty_dataframe_passes(self):
        check_not_empty(pd.DataFrame({"a": [1, 2]}), "df")

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match="empty"):
            check_not_empty(pd.DataFrame({"a": []}), "df")

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            check_not_empty([], "arr")

    def test_single_row_passes(self):
        check_not_empty(pd.DataFrame({"a": [42]}), "df")

    def test_param_name_in_message(self):
        with pytest.raises(ValueError) as exc_info:
            check_not_empty(pd.DataFrame(), "training_data")
        assert "training_data" in str(exc_info.value)


# ---------------------------------------------------------------------------
# check_columns_exist
# ---------------------------------------------------------------------------


class TestCheckColumnsExist:
    def test_all_present_passes(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        check_columns_exist(df, ["a", "b"], "df")

    def test_missing_column_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="insurance-causal"):
            check_columns_exist(df, ["a", "missing_col"], "df")

    def test_error_shows_missing_col(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="missing_col"):
            check_columns_exist(df, ["missing_col"], "df")

    def test_error_shows_available_cols(self):
        df = pd.DataFrame({"existing": [1]})
        with pytest.raises(ValueError) as exc_info:
            check_columns_exist(df, ["nope"], "df")
        assert "existing" in str(exc_info.value)

    def test_empty_columns_list_passes(self):
        df = pd.DataFrame({"a": [1]})
        check_columns_exist(df, [], "df")  # vacuously true

    def test_multiple_missing_reported(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError) as exc_info:
            check_columns_exist(df, ["b", "c"], "df")
        msg = str(exc_info.value)
        assert "b" in msg or "c" in msg


# ---------------------------------------------------------------------------
# check_column_exists
# ---------------------------------------------------------------------------


class TestCheckColumnExists:
    def test_present_passes(self):
        df = pd.DataFrame({"age": [25]})
        check_column_exists(df, "age", "age_col")

    def test_absent_raises(self):
        df = pd.DataFrame({"age": [25]})
        with pytest.raises(ValueError, match="insurance-causal"):
            check_column_exists(df, "ncd_years", "ncd_col")

    def test_param_name_in_error(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError) as exc_info:
            check_column_exists(df, "treatment", "my_treatment_param")
        assert "my_treatment_param" in str(exc_info.value)


# ---------------------------------------------------------------------------
# check_column_numeric
# ---------------------------------------------------------------------------


class TestCheckColumnNumeric:
    def test_int_column_passes(self):
        df = pd.DataFrame({"age": [25, 30, 45]})
        check_column_numeric(df, "age", "age_col")

    def test_float_column_passes(self):
        df = pd.DataFrame({"premium": [350.0, 420.5]})
        check_column_numeric(df, "premium", "premium_col")

    def test_string_column_raises(self):
        df = pd.DataFrame({"region": ["North", "South"]})
        with pytest.raises(TypeError, match="numeric"):
            check_column_numeric(df, "region", "region_col")

    def test_object_column_raises(self):
        df = pd.DataFrame({"cat": pd.Categorical(["A", "B"])})
        with pytest.raises(TypeError, match="numeric"):
            check_column_numeric(df, "cat", "cat_col")

    def test_param_name_in_error(self):
        df = pd.DataFrame({"channel": ["aggregator", "direct"]})
        with pytest.raises(TypeError) as exc_info:
            check_column_numeric(df, "channel", "my_channel_param")
        assert "my_channel_param" in str(exc_info.value)


# ---------------------------------------------------------------------------
# check_n_splits
# ---------------------------------------------------------------------------


class TestCheckNSplits:
    def test_valid_int_passes(self):
        check_n_splits(5)

    def test_minimum_value_passes(self):
        check_n_splits(2)

    def test_below_minimum_raises(self):
        with pytest.raises(ValueError, match=">= 2"):
            check_n_splits(1)

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            check_n_splits(0)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            check_n_splits(-1)

    def test_float_raises(self):
        with pytest.raises(TypeError, match="integer"):
            check_n_splits(5.0)

    def test_bool_raises(self):
        """True would pass isinstance(True, int) — we must guard against booleans."""
        with pytest.raises(TypeError):
            check_n_splits(True)

    def test_string_raises(self):
        with pytest.raises(TypeError):
            check_n_splits("5")

    def test_custom_minimum(self):
        check_n_splits(3, minimum=3)
        with pytest.raises(ValueError):
            check_n_splits(2, minimum=3)

    def test_custom_param_name_in_message(self):
        with pytest.raises((TypeError, ValueError)) as exc_info:
            check_n_splits(1.5, param="cv_folds")
        assert "cv_folds" in str(exc_info.value)


# ---------------------------------------------------------------------------
# check_n_estimators
# ---------------------------------------------------------------------------


class TestCheckNEstimators:
    def test_valid_passes(self):
        check_n_estimators(100)
        check_n_estimators(1)

    def test_zero_raises(self):
        with pytest.raises(ValueError, match=">= 1"):
            check_n_estimators(0)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            check_n_estimators(-10)

    def test_float_raises(self):
        with pytest.raises(TypeError, match="integer"):
            check_n_estimators(100.0)

    def test_bool_raises(self):
        with pytest.raises(TypeError):
            check_n_estimators(True)


# ---------------------------------------------------------------------------
# check_confounders
# ---------------------------------------------------------------------------


class TestCheckConfounders:
    def test_valid_list_passes(self):
        check_confounders(["age", "region", "ncd_years"])

    def test_single_element_passes(self):
        check_confounders(["age"])

    def test_none_raises(self):
        with pytest.raises(ValueError, match="must be provided"):
            check_confounders(None)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            check_confounders([])

    def test_string_raises(self):
        """A plain string should fail — must be a list."""
        with pytest.raises(TypeError, match="list"):
            check_confounders("age")

    def test_tuple_passes(self):
        """Tuples are accepted as equivalent to lists."""
        check_confounders(("age", "region"))

    def test_dict_raises(self):
        with pytest.raises(TypeError):
            check_confounders({"age": True})

    def test_error_message_has_example(self):
        """Error should suggest typical confounder columns so user knows what's expected."""
        with pytest.raises(ValueError) as exc_info:
            check_confounders(None)
        msg = str(exc_info.value)
        # Should mention that confounders are column names
        assert "column" in msg.lower() or "list" in msg.lower()


# ---------------------------------------------------------------------------
# check_alpha
# ---------------------------------------------------------------------------


class TestCheckAlpha:
    def test_standard_alpha_passes(self):
        check_alpha(0.05)
        check_alpha(0.01)
        check_alpha(0.10)

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            check_alpha(0.0)

    def test_one_raises(self):
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            check_alpha(1.0)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            check_alpha(-0.05)

    def test_greater_than_one_raises(self):
        with pytest.raises(ValueError):
            check_alpha(1.5)

    def test_string_numeric_coercion(self):
        """String '0.05' should be accepted via float() coercion."""
        check_alpha("0.05")

    def test_string_nonnumeric_raises(self):
        with pytest.raises(TypeError):
            check_alpha("not_a_number")

    def test_none_raises(self):
        with pytest.raises(TypeError):
            check_alpha(None)


# ---------------------------------------------------------------------------
# check_outcome_type
# ---------------------------------------------------------------------------


class TestCheckOutcomeType:
    def test_valid_types_pass(self):
        valid = ("continuous", "binary", "poisson", "gamma")
        for t in valid:
            check_outcome_type(t, valid)

    def test_invalid_raises(self):
        valid = ("continuous", "binary", "poisson", "gamma")
        with pytest.raises(ValueError, match="must be one of"):
            check_outcome_type("gaussian", valid)

    def test_error_shows_valid_options(self):
        valid = ("continuous", "binary")
        with pytest.raises(ValueError) as exc_info:
            check_outcome_type("lognormal", valid)
        msg = str(exc_info.value)
        assert "continuous" in msg or "binary" in msg

    def test_case_sensitive(self):
        """Validation is case-sensitive — 'Continuous' != 'continuous'."""
        valid = ("continuous", "binary", "poisson", "gamma")
        with pytest.raises(ValueError):
            check_outcome_type("Continuous", valid)

    def test_empty_string_raises(self):
        valid = ("continuous",)
        with pytest.raises(ValueError):
            check_outcome_type("", valid)
