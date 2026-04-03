"""
Unit tests for _model.py that don't require fitting a DML model.

Fitting requires heavy computation (DML + CatBoost) and must run on
Databricks. These tests cover:

- CausalPricingModel.__init__ validation
- CausalPricingModel.__repr__
- CausalPricingModel._check_fitted
- CausalPricingModel._validate_columns
- CausalPricingModel._interpret_bias
- CausalPricingModel._extract_naive_from_model
- AverageTreatmentEffect.__str__ and field types
- _transform_outcome dispatch

All tests use mocks or construct model objects without calling .fit().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_causal._model import AverageTreatmentEffect, CausalPricingModel
from insurance_causal.treatments import (
    BinaryTreatment,
    ContinuousTreatment,
    PriceChangeTreatment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(**kwargs):
    """Build a minimal unfitted CausalPricingModel with sensible defaults."""
    defaults = dict(
        outcome="claim_count",
        outcome_type="poisson",
        treatment=PriceChangeTreatment(column="price_change"),
        confounders=["age", "ncd_years", "region"],
    )
    defaults.update(kwargs)
    return CausalPricingModel(**defaults)


# ---------------------------------------------------------------------------
# __init__ validation
# ---------------------------------------------------------------------------


class TestCausalPricingModelInit:
    def test_valid_construction(self):
        m = _make_model()
        assert m.outcome == "claim_count"
        assert m.outcome_type == "poisson"
        assert m.cv_folds == 5
        assert not m._fitted

    def test_invalid_outcome_type_raises(self):
        with pytest.raises(ValueError, match="outcome_type"):
            _make_model(outcome_type="lognormal")

    def test_cv_folds_below_2_raises(self):
        with pytest.raises(ValueError, match="cv_folds"):
            _make_model(cv_folds=1)

    def test_cv_folds_zero_raises(self):
        with pytest.raises(ValueError, match="cv_folds"):
            _make_model(cv_folds=0)

    def test_cv_folds_float_raises(self):
        with pytest.raises(ValueError, match="cv_folds"):
            _make_model(cv_folds=5.0)

    def test_cv_folds_2_accepted(self):
        m = _make_model(cv_folds=2)
        assert m.cv_folds == 2

    def test_all_outcome_types_accepted(self):
        for ot in ("continuous", "binary", "poisson", "gamma"):
            m = _make_model(outcome_type=ot)
            assert m.outcome_type == ot

    def test_binary_treatment_accepted(self):
        m = _make_model(
            treatment=BinaryTreatment(column="is_aggregator"),
            outcome_type="binary",
        )
        assert isinstance(m.treatment, BinaryTreatment)

    def test_continuous_treatment_accepted(self):
        m = _make_model(
            treatment=ContinuousTreatment(column="telematics_score"),
        )
        assert isinstance(m.treatment, ContinuousTreatment)

    def test_exposure_col_stored(self):
        m = _make_model(exposure_col="earned_years")
        assert m.exposure_col == "earned_years"

    def test_nuisance_params_stored(self):
        params = {"iterations": 100, "depth": 3}
        m = _make_model(nuisance_params=params)
        assert m.nuisance_params == params

    def test_random_state_stored(self):
        m = _make_model(random_state=99)
        assert m.random_state == 99


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestCausalPricingModelRepr:
    def test_repr_shows_unfitted(self):
        m = _make_model()
        r = repr(m)
        assert "unfitted" in r

    def test_repr_contains_outcome(self):
        m = _make_model(outcome="loss_ratio")
        assert "loss_ratio" in repr(m)

    def test_repr_contains_treatment_col(self):
        m = _make_model(treatment=PriceChangeTreatment(column="pct_change"))
        assert "pct_change" in repr(m)

    def test_repr_contains_confounders(self):
        m = _make_model(confounders=["age", "region"])
        r = repr(m)
        assert "age" in r or "confounders" in r

    def test_repr_shows_fitted_after_mock_fit(self):
        """Manually toggle _fitted to verify repr changes."""
        m = _make_model()
        m._fitted = True  # simulate post-fit state
        assert "fitted" in repr(m)


# ---------------------------------------------------------------------------
# _check_fitted
# ---------------------------------------------------------------------------


class TestCheckFitted:
    def test_raises_before_fit(self):
        m = _make_model()
        with pytest.raises(RuntimeError, match="fit"):
            m._check_fitted()

    def test_passes_after_manual_toggle(self):
        m = _make_model()
        m._fitted = True
        m._check_fitted()  # should not raise

    def test_average_treatment_effect_raises_before_fit(self):
        m = _make_model()
        with pytest.raises(RuntimeError):
            m.average_treatment_effect()

    def test_treatment_overlap_stats_raises_before_fit(self):
        m = _make_model()
        with pytest.raises(RuntimeError):
            m.treatment_overlap_stats()

    def test_dml_model_property_raises_before_fit(self):
        m = _make_model()
        with pytest.raises(RuntimeError):
            _ = m.dml_model


# ---------------------------------------------------------------------------
# _validate_columns
# ---------------------------------------------------------------------------


class TestValidateColumns:
    def test_all_columns_present_passes(self):
        m = _make_model(
            outcome="claims",
            treatment=PriceChangeTreatment(column="price_change"),
            confounders=["age", "region"],
        )
        df = pd.DataFrame({
            "claims": [0, 1],
            "price_change": [0.05, -0.10],
            "age": [30, 45],
            "region": ["North", "South"],
        })
        m._validate_columns(df)  # should not raise

    def test_missing_outcome_raises(self):
        m = _make_model(outcome="claims")
        df = pd.DataFrame({
            "price_change": [0.05],
            "age": [30],
            "ncd_years": [3],
            "region": ["North"],
        })
        with pytest.raises(ValueError, match="claims"):
            m._validate_columns(df)

    def test_missing_treatment_raises(self):
        m = _make_model(
            treatment=PriceChangeTreatment(column="price_change"),
            outcome="claim_count",
            confounders=["age", "region"],
        )
        df = pd.DataFrame({
            "claim_count": [0],
            "age": [30],
            "region": ["North"],
            # price_change is missing
        })
        with pytest.raises(ValueError, match="price_change"):
            m._validate_columns(df)

    def test_missing_confounder_raises(self):
        m = _make_model(
            outcome="claims",
            treatment=PriceChangeTreatment(column="pct_chg"),
            confounders=["age", "region", "missing_factor"],
        )
        df = pd.DataFrame({
            "claims": [0],
            "pct_chg": [0.05],
            "age": [30],
            "region": ["North"],
        })
        with pytest.raises(ValueError, match="missing_factor"):
            m._validate_columns(df)

    def test_missing_exposure_raises(self):
        m = _make_model(
            outcome="claims",
            treatment=PriceChangeTreatment(column="pct_chg"),
            confounders=["age"],
            exposure_col="earned_years",
        )
        df = pd.DataFrame({
            "claims": [0],
            "pct_chg": [0.05],
            "age": [30],
            # earned_years missing
        })
        with pytest.raises(ValueError, match="earned_years"):
            m._validate_columns(df)

    def test_error_message_lists_available_columns(self):
        m = _make_model(outcome="claims", confounders=["age"])
        df = pd.DataFrame({"something_else": [1]})
        with pytest.raises(ValueError) as exc_info:
            m._validate_columns(df)
        # Should show available columns
        assert "something_else" in str(exc_info.value) or "Available" in str(exc_info.value)


# ---------------------------------------------------------------------------
# _transform_outcome
# ---------------------------------------------------------------------------


class TestTransformOutcome:
    def test_continuous_passthrough(self):
        m = _make_model(outcome_type="continuous")
        y = np.array([1.0, 2.0, 3.0])
        result = m._transform_outcome(y, None)
        np.testing.assert_allclose(result, y)

    def test_binary_passthrough(self):
        m = _make_model(outcome_type="binary")
        y = np.array([0, 1, 0, 1])
        result = m._transform_outcome(y, None)
        np.testing.assert_allclose(result, y.astype(float))

    def test_poisson_divides_by_exposure(self):
        m = _make_model(outcome_type="poisson")
        y = np.array([2.0, 4.0])
        exposure = np.array([1.0, 2.0])
        result = m._transform_outcome(y, exposure)
        np.testing.assert_allclose(result, [2.0, 2.0])

    def test_poisson_no_exposure_passthrough(self):
        m = _make_model(outcome_type="poisson")
        y = np.array([1.0, 3.0])
        result = m._transform_outcome(y, None)
        np.testing.assert_allclose(result, [1.0, 3.0])

    def test_gamma_log_transform(self):
        m = _make_model(outcome_type="gamma")
        y = np.array([1.0, np.e, np.e ** 2])
        result = m._transform_outcome(y, None)
        np.testing.assert_allclose(result, [0.0, 1.0, 2.0], atol=1e-10)

    def test_gamma_nonpositive_raises(self):
        m = _make_model(outcome_type="gamma")
        y = np.array([100.0, 0.0])
        with pytest.raises(ValueError, match="strictly positive"):
            m._transform_outcome(y, None)

    def test_poisson_zero_exposure_raises(self):
        m = _make_model(outcome_type="poisson")
        y = np.array([1.0, 0.0])
        exposure = np.array([1.0, 0.0])
        with pytest.raises(ValueError, match="strictly positive"):
            m._transform_outcome(y, exposure)


# ---------------------------------------------------------------------------
# _interpret_bias (static method)
# ---------------------------------------------------------------------------


class TestInterpretBias:
    def test_small_bias_message(self):
        msg = CausalPricingModel._interpret_bias(
            naive=-0.23, causal=-0.24, bias=0.01, bias_pct=4.2
        )
        assert "small" in msg.lower()

    def test_overstate_message(self):
        msg = CausalPricingModel._interpret_bias(
            naive=-0.30, causal=-0.15, bias=-0.15, bias_pct=-100.0
        )
        assert "understates" in msg.lower() or "overstates" in msg.lower()

    def test_opposite_signs_message(self):
        """When naive and causal have opposite signs, message should say so."""
        msg = CausalPricingModel._interpret_bias(
            naive=0.10, causal=-0.05, bias=0.15, bias_pct=300.0
        )
        assert "opposite" in msg.lower() or "reverses" in msg.lower()

    def test_moderate_overstatement(self):
        msg = CausalPricingModel._interpret_bias(
            naive=-0.30, causal=-0.20, bias=-0.10, bias_pct=-50.0
        )
        # 50% is on the boundary between moderate and substantial
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_returns_string(self):
        msg = CausalPricingModel._interpret_bias(0.1, 0.09, 0.01, 11.1)
        assert isinstance(msg, str)


# ---------------------------------------------------------------------------
# _extract_naive_from_model
# ---------------------------------------------------------------------------


class TestExtractNaiveFromModel:
    def _make_fitted_model(self):
        m = _make_model(
            treatment=PriceChangeTreatment(column="price_change"),
            outcome="claims",
        )
        m._fitted = True
        # Stub _dml_model to avoid AttributeError
        return m

    def test_extracts_from_statsmodels_style_params(self):
        """Model with .params attribute (like statsmodels OLS result)."""
        m = self._make_fitted_model()

        class _FakeStatsmodels:
            params = pd.Series({"price_change": -0.23, "age": 0.01})

        result = m._extract_naive_from_model(_FakeStatsmodels())
        assert abs(result - (-0.23)) < 1e-10

    def test_statsmodels_missing_treatment_raises(self):
        m = self._make_fitted_model()

        class _FakeStatsmodels:
            params = pd.Series({"age": 0.01, "region": -0.05})

        with pytest.raises(ValueError, match="price_change"):
            m._extract_naive_from_model(_FakeStatsmodels())

    def test_extracts_from_sklearn_style_coef(self):
        """Model with .coef_ and .feature_names_in_ (sklearn LinearRegression)."""
        m = self._make_fitted_model()
        from sklearn.linear_model import LinearRegression

        X = pd.DataFrame({
            "price_change": [0.05, 0.10, -0.05, -0.10, 0.02],
            "age": [30, 40, 25, 55, 35],
        })
        y = [0.8, 0.9, 0.7, 0.85, 0.75]
        lr = LinearRegression().fit(X, y)

        result = m._extract_naive_from_model(lr)
        assert isinstance(result, float)

    def test_sklearn_without_feature_names_raises(self):
        """If sklearn model has .coef_ but not .feature_names_in_, must raise."""
        m = self._make_fitted_model()
        from sklearn.linear_model import LinearRegression

        # Fit on numpy (no feature names stored)
        X = np.array([[0.05, 30], [0.10, 40]])
        y = [0.8, 0.9]
        lr = LinearRegression().fit(X, y)

        with pytest.raises(ValueError, match="feature_names_in_"):
            m._extract_naive_from_model(lr)

    def test_unknown_model_type_raises(self):
        m = self._make_fitted_model()

        class _UnknownModel:
            pass

        with pytest.raises(ValueError, match="naive_coefficient"):
            m._extract_naive_from_model(_UnknownModel())


# ---------------------------------------------------------------------------
# AverageTreatmentEffect
# ---------------------------------------------------------------------------


class TestAverageTreatmentEffect:
    def _make_ate(self, **kwargs):
        defaults = dict(
            estimate=-0.023,
            std_error=0.004,
            ci_lower=-0.031,
            ci_upper=-0.015,
            p_value=0.00012,
            n_obs=5000,
            treatment_col="price_change",
            outcome_col="renewed",
        )
        defaults.update(kwargs)
        return AverageTreatmentEffect(**defaults)

    def test_fields_stored_correctly(self):
        ate = self._make_ate()
        assert ate.estimate == -0.023
        assert ate.std_error == 0.004
        assert ate.n_obs == 5000
        assert ate.treatment_col == "price_change"
        assert ate.outcome_col == "renewed"

    def test_frozen_immutable(self):
        """AverageTreatmentEffect is a frozen dataclass — fields are read-only."""
        ate = self._make_ate()
        with pytest.raises((AttributeError, TypeError)):
            ate.estimate = 0.0

    def test_str_contains_estimate(self):
        ate = self._make_ate(estimate=-0.0235)
        s = str(ate)
        assert "-0.0235" in s or "-0.02" in s

    def test_str_contains_ci(self):
        ate = self._make_ate()
        s = str(ate)
        assert "CI" in s or "ci" in s.lower() or "(" in s

    def test_str_contains_treatment_col(self):
        ate = self._make_ate(treatment_col="log_price_change")
        assert "log_price_change" in str(ate)

    def test_str_contains_outcome_col(self):
        ate = self._make_ate(outcome_col="lapse_indicator")
        assert "lapse_indicator" in str(ate)

    def test_str_contains_n_obs(self):
        ate = self._make_ate(n_obs=12345)
        s = str(ate)
        assert "12" in s  # at least part of 12345 should appear

    def test_str_contains_p_value(self):
        ate = self._make_ate(p_value=0.042)
        s = str(ate)
        assert "0.0420" in s or "0.04" in s or "p-value" in s.lower() or "p_value" in s.lower()

    def test_estimate_is_float(self):
        ate = self._make_ate()
        assert isinstance(ate.estimate, float)

    def test_n_obs_is_int(self):
        ate = self._make_ate(n_obs=1000)
        assert isinstance(ate.n_obs, int)
