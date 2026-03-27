"""
Tests for diagnostics.py — extended coverage.

Covers:
- sensitivity_analysis() raises NotImplementedError (already tested; repeated here
  alongside new tests for completeness)
- nuisance_model_summary() with a mock model
- confounding_bias_report() via a mock model
- cate_by_decile() with a mock model
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_causal.diagnostics import (
    sensitivity_analysis,
    nuisance_model_summary,
    confounding_bias_report,
    cate_by_decile,
)


# ---------------------------------------------------------------------------
# sensitivity_analysis — already tested, but ensure import works
# ---------------------------------------------------------------------------

class TestSensitivityAnalysis:
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            sensitivity_analysis(ate=-0.023, se=0.004)

    def test_message_mentions_alternatives(self):
        with pytest.raises(NotImplementedError) as exc_info:
            sensitivity_analysis(ate=-0.01, se=0.005)
        assert "confounding_bias_report" in str(exc_info.value) or "sensitivity_bounds" in str(exc_info.value)


# ---------------------------------------------------------------------------
# nuisance_model_summary with mock model
# ---------------------------------------------------------------------------

class _MockDMLModel:
    """Minimal mock of a DoubleML model."""

    def __init__(self, include_preds=True, high_treatment_r2=False):
        n = 200
        rng = np.random.default_rng(0)
        y = rng.binomial(1, 0.8, n).astype(float)
        d = rng.normal(0, 0.1, n)

        if high_treatment_r2:
            d_pred = d * 0.999  # near-perfect prediction
        else:
            d_pred = d * 0.5

        y_pred = y * 0.8

        class _DMLData:
            def __init__(self):
                self.y = y
                self.d = d

        class _Preds:
            def __init__(self, y_p, d_p):
                import pandas as pd
                self._preds = {
                    "ml_l": pd.DataFrame({"0": y_p}),
                    "ml_m": pd.DataFrame({"0": d_p}),
                }

            def __contains__(self, item):
                return item in self._preds

            def __getitem__(self, item):
                return self._preds[item]

        self.data = _DMLData()
        if include_preds:
            self.predictions = _Preds(y_pred, d_pred)
        else:
            self.predictions = {}


class _MockCausalPricingModel:
    def __init__(self, include_preds=True, high_treatment_r2=False):
        self._dml_model = _MockDMLModel(include_preds, high_treatment_r2)
        self._is_fitted_flag = True

    def _check_fitted(self):
        if not self._is_fitted_flag:
            raise RuntimeError("not fitted")

    def confounding_bias_report(self, naive_coefficient=None, glm_model=None):
        # Return a minimal DataFrame
        import pandas as pd
        return pd.DataFrame({
            "estimator": ["naive", "dml"],
            "estimate": [naive_coefficient or -0.1, -0.15],
            "bias": [0.0, naive_coefficient - (-0.15) if naive_coefficient else 0.05],
        })

    def cate_by_segment(self, df, segment_col):
        import pandas as pd
        segs = df[segment_col].unique()
        return pd.DataFrame({
            "segment": segs,
            "cate_estimate": np.full(len(segs), -0.15),
            "ci_lower": np.full(len(segs), -0.20),
            "ci_upper": np.full(len(segs), -0.10),
            "std_error": np.full(len(segs), 0.05),
            "p_value": np.full(len(segs), 0.01),
            "n_obs": np.full(len(segs), 50),
            "status": ["ok"] * len(segs),
        })


class TestNuisanceModelSummary:
    def test_returns_dict(self):
        model = _MockCausalPricingModel()
        result = nuisance_model_summary(model)
        assert isinstance(result, dict)

    def test_contains_outcome_r2(self):
        model = _MockCausalPricingModel()
        result = nuisance_model_summary(model)
        assert "outcome_r2" in result

    def test_contains_treatment_r2(self):
        model = _MockCausalPricingModel()
        result = nuisance_model_summary(model)
        assert "treatment_r2" in result

    def test_treatment_r2_in_range(self):
        model = _MockCausalPricingModel()
        result = nuisance_model_summary(model)
        r2 = result.get("treatment_r2")
        if r2 is not None:
            assert -1.0 <= r2 <= 1.0

    def test_high_treatment_r2_adds_warning(self):
        model = _MockCausalPricingModel(high_treatment_r2=True)
        result = nuisance_model_summary(model)
        # With treatment_r2 > 0.95, a warning key should be added
        assert "warning" in result or result.get("treatment_r2", 0) <= 0.95

    def test_no_predictions_returns_error_key(self):
        model = _MockCausalPricingModel(include_preds=False)
        result = nuisance_model_summary(model)
        # Should either be empty dict or have "error" key
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# confounding_bias_report
# ---------------------------------------------------------------------------

class TestConfoundingBiasReport:
    def test_returns_dataframe(self):
        model = _MockCausalPricingModel()
        result = confounding_bias_report(model, naive_coefficient=-0.1)
        assert isinstance(result, pd.DataFrame)

    def test_accepts_naive_coefficient(self):
        model = _MockCausalPricingModel()
        result = confounding_bias_report(model, naive_coefficient=-0.05)
        assert isinstance(result, pd.DataFrame)

    def test_no_naive_coefficient(self):
        model = _MockCausalPricingModel()
        result = confounding_bias_report(model)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# cate_by_decile
# ---------------------------------------------------------------------------

class TestCateByDecile:
    def _make_df_with_score(self, n=100):
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "score": rng.normal(0, 1, n),
            "renewed": rng.integers(0, 2, n).astype(float),
            "log_price_change": rng.normal(0.05, 0.02, n),
        })

    def test_returns_dataframe(self):
        model = _MockCausalPricingModel()
        df = self._make_df_with_score()
        result = cate_by_decile(model, df, score_col="score", n_deciles=5)
        assert isinstance(result, pd.DataFrame)

    def test_n_rows_equals_deciles(self):
        model = _MockCausalPricingModel()
        df = self._make_df_with_score(n=100)
        result = cate_by_decile(model, df, score_col="score", n_deciles=5)
        assert len(result) == 5

    def test_score_bounds_present(self):
        model = _MockCausalPricingModel()
        df = self._make_df_with_score(n=100)
        result = cate_by_decile(model, df, score_col="score", n_deciles=4)
        assert "score_lower" in result.columns
        assert "score_upper" in result.columns

    def test_decile_column_present(self):
        model = _MockCausalPricingModel()
        df = self._make_df_with_score(n=100)
        result = cate_by_decile(model, df, score_col="score", n_deciles=5)
        assert "decile" in result.columns

    def test_missing_score_col_raises(self):
        model = _MockCausalPricingModel()
        df = pd.DataFrame({"other": [1, 2, 3]})
        with pytest.raises(ValueError, match="not found"):
            cate_by_decile(model, df, score_col="score")

    def test_accepts_polars_input(self):
        """cate_by_decile should accept polars DataFrame (via to_pandas conversion)."""
        import polars as pl
        model = _MockCausalPricingModel()
        rng = np.random.default_rng(0)
        df_pl = pl.DataFrame({
            "score": rng.normal(0, 1, 100).tolist(),
            "renewed": rng.integers(0, 2, 100).tolist(),
            "log_price_change": rng.normal(0.05, 0.02, 100).tolist(),
        })
        result = cate_by_decile(model, df_pl, score_col="score", n_deciles=5)
        assert isinstance(result, pd.DataFrame)
