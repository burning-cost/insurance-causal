"""
Tests for the diagnostics module.

sensitivity_analysis() has no model dependency and can be tested locally.
confounding_bias_report() and cate_by_decile() require a fitted model —
those are tested on Databricks with synthetic data.
"""

import numpy as np
import pandas as pd
import pytest

from insurance_causal.diagnostics import sensitivity_analysis


class TestSensitivityAnalysis:
    """
    Test the Rosenbaum-style sensitivity analysis.

    We test with a negative ATE (typical for price elasticity: price increase
    reduces renewal) and verify the output structure and monotonicity properties.
    """

    def test_output_structure(self):
        result = sensitivity_analysis(ate=-0.023, se=0.004)
        expected_cols = {
            "gamma", "bias_bound", "bound_lower", "bound_upper",
            "ci_lower", "ci_upper", "conclusion_holds", "p_value_worst_case",
        }
        assert set(result.columns) == expected_cols

    def test_default_gamma_values(self):
        result = sensitivity_analysis(ate=-0.023, se=0.004)
        assert len(result) == 7  # 7 default gamma values
        assert result["gamma"].iloc[0] == 1.0

    def test_custom_gamma_values(self):
        result = sensitivity_analysis(ate=0.1, se=0.02, gamma_values=[1.0, 2.0, 3.0])
        assert len(result) == 3
        assert list(result["gamma"]) == [1.0, 2.0, 3.0]

    def test_gamma_1_is_baseline(self):
        """At Γ=1 (no unobserved confounding), bias bound should be zero."""
        result = sensitivity_analysis(ate=-0.05, se=0.01)
        row = result[result["gamma"] == 1.0].iloc[0]
        assert row["bias_bound"] == pytest.approx(0.0, abs=1e-10)
        assert row["bound_lower"] == pytest.approx(-0.05, abs=1e-10)
        assert row["bound_upper"] == pytest.approx(-0.05, abs=1e-10)

    def test_conclusion_holds_at_gamma_1_for_significant_ate(self):
        """A significant ATE should survive at Γ=1 (no confounding)."""
        # ATE=-0.05, SE=0.005 → z = 10, clearly significant
        result = sensitivity_analysis(ate=-0.05, se=0.005)
        row = result[result["gamma"] == 1.0].iloc[0]
        assert row["conclusion_holds"] is True

    def test_conclusion_overturned_at_large_gamma(self):
        """A weak ATE should be overturned by large unobserved confounding."""
        # ATE=-0.02, SE=0.01 → marginal significance; should fail at large Γ
        result = sensitivity_analysis(ate=-0.02, se=0.01, gamma_values=[1.0, 3.0])
        row_gamma3 = result[result["gamma"] == 3.0].iloc[0]
        assert row_gamma3["conclusion_holds"] is False

    def test_bias_bound_monotone_in_gamma(self):
        """Bias bound should increase with Γ."""
        result = sensitivity_analysis(ate=-0.05, se=0.01)
        bounds = result["bias_bound"].values
        assert np.all(np.diff(bounds) >= 0)

    def test_positive_ate(self):
        """Works correctly with positive ATE (e.g. telematics effect on claims)."""
        result = sensitivity_analysis(ate=0.10, se=0.02)
        row_gamma1 = result[result["gamma"] == 1.0].iloc[0]
        assert row_gamma1["conclusion_holds"] is True

    def test_returns_dataframe(self):
        result = sensitivity_analysis(ate=-0.03, se=0.005)
        assert isinstance(result, pd.DataFrame)
