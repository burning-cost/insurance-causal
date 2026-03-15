"""
Tests for the diagnostics module.

sensitivity_analysis() was redesigned in v0.2.2 — the previous implementation
used a formula (bias_bound = log(Gamma) * SE) that has no statistical basis.
It now raises NotImplementedError.

confounding_bias_report() and cate_by_decile() require a fitted model —
those are tested on Databricks with synthetic data.
"""

import pytest

from insurance_causal.diagnostics import sensitivity_analysis


class TestSensitivityAnalysis:
    """
    Tests for the redesigned sensitivity_analysis function.

    In v0.2.2 the previous Rosenbaum implementation was removed because
    the formula (bias_bound = log(Gamma) * SE) has no statistical basis.
    The function now raises NotImplementedError to prevent users from
    getting misleading results.

    These tests verify the new behaviour and guard against re-introduction
    of the old formula.
    """

    def test_raises_not_implemented(self):
        """sensitivity_analysis() must raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            sensitivity_analysis(ate=-0.023, se=0.004)

    def test_raises_for_any_inputs(self):
        """Should raise regardless of ate/se values."""
        with pytest.raises(NotImplementedError):
            sensitivity_analysis(ate=0.0, se=0.0)

    def test_raises_with_gamma_values(self):
        """Should raise regardless of gamma_values parameter."""
        with pytest.raises(NotImplementedError):
            sensitivity_analysis(ate=-0.05, se=0.01, gamma_values=[1.0, 2.0, 3.0])

    def test_error_message_references_alternatives(self):
        """Error message must point users to valid alternatives."""
        with pytest.raises(NotImplementedError) as exc_info:
            sensitivity_analysis(ate=-0.023, se=0.004)
        msg = str(exc_info.value)
        assert "sensitivity_bounds" in msg or "confounding_bias_report" in msg

    def test_old_formula_not_in_source(self):
        """The log(Gamma)*SE formula must not be present in the implementation."""
        import inspect
        src = inspect.getsource(sensitivity_analysis)
        assert "np.log(gamma) * se" not in src, (
            "The formula np.log(gamma) * se must not appear in sensitivity_analysis. "
            "It was removed in v0.2.2 because it has no statistical basis."
        )
