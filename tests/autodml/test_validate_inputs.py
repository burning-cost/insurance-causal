"""
Tests for the validate_inputs() function in _crossfit.py.

These tests guard against regressions where NaN inputs silently propagate
through the autodml pipeline and produce numerically incorrect results
without any informative error message.
"""
import numpy as np
import pytest

from insurance_causal.autodml._crossfit import validate_inputs


def _clean(n=100, p=4, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    D = rng.uniform(200, 600, n)
    Y = -0.002 * D + rng.randn(n) * 0.1
    return X, D, Y


class TestValidateInputs:
    def test_clean_inputs_pass(self):
        X, D, Y = _clean()
        validate_inputs(X, D, Y)  # must not raise

    def test_nan_in_X_raises(self):
        X, D, Y = _clean()
        X[5, 2] = np.nan
        with pytest.raises(ValueError, match="X contains"):
            validate_inputs(X, D, Y)

    def test_nan_in_D_raises(self):
        X, D, Y = _clean()
        D[10] = np.nan
        with pytest.raises(ValueError, match="D.*treatment.*NaN"):
            validate_inputs(X, D, Y)

    def test_nan_in_Y_raises_by_default(self):
        X, D, Y = _clean()
        Y[0] = np.nan
        with pytest.raises(ValueError, match="Y.*outcome.*NaN"):
            validate_inputs(X, D, Y)

    def test_nan_in_Y_allowed_when_flag_set(self):
        X, D, Y = _clean()
        Y[0] = np.nan
        validate_inputs(X, D, Y, allow_nan_Y=True)  # must not raise

    def test_multiple_nans_in_X_reports_count(self):
        X, D, Y = _clean()
        X[0, 0] = np.nan
        X[3, 1] = np.nan
        X[7, 2] = np.nan
        with pytest.raises(ValueError) as exc_info:
            validate_inputs(X, D, Y)
        assert "3" in str(exc_info.value)

    def test_error_messages_mention_fix(self):
        """Error messages should give actionable guidance."""
        X, D, Y = _clean()
        X[0, 0] = np.nan
        with pytest.raises(ValueError) as exc_info:
            validate_inputs(X, D, Y)
        msg = str(exc_info.value)
        # Should mention impute or drop
        assert "impute" in msg.lower() or "drop" in msg.lower()
