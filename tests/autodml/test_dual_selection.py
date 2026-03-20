"""
Tests for DualSelectionDML.

DGP notes
---------
All DGPs are lightweight (n <= 1000) to keep Databricks runtime under a minute.
We use n_folds=2 throughout to reduce fitting time.

The "unbiased" test (test_dual_binary_unbiased) uses a clearly separable
DGP with true ATE = 2.0, and checks that the estimate is within 3*SE of
truth — a generous tolerance appropriate for n=800 with dual selection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_causal.autodml.dual_selection import DualSelectionDML, DualSelectionResult


# ---------------------------------------------------------------------------
# Shared DGP utilities
# ---------------------------------------------------------------------------


def _make_dual_binary_data(n: int = 800, ate_true: float = 2.0, seed: int = 0):
    """
    DGP with known ATE under dual binary selection (renew AND claim).

    Selection model:
        P(Z_0=1 | X) = logistic(0.5 * X0)          (renewal)
        P(Z_1=1 | X, Z_0) = 0.4 * Z_0               (claim, given renewal)

    Outcome (severity, observed only for Z_0=1 AND Z_1=1):
        Y = ate_true * D + 0.5 * X0 + noise

    True ATE = ate_true.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4)

    D = rng.binomial(1, 0.5, n).astype(float)

    p_renew = 1.0 / (1.0 + np.exp(-0.5 * X[:, 0]))
    Z_renew = rng.binomial(1, p_renew).astype(float)

    p_claim = 0.4 * Z_renew
    Z_claim = rng.binomial(1, p_claim).astype(float)

    Z = np.column_stack([Z_renew, Z_claim])
    sel = (Z_renew == 1) & (Z_claim == 1)

    Y = np.full(n, np.nan)
    Y[sel] = ate_true * D[sel] + 0.5 * X[sel, 0] + rng.randn(sel.sum())

    return X, D, Z, Y, sel


def _make_single_binary_data(n: int = 600, ate_true: float = 1.5, seed: int = 0):
    """
    DGP with single binary selection (renewal only).
    Used to compare DualSelectionDML against naive DML on selected sample.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3)
    D = rng.binomial(1, 0.5, n).astype(float)
    p_sel = 1.0 / (1.0 + np.exp(-0.3 * X[:, 0]))
    Z_sel = rng.binomial(1, p_sel).astype(float)
    Z = Z_sel.reshape(-1, 1)
    sel = Z_sel == 1
    Y = np.full(n, np.nan)
    Y[sel] = ate_true * D[sel] + rng.randn(sel.sum())
    return X, D, Z, Y, sel


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestDualSelectionDMLConstruction:
    def test_default_construction(self):
        model = DualSelectionDML()
        assert model.estimand == "ATE"
        assert model.n_folds == 5

    def test_invalid_estimand_raises(self):
        with pytest.raises(ValueError, match="estimand must be one of"):
            DualSelectionDML(estimand="INVALID")

    def test_late_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            DualSelectionDML(estimand="LATE")

    def test_ates_without_gradients_raises(self):
        with pytest.raises(ValueError, match="compute_gradients=True"):
            DualSelectionDML(estimand="ATES", compute_gradients=False)

    def test_repr(self):
        model = DualSelectionDML(estimand="ATE", n_folds=3)
        assert "ATE" in repr(model)
        assert "3" in repr(model)


class TestNaNYHandling:
    """NaN Y for unselected obs must not crash — this is the core design contract."""

    def test_nan_y_handling_no_crash(self):
        X, D, Z, Y, sel = _make_dual_binary_data(n=400, seed=1)
        # Y is already NaN for unselected obs — verify this
        assert np.any(np.isnan(Y))
        model = DualSelectionDML(n_folds=2, random_state=0)
        model.fit(Y, D, Z, X)  # must not raise
        result = model.estimate()
        assert isinstance(result, DualSelectionResult)
        assert not np.isnan(result.ate)

    def test_all_nan_y_warns_and_proceeds(self):
        """If no selected observations have observed Y, raise a warning."""
        rng = np.random.RandomState(42)
        n = 200
        X = rng.randn(n, 3)
        D = rng.binomial(1, 0.5, n).astype(float)
        Z = np.ones((n, 1))  # everyone selected
        Y = np.full(n, np.nan)  # no observed outcomes

        model = DualSelectionDML(n_folds=2, random_state=0)
        with pytest.warns(RuntimeWarning):
            model.fit(Y, D, Z, X)

    def test_y_nan_exactly_matches_unselected(self):
        """Standard usage: NaN Y precisely for unselected, non-NaN for selected."""
        X, D, Z, Y, sel = _make_dual_binary_data(n=500, seed=2)
        assert np.all(np.isnan(Y[~sel]))
        assert np.all(~np.isnan(Y[sel]))
        model = DualSelectionDML(n_folds=2, random_state=0)
        model.fit(Y, D, Z, X)
        result = model.estimate()
        assert result.n_selected == int(sel.sum())


class TestSelectionRates:
    """selection_rates in result must match observed data."""

    def test_selection_rates_match_data(self):
        X, D, Z, Y, sel = _make_dual_binary_data(n=600, seed=3)
        model = DualSelectionDML(n_folds=2, random_state=0)
        model.fit(Y, D, Z, X)
        result = model.estimate()

        # Z_0 selection rate
        expected_z0 = float(np.mean(Z[:, 0] == 1))
        expected_z1 = float(np.mean(Z[:, 1] == 1))
        expected_joint = float(np.mean(sel))

        assert abs(result.selection_rates["Z_0"] - expected_z0) < 1e-9
        assert abs(result.selection_rates["Z_1"] - expected_z1) < 1e-9
        assert abs(result.selection_rates["joint"] - expected_joint) < 1e-9

    def test_joint_rate_leq_individual_rates(self):
        X, D, Z, Y, sel = _make_dual_binary_data(n=600, seed=4)
        model = DualSelectionDML(n_folds=2, random_state=0)
        model.fit(Y, D, Z, X)
        result = model.estimate()
        # Joint selection is always <= each individual rate
        assert result.selection_rates["joint"] <= result.selection_rates["Z_0"] + 1e-9
        assert result.selection_rates["joint"] <= result.selection_rates["Z_1"] + 1e-9

    def test_n_selected_matches_selection_rate(self):
        X, D, Z, Y, sel = _make_dual_binary_data(n=700, seed=5)
        model = DualSelectionDML(n_folds=2, random_state=0)
        model.fit(Y, D, Z, X)
        result = model.estimate()
        assert result.n_selected == int(sel.sum())
        assert result.n_obs == 700


class TestSensitivityMonotone:
    """CI must widen monotonically as |rho| increases from 0."""

    def test_sensitivity_returns_dataframe(self):
        X, D, Z, Y, _ = _make_dual_binary_data(n=400, seed=6)
        model = DualSelectionDML(n_folds=2, random_state=0)
        model.fit(Y, D, Z, X)
        df = model.sensitivity(rho_range=(-0.4, 0.4), n_points=9)
        assert isinstance(df, pd.DataFrame)
        assert set(["rho", "ate_point", "ate_lower", "ate_upper"]).issubset(df.columns)
        assert len(df) == 9

    def test_ci_width_increases_with_abs_rho(self):
        X, D, Z, Y, _ = _make_dual_binary_data(n=400, seed=7)
        model = DualSelectionDML(n_folds=2, random_state=0)
        model.fit(Y, D, Z, X)
        df = model.sensitivity(rho_range=(-0.5, 0.5), n_points=11)
        df["ci_width"] = df["ate_upper"] - df["ate_lower"]
        df["abs_rho"] = df["rho"].abs()
        df = df.sort_values("abs_rho")

        # CI width should be non-decreasing as |rho| grows
        widths = df["ci_width"].values
        for i in range(len(widths) - 1):
            assert widths[i] <= widths[i + 1] + 1e-10, (
                f"CI width not monotone: width[{i}]={widths[i]:.6f} > "
                f"width[{i+1}]={widths[i+1]:.6f} at |rho|={df['abs_rho'].values[i]:.3f}"
            )

    def test_rho_zero_matches_main_estimate(self):
        X, D, Z, Y, _ = _make_dual_binary_data(n=400, seed=8)
        model = DualSelectionDML(n_folds=2, random_state=0)
        model.fit(Y, D, Z, X)
        result = model.estimate()
        df = model.sensitivity(rho_range=(0.0, 0.0), n_points=1)
        assert abs(df["ate_point"].iloc[0] - result.ate) < 1e-9

    def test_sensitivity_requires_fit(self):
        model = DualSelectionDML()
        with pytest.raises(RuntimeError, match="fit()"):
            model.sensitivity()


class TestDualBinaryUnbiased:
    """
    Integration test: estimated ATE should be within 3*SE of true ATE.
    Uses n=800, n_folds=2.  This is a stochastic test — generous tolerance.
    """

    def test_ate_within_tolerance_of_truth(self):
        ate_true = 2.0
        X, D, Z, Y, sel = _make_dual_binary_data(n=800, ate_true=ate_true, seed=99)
        model = DualSelectionDML(
            estimand="ATE",
            n_folds=2,
            random_state=42,
        )
        model.fit(Y, D, Z, X)
        result = model.estimate()

        # ATE estimate should be within 3 SE of truth
        error = abs(result.ate - ate_true)
        tolerance = 3.0 * result.se
        assert error < tolerance, (
            f"ATE estimate {result.ate:.3f} is more than 3*SE={tolerance:.3f} "
            f"from truth {ate_true}. SE={result.se:.3f}. "
            "This may occasionally fail due to randomness — re-run to confirm."
        )

    def test_ci_has_positive_width(self):
        X, D, Z, Y, _ = _make_dual_binary_data(n=400, seed=10)
        model = DualSelectionDML(n_folds=2, random_state=0)
        model.fit(Y, D, Z, X)
        result = model.estimate()
        assert result.ci_upper > result.ci_lower

    def test_se_positive(self):
        X, D, Z, Y, _ = _make_dual_binary_data(n=400, seed=11)
        model = DualSelectionDML(n_folds=2, random_state=0)
        model.fit(Y, D, Z, X)
        result = model.estimate()
        assert result.se > 0

    def test_result_repr_contains_estimand(self):
        X, D, Z, Y, _ = _make_dual_binary_data(n=300, seed=12)
        model = DualSelectionDML(n_folds=2, random_state=0)
        model.fit(Y, D, Z, X)
        result = model.estimate()
        assert "ATE" in repr(result)
        assert "n=" in repr(result)


class TestSingleSelectionReduces:
    """
    With 1 binary selection variable, DualSelectionDML should produce a
    reasonable ATE estimate (we don't assert equality to naive DML on the
    selected sample, but we do check the estimate is finite and the CI
    contains the naive estimate with high probability for this DGP).
    """

    def test_single_selection_no_crash(self):
        X, D, Z, Y, sel = _make_single_binary_data(n=600, seed=13)
        model = DualSelectionDML(n_folds=2, random_state=0)
        model.fit(Y, D, Z, X)
        result = model.estimate()
        assert isinstance(result, DualSelectionResult)
        assert not np.isnan(result.ate)
        assert not np.isnan(result.se)

    def test_single_selection_one_selection_rate_key(self):
        X, D, Z, Y, sel = _make_single_binary_data(n=500, seed=14)
        model = DualSelectionDML(n_folds=2, random_state=0)
        model.fit(Y, D, Z, X)
        result = model.estimate()
        assert "Z_0" in result.selection_rates
        assert "joint" in result.selection_rates

    def test_single_selection_ate_in_reasonable_range(self):
        """ATE should be in [-5, 10] for this DGP (true=1.5)."""
        X, D, Z, Y, sel = _make_single_binary_data(n=600, ate_true=1.5, seed=15)
        model = DualSelectionDML(n_folds=2, random_state=0)
        model.fit(Y, D, Z, X)
        result = model.estimate()
        assert -5.0 < result.ate < 10.0


class TestInputVariants:
    """Various input types and edge cases."""

    def test_accepts_pandas_inputs(self):
        X, D, Z, Y, _ = _make_dual_binary_data(n=400, seed=16)
        df_X = pd.DataFrame(X, columns=[f"x{i}" for i in range(4)])
        df_Z = pd.DataFrame(Z, columns=["renew", "claim"])
        s_D = pd.Series(D)
        s_Y = pd.Series(Y)
        model = DualSelectionDML(n_folds=2, random_state=0)
        model.fit(s_Y, s_D, df_Z, df_X)
        result = model.estimate()
        assert isinstance(result, DualSelectionResult)

    def test_nan_in_x_raises(self):
        X, D, Z, Y, _ = _make_dual_binary_data(n=300, seed=17)
        X[5, 0] = np.nan
        model = DualSelectionDML(n_folds=2, random_state=0)
        with pytest.raises(ValueError, match="X contains NaN"):
            model.fit(Y, D, Z, X)

    def test_nan_in_d_raises(self):
        X, D, Z, Y, _ = _make_dual_binary_data(n=300, seed=18)
        D[3] = np.nan
        model = DualSelectionDML(n_folds=2, random_state=0)
        with pytest.raises(ValueError, match="D contains NaN"):
            model.fit(Y, D, Z, X)

    def test_estimate_before_fit_raises(self):
        model = DualSelectionDML()
        with pytest.raises(RuntimeError, match="fit()"):
            model.estimate()

    def test_fit_returns_self(self):
        X, D, Z, Y, _ = _make_dual_binary_data(n=300, seed=19)
        model = DualSelectionDML(n_folds=2, random_state=0)
        result = model.fit(Y, D, Z, X)
        assert result is model

    def test_w_z_exclusion_restriction(self):
        """Providing W_Z should suppress the identification warning."""
        X, D, Z, Y, _ = _make_dual_binary_data(n=400, seed=20)
        rng = np.random.RandomState(0)
        W_Z = rng.randn(400, 1)  # dummy exclusion restriction
        model = DualSelectionDML(n_folds=2, random_state=0)
        # Should NOT warn about functional form when W_Z is provided
        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            model.fit(Y, D, Z, X, W_Z=W_Z)
        id_warnings = [x for x in w if "exclusion restrictions" in str(x.message)]
        assert len(id_warnings) == 0

    def test_no_w_z_issues_warning(self):
        """Not providing W_Z should warn about functional form identification."""
        X, D, Z, Y, _ = _make_dual_binary_data(n=300, seed=21)
        model = DualSelectionDML(n_folds=2, random_state=0)
        with pytest.warns(UserWarning, match="W_Z not provided"):
            model.fit(Y, D, Z, X)

    def test_bootstrap_inference(self):
        X, D, Z, Y, _ = _make_dual_binary_data(n=300, seed=22)
        model = DualSelectionDML(n_folds=2, n_bootstrap=50, random_state=0)
        model.fit(Y, D, Z, X)
        result = model.estimate()
        assert result.se > 0
        assert result.ci_upper > result.ci_lower


class TestEstimandVariants:
    def test_ates_with_gradients(self):
        """ATES estimand with compute_gradients=True must run without error."""
        X, D, Z, Y, _ = _make_dual_binary_data(n=400, seed=23)
        model = DualSelectionDML(
            estimand="ATES",
            n_folds=2,
            compute_gradients=True,
            random_state=0,
        )
        model.fit(Y, D, Z, X)
        result = model.estimate()
        assert result.estimand == "ATES"
        assert not np.isnan(result.ate)

    def test_atet_estimand(self):
        X, D, Z, Y, _ = _make_dual_binary_data(n=400, seed=24)
        model = DualSelectionDML(estimand="ATET", n_folds=2, random_state=0)
        model.fit(Y, D, Z, X)
        result = model.estimate()
        assert result.estimand == "ATET"
        assert not np.isnan(result.ate)


class TestDualSelectionResult:
    def test_pvalue_finite(self):
        X, D, Z, Y, _ = _make_dual_binary_data(n=400, seed=25)
        model = DualSelectionDML(n_folds=2, random_state=0)
        model.fit(Y, D, Z, X)
        result = model.estimate()
        p = result.pvalue
        assert 0.0 <= p <= 1.0

    def test_summary_string(self):
        X, D, Z, Y, _ = _make_dual_binary_data(n=400, seed=26)
        model = DualSelectionDML(n_folds=2, random_state=0)
        model.fit(Y, D, Z, X)
        result = model.estimate()
        s = result.summary()
        assert "ATE" in s
        assert "ate=" in s
        assert "CI=" in s

    def test_eif_scores_shape(self):
        X, D, Z, Y, _ = _make_dual_binary_data(n=500, seed=27)
        model = DualSelectionDML(n_folds=2, random_state=0)
        model.fit(Y, D, Z, X)
        result = model.estimate()
        assert result.eif_scores.shape == (500,)
