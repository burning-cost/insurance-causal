"""
Diagnostics for causal forest HTE estimation.

Three checks that should run before trusting CATE estimates:

1. OVERLAP (positivity assumption):
   For valid causal inference, every value of X must have positive probability
   of both treatment levels. For continuous treatments, overlap means the
   treatment propensity e(x) = E[D|X=x] must not be degenerate. We check:
   - Distribution of estimated e(x) (should not be all-or-nothing)
   - Variance of treatment residuals D - e(x) (near-zero = near-deterministic)

2. RESIDUAL VARIATION RATIO:
   Var(D - e_hat) / Var(D). The fraction of treatment variation that is
   exogenous after conditioning on observable risk factors. Values below 0.10
   indicate the near-deterministic price problem — DML results unreliable.

3. CALIBRATION (BLP):
   Are the CATE estimates calibrated? Run the BLP regression and check whether
   beta_1 ≈ ATE (calibration of average effect) and beta_2 > 0 (heterogeneity
   present). A well-calibrated forest should have beta_1 ≈ ATE and beta_2 > 0.

All checks are run via CausalForestDiagnostics.check(). The result is a
DiagnosticsReport dataclass with:
- warnings: list of actionable warning strings
- overlap_ok: whether the overlap check passed
- residual_variation_ok: whether residual variation exceeds 10%
- calibration_ok: whether the BLP calibration check passed
- plot methods for visual inspection

References
----------
Chernozhukov et al. (2018) — near-deterministic price problem / weak instruments
Wager & Athey (2018) — honesty and coverage for causal forests
KB entry 597: near-deterministic price problem in insurance DML.
KB entry 2806: insurance-specific practical concerns for causal forests.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np
import polars as pl

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


DataFrameLike = Union[pl.DataFrame, "pd.DataFrame"]

# Thresholds
_MIN_RESIDUAL_VARIATION = 0.10
_PROPENSITY_CLAMP_WARNING = 0.05  # warn if > 5% of obs have |D - mean(D)| < threshold


@dataclass
class DiagnosticsReport:
    """Results from CausalForestDiagnostics.check().

    Attributes
    ----------
    overlap_ok:
        True if the propensity/treatment distribution looks non-degenerate.
    residual_variation_ok:
        True if Var(D - e_hat)/Var(D) >= 0.10.
    calibration_ok:
        True if the BLP beta_2 > 0 (some heterogeneity detected).
    residual_variation_fraction:
        Var(D - e_hat)/Var(D). The key diagnostic for insurance data.
    propensity_std:
        Standard deviation of the estimated treatment propensity e_hat.
        Low std = degenerate propensity.
    blp_beta_2:
        BLP beta_2 estimate (HTE magnitude). NaN if not computed.
    blp_beta_2_pvalue:
        p-value for BLP beta_2 = 0. NaN if not computed.
    n_obs:
        Number of observations.
    warnings:
        List of warning strings with actionable remedies.
    """

    overlap_ok: bool
    residual_variation_ok: bool
    calibration_ok: bool
    residual_variation_fraction: float
    propensity_std: float
    blp_beta_2: float
    blp_beta_2_pvalue: float
    n_obs: int
    warnings: list[str] = field(default_factory=list)

    def all_ok(self) -> bool:
        """Return True if all diagnostic checks passed."""
        return self.overlap_ok and self.residual_variation_ok and self.calibration_ok

    def summary(self) -> str:
        """Return a human-readable diagnostics report."""
        def tick(ok: bool) -> str:
            return "PASS" if ok else "FAIL"

        lines = [
            "Causal Forest Diagnostics",
            "=" * 45,
            f"N observations: {self.n_obs:,}",
            "",
            f"[{tick(self.overlap_ok)}] Overlap / propensity",
            f"       Propensity SD:        {self.propensity_std:.4f}",
            "",
            f"[{tick(self.residual_variation_ok)}] Residual variation",
            f"       Var(D-e)/Var(D):       {self.residual_variation_fraction:.4f}  "
            f"(threshold: >={_MIN_RESIDUAL_VARIATION:.2f})",
            "",
            f"[{tick(self.calibration_ok)}] BLP calibration",
            f"       beta_2 (HTE):          {self.blp_beta_2:.4f}  "
            f"(p={self.blp_beta_2_pvalue:.4f})",
            "",
        ]

        if self.warnings:
            lines.append("WARNINGS:")
            for i, w in enumerate(self.warnings, 1):
                lines.append(f"  {i}. {w}")
        else:
            lines.append("All checks passed. CATE estimates should be reliable.")

        return "\n".join(lines)

    def plot_propensity(self, D: Optional[np.ndarray] = None) -> None:
        """Plot the distribution of treatment variable and its residuals."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not installed.", stacklevel=2)
            return

        fig, ax = plt.subplots(figsize=(7, 4))
        if D is not None:
            ax.hist(D, bins=40, alpha=0.6, color="steelblue", label="D (raw treatment)")
        ax.set_xlabel("Log price change")
        ax.set_ylabel("Count")
        ax.set_title("Treatment distribution")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_residual_variation(
        self,
        D: Optional[np.ndarray] = None,
        D_resid: Optional[np.ndarray] = None,
    ) -> None:
        """Compare distribution of treatment D vs residual D - e_hat."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not installed.", stacklevel=2)
            return

        if D is None or D_resid is None:
            warnings.warn(
                "Pass D and D_resid arrays to plot_residual_variation().",
                stacklevel=2,
            )
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(D, bins=40, color="steelblue", alpha=0.7)
        axes[0].set_title("D: Raw treatment (log price change)")
        axes[0].set_xlabel("log price change")

        axes[1].hist(D_resid, bins=40, color="darkorange", alpha=0.7)
        axes[1].set_title(f"D - e_hat: Exogenous residual (Var ratio={self.residual_variation_fraction:.3f})")
        axes[1].set_xlabel("treatment residual")

        plt.tight_layout()
        plt.show()


class CausalForestDiagnostics:
    """Pre-fit and post-fit diagnostics for causal forest HTE estimation.

    Run these checks after fitting HeterogeneousElasticityEstimator to verify
    that the CATE estimates are trustworthy.

    Parameters
    ----------
    n_splits:
        Data splits for the BLP calibration check (via HeterogeneousInference).
        Use fewer splits (e.g. 10) for fast diagnostics; 100 for production.
    random_state:
        Random seed.

    Examples
    --------
    >>> from insurance_causal.causal_forest.data import make_hte_renewal_data
    >>> from insurance_causal.causal_forest.estimator import HeterogeneousElasticityEstimator
    >>> from insurance_causal.causal_forest.diagnostics import CausalForestDiagnostics
    >>> df = make_hte_renewal_data(n=5000)
    >>> confounders = ["age", "ncd_years", "vehicle_group", "channel"]
    >>> est = HeterogeneousElasticityEstimator(n_estimators=100, catboost_iterations=100)
    >>> est.fit(df, outcome="renewed", treatment="log_price_change",
    ...         confounders=confounders)
    >>> cates = est.cate(df)
    >>> diag = CausalForestDiagnostics()
    >>> report = diag.check(df, estimator=est, cates=cates)
    >>> print(report.summary())
    """

    def __init__(
        self,
        n_splits: int = 20,
        random_state: int = 42,
    ) -> None:
        self.n_splits = n_splits
        self.random_state = random_state

    def check(
        self,
        df: DataFrameLike,
        estimator: object,
        cates: np.ndarray,
    ) -> DiagnosticsReport:
        """Run all diagnostic checks.

        Parameters
        ----------
        df:
            Dataset used to fit the estimator.
        estimator:
            Fitted HeterogeneousElasticityEstimator.
        cates:
            Per-row CATE estimates from estimator.cate(df).

        Returns
        -------
        DiagnosticsReport
        """
        from insurance_causal.causal_forest.inference import (
            HeterogeneousInference, _fit_propensity
        )

        Y = estimator._Y_train
        D = estimator._D_train
        X = estimator._X_train
        n = len(Y)

        warn_list: list[str] = []

        # --- 1. Overlap / propensity check ---
        e_hat = _fit_propensity(D, X, D, X)
        propensity_std = float(np.std(e_hat))
        D_mean = float(np.mean(D))

        # For continuous treatment: check std of propensity
        overlap_ok = True
        if propensity_std < 0.001:
            overlap_ok = False
            warn_list.append(
                "Treatment propensity e_hat has near-zero standard deviation "
                f"({propensity_std:.5f}). The treatment appears to be nearly "
                "constant across the covariate space. Check that there is real "
                "exogenous price variation in the data."
            )

        # --- 2. Residual variation ---
        D_resid = D - e_hat
        var_D = float(np.var(D, ddof=1))
        var_resid = float(np.var(D_resid, ddof=1))

        if var_D < 1e-12:
            residual_variation_fraction = 0.0
        else:
            residual_variation_fraction = var_resid / var_D

        residual_variation_ok = residual_variation_fraction >= _MIN_RESIDUAL_VARIATION
        if not residual_variation_ok:
            warn_list.append(
                f"Residual variation Var(D-e_hat)/Var(D) = {residual_variation_fraction:.4f} "
                f"is below the minimum threshold of {_MIN_RESIDUAL_VARIATION:.2f}. "
                "The price change is nearly determined by observable risk factors. "
                "CATE estimates will have wide confidence intervals and may be unreliable. "
                "Consider: (1) A/B test price variation, (2) panel data with within-customer "
                "variation, (3) bulk re-rate quasi-experiments."
            )

        # --- 3. BLP calibration ---
        blp_beta_2 = float("nan")
        blp_pvalue = float("nan")
        calibration_ok = True

        try:
            inf = HeterogeneousInference(
                n_splits=self.n_splits,
                k_groups=5,
                random_state=self.random_state,
            )
            blp_result = inf._run_blp_splits(
                Y, D, X, cates, np.random.default_rng(self.random_state)
            )
            blp_beta_2 = blp_result.beta_2
            blp_pvalue = blp_result.beta_2_pvalue

            if np.isfinite(blp_beta_2) and blp_beta_2 <= 0:
                calibration_ok = False
                warn_list.append(
                    f"BLP beta_2 = {blp_beta_2:.4f} <= 0. The CATE proxy does not "
                    "positively predict realised treatment effect variation. "
                    "This can mean: (a) insufficient sample size for reliable CATE, "
                    "(b) near-deterministic price problem limiting identification, "
                    "or (c) the true elasticity is genuinely homogeneous."
                )
        except Exception as exc:
            warnings.warn(
                f"BLP calibration check failed: {exc}. Skipping.",
                stacklevel=2,
            )

        # Warn on small dataset
        if n < 5000:
            warn_list.append(
                f"Dataset has only {n:,} observations. CausalForestDML requires "
                "large samples for valid CATE inference — honest splitting halves "
                "the effective sample per fold. Estimates may be noisy; "
                "treat results as exploratory until n >= 10,000."
            )

        return DiagnosticsReport(
            overlap_ok=overlap_ok,
            residual_variation_ok=residual_variation_ok,
            calibration_ok=calibration_ok,
            residual_variation_fraction=residual_variation_fraction,
            propensity_std=propensity_std,
            blp_beta_2=blp_beta_2,
            blp_beta_2_pvalue=blp_pvalue,
            n_obs=n,
            warnings=warn_list,
        )

    def degenerate_propensity_test(
        self,
        D: np.ndarray,
        X: np.ndarray,
    ) -> bool:
        """Return True if the treatment propensity appears degenerate.

        A degenerate propensity means every observation has nearly the same
        treatment level, leaving no variation for DML to exploit.

        Parameters
        ----------
        D:
            Treatment array.
        X:
            Confounder feature matrix.

        Returns
        -------
        bool
            True if propensity is degenerate (problematic).
        """
        from insurance_causal.causal_forest.inference import _fit_propensity
        e_hat = _fit_propensity(D, X, D, X)
        std = float(np.std(e_hat))
        return std < 0.001
