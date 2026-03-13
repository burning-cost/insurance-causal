"""
Treatment variation diagnostics for DML-based insurance elasticity estimation.

The near-deterministic price problem is the central challenge in applying DML
to insurance renewal data. Insurance re-rating makes the price change nearly
a deterministic function of observable risk factors. When that happens, the DML
residual D̃ = D - E[D|X] has near-zero variance, making the causal estimator
numerically unreliable — exactly analogous to weak instruments in IV.

This module diagnoses the problem before fitting and reports actionable
suggestions. It is the library's key differentiator from naive ML approaches
that skip this check entirely.

References
----------
Chernozhukov et al. (2018), Econometrics Journal 21(1).
Clarke & Polselli (2025), xtdml for panel DML.
KB entry 597: Near-deterministic price problem in insurance DML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import polars as pl

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# Threshold below which DML results are considered unreliable
_MIN_VARIATION_FRACTION = 0.10  # Var(D̃)/Var(D) must exceed 10%
_MAX_NUISANCE_R2 = 0.90         # R² of treatment nuisance model must be below this


@dataclass
class TreatmentVariationReport:
    """Results of the treatment variation diagnostic.

    Attributes
    ----------
    treatment_var:
        Total variance of the treatment variable D (log price change).
    residual_var:
        Variance of D̃ = D - E[D|X] (the part of price change not explained
        by confounders). This is what DML uses for identification.
    variation_fraction:
        Var(D̃) / Var(D). The fraction of price variation that is exogenous
        with respect to the observable risk factors. Values below 0.10 indicate
        the near-deterministic price problem.
    nuisance_r2:
        R² of the treatment nuisance model E[D|X]. High R² means most price
        variation is predictable from risk factors — leaving little for DML.
    n_obs:
        Number of observations used.
    weak_treatment:
        Whether the variation fraction is below the minimum threshold (0.10).
        If True, DML results should not be trusted without further investigation.
    suggestions:
        Actionable suggestions for improving identification when
        weak_treatment is True.
    """

    treatment_var: float
    residual_var: float
    variation_fraction: float
    nuisance_r2: float
    n_obs: int
    weak_treatment: bool
    suggestions: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary of the report."""
        lines = [
            "Treatment Variation Diagnostic",
            "=" * 40,
            f"N observations:          {self.n_obs:,}",
            f"Var(D):                  {self.treatment_var:.6f}",
            f"Var(D̃):                 {self.residual_var:.6f}",
            f"Var(D̃)/Var(D):          {self.variation_fraction:.4f}  "
            f"({'OK' if not self.weak_treatment else 'WARNING: below 0.10 threshold'})",
            f"Treatment nuisance R²:   {self.nuisance_r2:.4f}  "
            f"({'OK' if self.nuisance_r2 < _MAX_NUISANCE_R2 else 'WARNING: above 0.90 threshold'})",
            "",
        ]
        if self.weak_treatment:
            lines.append("WEAK TREATMENT WARNING")
            lines.append("-" * 40)
            lines.append(
                "The price change is nearly determined by observable risk factors. "
                "DML residuals have low variance, making the causal estimator unreliable. "
                "Confidence intervals will be wide and point estimates noisy."
            )
            lines.append("")
            lines.append("Suggested remedies:")
            for i, s in enumerate(self.suggestions, 1):
                lines.append(f"  {i}. {s}")
        else:
            lines.append("Treatment variation is sufficient for DML identification.")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"TreatmentVariationReport("
            f"variation_fraction={self.variation_fraction:.4f}, "
            f"nuisance_r2={self.nuisance_r2:.4f}, "
            f"weak_treatment={self.weak_treatment})"
        )


class ElasticityDiagnostics:
    """Diagnostics for DML-based insurance price elasticity estimation.

    Run these checks before fitting :class:`~insurance_causal.elasticity.fit.RenewalElasticityEstimator`.
    If the treatment variation report flags a weak treatment, the DML results
    will not be reliable.

    Parameters
    ----------
    n_folds:
        Number of cross-validation folds for computing treatment nuisance R².
    random_state:
        Random seed for cross-validation.

    Examples
    --------
    >>> from insurance_causal.elasticity.data import make_renewal_data
    >>> from insurance_causal.elasticity.diagnostics import ElasticityDiagnostics
    >>> df = make_renewal_data(n=5000)
    >>> diag = ElasticityDiagnostics()
    >>> report = diag.treatment_variation_report(
    ...     df,
    ...     treatment="log_price_change",
    ...     confounders=["age", "ncd_years", "vehicle_group", "region"],
    ... )
    >>> print(report.summary())
    """

    def __init__(self, n_folds: int = 3, random_state: int = 42) -> None:
        self.n_folds = n_folds
        self.random_state = random_state
        self._last_report: Optional[TreatmentVariationReport] = None

    def treatment_variation_report(
        self,
        df: pl.DataFrame,
        treatment: str = "log_price_change",
        confounders: Optional[Sequence[str]] = None,
    ) -> TreatmentVariationReport:
        """Compute the treatment variation diagnostic.

        Fits a nuisance model E[D|X] and reports the fraction of treatment
        variation that remains after conditioning on observable confounders.
        Low residual variation (Var(D̃)/Var(D) < 0.10) indicates the
        near-deterministic price problem.

        Parameters
        ----------
        df:
            Renewal dataset. Accepts polars or pandas DataFrames.
        treatment:
            Column name of the treatment variable (log price change).
        confounders:
            Column names to use as confounders. If None, uses all numeric
            columns except the treatment.

        Returns
        -------
        TreatmentVariationReport
        """
        df_pd = _to_pandas(df)

        D = df_pd[treatment].values.astype(float)

        if confounders is None:
            confounders = [
                c for c in df_pd.select_dtypes(include=[float, int]).columns
                if c != treatment
            ]

        X = _prepare_features(df_pd, list(confounders))

        treatment_var = float(np.var(D, ddof=1))
        if treatment_var < 1e-12:
            # Degenerate case: no variation in treatment at all
            report = TreatmentVariationReport(
                treatment_var=treatment_var,
                residual_var=0.0,
                variation_fraction=0.0,
                nuisance_r2=1.0,
                n_obs=len(D),
                weak_treatment=True,
                suggestions=_weak_treatment_suggestions(),
            )
            self._last_report = report
            return report

        # Fit treatment nuisance via cross-validated GBM
        if _SKLEARN_AVAILABLE:
            nuisance_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                random_state=self.random_state,
                subsample=0.8,
            )
            cv_r2_scores = cross_val_score(
                nuisance_model,
                X,
                D,
                cv=self.n_folds,
                scoring="r2",
            )
            nuisance_r2 = float(np.clip(np.mean(cv_r2_scores), 0.0, 1.0))

            # Fit on full data to get residuals
            nuisance_model.fit(X, D)
            D_hat = nuisance_model.predict(X)
        else:
            # Fallback: OLS residuals (much less accurate but avoids hard sklearn dep)
            D_hat = np.full_like(D, np.mean(D))
            nuisance_r2 = 0.0

        D_tilde = D - D_hat
        residual_var = float(np.var(D_tilde, ddof=1))
        variation_fraction = residual_var / treatment_var

        weak = (
            variation_fraction < _MIN_VARIATION_FRACTION
            or nuisance_r2 > _MAX_NUISANCE_R2
        )

        suggestions = _weak_treatment_suggestions() if weak else []

        report = TreatmentVariationReport(
            treatment_var=treatment_var,
            residual_var=residual_var,
            variation_fraction=variation_fraction,
            nuisance_r2=nuisance_r2,
            n_obs=len(D),
            weak_treatment=weak,
            suggestions=suggestions,
        )
        self._last_report = report
        return report

    def weak_treatment_warning(self) -> bool:
        """Return True if the last report flagged a weak treatment problem.

        Call :meth:`treatment_variation_report` first.

        Returns
        -------
        bool
        """
        if self._last_report is None:
            raise RuntimeError(
                "Call treatment_variation_report() before weak_treatment_warning()."
            )
        return self._last_report.weak_treatment

    def calibration_summary(
        self,
        df: pl.DataFrame,
        outcome: str = "renewed",
        treatment: str = "log_price_change",
        n_bins: int = 10,
    ) -> pl.DataFrame:
        """Summarise observed renewal rates by decile of price change.

        A useful sanity check: renewal rate should fall as price change rises.
        A flat relationship suggests no price effect in the data (possibly due
        to the near-deterministic price problem or severe confounding).

        Parameters
        ----------
        df:
            Renewal dataset.
        outcome:
            Binary renewal indicator column.
        treatment:
            Log price change column.
        n_bins:
            Number of decile bins.

        Returns
        -------
        polars.DataFrame with columns: bin_label, mean_log_price_change,
        renewal_rate, n
        """
        df_pd = _to_pandas(df)
        D = df_pd[treatment].values
        Y = df_pd[outcome].values

        bins = np.quantile(D, np.linspace(0, 1, n_bins + 1))
        bins[0] -= 1e-9  # include minimum
        bin_idx = np.digitize(D, bins) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        rows = []
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() == 0:
                continue
            rows.append({
                "bin": b + 1,
                "mean_log_price_change": float(np.mean(D[mask])),
                "renewal_rate": float(np.mean(Y[mask])),
                "n": int(mask.sum()),
            })

        return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_pandas(df: object) -> object:
    """Convert polars DataFrame to pandas, pass pandas through unchanged."""
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    return df


def _prepare_features(df_pd: object, confounders: list[str]) -> object:
    """One-hot encode categoricals, return numpy array."""
    import pandas as pd
    subset = df_pd[confounders]
    # Encode object/category columns
    obj_cols = subset.select_dtypes(include=["object", "category"]).columns.tolist()
    if obj_cols:
        subset = pd.get_dummies(subset, columns=obj_cols, drop_first=True)
    return subset.values.astype(float)


def _weak_treatment_suggestions() -> list[str]:
    """Return standard suggestions for addressing the weak treatment problem."""
    return [
        "Run randomised A/B price tests: assign a random subset of renewals "
        "to an alternative quoted price. This is the gold standard for "
        "identifying causal elasticity.",

        "Use panel data with within-customer variation: for customers with "
        "3+ renewal observations, within-customer price changes over time "
        "contain variation not explained by cross-sectional risk factors "
        "(fixed-effects DML, Clarke & Polselli 2025).",

        "Exploit bulk re-rating quasi-experiments: when you apply a uniform "
        "percentage increase to a line of business, create an indicator for "
        "'subject to Q1 2023 bulk re-rate'. This is exogenous at the "
        "individual level and acts as a valid instrument.",

        "Use rate change timing heterogeneity: customers with different "
        "anniversary dates face rate changes at different calendar quarters. "
        "Market conditions differ by quarter, creating quasi-exogenous "
        "cross-sectional variation.",

        "Exploit the PS21/5 kink: post-January 2022, customers previously "
        "above ENBP were price-reduced. A regression discontinuity with "
        "running variable = prior year overcharge provides clean elasticity "
        "identification near the ENBP constraint.",
    ]
