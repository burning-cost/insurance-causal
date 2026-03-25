"""
Result dataclasses for RateChangeEvaluator.

These are the outputs of .summary() — frozen dataclasses with a __str__
that produces a formatted brief suitable for a pricing team.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class DiDResult:
    """
    Internal DiD estimation results. Accessed via RateChangeResult.method_detail.

    Attributes
    ----------
    att : float
        Average Treatment Effect on the Treated.
    se : float
        Standard error.
    ci_lower, ci_upper : float
        Confidence interval bounds.
    p_value : float
        Two-sided p-value.
    formula : str
        The statsmodels formula string used.
    n_units_treated : int
        Number of treated units (segments).
    n_units_control : int
        Number of control units (segments).
    n_periods : int
        Total number of periods in the data.
    event_study_df : pd.DataFrame
        Columns: event_time, att_e, se_e, ci_lower_e, ci_upper_e.
    joint_pt_fstat : float
        F-statistic for joint pre-treatment test.
    joint_pt_pvalue : float
        P-value for joint pre-treatment test.
    """

    att: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    formula: str
    n_units_treated: int
    n_units_control: int
    n_periods: int
    event_study_df: pd.DataFrame
    joint_pt_fstat: float
    joint_pt_pvalue: float


@dataclass(frozen=True)
class ITSResult:
    """
    Internal ITS estimation results. Accessed via RateChangeResult.method_detail.

    Attributes
    ----------
    level_shift : float
        beta_2: immediate level change at the intervention.
    level_shift_se : float
        Standard error of level_shift.
    level_shift_ci_lower, level_shift_ci_upper : float
        Confidence interval bounds for level_shift.
    level_shift_pvalue : float
        P-value for level_shift.
    slope_change : float
        beta_3: change in slope post-intervention.
    slope_change_se : float
        Standard error of slope_change.
    slope_change_pvalue : float
        P-value for slope_change.
    pre_trend : float
        beta_1: estimated pre-intervention slope.
    pre_trend_se : float
        Standard error of pre_trend.
    effect_at_periods : dict[int, float]
        Effect estimate at k periods post-intervention: beta_2 + beta_3*k.
        Keys: 1, 4, 8.
    seasonal_adjustment : bool
        Whether quarter dummies were included.
    shocks_near_intervention : list[str]
        UK shock names within 2 quarters of the change period.
    """

    level_shift: float
    level_shift_se: float
    level_shift_ci_lower: float
    level_shift_ci_upper: float
    level_shift_pvalue: float
    slope_change: float
    slope_change_se: float
    slope_change_pvalue: float
    pre_trend: float
    pre_trend_se: float
    effect_at_periods: dict[int, float]
    seasonal_adjustment: bool
    shocks_near_intervention: list[str]


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.1:
        return "."
    return ""


@dataclass(frozen=True)
class RateChangeResult:
    """
    Output of RateChangeEvaluator.summary().

    Attributes
    ----------
    method : str
        "did" or "its" — which estimator was used.
    outcome_col : str
        Name of the outcome variable.
    att : float
        Average Treatment Effect on the Treated. Interpretation depends on outcome:
        - Loss ratio: absolute change (e.g. -0.05 = 5pp reduction)
        - Conversion: absolute change in probability (e.g. -0.03 = 3pp drop)
        - Frequency: absolute change in claims per exposure year
    att_pct : float | None
        ATT as a percentage of the pre-treatment treated mean.
    se : float
        Standard error of the ATT estimate.
    ci_lower : float
        Lower bound of the (1-alpha)% confidence interval.
    ci_upper : float
        Upper bound of the (1-alpha)% confidence interval.
    p_value : float
        Two-sided p-value for H0: ATT = 0.
    n_treated : int
        Number of observations in treated group.
    n_control : int
        Number of observations in control group. 0 for ITS.
    n_periods_pre : int
        Number of pre-treatment periods.
    n_periods_post : int
        Number of post-treatment periods.
    pre_mean_treated : float
        Exposure-weighted mean outcome for treated group pre-treatment.
    pre_mean_control : float | None
        Exposure-weighted mean outcome for control group pre-treatment.
        None for ITS.
    parallel_trends_pvalue : float | None
        P-value from joint F-test of pre-treatment event study coefficients.
        None if ITS or insufficient pre-periods.
    staggered_adoption_detected : bool
        Whether multiple treatment cohorts were detected.
    cluster_se_used : bool
        True if clustered SE were used; False if HC3 (due to few clusters).
    n_clusters : int | None
        Number of clusters used for SE clustering. None for ITS.
    warnings : list[str]
        Warnings generated during estimation.
    method_detail : DiDResult | ITSResult
        Detailed internal results for the specific estimation method.
    """

    method: str
    outcome_col: str
    att: float
    att_pct: float | None
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_treated: int
    n_control: int
    n_periods_pre: int
    n_periods_post: int
    pre_mean_treated: float
    pre_mean_control: float | None
    parallel_trends_pvalue: float | None
    staggered_adoption_detected: bool
    cluster_se_used: bool
    n_clusters: int | None
    warnings: list
    method_detail: "DiDResult | ITSResult"

    def __str__(self) -> str:
        method_label = "DiD" if self.method == "did" else self.method.upper()
        title = f"Rate Change Evaluation ({method_label})"
        sep = "=" * len(title)

        lines = [title, sep, ""]

        if self.method == "did":
            return self._str_did(title, sep)
        else:
            return self._str_its(title, sep)

    def _str_did(self, title: str, sep: str) -> str:
        lines = [title, sep, ""]

        pct_str = ""
        if self.att_pct is not None:
            pct_str = f"  ({self.att_pct:+.1f}% vs pre-treatment mean of {self.pre_mean_treated:.3f})"

        stars = _sig_stars(self.p_value)
        ci_alpha = 95  # hard-coded display; stored alpha drives the actual CI

        lines.append(f"Outcome:        {self.outcome_col}")
        lines.append(f"ATT:            {self.att:+.4f}{pct_str}")
        lines.append(f"{ci_alpha}% CI:         ({self.ci_lower:+.4f}, {self.ci_upper:+.4f})")
        lines.append(f"p-value:        {self.p_value:.4f}  {stars}")
        lines.append("")

        d = self.method_detail
        lines.append(
            f"Treated units:  {d.n_units_treated} segments"
            f" | Control units: {d.n_units_control} segments"
        )
        lines.append(
            f"Pre-periods:    {self.n_periods_pre}"
            f" | Post-periods: {self.n_periods_post}"
        )

        # Parallel trends
        if self.parallel_trends_pvalue is not None:
            pt_pass = "PASS" if self.parallel_trends_pvalue > 0.05 else "FAIL — treat estimate with caution"
            lines.append(
                f"Parallel trends (joint F-test): p={self.parallel_trends_pvalue:.2f}  [{pt_pass}]"
            )
        else:
            lines.append("Parallel trends: insufficient pre-periods for joint test")

        # SE method
        if self.cluster_se_used:
            lines.append(
                f"SE method:      clustered ({self.n_clusters} clusters)"
            )
        else:
            lines.append(
                f"SE method:      HC3 heteroskedasticity-robust"
                f" (only {self.n_clusters} clusters — clustered SE unreliable)"
            )

        # Staggered adoption
        if self.staggered_adoption_detected:
            lines.append("")
            lines.append(
                "WARNING: Staggered adoption detected. TWFE estimate may be biased."
            )
            lines.append(
                "  For valid staggered DiD, use:"
                " insurance_causal_policy.StaggeredEstimator"
            )

        # Warnings
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")

        return "\n".join(lines)

    def _str_its(self, title: str, sep: str) -> str:
        lines = [title, sep, ""]

        d = self.method_detail
        stars_level = _sig_stars(d.level_shift_pvalue)
        stars_slope = _sig_stars(d.slope_change_pvalue)

        lines.append(f"Outcome:       {self.outcome_col}")
        lines.append(
            f"Level shift:   {d.level_shift:+.4f}"
            f"  (immediate {'reduction' if d.level_shift < 0 else 'increase'})"
        )
        lines.append(f"p-value:       {d.level_shift_pvalue:.4f}  {stars_level}")
        lines.append(f"95% CI:        ({d.level_shift_ci_lower:+.4f}, {d.level_shift_ci_upper:+.4f})")
        lines.append(
            f"Slope change:  {d.slope_change:+.4f} per period  "
            f"({'trend change' if abs(d.slope_change) > 0.0001 else 'no trend change'})"
        )
        lines.append(f"p-value:       {d.slope_change_pvalue:.4f}  {stars_slope}")
        lines.append(f"Pre-periods:   {self.n_periods_pre} | Post-periods: {self.n_periods_post}")
        lines.append(f"Pre-trend:     {d.pre_trend:+.4f} per period (pre-intervention slope)")
        lines.append(f"Seasonal adj:  {'Yes (quarter dummies)' if d.seasonal_adjustment else 'No'}")
        lines.append("SE method:     HAC (Newey-West)")

        # Effects at horizons
        if d.effect_at_periods:
            lines.append("")
            lines.append("Cumulative effects post-intervention:")
            for k, eff in sorted(d.effect_at_periods.items()):
                lines.append(f"  +{k} period(s): {eff:+.4f}")

        # Shock warnings
        if d.shocks_near_intervention:
            lines.append("")
            lines.append("Potential confounders (known UK insurance shocks nearby):")
            for shock in d.shocks_near_intervention:
                lines.append(f"  - {shock}")

        # Other warnings
        non_shock_warnings = [
            w for w in self.warnings
            if not any(shock in w for shock in (d.shocks_near_intervention or []))
        ]
        if non_shock_warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in non_shock_warnings:
                lines.append(f"  - {w}")

        if not self.warnings:
            lines.append("Warnings:      None")

        return "\n".join(lines)
