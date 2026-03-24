"""RateChangeEvaluator: post-hoc causal evaluation of insurance rate changes.

Answers the question every pricing team asks after a rate change:
'We raised rates 8% on young drivers in January. What actually happened
to conversion and loss ratio?'

Two methods are supported:
- DiD (Difference-in-Differences): when a control group exists (segment-
  specific rate change). Primary method. Stronger identification.
- ITS (Interrupted Time Series): when the entire book was treated (no
  control group). Fallback method with explicit confounding caveats.

Method selection is automatic based on whether the ``treated_col`` contains
both 0 and 1 values.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Union
import numpy as np
import pandas as pd

from ._result import RateChangeResult, DiDResult, ITSResult
from ._did import DiDEstimator
from ._its import ITSEstimator
from ._shocks import check_shock_proximity


class RateChangeEvaluator:
    """Post-hoc causal evaluation of insurance rate changes.

    Automatically selects DiD or ITS based on the structure of the data,
    applies exposure weighting, checks for known UK market confounders, and
    provides parallel trends diagnostics.

    Parameters
    ----------
    outcome_col : str
        Column name for the outcome to evaluate (e.g. 'loss_ratio',
        'conversion_rate', 'claim_frequency').
    treated_col : str, default 'treated'
        Column indicating treatment group (0 = control, 1 = treated).
        If all values are 1, ITS is used automatically.
    period_col : str, default 'period'
        Column identifying time periods. Must be sortable.
    treatment_period : any
        The period in which the rate change took effect. Must be present
        in the ``period_col`` column.
    unit_col : str or None
        Column identifying cross-sectional units (segments, channels, etc.).
        Required for cluster-robust standard errors in DiD.
    weight_col : str or None
        Column for exposure weights (earned policy-years or earned premium).
        Strongly recommended for insurance data — unweighted estimates treat
        a 10-policy segment the same as a 10,000-policy segment.
    add_seasonality : bool, default True
        (ITS only) Whether to include quarterly seasonal dummies.
    season_col : str or None
        (ITS only) Column identifying the quarter/season. Derived from
        ``period_col`` automatically if not provided.
    n_pre_periods : int, default 4
        Number of pre-treatment periods for the DiD event study.
    shock_proximity_quarters : int, default 2
        Number of quarters within which to warn about known UK market shocks.
    min_clusters_for_cluster_se : int, default 20
        Minimum clusters for cluster-robust SE; HC3 used below this threshold.

    Examples
    --------
    DiD evaluation (segment-specific rate change):

    >>> from insurance_causal.rate_change import RateChangeEvaluator, make_rate_change_data
    >>> df = make_rate_change_data(n_segments=40, true_att=-0.05, seed=0)
    >>> evaluator = RateChangeEvaluator(
    ...     outcome_col='outcome',
    ...     treatment_period=9,
    ...     unit_col='segment',
    ...     weight_col='earned_exposure',
    ... )
    >>> result = evaluator.fit(df)
    >>> print(result.summary())

    ITS evaluation (whole-book rate change):

    >>> df_its = make_rate_change_data(mode='its', true_level_shift=-0.02, seed=0)
    >>> evaluator = RateChangeEvaluator(
    ...     outcome_col='outcome',
    ...     treatment_period=9,
    ...     weight_col='earned_exposure',
    ... )
    >>> result = evaluator.fit(df_its)
    >>> print(result.summary())
    """

    def __init__(
        self,
        outcome_col: str,
        treatment_period: Any,
        treated_col: str = "treated",
        period_col: str = "period",
        unit_col: Optional[str] = None,
        weight_col: Optional[str] = None,
        add_seasonality: bool = True,
        season_col: Optional[str] = None,
        n_pre_periods: int = 4,
        shock_proximity_quarters: int = 2,
        min_clusters_for_cluster_se: int = 20,
    ) -> None:
        self.outcome_col = outcome_col
        self.treatment_period = treatment_period
        self.treated_col = treated_col
        self.period_col = period_col
        self.unit_col = unit_col
        self.weight_col = weight_col
        self.add_seasonality = add_seasonality
        self.season_col = season_col
        self.n_pre_periods = n_pre_periods
        self.shock_proximity_quarters = shock_proximity_quarters
        self.min_clusters_for_cluster_se = min_clusters_for_cluster_se

        self._result: Optional[RateChangeResult] = None
        self._did_estimator: Optional[DiDEstimator] = None
        self._its_estimator: Optional[ITSEstimator] = None

    def fit(self, df: pd.DataFrame) -> "RateChangeEvaluator":
        """Fit the rate change evaluator on panel or time series data.

        Automatically selects DiD or ITS based on treatment group structure,
        checks for UK market shocks near the treatment period, and raises
        warnings for staggered adoption.

        Parameters
        ----------
        df : pd.DataFrame
            Insurance panel data. For DiD: one row per segment-period with
            both treated (1) and control (0) observations. For ITS: one row
            per period with all observations treated.

        Returns
        -------
        RateChangeEvaluator
            Self for method chaining.
        """
        self._validate_columns(df)

        collected_warnings: list[str] = []
        notes: list[str] = []

        # Check for UK market shocks
        shock_warnings = check_shock_proximity(
            self.treatment_period,
            proximity_quarters=self.shock_proximity_quarters,
        )
        collected_warnings.extend(shock_warnings)
        for w in shock_warnings:
            warnings.warn(w, stacklevel=2)

        # Determine method
        has_control = (
            self.treated_col in df.columns
            and df[self.treated_col].nunique() >= 2
            and 0 in df[self.treated_col].values
            and 1 in df[self.treated_col].values
        )

        if has_control:
            method = "DiD"
            notes.append(
                "DiD selected: both treated and control units present. "
                "Identification relies on parallel trends assumption."
            )
        else:
            method = "ITS"
            notes.append(
                "ITS selected: no control group detected (all units treated). "
                "Counterfactual is the pre-intervention trend extrapolated forward. "
                "Any concurrent market event invalidates this assumption."
            )
            warnings.warn(
                "No control group found. Using ITS (Interrupted Time Series). "
                "ITS identification is weaker than DiD — the causal estimate "
                "is valid only if no concurrent events affected the outcome.",
                stacklevel=2,
            )

        if method == "DiD":
            self._did_estimator = DiDEstimator(
                outcome_col=self.outcome_col,
                treated_col=self.treated_col,
                period_col=self.period_col,
                treatment_period=self.treatment_period,
                unit_col=self.unit_col,
                weight_col=self.weight_col,
                min_clusters_for_cluster_se=self.min_clusters_for_cluster_se,
                run_event_study=True,
                n_pre_periods=self.n_pre_periods,
            )
            self._did_estimator.fit(df)
            did_result = self._did_estimator.result()

            # Collect staggered adoption warning
            stag = self._did_estimator.staggered_info()
            if stag and stag.get("is_staggered"):
                msg = stag["message"]
                collected_warnings.append(msg)
                warnings.warn(msg, stacklevel=2)

            # Warn if parallel trends test fails
            if (
                did_result.parallel_trends_p_value is not None
                and did_result.parallel_trends_p_value < 0.1
            ):
                msg = (
                    f"Parallel trends test p={did_result.parallel_trends_p_value:.3f} "
                    f"(F={did_result.parallel_trends_f_stat:.2f}). "
                    f"Pre-trend coefficients are not jointly zero. "
                    f"The parallel trends assumption may be violated."
                )
                collected_warnings.append(msg)
                warnings.warn(msg, stacklevel=2)

            self._result = RateChangeResult(
                method="DiD",
                outcome=self.outcome_col,
                treatment_period=self.treatment_period,
                did=did_result,
                its=None,
                warnings=collected_warnings,
                notes=notes,
            )

        else:  # ITS
            self._its_estimator = ITSEstimator(
                outcome_col=self.outcome_col,
                period_col=self.period_col,
                treatment_period=self.treatment_period,
                weight_col=self.weight_col,
                add_seasonality=self.add_seasonality,
                season_col=self.season_col,
            )
            self._its_estimator.fit(df)
            its_result = self._its_estimator.result()

            notes.append(
                f"ITS used HAC standard errors with {its_result.hac_lags} lags "
                f"(Newey-West) to account for autocorrelation."
            )
            if not its_result.has_seasonality:
                notes.append(
                    "Seasonal quarter dummies could not be included. "
                    "If outcomes have strong seasonality, results may be biased "
                    "by seasonal patterns coinciding with the rate change."
                )

            self._result = RateChangeResult(
                method="ITS",
                outcome=self.outcome_col,
                treatment_period=self.treatment_period,
                did=None,
                its=its_result,
                warnings=collected_warnings,
                notes=notes,
            )

        return self

    def summary(self) -> str:
        """Return a formatted text summary of the rate change evaluation.

        Returns
        -------
        str

        Raises
        ------
        RuntimeError
            If fit() has not been called.
        """
        result = self._get_result()
        lines = []
        lines.append("=" * 60)
        lines.append("Rate Change Evaluation Summary")
        lines.append("=" * 60)
        lines.append(f"Method         : {result.method}")
        lines.append(f"Outcome        : {result.outcome}")
        lines.append(f"Treatment at   : period {result.treatment_period}")
        lines.append("")

        if result.method == "DiD" and result.did:
            d = result.did
            sig = "***" if d.p_value < 0.01 else ("**" if d.p_value < 0.05 else ("*" if d.p_value < 0.1 else ""))
            lines.append("Difference-in-Differences (TWFE)")
            lines.append("-" * 40)
            lines.append(f"ATT             : {d.att:+.4f}{sig}")
            lines.append(f"Std error       : {d.se:.4f}")
            lines.append(f"95% CI          : [{d.ci_lower:+.4f}, {d.ci_upper:+.4f}]")
            lines.append(f"p-value         : {d.p_value:.4f}")
            lines.append(f"SE type         : {d.se_type}")
            lines.append(f"Observations    : {d.n_obs:,}")
            lines.append(f"Units           : {d.n_units} ({d.n_treated} treated)")
            lines.append(f"Periods         : {d.n_periods}")
            if d.parallel_trends_p_value is not None:
                lines.append(
                    f"Parallel trends : F={d.parallel_trends_f_stat:.2f}, "
                    f"p={d.parallel_trends_p_value:.3f} "
                    f"({'pass' if d.parallel_trends_p_value >= 0.1 else 'FAIL'})"
                )

        elif result.method == "ITS" and result.its:
            s = result.its
            sig_l = "***" if s.level_shift_p_value < 0.01 else ("**" if s.level_shift_p_value < 0.05 else ("*" if s.level_shift_p_value < 0.1 else ""))
            sig_s = "***" if s.slope_change_p_value < 0.01 else ("**" if s.slope_change_p_value < 0.05 else ("*" if s.slope_change_p_value < 0.1 else ""))
            lines.append("Interrupted Time Series (Segmented Regression)")
            lines.append("-" * 40)
            lines.append(f"Level shift     : {s.level_shift:+.4f}{sig_l}")
            lines.append(f"  Std error     :  {s.level_shift_se:.4f}")
            lines.append(f"  95% CI        : [{s.level_shift_ci_lower:+.4f}, {s.level_shift_ci_upper:+.4f}]")
            lines.append(f"  p-value       :  {s.level_shift_p_value:.4f}")
            lines.append(f"Slope change    : {s.slope_change:+.6f}{sig_s}")
            lines.append(f"  Std error     :  {s.slope_change_se:.6f}")
            lines.append(f"  95% CI        : [{s.slope_change_ci_lower:+.6f}, {s.slope_change_ci_upper:+.6f}]")
            lines.append(f"  p-value       :  {s.slope_change_p_value:.4f}")
            lines.append(f"Pre-trend slope : {s.pre_trend:+.6f}")
            lines.append(f"Pre periods     : {s.n_pre}")
            lines.append(f"Post periods    : {s.n_post}")
            lines.append(f"R-squared       : {s.r_squared:.3f}")
            lines.append(f"HAC lags        : {s.hac_lags}")
            lines.append(f"Seasonality     : {'yes' if s.has_seasonality else 'no'}")

        if result.warnings:
            lines.append("")
            lines.append("Warnings")
            lines.append("-" * 40)
            for w in result.warnings:
                lines.append(f"  ! {w}")

        if result.notes:
            lines.append("")
            lines.append("Notes")
            lines.append("-" * 40)
            for n in result.notes:
                lines.append(f"  - {n}")

        lines.append("")
        lines.append("Significance: *** p<0.01  ** p<0.05  * p<0.1")
        lines.append("=" * 60)

        return "\n".join(lines)

    def parallel_trends_test(self) -> dict:
        """Return the parallel trends test results from the DiD event study.

        Returns
        -------
        dict
            Keys: 'f_stat', 'p_value', 'coefs', 'ses', 'periods', 'passed'.
            Returns an empty dict with 'passed': True if method is ITS
            (no parallel trends assumption).

        Raises
        ------
        RuntimeError
            If fit() has not been called.
        """
        result = self._get_result()
        if result.method == "ITS":
            return {
                "f_stat": None,
                "p_value": None,
                "coefs": [],
                "ses": [],
                "periods": [],
                "passed": True,
                "note": "Parallel trends assumption not applicable for ITS.",
            }

        d = result.did
        return {
            "f_stat": d.parallel_trends_f_stat,
            "p_value": d.parallel_trends_p_value,
            "coefs": d.event_study_coefs,
            "ses": d.event_study_se,
            "periods": d.event_study_periods,
            "passed": (
                d.parallel_trends_p_value is None
                or d.parallel_trends_p_value >= 0.1
            ),
        }

    def plot_event_study(self, ax=None, **kwargs):
        """Plot the DiD event study (pre-trend parallel trends check).

        Parameters
        ----------
        ax : matplotlib Axes, optional
        **kwargs
            Additional keyword arguments passed to ``plot_event_study``.

        Returns
        -------
        matplotlib.axes.Axes

        Raises
        ------
        RuntimeError
            If fit() has not been called or method is ITS.
        """
        result = self._get_result()
        if result.method != "DiD":
            raise RuntimeError(
                "plot_event_study is only available for DiD. "
                f"Current method is {result.method}."
            )
        from ._plots import plot_event_study
        return plot_event_study(result.did, ax=ax, **kwargs)

    def plot_pre_post(self, df: pd.DataFrame, ax=None, **kwargs):
        """Plot treated/control mean outcomes over time.

        Parameters
        ----------
        df : pd.DataFrame
            The data used in fit().
        ax : matplotlib Axes, optional
        **kwargs
            Additional keyword arguments passed to ``plot_pre_post``.

        Returns
        -------
        matplotlib.axes.Axes

        Raises
        ------
        RuntimeError
            If fit() has not been called.
        """
        result = self._get_result()
        from ._plots import plot_pre_post
        return plot_pre_post(
            df=df,
            outcome_col=self.outcome_col,
            treated_col=self.treated_col,
            period_col=self.period_col,
            treatment_period=self.treatment_period,
            weight_col=self.weight_col,
            did_result=result.did,
            ax=ax,
            **kwargs,
        )

    def plot_its(self, df: pd.DataFrame, ax=None, **kwargs):
        """Plot the ITS observed vs counterfactual time series.

        Parameters
        ----------
        df : pd.DataFrame
            The data used in fit().
        ax : matplotlib Axes, optional
        **kwargs
            Additional keyword arguments passed to ``plot_its``.

        Returns
        -------
        matplotlib.axes.Axes

        Raises
        ------
        RuntimeError
            If fit() has not been called or method is DiD.
        """
        result = self._get_result()
        if result.method != "ITS":
            raise RuntimeError(
                "plot_its is only available for ITS. "
                f"Current method is {result.method}."
            )
        from ._plots import plot_its
        return plot_its(
            df=df,
            outcome_col=self.outcome_col,
            period_col=self.period_col,
            treatment_period=self.treatment_period,
            its_estimator=self._its_estimator,
            its_result=result.its,
            weight_col=self.weight_col,
            ax=ax,
            **kwargs,
        )

    def _get_result(self) -> RateChangeResult:
        if self._result is None:
            raise RuntimeError("Call fit() before accessing results.")
        return self._result

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Basic column validation before fitting."""
        if self.outcome_col not in df.columns:
            raise ValueError(
                f"outcome_col {self.outcome_col!r} not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )
        if self.period_col not in df.columns:
            raise ValueError(
                f"period_col {self.period_col!r} not found in DataFrame."
            )
        if self.treatment_period not in df[self.period_col].values:
            raise ValueError(
                f"treatment_period {self.treatment_period!r} not found in "
                f"column {self.period_col!r}. "
                f"Available periods: {sorted(df[self.period_col].unique()).tolist()}"
            )
        if self.weight_col and self.weight_col not in df.columns:
            raise ValueError(
                f"weight_col {self.weight_col!r} not found in DataFrame."
            )
