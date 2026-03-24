"""Visualisation functions for rate change evaluation results.

Three plot types:
- plot_event_study: pre/post event study coefficients from DiD
- plot_pre_post: outcome means before/after with CIs
- plot_its: time series with fitted counterfactual and observed line
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import matplotlib.axes


def plot_event_study(
    did_result: Any,
    ax: Optional["matplotlib.axes.Axes"] = None,
    title: str = "Event Study: Pre-Treatment Parallel Trends",
    outcome_label: str = "Outcome",
) -> "matplotlib.axes.Axes":
    """Plot event study coefficients for parallel trends assessment.

    Displays the event study coefficients (pre-treatment period dummies
    for treated units) with 95% confidence intervals. Coefficients near
    zero in pre-treatment periods support the parallel trends assumption.

    Parameters
    ----------
    did_result : DiDResult
        Result from DiDEstimator.
    ax : matplotlib Axes, optional
        Axes object to plot on. Creates a new figure if not provided.
    title : str
        Plot title.
    outcome_label : str
        Y-axis label.

    Returns
    -------
    matplotlib.axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install matplotlib"
        ) from exc

    if did_result.event_study_coefs is None or len(did_result.event_study_coefs) == 0:
        raise ValueError(
            "No event study coefficients available. "
            "Run DiDEstimator with run_event_study=True."
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    periods = np.array(did_result.event_study_periods)
    coefs = np.array(did_result.event_study_coefs)
    ses = np.array(did_result.event_study_se)
    ci_width = 1.96 * ses

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(-0.5, color="grey", linewidth=0.8, linestyle=":", alpha=0.5,
               label="Treatment onset")

    ax.errorbar(
        periods, coefs,
        yerr=ci_width,
        fmt="o",
        color="#1f77b4",
        ecolor="#1f77b4",
        capsize=4,
        linewidth=1.5,
        markersize=6,
        label="Event study coef (95% CI)",
    )

    ax.set_xlabel("Period relative to treatment (0 = treatment onset)")
    ax.set_ylabel(f"Effect on {outcome_label}")
    ax.set_title(title)
    ax.legend(framealpha=0.8)

    if did_result.parallel_trends_p_value is not None:
        sig_str = (
            "Parallel trends: FAIL (p={:.3f})".format(did_result.parallel_trends_p_value)
            if did_result.parallel_trends_p_value < 0.1
            else "Parallel trends: pass (p={:.3f})".format(did_result.parallel_trends_p_value)
        )
        ax.text(
            0.02, 0.97, sig_str,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=9,
            color="red" if did_result.parallel_trends_p_value < 0.1 else "green",
        )

    return ax


def plot_pre_post(
    df: pd.DataFrame,
    outcome_col: str,
    treated_col: str,
    period_col: str,
    treatment_period: Any,
    weight_col: Optional[str] = None,
    did_result: Optional[Any] = None,
    ax: Optional["matplotlib.axes.Axes"] = None,
    title: str = "Pre/Post Outcomes by Group",
) -> "matplotlib.axes.Axes":
    """Plot exposure-weighted mean outcomes by group (treated/control) over time.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data.
    outcome_col : str
        Outcome column name.
    treated_col : str
        Treatment group indicator column (0/1).
    period_col : str
        Period column name.
    treatment_period : any
        Treatment onset period.
    weight_col : str or None
        Exposure weight column. If None, uses unweighted means.
    did_result : DiDResult or None
        If provided, annotates the ATT on the plot.
    ax : matplotlib Axes, optional
    title : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting.") from exc

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    sorted_periods = sorted(df[period_col].unique())
    colors = {0: "#d62728", 1: "#1f77b4"}
    labels = {0: "Control", 1: "Treated"}

    for grp in sorted(df[treated_col].unique()):
        sub = df[df[treated_col] == grp]
        means = []
        for p in sorted_periods:
            p_data = sub[sub[period_col] == p]
            if len(p_data) == 0:
                means.append(np.nan)
            elif weight_col and weight_col in p_data.columns:
                w = p_data[weight_col].values
                means.append(float(np.average(p_data[outcome_col].values, weights=w)))
            else:
                means.append(float(p_data[outcome_col].mean()))

        ax.plot(
            sorted_periods, means,
            marker="o",
            markersize=5,
            color=colors.get(grp, "grey"),
            linewidth=1.8,
            label=labels.get(grp, f"Group {grp}"),
        )

    # Mark treatment onset
    ax.axvline(
        treatment_period,
        color="black",
        linewidth=1.2,
        linestyle="--",
        alpha=0.7,
        label=f"Treatment onset ({treatment_period})",
    )

    if did_result is not None:
        sig = "***" if did_result.p_value < 0.01 else ("**" if did_result.p_value < 0.05 else ("*" if did_result.p_value < 0.1 else ""))
        att_str = f"ATT = {did_result.att:.4f}{sig} (p={did_result.p_value:.3f})"
        ax.text(
            0.97, 0.03, att_str,
            transform=ax.transAxes,
            ha="right",
            fontsize=9,
        )

    ax.set_xlabel("Period")
    ax.set_ylabel(outcome_col)
    ax.set_title(title)
    ax.legend(framealpha=0.8)
    return ax


def plot_its(
    df: pd.DataFrame,
    outcome_col: str,
    period_col: str,
    treatment_period: Any,
    its_estimator: Optional[Any] = None,
    its_result: Optional[Any] = None,
    weight_col: Optional[str] = None,
    ax: Optional["matplotlib.axes.Axes"] = None,
    title: str = "Interrupted Time Series",
) -> "matplotlib.axes.Axes":
    """Plot ITS fitted values with counterfactual trend line.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data (one row per period).
    outcome_col : str
        Outcome column.
    period_col : str
        Period column.
    treatment_period : any
        Intervention period.
    its_estimator : ITSEstimator or None
        If provided, plots the counterfactual trend from the fitted model.
    its_result : ITSResult or None
        If provided, annotates level shift and slope change.
    weight_col : str or None
        For computing weighted observed values (informational only).
    ax : matplotlib Axes, optional
    title : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting.") from exc

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    sorted_periods = sorted(df[period_col].unique())

    # Observed outcomes
    obs_y = []
    for p in sorted_periods:
        p_data = df[df[period_col] == p]
        if weight_col and weight_col in p_data.columns:
            w = p_data[weight_col].values
            obs_y.append(float(np.average(p_data[outcome_col].values, weights=w)))
        else:
            obs_y.append(float(p_data[outcome_col].mean()))

    ax.plot(sorted_periods, obs_y, "o-", color="#1f77b4",
            linewidth=1.8, markersize=5, label="Observed", zorder=3)

    # Counterfactual from fitted estimator
    if its_estimator is not None:
        cf = its_estimator.counterfactual()
        if cf is not None:
            cf_sorted = [cf.get(p, np.nan) for p in sorted_periods]
            ax.plot(
                sorted_periods, cf_sorted,
                "--", color="#d62728", linewidth=1.5,
                label="Counterfactual (pre-trend extrapolated)",
            )

    # Mark treatment onset
    ax.axvline(
        treatment_period,
        color="black",
        linewidth=1.2,
        linestyle=":",
        alpha=0.7,
        label=f"Rate change ({treatment_period})",
    )

    if its_result is not None:
        sig = "***" if its_result.level_shift_p_value < 0.01 else (
            "**" if its_result.level_shift_p_value < 0.05 else (
                "*" if its_result.level_shift_p_value < 0.1 else ""
            )
        )
        ann_str = (
            f"Level shift = {its_result.level_shift:.4f}{sig}\n"
            f"Slope change = {its_result.slope_change:.4f}"
        )
        ax.text(
            0.97, 0.97, ann_str,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )

    ax.set_xlabel("Period")
    ax.set_ylabel(outcome_col)
    ax.set_title(title)
    ax.legend(framealpha=0.8)
    return ax
