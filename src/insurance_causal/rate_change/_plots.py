"""
Plotting functions for RateChangeEvaluator.

All functions return matplotlib Axes. They accept an optional ax parameter
so callers can integrate into their own figure layouts.

Requires matplotlib>=3.5 (not installed by default — part of rate_change extra).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import matplotlib.axes


def _get_ax(ax):
    """Return ax if provided, else create a new figure and return its axes."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib>=3.5 is required for plotting. "
            "Install with: pip install 'insurance-causal[rate_change]'"
        ) from exc
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    return ax


def plot_event_study(
    event_study_df: pd.DataFrame,
    ax: "matplotlib.axes.Axes | None" = None,
    title: str | None = None,
    alpha: float = 0.05,
) -> "matplotlib.axes.Axes":
    """
    Plot event study coefficients with confidence intervals.

    Pre-treatment periods are plotted in grey; post-treatment in blue.
    A dashed vertical line marks e=0 (the intervention period).
    A horizontal reference line at zero is shown for pre-treatment periods.

    Parameters
    ----------
    event_study_df : pd.DataFrame
        Output from DiDResult.event_study_df or parallel_trends_test().event_study_df.
        Columns: event_time, att_e, se_e, ci_lower_e, ci_upper_e.
    ax : matplotlib.axes.Axes | None
        Axes to plot on. Created if None.
    title : str | None
        Plot title. Defaults to "Event Study: Pre- and Post-Treatment Effects".
    alpha : float
        Not used directly here (CIs are taken from event_study_df), kept for
        API consistency.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _get_ax(ax)

    pre = event_study_df[event_study_df["event_time"] < 0]
    post = event_study_df[event_study_df["event_time"] >= 0]
    ref = event_study_df[event_study_df["event_time"] == -1]

    # Pre-treatment (grey)
    if not pre.empty:
        ax.errorbar(
            pre["event_time"],
            pre["att_e"],
            yerr=[pre["att_e"] - pre["ci_lower_e"], pre["ci_upper_e"] - pre["att_e"]],
            fmt="o",
            color="grey",
            label="Pre-treatment",
            capsize=4,
            zorder=3,
        )

    # Post-treatment (blue) — exclude e=0 reference if it's there
    post_nonref = post[post["event_time"] != -1]
    if not post_nonref.empty:
        ax.errorbar(
            post_nonref["event_time"],
            post_nonref["att_e"],
            yerr=[
                post_nonref["att_e"] - post_nonref["ci_lower_e"],
                post_nonref["ci_upper_e"] - post_nonref["att_e"],
            ],
            fmt="o",
            color="steelblue",
            label="Post-treatment",
            capsize=4,
            zorder=3,
        )

    # Reference period marker
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(-0.5, color="darkred", linewidth=1.2, linestyle="--", alpha=0.7,
               label="Rate change")

    ax.set_xlabel("Event time (periods relative to rate change)")
    ax.set_ylabel("ATT estimate")
    ax.set_title(title or "Event Study: Pre- and Post-Treatment Effects")
    ax.legend(loc="upper left", fontsize=9)

    return ax


def plot_pre_post_did(
    df_agg: pd.DataFrame,
    period_col: str,
    outcome_col: str,
    treated_col: str,
    change_period: int,
    exposure_col: str | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
    title: str | None = None,
) -> "matplotlib.axes.Axes":
    """
    Plot observed outcomes for treated and control groups over time (DiD).

    Parameters
    ----------
    df_agg : pd.DataFrame
        Segment-period aggregated data.
    period_col : str
    outcome_col : str
    treated_col : str
    change_period : int
        Integer-encoded change period.
    exposure_col : str | None
    ax : matplotlib.axes.Axes | None
    title : str | None

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _get_ax(ax)

    def _weighted_mean(grp):
        if exposure_col and exposure_col in grp.columns:
            w = grp[exposure_col].clip(lower=0)
            total = w.sum()
            if total > 0:
                return np.average(grp[outcome_col], weights=w)
        return grp[outcome_col].mean()

    # Aggregate to period x treated level
    period_treated_mean = (
        df_agg.groupby([period_col, treated_col])
        .apply(_weighted_mean)
        .reset_index()
        .rename(columns={0: outcome_col})
    )

    treated_ts = period_treated_mean[period_treated_mean[treated_col] == 1]
    control_ts = period_treated_mean[period_treated_mean[treated_col] == 0]

    ax.plot(
        treated_ts[period_col], treated_ts[outcome_col],
        "o-", color="steelblue", label="Treated", linewidth=2,
    )
    if not control_ts.empty:
        ax.plot(
            control_ts[period_col], control_ts[outcome_col],
            "s--", color="grey", label="Control", linewidth=1.5,
        )

    ax.axvline(change_period - 0.5, color="darkred", linewidth=1.2, linestyle="--",
               alpha=0.8, label="Rate change")

    ax.set_xlabel("Period")
    ax.set_ylabel(outcome_col)
    ax.set_title(title or "Pre-Post Outcome: Treated vs Control (DiD)")
    ax.legend(fontsize=9)

    return ax


def plot_pre_post_its(
    df_ts: pd.DataFrame,
    period_col: str,
    outcome_col: str,
    change_period: int,
    t_change: int,
    pre_trend: float,
    intercept: float,
    ax: "matplotlib.axes.Axes | None" = None,
    title: str | None = None,
) -> "matplotlib.axes.Axes":
    """
    Plot observed vs counterfactual trend (ITS).

    The counterfactual is the pre-intervention trend extrapolated forward.

    Parameters
    ----------
    df_ts : pd.DataFrame
        Time series data with _time_ column (1-indexed time counter).
    period_col : str
    outcome_col : str
    change_period : int
        Integer-encoded change period (for x-axis alignment).
    t_change : int
        _time_ value corresponding to change_period.
    pre_trend : float
        Estimated pre-intervention slope (beta_1).
    intercept : float
        Estimated intercept (beta_0).
    ax : matplotlib.axes.Axes | None
    title : str | None

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _get_ax(ax)

    ax.plot(
        df_ts[period_col], df_ts[outcome_col],
        "o-", color="steelblue", label="Observed", linewidth=2,
    )

    # Counterfactual: intercept + pre_trend * _time_
    counterfactual = intercept + pre_trend * df_ts["_time_"]
    ax.plot(
        df_ts[period_col], counterfactual,
        "--", color="grey", label="Counterfactual (pre-trend extrapolated)", linewidth=1.5,
    )

    ax.axvline(change_period - 0.5, color="darkred", linewidth=1.2, linestyle="--",
               alpha=0.8, label="Rate change")

    ax.set_xlabel("Period")
    ax.set_ylabel(outcome_col)
    ax.set_title(title or "ITS: Observed vs Counterfactual Trend")
    ax.legend(fontsize=9)

    return ax
