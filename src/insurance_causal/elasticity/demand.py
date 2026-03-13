"""
Demand curve utilities for renewal pricing.

These utilities sweep the price range and compute the predicted renewal rate
and expected profit across that range using the fitted elasticity model.

The demand curve is a portfolio-level summary: it averages the individual-level
linear demand predictions across all customers. It answers the question: "if I
apply a uniform X% price change to my whole book, what renewal rate and profit
would I expect?"

This is useful for:
    - Setting the overall renewal price change assumption for a pricing review
    - Understanding the trade-off between retention and profit at portfolio level
    - Communicating the price-demand relationship to senior stakeholders

Note: individual optimal pricing (per customer) is handled in optimise.py.
      The demand curve here is a management information tool, not an optimiser.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import polars as pl


def demand_curve(
    estimator: object,
    df: pl.DataFrame,
    price_range: Tuple[float, float, int] = (-0.30, 0.30, 50),
) -> pl.DataFrame:
    """Compute the portfolio demand curve across a range of price changes.

    For each price change delta in the specified range, predicts the renewal
    probability for every customer using the linear demand approximation, then
    averages across the portfolio.

    Parameters
    ----------
    estimator:
        A fitted :class:`~insurance_causal.elasticity.fit.RenewalElasticityEstimator`.
    df:
        Renewal dataset. Must contain confounder columns plus ``log_price_change``,
        ``renewed``, ``tech_prem``, and ``last_premium``.
    price_range:
        (min_delta, max_delta, n_points). Log price changes to sweep. Defaults to
        -30% to +30% in 50 steps.

    Returns
    -------
    polars.DataFrame with columns:
        - ``log_price_change``: the log price change applied
        - ``pct_price_change``: approximate percentage price change (exp(delta) - 1)
        - ``predicted_renewal_rate``: expected portfolio renewal rate at this price
        - ``predicted_profit``: expected portfolio profit per policy (using tech_prem)
        - ``predicted_revenue``: expected portfolio revenue per policy
    """
    if not hasattr(estimator, '_is_fitted') or not estimator._is_fitted:
        raise RuntimeError("Estimator is not fitted. Call .fit() first.")

    if "log_price_change" not in df.columns:
        raise ValueError("DataFrame must contain 'log_price_change' column.")

    min_delta, max_delta, n_points = price_range
    deltas = np.linspace(min_delta, max_delta, n_points)

    # Per-customer CATE and baseline renewal probability
    cate = estimator.cate(df)
    current_log_change = df["log_price_change"].to_numpy()
    y_observed = df["renewed"].to_numpy().astype(float)
    overall_rate = float(np.mean(y_observed))

    # Smoothed baseline renewal probability
    p0 = 0.2 * overall_rate + 0.8 * y_observed

    tech_prem = df["tech_prem"].to_numpy() if "tech_prem" in df.columns else np.ones(len(df))
    last_prem = df["last_premium"].to_numpy() if "last_premium" in df.columns else np.ones(len(df))

    rows = []
    for delta in deltas:
        # Delta relative to current offer
        additional_delta = delta - current_log_change
        pred_renewal = np.clip(p0 + cate * additional_delta, 0.0, 1.0)
        portfolio_renewal_rate = float(np.mean(pred_renewal))

        new_price = last_prem * np.exp(delta)
        expected_profit_per_policy = float(np.mean((new_price - tech_prem) * pred_renewal))
        expected_revenue_per_policy = float(np.mean(new_price * pred_renewal))

        rows.append({
            "log_price_change": float(delta),
            "pct_price_change": float(np.exp(delta) - 1),
            "predicted_renewal_rate": portfolio_renewal_rate,
            "predicted_profit": expected_profit_per_policy,
            "predicted_revenue": expected_revenue_per_policy,
        })

    return pl.DataFrame(rows)


def plot_demand_curve(
    demand_df: pl.DataFrame,
    ax: Optional[object] = None,
    show_profit: bool = True,
) -> object:
    """Plot the portfolio demand curve.

    Produces a two-axis plot: renewal rate on the left axis, expected profit
    on the right axis (if show_profit=True), both against percentage price
    change.

    Parameters
    ----------
    demand_df:
        Output of :func:`demand_curve`.
    ax:
        Existing matplotlib Axes. If None, creates a new figure. If show_profit
        is True, a secondary axis is always created regardless of ax.
    show_profit:
        Whether to overlay the expected profit curve on a secondary axis.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    pct_change = demand_df["pct_price_change"].to_numpy() * 100  # convert to %
    renewal_rate = demand_df["predicted_renewal_rate"].to_numpy() * 100  # as %
    profit = demand_df["predicted_profit"].to_numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.get_figure()

    color_renewal = "#1f77b4"
    color_profit = "#d62728"

    line1, = ax.plot(pct_change, renewal_rate, color=color_renewal, linewidth=2, label="Renewal rate")
    ax.set_xlabel("Price change (%)")
    ax.set_ylabel("Predicted renewal rate (%)", color=color_renewal)
    ax.tick_params(axis="y", labelcolor=color_renewal)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_title("Portfolio demand curve")

    handles = [line1]
    labels = ["Renewal rate"]

    if show_profit:
        ax2 = ax.twinx()
        line2, = ax2.plot(
            pct_change, profit, color=color_profit, linewidth=2,
            linestyle="--", label="Expected profit/policy"
        )
        ax2.set_ylabel("Expected profit per policy (£)", color=color_profit)
        ax2.tick_params(axis="y", labelcolor=color_profit)

        # Mark profit maximum
        max_profit_idx = np.argmax(profit)
        ax2.scatter(
            [pct_change[max_profit_idx]], [profit[max_profit_idx]],
            color=color_profit, s=80, zorder=5,
            label=f"Max profit at {pct_change[max_profit_idx]:.1f}%"
        )
        handles.append(line2)
        labels.append("Expected profit/policy")

    ax.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    return fig
