"""Synthetic data generator for RateChangeEvaluator testing and demos.

Generates realistic insurance panel data with a known treatment effect,
suitable for validating DiD and ITS recovery.
"""

from __future__ import annotations

from typing import Any, Optional, Union
import numpy as np
import pandas as pd


def make_rate_change_data(
    n_segments: int = 20,
    n_periods: int = 16,
    treatment_period: int = 9,
    true_att: float = -0.05,
    true_level_shift: float = -0.03,
    true_slope_change: float = -0.004,
    treated_fraction: float = 0.5,
    base_outcome: float = 0.08,
    pre_trend_slope: float = 0.001,
    noise_scale: float = 0.01,
    exposure_mean: float = 500.0,
    exposure_cv: float = 0.5,
    add_seasonality: bool = True,
    seed: Optional[int] = 42,
    mode: str = "did",
) -> pd.DataFrame:
    """Generate synthetic insurance panel data for rate change evaluation.

    Creates a balanced panel with known treatment effects for validating
    DiD and ITS estimation. The data structure mirrors what you would
    see after a segment-specific rate change (e.g., raising rates on
    young drivers in one product line).

    Parameters
    ----------
    n_segments : int, default 20
        Number of risk segments (cross-sectional units).
    n_periods : int, default 16
        Number of time periods (quarters).
    treatment_period : int, default 9
        Period index (1-based) at which treatment starts.
    true_att : float, default -0.05
        True average treatment effect on the treated. For a conversion
        rate outcome, -0.05 means a 5 percentage point reduction.
    true_level_shift : float, default -0.03
        For ITS mode: true immediate level shift at intervention.
    true_slope_change : float, default -0.004
        For ITS mode: true change in slope per period after intervention.
    treated_fraction : float, default 0.5
        Fraction of segments in the treatment group.
    base_outcome : float, default 0.08
        Baseline outcome level (e.g., loss ratio or claim frequency).
    pre_trend_slope : float, default 0.001
        Common pre-trend slope (shared by treated and control in DiD).
    noise_scale : float, default 0.01
        Standard deviation of idiosyncratic noise.
    exposure_mean : float, default 500.0
        Mean earned exposure (policy-years) per segment-period.
    exposure_cv : float, default 0.5
        Coefficient of variation for exposure. Controls heterogeneity.
    add_seasonality : bool, default True
        Whether to add quarterly seasonal effects.
    seed : int or None, default 42
        Random seed for reproducibility.
    mode : str, default 'did'
        One of 'did' (panel with treated/control groups) or 'its'
        (aggregate time series, all treated — for ITS evaluation).

    Returns
    -------
    pd.DataFrame
        Panel DataFrame with columns:
        - ``segment``: segment identifier (str)
        - ``period``: period index (int, 1-based)
        - ``quarter``: quarter of year (1-4) derived from period
        - ``treated``: treatment group indicator (0/1) — DiD only
        - ``outcome``: observed outcome (loss ratio / claim frequency)
        - ``earned_exposure``: exposure weights
        - ``rate_change``: treatment indicator D_it (1 after treatment for treated)

    Examples
    --------
    >>> df = make_rate_change_data(n_segments=40, true_att=-0.05, seed=0)
    >>> df.head()

    >>> # ITS mode (aggregate time series)
    >>> df_its = make_rate_change_data(mode='its', n_periods=20,
    ...                                true_level_shift=-0.02, seed=0)
    """
    if mode not in ("did", "its"):
        raise ValueError(f"mode must be 'did' or 'its', got {mode!r}")

    rng = np.random.default_rng(seed)

    # Seasonal effects by quarter
    seasonal_effects = {1: 0.005, 2: -0.003, 3: -0.002, 4: 0.008}

    if mode == "did":
        return _make_did_data(
            rng=rng,
            n_segments=n_segments,
            n_periods=n_periods,
            treatment_period=treatment_period,
            true_att=true_att,
            treated_fraction=treated_fraction,
            base_outcome=base_outcome,
            pre_trend_slope=pre_trend_slope,
            noise_scale=noise_scale,
            exposure_mean=exposure_mean,
            exposure_cv=exposure_cv,
            add_seasonality=add_seasonality,
            seasonal_effects=seasonal_effects,
        )
    else:
        return _make_its_data(
            rng=rng,
            n_periods=n_periods,
            treatment_period=treatment_period,
            true_level_shift=true_level_shift,
            true_slope_change=true_slope_change,
            base_outcome=base_outcome,
            pre_trend_slope=pre_trend_slope,
            noise_scale=noise_scale,
            exposure_mean=exposure_mean,
            add_seasonality=add_seasonality,
            seasonal_effects=seasonal_effects,
        )


def _make_did_data(
    rng: np.random.Generator,
    n_segments: int,
    n_periods: int,
    treatment_period: int,
    true_att: float,
    treated_fraction: float,
    base_outcome: float,
    pre_trend_slope: float,
    noise_scale: float,
    exposure_mean: float,
    exposure_cv: float,
    add_seasonality: bool,
    seasonal_effects: dict,
) -> pd.DataFrame:
    """Build balanced panel data for DiD recovery tests."""
    n_treated = max(1, int(n_segments * treated_fraction))
    segment_ids = [f"seg_{i:03d}" for i in range(n_segments)]

    # Assign treatment groups
    treated_segs = set(segment_ids[:n_treated])

    # Segment-level fixed effects
    seg_fe = rng.normal(0, noise_scale * 2, n_segments)
    seg_fe_map = dict(zip(segment_ids, seg_fe))

    rows = []
    for period in range(1, n_periods + 1):
        quarter = ((period - 1) % 4) + 1
        season_eff = seasonal_effects[quarter] if add_seasonality else 0.0

        for i, seg in enumerate(segment_ids):
            is_treated = int(seg in treated_segs)
            is_post = int(period >= treatment_period)
            treatment_active = is_treated * is_post

            # Exposure: log-normal with given mean and CV
            sigma_log = np.sqrt(np.log(1 + exposure_cv ** 2))
            mu_log = np.log(exposure_mean) - 0.5 * sigma_log ** 2
            exposure = float(rng.lognormal(mu_log, sigma_log))

            # Outcome: base + trend + FE + seasonality + treatment effect + noise
            outcome = (
                base_outcome
                + pre_trend_slope * period
                + seg_fe_map[seg]
                + season_eff
                + true_att * treatment_active
                + rng.normal(0, noise_scale / np.sqrt(max(exposure / 100, 1)))
            )
            # Clip to avoid negative loss ratios
            outcome = max(0.001, outcome)

            rows.append({
                "segment": seg,
                "period": period,
                "quarter": quarter,
                "treated": is_treated,
                "outcome": outcome,
                "earned_exposure": exposure,
                "rate_change": treatment_active,
            })

    return pd.DataFrame(rows)


def _make_its_data(
    rng: np.random.Generator,
    n_periods: int,
    treatment_period: int,
    true_level_shift: float,
    true_slope_change: float,
    base_outcome: float,
    pre_trend_slope: float,
    noise_scale: float,
    exposure_mean: float,
    add_seasonality: bool,
    seasonal_effects: dict,
) -> pd.DataFrame:
    """Build aggregate time series data for ITS recovery tests."""
    rows = []
    for period in range(1, n_periods + 1):
        quarter = ((period - 1) % 4) + 1
        season_eff = seasonal_effects[quarter] if add_seasonality else 0.0

        is_post = int(period >= treatment_period)
        time_since = (period - treatment_period) * is_post

        outcome = (
            base_outcome
            + pre_trend_slope * period
            + season_eff
            + true_level_shift * is_post
            + true_slope_change * time_since
            + rng.normal(0, noise_scale)
        )
        outcome = max(0.001, outcome)

        rows.append({
            "period": period,
            "quarter": quarter,
            "treated": 1,
            "outcome": outcome,
            "earned_exposure": exposure_mean + rng.normal(0, exposure_mean * 0.05),
            "rate_change": is_post,
        })

    return pd.DataFrame(rows)
