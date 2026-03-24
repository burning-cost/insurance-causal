"""
Synthetic data generators for testing and demonstrating RateChangeEvaluator.

make_rate_change_data() — panel data for DiD tests (policy-level)
make_its_data()         — aggregate time series for ITS tests
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_rate_change_data(
    n_policies: int = 10_000,
    n_segments: int = 20,
    n_periods: int = 12,
    change_period: int = 7,
    treated_fraction: float = 0.4,
    true_att: float = -0.03,
    outcome: str = "loss_ratio",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic insurance panel data for testing RateChangeEvaluator.

    Creates ``n_policies`` across ``n_segments`` and ``n_periods``. At
    ``change_period``, ``treated_fraction`` of segments receive a rate change
    with true causal effect ``true_att`` on the outcome.

    The DGP satisfies parallel trends by construction:

    - Segment-level random effects (between-segment heterogeneity in levels)
    - Common time trend shared by treated and control
    - Heterogeneous exposure across segments (lognormal draws)
    - No staggered adoption (all treated segments change at ``change_period``)
    - ``true_att`` added only to treated segment outcomes for period >=
      ``change_period``

    Parameters
    ----------
    n_policies : int
        Total number of policies to generate. Distributed across segments.
    n_segments : int
        Number of distinct segments (e.g. age bands, regions, channels).
    n_periods : int
        Number of time periods.
    change_period : int
        Period at which the rate change is implemented (1-indexed).
    treated_fraction : float
        Fraction of segments in the treated group.
    true_att : float
        True ATT. For loss_ratio: absolute change in loss ratio.
        E.g. -0.03 means a 3pp reduction.
    outcome : str
        Outcome variable name in the returned DataFrame.
        Currently only affects column naming; the DGP is always
        a continuous outcome in [0, 1] range appropriate for loss ratio.
    random_state : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: policy_id, segment_id, period, treated, <outcome>, exposure

    Notes
    -----
    The returned DataFrame is at policy-period level. For DiD estimation,
    RateChangeEvaluator will aggregate to segment-period level internally.
    """
    rng = np.random.default_rng(random_state)

    n_treated = max(1, int(n_segments * treated_fraction))
    n_control = n_segments - n_treated

    # Segment-level random effects (heterogeneity in base loss ratio)
    segment_effects = rng.normal(0.0, 0.05, size=n_segments)

    # Common time trend (shared by all segments)
    time_trend = np.linspace(0.0, 0.02, n_periods)

    # Segment treatment assignment
    all_segments = np.arange(n_segments)
    treated_segments = set(all_segments[:n_treated])

    # Policies per segment (roughly equal)
    policies_per_segment = n_policies // n_segments
    remainder = n_policies - policies_per_segment * n_segments

    rows = []
    policy_id = 0

    for seg_idx in range(n_segments):
        is_treated = 1 if seg_idx in treated_segments else 0
        n_pol = policies_per_segment + (1 if seg_idx < remainder else 0)

        # Segment base exposure: lognormal, mean ~1.0 year
        base_exposure = rng.lognormal(mean=0.0, sigma=0.5, size=n_pol)

        for period in range(1, n_periods + 1):
            # Base outcome: segment effect + common trend
            base_outcome = 0.65 + segment_effects[seg_idx] + time_trend[period - 1]

            # Add treatment effect
            treatment_active = is_treated and period >= change_period
            treatment_effect = true_att if treatment_active else 0.0

            # Policy-level noise
            noise = rng.normal(0.0, 0.08, size=n_pol)

            policy_outcomes = base_outcome + treatment_effect + noise

            for i in range(n_pol):
                rows.append({
                    "policy_id": policy_id + i,
                    "segment_id": seg_idx,
                    "period": period,
                    "treated": is_treated,
                    outcome: policy_outcomes[i],
                    "exposure": base_exposure[i],
                })

        policy_id += n_pol

    df = pd.DataFrame(rows)
    # Clip outcome to reasonable range (loss ratios don't go negative)
    df[outcome] = df[outcome].clip(0.0, 3.0)

    return df.reset_index(drop=True)


def make_its_data(
    n_periods: int = 16,
    change_period: int = 9,
    true_level_shift: float = -0.04,
    true_slope_change: float = 0.0,
    true_pre_trend: float = 0.002,
    base_outcome: float = 0.65,
    exposure_per_period: float = 50_000.0,
    add_seasonality: bool = True,
    noise_scale: float = 0.005,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic aggregate time series for testing ITSEstimator.

    Creates a time series with:
    - A linear pre-trend (``true_pre_trend`` per period)
    - An immediate level shift of ``true_level_shift`` at ``change_period``
    - A slope change of ``true_slope_change`` post-intervention
    - Optional quarterly seasonality
    - HAC-appropriate autocorrelated noise

    Parameters
    ----------
    n_periods : int
        Total number of periods (quarters).
    change_period : int
        Period of rate change (1-indexed). Pre-treatment periods are
        1, ..., change_period-1.
    true_level_shift : float
        True immediate level shift (beta_2 in the ITS model).
    true_slope_change : float
        True change in slope post-intervention (beta_3).
    true_pre_trend : float
        True pre-intervention slope (beta_1).
    base_outcome : float
        Outcome at period 0 before any trend.
    exposure_per_period : float
        Earned exposure in each period (constant for simplicity).
    add_seasonality : bool
        Add quarterly seasonal effects.
    noise_scale : float
        Standard deviation of the noise term.
    random_state : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: period, outcome, exposure, quarter
    """
    rng = np.random.default_rng(random_state)

    # Seasonal effects by quarter (sum to zero)
    seasonal = {1: 0.01, 2: -0.005, 3: -0.008, 4: 0.003}

    # AR(1) noise to mimic autocorrelation in insurance time series
    noise = np.zeros(n_periods)
    noise[0] = rng.normal(0, noise_scale)
    for t in range(1, n_periods):
        noise[t] = 0.4 * noise[t - 1] + rng.normal(0, noise_scale)

    outcomes = []
    quarters = []

    for t_idx in range(n_periods):
        t = t_idx + 1  # 1-indexed
        q = ((t - 1) % 4) + 1
        quarters.append(q)

        # Pre-trend
        y = base_outcome + true_pre_trend * t

        # Post-intervention effects: level shift + slope change
        if t >= change_period:
            y += true_level_shift + true_slope_change * (t - change_period)

        # Seasonality
        if add_seasonality:
            y += seasonal[q]

        # Noise
        y += noise[t_idx]

        outcomes.append(y)

    df = pd.DataFrame({
        "period": np.arange(1, n_periods + 1),
        "outcome": outcomes,
        "exposure": [exposure_per_period] * n_periods,
        "quarter": quarters,
    })

    return df
