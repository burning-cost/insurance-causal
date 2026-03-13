"""
Synthetic UK motor renewal data generator.

Generates realistic renewal datasets with a known data-generating process (DGP),
useful for validating the elasticity estimator and running end-to-end demos.

The true elasticity varies by NCD band and age group, so the generator serves as
a ground-truth benchmark: fit the estimator on generated data and compare GATE
estimates against the known truth.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from typing import Optional


# True elasticities by segment in the DGP.
# These are semi-elasticities: a 1-unit increase in log_price_change
# changes renewal probability by this many percentage points.
_TRUE_ATE: float = -2.0  # overall average

_TRUE_ELASTICITY_BY_NCD: dict[int, float] = {
    0: -3.5,  # no NCD: most elastic (young, price-constrained)
    1: -3.0,
    2: -2.5,
    3: -2.0,
    4: -1.5,
    5: -1.0,  # max NCD: most inelastic (value their NCD protection)
}

_TRUE_ELASTICITY_BY_AGE: dict[str, float] = {
    "17-24": -3.0,   # price-constrained, PCW-native
    "25-34": -2.5,
    "35-44": -2.0,
    "45-54": -1.8,
    "55-64": -1.5,
    "65+":   -1.2,   # less price-sensitive, lower switching behaviour
}

_CHANNELS = ["pcw", "direct", "broker"]
_REGIONS = ["London", "South East", "South West", "Midlands", "North West",
            "North East", "Scotland", "Wales"]
_VEHICLE_GROUPS = ["A", "B", "C", "D", "E", "F"]


def make_renewal_data(
    n: int = 10_000,
    seed: int = 42,
    price_variation_sd: float = 0.08,
    near_deterministic: bool = False,
) -> pl.DataFrame:
    """Generate a synthetic UK motor renewal dataset with known elasticity DGP.

    The dataset mimics a real insurer's renewal book: risk factors determine
    the base technical premium, the offer price is close to the technical price
    (optionally with near-zero exogenous variation), and the renewal decision
    follows a logistic model with heterogeneous price sensitivity.

    Parameters
    ----------
    n:
        Number of renewal records.
    seed:
        Random seed for reproducibility.
    price_variation_sd:
        Standard deviation of the exogenous log price change around the
        re-rated technical change. Larger values give more treatment variation
        for DML to exploit. Set low (e.g. 0.01) to simulate the
        near-deterministic price problem.
    near_deterministic:
        Convenience flag. If True, sets price_variation_sd=0.01 to simulate
        the near-deterministic price problem described in KB 597.

    Returns
    -------
    polars.DataFrame with columns:
        policy_id, age, age_band, ncd_years, region, vehicle_group, channel,
        payment_method, last_premium, tech_prem, enbp, log_price_change,
        renewed, true_elasticity
    """
    if near_deterministic:
        price_variation_sd = 0.01

    rng = np.random.default_rng(seed)

    # --- Demographics ---
    age = rng.integers(17, 80, size=n)
    age_band = _age_to_band(age)

    ncd_years = np.clip(rng.integers(0, 8, size=n), 0, 5)

    region = rng.choice(_REGIONS, size=n)
    vehicle_group = rng.choice(_VEHICLE_GROUPS, size=n, p=[0.25, 0.25, 0.20, 0.15, 0.10, 0.05])
    channel = rng.choice(_CHANNELS, size=n, p=[0.55, 0.35, 0.10])
    payment_method = rng.choice(["direct_debit", "annual"], size=n, p=[0.65, 0.35])

    # --- Technical premium ---
    # Base premium driven by risk factors (realistic range £300–£2,500)
    base_log_prem = (
        5.5  # ~£245 base
        + 0.8 * (age < 25).astype(float)           # young driver loading
        + 0.3 * (age > 70).astype(float)            # elderly loading
        - 0.06 * ncd_years                           # NCD discount
        + 0.5 * (channel == "pcw").astype(float)    # PCW tends to attract higher-risk
        + rng.normal(0, 0.15, size=n)               # residual risk noise
    )

    # Vehicle group loading
    vg_loading = {"A": 0.0, "B": 0.1, "C": 0.2, "D": 0.35, "E": 0.5, "F": 0.7}
    vg_load_arr = np.array([vg_loading[v] for v in vehicle_group])
    base_log_prem = base_log_prem + vg_load_arr

    # Region loading
    region_loading = {
        "London": 0.4, "South East": 0.15, "South West": 0.0,
        "Midlands": 0.1, "North West": 0.2, "North East": 0.05,
        "Scotland": -0.05, "Wales": -0.1,
    }
    region_load_arr = np.array([region_loading[r] for r in region])
    base_log_prem = base_log_prem + region_load_arr

    last_premium = np.exp(base_log_prem)

    # Technical premium = last year's + risk re-rating (nearly deterministic)
    # Re-rating is a function of risk factors only
    rerate_pct = (
        0.05                                         # market-wide 5% increase
        + 0.01 * (ncd_years < 2).astype(float)      # young/low NCD surcharge
        + rng.normal(0, 0.005, size=n)              # pricing grid rounding noise
    )
    tech_prem = last_premium * np.exp(rerate_pct)

    # ENBP: NB model applied to renewal customer's profile.
    # In practice this comes from the firm; here we simulate as tech_prem
    # times a channel factor, minus any cashback deduction.
    enbp_factor = np.where(channel == "pcw", 0.98, 1.02)
    enbp = tech_prem * enbp_factor * np.exp(rng.normal(0, 0.02, size=n))

    # --- Offer price ---
    # The exogenous variation (small, representing A/B testing or UW override)
    exog_variation = rng.normal(0, price_variation_sd, size=n)
    log_price_change = rerate_pct + exog_variation  # D_i

    offer_price = last_premium * np.exp(log_price_change)

    # Enforce ENBP constraint (post-PS21/5)
    offer_price = np.minimum(offer_price, enbp)
    log_price_change = np.log(offer_price / last_premium)

    # --- True elasticity (heterogeneous by NCD and age) ---
    ncd_elasticity = np.array([_TRUE_ELASTICITY_BY_NCD[int(n_)] for n_ in ncd_years])
    age_elasticity = np.array([_TRUE_ELASTICITY_BY_AGE[b] for b in age_band])
    true_elasticity = 0.6 * ncd_elasticity + 0.4 * age_elasticity

    # Channel moderation: PCW customers are 30% more elastic
    pcw_mask = channel == "pcw"
    true_elasticity = np.where(pcw_mask, true_elasticity * 1.3, true_elasticity)

    # --- Renewal decision ---
    # P(renew) = logistic(intercept + elasticity * log_price_change + confounders)
    intercept = (
        2.0                                          # ~88% base renewal at no price change
        - 0.3 * (channel == "pcw").astype(float)    # PCW base churn higher
        + 0.1 * np.minimum(ncd_years, 3) * 0.1     # long-term NCD loyalty
        + rng.normal(0, 0.05, size=n)               # individual unobserved heterogeneity
    )
    log_odds = intercept + true_elasticity * log_price_change
    renewal_prob = 1 / (1 + np.exp(-log_odds))
    renewed = rng.binomial(1, renewal_prob)

    return pl.DataFrame({
        "policy_id": np.arange(1, n + 1),
        "age": age,
        "age_band": age_band,
        "ncd_years": ncd_years,
        "region": region,
        "vehicle_group": vehicle_group,
        "channel": channel,
        "payment_method": payment_method,
        "last_premium": np.round(last_premium, 2),
        "tech_prem": np.round(tech_prem, 2),
        "enbp": np.round(enbp, 2),
        "offer_price": np.round(offer_price, 2),
        "log_price_change": log_price_change,
        "renewal_prob_true": renewal_prob,
        "renewed": renewed,
        "true_elasticity": true_elasticity,
    })


def true_gate_by_ncd(df: pl.DataFrame) -> pl.DataFrame:
    """Return the ground-truth group average elasticity by NCD band.

    Useful for comparing estimated GATEs against the known DGP after fitting
    on synthetic data.

    Parameters
    ----------
    df:
        DataFrame returned by :func:`make_renewal_data`.

    Returns
    -------
    polars.DataFrame with columns: ncd_years, true_elasticity_mean, n
    """
    return (
        df
        .group_by("ncd_years")
        .agg([
            pl.col("true_elasticity").mean().alias("true_elasticity_mean"),
            pl.len().alias("n"),
        ])
        .sort("ncd_years")
    )


def true_gate_by_age(df: pl.DataFrame) -> pl.DataFrame:
    """Return the ground-truth group average elasticity by age band.

    Parameters
    ----------
    df:
        DataFrame returned by :func:`make_renewal_data`.

    Returns
    -------
    polars.DataFrame with columns: age_band, true_elasticity_mean, n
    """
    age_order = ["17-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    result = (
        df
        .group_by("age_band")
        .agg([
            pl.col("true_elasticity").mean().alias("true_elasticity_mean"),
            pl.len().alias("n"),
        ])
    )
    # Sort by defined age band order
    order_map = {band: i for i, band in enumerate(age_order)}
    result = result.with_columns(
        pl.col("age_band").replace(order_map).alias("_order")
    ).sort("_order").drop("_order")
    return result


def _age_to_band(age: np.ndarray) -> list[str]:
    """Map integer ages to age band strings."""
    bands = []
    for a in age:
        if a < 25:
            bands.append("17-24")
        elif a < 35:
            bands.append("25-34")
        elif a < 45:
            bands.append("35-44")
        elif a < 55:
            bands.append("45-54")
        elif a < 65:
            bands.append("55-64")
        else:
            bands.append("65+")
    return bands
