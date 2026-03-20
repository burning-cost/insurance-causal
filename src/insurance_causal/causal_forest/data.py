"""
Synthetic DGP with known heterogeneous treatment effects for testing the
causal_forest subpackage.

The DGP mimics UK motor renewal data where the true price elasticity
varies by NCD (no-claims discount) band. NCD band 0 customers are most
elastic; NCD band 5+ are most inelastic. This known heterogeneity allows
ground-truth validation of GATES and RATE estimates.

The treatment is log_price_change, which has both a confounded component
(determined by risk factors) and an exogenous component (random variation,
representing A/B test nudges or rounding differences). The exogenous
component is what DML exploits for identification.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from typing import Optional


# True CATE by NCD band (semi-elasticity: unit = change in renewal prob
# per 1-unit increase in log price change).
_TRUE_CATE_BY_NCD: dict[int, float] = {
    0: -0.30,  # no NCD: most price-elastic (young, price-constrained drivers)
    1: -0.26,
    2: -0.22,
    3: -0.18,
    4: -0.14,
    5: -0.10,  # max NCD: least price-elastic (value their NCD protection)
}

_CHANNELS = ["pcw", "direct", "broker"]
_REGIONS = ["London", "South East", "Midlands", "North West", "Scotland"]
_VEHICLE_GROUPS = ["A", "B", "C", "D", "E"]


def make_hte_renewal_data(
    n: int = 10_000,
    seed: int = 42,
    price_sd: float = 0.10,
) -> pl.DataFrame:
    """Generate synthetic UK motor renewal data with known heterogeneous elasticity.

    The data-generating process produces treatment effect heterogeneity driven by
    NCD band: ``tau(x) = -0.30`` for NCD=0 and ``tau(x) = -0.10`` for NCD=5+.
    This known structure allows BLP, GATES, and RATE tests to verify that the
    estimators recover the true pattern.

    The treatment ``log_price_change`` is confounded: risk factors (NCD, age,
    channel) determine most of the price change, but there is an exogenous
    component of standard deviation ``price_sd``. DML cross-fitting removes the
    confounded portion.

    Parameters
    ----------
    n:
        Number of renewal records.
    seed:
        Random seed for reproducibility.
    price_sd:
        Standard deviation of the exogenous price variation around the technical
        re-rate. Larger values give more treatment variation; lower values simulate
        the near-deterministic price problem.

    Returns
    -------
    polars.DataFrame with columns:
        policy_id, age, ncd_years, region, vehicle_group, channel,
        log_price_change, renewed, true_cate, exposure
    """
    rng = np.random.default_rng(seed)

    # --- Risk factors ---
    age = rng.integers(17, 80, size=n).astype(float)
    ncd_years = np.clip(rng.integers(0, 8, size=n), 0, 5)
    region = rng.choice(_REGIONS, size=n)
    vehicle_group = rng.choice(_VEHICLE_GROUPS, size=n, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    channel = rng.choice(_CHANNELS, size=n, p=[0.55, 0.35, 0.10])

    # --- Confounded treatment (log price change) ---
    # Technical re-rate: driven by observable risk factors
    tech_rerate = (
        0.06                                          # market-wide 6% increase
        + 0.015 * (ncd_years < 2).astype(float)      # low-NCD surcharge
        - 0.008 * ncd_years                           # NCD discount
        + 0.010 * (channel == "pcw").astype(float)   # PCW channel adjustment
        + 0.005 * (age < 25).astype(float)            # young driver loading
        + rng.normal(0, 0.005, size=n)               # pricing grid noise
    )

    # Exogenous variation (the variation DML uses for identification)
    exog = rng.normal(0, price_sd, size=n)

    log_price_change = tech_rerate + exog  # W_i in the DML model

    # --- True CATE (heterogeneous by NCD) ---
    true_cate = np.array([_TRUE_CATE_BY_NCD[int(ncd)] for ncd in ncd_years])

    # --- Renewal outcome (binary) ---
    # P(renewed) = sigmoid(intercept + tau(x)*W + unobserved noise)
    intercept = (
        2.2                                           # ~90% base renewal rate
        - 0.4 * (channel == "pcw").astype(float)     # PCW base churn
        + 0.05 * np.minimum(ncd_years, 3)            # NCD loyalty
        + 0.3 * (age > 50).astype(float)             # older drivers more loyal
        + rng.normal(0, 0.1, size=n)                 # individual heterogeneity
    )
    log_odds = intercept + true_cate * log_price_change
    renewal_prob = 1.0 / (1.0 + np.exp(-log_odds))
    renewed = rng.binomial(1, renewal_prob).astype(float)

    # Exposure (years at risk) — for Poisson/rate use case
    exposure = np.ones(n)  # renewal data: 1 year each

    return pl.DataFrame({
        "policy_id": np.arange(1, n + 1),
        "age": age,
        "ncd_years": ncd_years,
        "region": region,
        "vehicle_group": vehicle_group,
        "channel": channel,
        "log_price_change": log_price_change,
        "renewed": renewed,
        "true_cate": true_cate,
        "exposure": exposure,
    })


def true_cate_by_ncd(df: pl.DataFrame) -> pl.DataFrame:
    """Return ground-truth group average CATE by NCD band.

    Use this to benchmark GATES estimates against the known DGP after fitting
    HeterogeneousElasticityEstimator on data from :func:`make_hte_renewal_data`.

    Parameters
    ----------
    df:
        DataFrame returned by :func:`make_hte_renewal_data`.

    Returns
    -------
    polars.DataFrame with columns: ncd_years, true_cate_mean, n
    """
    return (
        df
        .group_by("ncd_years")
        .agg([
            pl.col("true_cate").mean().alias("true_cate_mean"),
            pl.len().alias("n"),
        ])
        .sort("ncd_years")
    )
