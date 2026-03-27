"""
Binary treatment: causal effect of aggregator channel on claim frequency
========================================================================

This script demonstrates DML with a binary treatment on a realistic UK
personal lines scenario:

    Did quoting via a price comparison website (PCW) cause higher claim
    frequency, or does the PCW channel simply attract a higher-risk
    customer mix that would have been more expensive regardless?

The two effects are:
  (a) Causal channel effect — the PCW itself changes behaviour (faster
      claims, lower loyalty, higher switching). This is real.
  (b) Selection effect — riskier customers are more likely to search via
      PCW. This is confounding.

A naive GLM cannot separate (a) from (b). DML can, by partialling out
the influence of the risk factors on both channel assignment and the
outcome before regressing one residual on the other.

Run on Databricks (recommended):
    Import via Repos, attach to a cluster with insurance-causal installed.
    Expected runtime: under 2 minutes.

Run locally:
    pip install "insurance-causal"
    python examples/binary_treatment.py

The true causal channel effect is embedded in the data generation process
below. Check that the DML estimate covers it and the naive GLM does not.
"""

import numpy as np
import polars as pl
from sklearn.linear_model import PoissonRegressor

from insurance_causal import CausalPricingModel
from insurance_causal.treatments import BinaryTreatment


# ---------------------------------------------------------------------------
# 1. Synthetic UK motor new business book
# ---------------------------------------------------------------------------
# Confounding structure:
#   - Young drivers with zero NCD are more likely to quote via PCW
#   - Young drivers with zero NCD are also higher-risk independently
#   - The naive GLM conflates these two channels

RNG = np.random.default_rng(2024)
N = 15_000
TRUE_CAUSAL_CHANNEL_EFFECT = 0.08   # PCW causes ~8pp higher claim frequency
                                     # (due to adverse selection within the
                                     # "attracted" pool and faster claims culture)

# Rating factors
driver_age   = RNG.integers(18, 75, N)
ncb_years    = RNG.integers(0, 9, N)
vehicle_age  = RNG.integers(1, 15, N)
prior_claims = RNG.integers(0, 3, N)
region       = RNG.choice(
    ["London", "SE", "Midlands", "North", "Scotland"], N,
    p=[0.18, 0.22, 0.25, 0.25, 0.10],
)

# Risk score — partially observed through rating factors
latent_risk = (
    0.05 * np.maximum(30 - driver_age, 0)   # young driver surcharge
    + 0.15 * prior_claims
    - 0.04 * ncb_years
    + 0.02 * vehicle_age
    + RNG.normal(0, 0.20, N)
)

# Treatment: PCW channel indicator.
# High-risk customers are MORE likely to quote via PCW (the confounding).
# Logistic probability of PCW usage:
pcw_logit = -0.5 + 1.2 * latent_risk
pcw_prob  = 1 / (1 + np.exp(-pcw_logit))
is_pcw    = (RNG.uniform(N) < pcw_prob).astype(int)

# Outcome: claim frequency (claims per policy year).
# Two drivers:
#   (a) latent risk: higher-risk customers claim more regardless of channel
#   (b) causal channel effect: PCW customers have slightly higher frequency
lam = np.exp(
    -2.5                                        # base log-frequency
    + TRUE_CAUSAL_CHANNEL_EFFECT * is_pcw       # causal channel effect
    + 0.80 * latent_risk                        # risk-driven frequency
    + 0.01 * np.maximum(30 - driver_age, 0)
    + RNG.normal(0, 0.05, N)
)
claim_count = RNG.poisson(lam)
exposure    = RNG.uniform(0.5, 1.0, N)  # earned years (0.5–1.0)

age_band = np.where(driver_age < 25, "17-24",
           np.where(driver_age < 35, "25-34",
           np.where(driver_age < 55, "35-54", "55+")))

df = pl.DataFrame({
    "claim_count":  claim_count.astype(float),
    "exposure":     exposure,
    "is_pcw":       is_pcw,
    "age_band":     age_band,
    "ncb_years":    ncb_years.astype(float),
    "vehicle_age":  vehicle_age.astype(float),
    "prior_claims": prior_claims.astype(float),
    "region":       region,
})

pcw_share = is_pcw.mean()
print(f"Dataset: {N:,} policies")
print(f"PCW share: {pcw_share:.1%}")
print(f"Overall claim frequency: {(claim_count / exposure).mean():.3f} per year")
print(f"PCW frequency:    {(claim_count[is_pcw == 1] / exposure[is_pcw == 1]).mean():.3f} per year")
print(f"Direct frequency: {(claim_count[is_pcw == 0] / exposure[is_pcw == 0]).mean():.3f} per year")
print(f"True causal effect: {TRUE_CAUSAL_CHANNEL_EFFECT:.4f}")
print()


# ---------------------------------------------------------------------------
# 2. Naive Poisson GLM — biased
# ---------------------------------------------------------------------------
# Regressing claim frequency on the PCW indicator gives a biased estimate
# because we cannot fully account for the risk selection into the PCW channel.

df_pd = df.to_pandas()
freq = df_pd["claim_count"] / df_pd["exposure"]

naive_glm = PoissonRegressor(max_iter=500)
naive_glm.fit(df_pd[["is_pcw"]], freq)
naive_coef = float(naive_glm.coef_[0])

print("--- Naive Poisson GLM (biased) ---")
print(f"Coefficient: {naive_coef:.4f}  (true causal = {TRUE_CAUSAL_CHANNEL_EFFECT})")
print(f"Bias:        {naive_coef - TRUE_CAUSAL_CHANNEL_EFFECT:+.4f}")
print()


# ---------------------------------------------------------------------------
# 3. DML estimate — confounding removed
# ---------------------------------------------------------------------------
# BinaryTreatment uses DoubleMLIRM (Interactive Regression Model) under the
# hood, which is the appropriate DML variant for binary treatments. The
# nuisance models are:
#   - E[Y|X, D=1] and E[Y|X, D=0]: outcome models for treated and control
#   - E[D|X]: propensity score model
# The ATE is the average causal effect over the full population.

model = CausalPricingModel(
    outcome="claim_count",
    outcome_type="poisson",
    exposure_col="exposure",
    treatment=BinaryTreatment(
        column="is_pcw",
        positive_label="PCW",
        negative_label="direct",
    ),
    confounders=["age_band", "ncb_years", "vehicle_age", "prior_claims", "region"],
    cv_folds=5,
    random_state=42,
)

print("Fitting DML model (5-fold cross-fitting)...")
model.fit(df)

ate = model.average_treatment_effect()
print()
print("--- DML estimate ---")
print(ate)
print()

inside_ci = ate.ci_lower <= TRUE_CAUSAL_CHANNEL_EFFECT <= ate.ci_upper
print(f"True value ({TRUE_CAUSAL_CHANNEL_EFFECT}) inside 95% CI: {inside_ci}")
print()


# ---------------------------------------------------------------------------
# 4. What does this mean commercially?
# ---------------------------------------------------------------------------
# If the naive GLM says PCW causes +15pp higher frequency but DML says the
# causal effect is only +8pp, the pricing team has been loading PCW customers
# by 7pp too much. Under FCA Consumer Duty, that loading needs a causal
# justification — and "our GLM coefficient is large" is not sufficient if
# most of the coefficient is selection, not causation.

print("--- Commercial interpretation ---")
print(f"Naive GLM PCW loading:  +{naive_coef:.1%} frequency uplift")
print(f"DML causal effect:      +{ate.estimate:.1%} frequency uplift")
print()
if naive_coef > ate.estimate * 1.2:
    print("The naive loading is materially overstated.")
    print("Applying it to new PCW business would over-price by roughly")
    print(f"  {naive_coef - ate.estimate:.1%} on claim frequency.")
    print("This is commercially significant and creates a fairness risk under Consumer Duty.")


# ---------------------------------------------------------------------------
# 5. CATE by segment
# ---------------------------------------------------------------------------
# Does the PCW effect vary by age band? Young drivers on PCW may be
# qualitatively different from the PCW population average.

print()
print("Fitting CATE by age band...")
cate_df = model.cate_by_segment(df, segment_col="age_band")

print()
print("--- PCW causal effect by age band ---")
print(cate_df[["segment", "cate_estimate", "ci_lower", "ci_upper",
               "p_value", "n_obs"]].to_string(index=False))
print()
print("If 17-24 has a larger CATE than 55+, a segment-specific PCW loading")
print("would be more accurate than applying the book-level ATE uniformly.")
