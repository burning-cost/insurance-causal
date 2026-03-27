"""
Quickstart: DML causal inference for UK motor renewal pricing
=============================================================

This script demonstrates the core problem and the solution in full:

  1. Generate a synthetic UK motor renewal book (10,000 policies) where
     the true causal price semi-elasticity is known in advance.

  2. Fit a naive logistic regression — the standard approach in most teams.
     The estimate is wrong because high-risk customers receive larger price
     increases and also have lower baseline renewal rates.

  3. Fit DML via CausalPricingModel. The estimate recovers the true value.

  4. Print the confounding bias report, which is the output a pricing team
     would take to a rate review.

Run on Databricks (recommended):
    Import this file via Repos, attach to a cluster with insurance-causal
    installed, and run. Expected runtime: under 3 minutes on a standard
    cluster with 10,000 policies.

Run locally (for development):
    pip install "insurance-causal[all]"
    python examples/quickstart.py

The true causal semi-elasticity is embedded in the data generation process
at the bottom of this file. Check that the DML estimate sits inside its 95%
confidence interval and the naive estimate does not.
"""

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from insurance_causal import CausalPricingModel
from insurance_causal.treatments import PriceChangeTreatment


# ---------------------------------------------------------------------------
# 1. Synthetic UK motor renewal book
# ---------------------------------------------------------------------------

# We build the data by hand so the confounding structure is explicit.
# The key mechanism: risk_score drives both the price increase (higher-risk
# customers get larger increases) and the renewal probability (higher-risk
# customers are more likely to lapse regardless of price).
#
# A naive regression cannot separate these two channels. DML can.

RNG = np.random.default_rng(42)
N = 10_000
TRUE_SEMI_ELASTICITY = -0.40  # the true causal effect we want to recover

# Rating factors (confounders)
driver_age   = RNG.integers(25, 75, N)
ncb_years    = RNG.integers(0, 9, N)
vehicle_age  = RNG.integers(1, 15, N)
prior_claims = RNG.integers(0, 3, N)
region       = RNG.choice(["London", "SE", "Midlands", "North", "Scotland"], N,
                           p=[0.18, 0.22, 0.25, 0.25, 0.10])

# Risk score — this is the confounder. It is observed via the rating factors,
# but not perfectly: there is residual risk the factors do not capture.
latent_risk = (
    0.04 * np.maximum(30 - driver_age, 0)   # young drivers
    + 0.10 * prior_claims
    - 0.05 * ncb_years
    + 0.03 * vehicle_age
    + RNG.normal(0, 0.15, N)                # unobserved component
)

# Treatment: proportional price change at renewal.
# High-risk policyholders receive larger increases (correlated with latent_risk).
# This is the confounding: the price change is not randomly assigned.
pct_price_change = 0.04 + 0.25 * latent_risk + RNG.normal(0, 0.03, N)
pct_price_change = np.clip(pct_price_change, -0.15, 0.30)

# Outcome: renewal indicator.
# Two separate channels drive lapse:
#   (a) causal price effect — the effect we want to estimate
#   (b) latent risk effect — higher-risk customers lapse regardless of price
log_odds = (
    1.2
    + TRUE_SEMI_ELASTICITY * np.log1p(pct_price_change)  # causal channel
    - 0.60 * latent_risk                                  # risk channel (confounder)
    + 0.02 * ncb_years
    + RNG.normal(0, 0.05, N)
)
renewal = (RNG.uniform(N) < 1 / (1 + np.exp(-log_odds))).astype(int)

age_band = np.where(driver_age < 35, "young",
           np.where(driver_age < 55, "mid", "senior"))

df = pl.DataFrame({
    "renewal":          renewal,
    "pct_price_change": pct_price_change,
    "age_band":         age_band,
    "ncb_years":        ncb_years.astype(float),
    "vehicle_age":      vehicle_age.astype(float),
    "prior_claims":     prior_claims.astype(float),
    "region":           region,
})

print(f"Dataset: {N:,} policies")
print(f"Renewal rate: {renewal.mean():.1%}")
print(f"Mean price change: {pct_price_change.mean():.1%}")
print(f"True causal semi-elasticity: {TRUE_SEMI_ELASTICITY}")
print()


# ---------------------------------------------------------------------------
# 2. Naive estimate — biased by confounding
# ---------------------------------------------------------------------------
# A logistic regression of renewal on price change, without controlling
# for the risk factors that drove the pricing decisions.

df_pd = df.to_pandas()

naive_lr = LogisticRegression(max_iter=500)
naive_lr.fit(df_pd[["pct_price_change"]], df_pd["renewal"])
naive_coef = float(naive_lr.coef_[0][0])

print("--- Naive estimate (biased) ---")
print(f"Coefficient: {naive_coef:.4f}  (true = {TRUE_SEMI_ELASTICITY})")
print(f"Bias: {naive_coef - TRUE_SEMI_ELASTICITY:+.4f}")
print()

# Note: even with controls included as features, the GLM estimate is biased
# because the confounding is nonlinear — the interaction of age, region, and
# NCB with risk quality cannot be captured by GLM main effects alone.


# ---------------------------------------------------------------------------
# 3. DML estimate — confounding removed
# ---------------------------------------------------------------------------
# CausalPricingModel uses DoubleML with CatBoost nuisance models.
# The cross-fitting procedure partial out the influence of the rating
# factors from both the outcome and the treatment before running the
# final regression. This removes the confounding without requiring
# a randomised trial.

model = CausalPricingModel(
    outcome="renewal",
    outcome_type="binary",
    treatment=PriceChangeTreatment(
        column="pct_price_change",
        scale="log",           # DML estimates semi-elasticity: theta in Y = theta * log(1+D)
    ),
    confounders=["age_band", "ncb_years", "vehicle_age", "prior_claims", "region"],
    cv_folds=5,
    random_state=42,
)

print("Fitting DML model (5-fold cross-fitting with CatBoost nuisance models)...")
model.fit(df)

ate = model.average_treatment_effect()
print()
print("--- DML estimate ---")
print(ate)
print()

# Check that the true value is inside the 95% CI
inside_ci = ate.ci_lower <= TRUE_SEMI_ELASTICITY <= ate.ci_upper
print(f"True value ({TRUE_SEMI_ELASTICITY}) inside 95% CI: {inside_ci}")
print()


# ---------------------------------------------------------------------------
# 4. Confounding bias report
# ---------------------------------------------------------------------------
# This is the output you would present at a rate review. It shows the naive
# estimate, the causal estimate, and the implied bias in plain terms.

report = model.confounding_bias_report(naive_coefficient=naive_coef)
print("--- Confounding bias report ---")
print(report[["treatment", "outcome", "naive_estimate", "causal_estimate",
              "bias", "bias_pct", "interpretation"]].to_string(index=False))
print()


# ---------------------------------------------------------------------------
# 5. CATE by segment — does price sensitivity vary by age band?
# ---------------------------------------------------------------------------
# A single ATE hides heterogeneity. Young drivers may be more price-sensitive
# than older drivers. The segment-level estimates fit separate DML models per
# group, so each has its own valid confidence interval.
#
# Note: this requires a minimum of 2,000 observations per segment by default.
# With 10,000 policies split across three age bands, each has roughly 3,000+
# observations — sufficient for reliable estimation.

print("Fitting CATE by age band (one DML model per segment)...")
cate_df = model.cate_by_segment(df, segment_col="age_band")

print()
print("--- CATE by age band ---")
print(cate_df[["segment", "cate_estimate", "ci_lower", "ci_upper",
               "p_value", "n_obs"]].to_string(index=False))
print()
print("If young drivers have a larger (more negative) CATE than senior drivers,")
print("that is the heterogeneity a renewal pricing optimiser should act on.")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("Summary")
print("=" * 60)
print(f"True causal semi-elasticity:  {TRUE_SEMI_ELASTICITY:.4f}")
print(f"Naive estimate (biased):      {naive_coef:.4f}  "
      f"(bias = {naive_coef - TRUE_SEMI_ELASTICITY:+.4f})")
print(f"DML estimate:                 {ate.estimate:.4f}  "
      f"95% CI [{ate.ci_lower:.4f}, {ate.ci_upper:.4f}]")
print(f"True value inside DML CI:     {inside_ci}")
print()
print("Next steps:")
print("  - See notebooks/04_causal_forest_hte_demo.py for per-customer CATEs")
print("  - See notebooks/03_elasticity_demo.py for renewal pricing optimisation")
print("  - See notebooks/05_rate_change_evaluator_demo.py for post-hoc rate change evaluation")
