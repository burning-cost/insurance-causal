"""
Benchmark: Causal Forest GATE vs Uniform ATE for heterogeneous treatment effect recovery.

The question: when price elasticity genuinely varies by segment — young urban drivers
are highly elastic, older rural drivers barely respond to price — does a causal forest
with GATE estimation recover those segment-level effects better than using a single
population-average ATE?

Both estimators use the same causal forest (HeterogeneousElasticityEstimator): the
comparison is fair because they share the same nuisance models and estimand. The only
difference is whether you report a single ATE or per-segment GATEs. This isolates the
cost of ignoring heterogeneity.

Scale note
----------
CausalForestDML for a binary outcome estimates dY/dW (the local effect on renewal
probability per unit log-price-change), not the log-odds coefficient. The DGP uses
a logistic model with log-odds elasticities. The TRUE probability-scale effects are
computed as p_i(1 - p_i) * log_odds_coefficient, evaluated at each policy's baseline
renewal probability. These are the ground truth we compare against.

Setup
-----
- 20,000 synthetic UK motor renewal policies
- Features: driver_age, vehicle_value, urban indicator, ncd_years, region
- Treatment: log price change (continuous, mean ~0.07, std ~0.04)
- Outcome: binary renewal indicator (logistic DGP)
- Baseline renewal rate: ~80% (more realistic than 92%, gives cleaner signal)
- Log-odds semi-elasticities by segment (DGP parameters):
    - Young (< 35) + Urban: -6.0
    - Young (< 35) + Rural: -4.0
    - Mid (35-54) + Urban:  -3.5
    - Mid (35-54) + Rural:  -2.5
    - Senior (55+) + Urban: -2.0
    - Senior (55+) + Rural: -1.0
- Implied TRUE probability-scale ATE: computed from the DGP

Metrics
-------
- Segment RMSE: sqrt(mean((estimated_gate - true_gate)^2)) across 6 segments
- Extreme segment bias: |estimated - true| for most/least elastic segment
- Coverage: does the GATE CI contain the true segment effect?
- AUTOC (RATE): does the CATE ranking identify genuinely elastic customers?
- BLP beta2: is the heterogeneity statistically significant?

Run
---
    Executed on Databricks serverless, 2026-03-20.
"""

from __future__ import annotations

import time
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Data-generating process
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N = 20_000
N_REGIONS = 12

print("=" * 70)
print("insurance-causal benchmark")
print("Causal Forest GATE vs Uniform ATE — heterogeneous treatment effects")
print("=" * 70)

# Observed covariates
driver_age = RNG.integers(17, 80, N).astype(float)
vehicle_value = RNG.uniform(5_000, 40_000, N)
ncd_years = RNG.integers(0, 9, N).astype(float)
region = RNG.integers(0, N_REGIONS, N)
urban_base = 0.6 + 0.3 * (driver_age < 35).astype(float) - 0.2 * (region >= 8).astype(float)
urban = (RNG.uniform(size=N) < np.clip(urban_base, 0.05, 0.95)).astype(float)

age_band = np.select(
    [driver_age < 35, driver_age < 55],
    ["young", "mid"],
    default="senior",
)

# ---------------------------------------------------------------------------
# Log-odds semi-elasticity coefficients (DGP parameters)
# These are NOT what the forest estimates. See probability-scale computation below.
# ---------------------------------------------------------------------------

LOG_ODDS_ELASTICITY: Dict[Tuple[str, int], float] = {
    ("young",  1): -6.0,
    ("young",  0): -4.0,
    ("mid",    1): -3.5,
    ("mid",    0): -2.5,
    ("senior", 1): -2.0,
    ("senior", 0): -1.0,
}

log_odds_elasticity = np.array([
    LOG_ODDS_ELASTICITY[(b, int(u))]
    for b, u in zip(age_band, urban.astype(int))
])

# ---------------------------------------------------------------------------
# Confounded treatment: risky profiles receive larger technical rerates
# ---------------------------------------------------------------------------

ncb_risk = np.exp(-0.10 * ncd_years)
age_risk = np.exp(0.015 * np.maximum(35 - driver_age, 0))

technical_rerate = (
    0.05
    + 0.04 * ncb_risk
    + 0.03 * age_risk
    + 0.01 * urban
    + RNG.normal(0, 0.005, N)
)
exog = RNG.normal(0, 0.04, N)
log_price_change = technical_rerate + exog

# ---------------------------------------------------------------------------
# Renewal outcome: logistic model with baseline ~80% renewal
# Lower intercept than prior version (1.3 not 2.5) gives more signal
# ---------------------------------------------------------------------------

# Baseline intercept: intercept around 1.3 gives ~80% renewal before price effects
intercept = (
    1.3
    + 0.04 * ncd_years
    - 0.10 * urban
    + 0.005 * (vehicle_value / 10_000)
    + RNG.normal(0, 0.08, N)
)
log_odds = intercept + log_odds_elasticity * log_price_change
renewal_prob = 1.0 / (1.0 + np.exp(-log_odds))
renewed = RNG.binomial(1, renewal_prob).astype(float)

# ---------------------------------------------------------------------------
# TRUE probability-scale CATE: dY/dW = p(1-p) * log_odds_coefficient
# This is what CausalForestDML (linear DML) actually estimates.
# ---------------------------------------------------------------------------

# Per-policy true effect: p_i(1-p_i) * log_odds_elasticity_i
true_prob_cate = renewal_prob * (1.0 - renewal_prob) * log_odds_elasticity

TRUE_ATE_PROB = float(np.mean(true_prob_cate))

# True GATE per segment (mean of per-policy true probability effects)
segment = pd.Series(age_band) + "_" + pd.Series(urban.astype(int)).map({1: "urban", 0: "rural"})
segment = segment.values

TRUE_GATE_PROB: Dict[str, float] = {}
for (band, urb), _ in LOG_ODDS_ELASTICITY.items():
    seg_label = f"{band}_{'urban' if urb else 'rural'}"
    mask = segment == seg_label
    TRUE_GATE_PROB[seg_label] = float(np.mean(true_prob_cate[mask]))

print(f"\nDGP: {N:,} policies, {N_REGIONS} regions")
print(f"Renewal rate: {renewed.mean():.1%}")
print(f"Price change: mean={log_price_change.mean():.4f}  std={log_price_change.std():.4f}")
print()
print(f"True ATE (probability scale): {TRUE_ATE_PROB:.4f}")
print()
print("True segment effects (probability-scale dY/dW):")
print(f"  {'Segment':<22} {'Log-odds coeff':>16} {'Prob-scale GATE':>16} {'N':>6}")
print("  " + "-" * 64)
for seg_label, prob_gate in sorted(TRUE_GATE_PROB.items()):
    parts = seg_label.rsplit("_", 1)
    band, urb_str = parts[0], parts[1]
    lo_coeff = LOG_ODDS_ELASTICITY[(band, 1 if urb_str == "urban" else 0)]
    n_seg = int(np.sum(segment == seg_label))
    print(f"  {seg_label:<22} {lo_coeff:>16.1f} {prob_gate:>16.4f} {n_seg:>6,}")
print()
print(f"Correlation(price, log_odds_elasticity): {np.corrcoef(log_price_change, log_odds_elasticity)[0,1]:.3f}")
print()

df = pd.DataFrame({
    "driver_age":       driver_age,
    "vehicle_value":    vehicle_value,
    "ncd_years":        ncd_years,
    "region":           region.astype(float),
    "urban":            urban,
    "log_price_change": log_price_change,
    "renewed":          renewed,
    "true_prob_cate":   true_prob_cate,
    "segment":          segment,
})

CONFOUNDERS = ["driver_age", "vehicle_value", "ncd_years", "region", "urban"]

# ---------------------------------------------------------------------------
# 2. Fit one causal forest — used for both ATE and GATE comparison
#    nuisance_backend="sklearn" is required; catboost incompatible with
#    econml==0.15.1's score() API (CatBoostRegressor.score does not accept
#    sample_weight kwarg).
# ---------------------------------------------------------------------------

print("=" * 70)
print("Fitting HeterogeneousElasticityEstimator (shared model for ATE + GATE)")
print("=" * 70)

from insurance_causal.causal_forest.estimator import HeterogeneousElasticityEstimator
from insurance_causal.causal_forest.inference import HeterogeneousInference
from insurance_causal.causal_forest.targeting import TargetingEvaluator

t0 = time.perf_counter()
est = HeterogeneousElasticityEstimator(
    outcome_family="binary",
    n_estimators=500,
    n_folds=5,
    nuisance_backend="sklearn",   # catboost incompatible with econml==0.15.1 score()
    min_samples_leaf=20,
    random_state=42,
)
est.fit(
    df,
    outcome="renewed",
    treatment="log_price_change",
    confounders=CONFOUNDERS,
)
fit_time = time.perf_counter() - t0
print(f"Fit time: {fit_time:.1f}s")
print()

# ---------------------------------------------------------------------------
# 3. Uniform ATE (forest ATE — same estimand as GATE, single number)
# ---------------------------------------------------------------------------

print("=" * 70)
print("Estimator 1: Uniform ATE (forest.ate() — ignores heterogeneity)")
print("=" * 70)

ate_val, ate_lb, ate_ub = est.ate()
ate_bias = abs(ate_val - TRUE_ATE_PROB)

print(f"  Forest ATE estimate:  {ate_val:.4f}")
print(f"  True ATE (prob scale):{TRUE_ATE_PROB:.4f}")
print(f"  ATE bias:             {ate_bias:.4f} ({ate_bias/abs(TRUE_ATE_PROB)*100:.1f}%)")
print(f"  95% CI:               ({ate_lb:.4f}, {ate_ub:.4f})")
print(f"  CI covers true ATE:   {ate_lb <= TRUE_ATE_PROB <= ate_ub}")
print()

print("  Implied segment-level error when using uniform ATE for all segments:")
print(f"  {'Segment':<22} {'True prob-scale':>16} {'ATE (uniform)':>14} {'Bias':>10}")
print("  " + "-" * 65)
uniform_biases = {}
for seg_label, true_gate in sorted(TRUE_GATE_PROB.items()):
    bias = ate_val - true_gate
    uniform_biases[seg_label] = bias
    print(f"  {seg_label:<22} {true_gate:>16.4f} {ate_val:>14.4f} {bias:>+10.4f}")

uniform_rmse = float(np.sqrt(np.mean(list(b**2 for b in uniform_biases.values()))))
print(f"\n  Segment RMSE (uniform ATE vs true segment effects): {uniform_rmse:.4f}")
print()

# ---------------------------------------------------------------------------
# 4. GATE by segment
# ---------------------------------------------------------------------------

print("=" * 70)
print("Estimator 2: GATE by age_band x urban segment")
print("=" * 70)

gate_df = est.gate(df, by="segment")
gate_rows = gate_df.to_pandas() if hasattr(gate_df, "to_pandas") else gate_df

print(f"  {'Segment':<22} {'True':>8} {'GATE':>8} {'CI lo':>8} {'CI hi':>8} {'Bias':>8} {'Covers':>8} {'N':>6}")
print("  " + "-" * 80)

gate_biases = []
n_covered = 0

for _, row in gate_rows.iterrows():
    seg = row["segment"]
    true_eff = TRUE_GATE_PROB.get(seg, np.nan)
    gate_est = row["elasticity"]
    ci_lo = row["ci_lower"]
    ci_hi = row["ci_upper"]
    bias = gate_est - true_eff
    covers = ci_lo <= true_eff <= ci_hi

    if not np.isnan(bias):
        gate_biases.append(bias)
    if covers:
        n_covered += 1

    print(
        f"  {seg:<22} {true_eff:>8.4f} {gate_est:>8.4f} "
        f"{ci_lo:>8.4f} {ci_hi:>8.4f} {bias:>+8.4f} "
        f"{'YES' if covers else 'NO':>8} {int(row['n']):>6}"
    )

gate_rmse = float(np.sqrt(np.mean([b**2 for b in gate_biases])))
n_segments = len(gate_rows)
coverage_rate = n_covered / n_segments

print(f"\n  GATE RMSE vs true segment effects: {gate_rmse:.4f}")
print(f"  CI coverage rate: {n_covered}/{n_segments} = {coverage_rate:.0%}")
print()

# CATE correlation with ground truth
cate_vals = est.cate(df)
cate_corr = np.corrcoef(cate_vals, df["true_prob_cate"].values)[0, 1]
print(f"  Correlation(estimated CATE, true prob-scale CATE): {cate_corr:.3f}")
print(f"  Per-row CATE: mean={cate_vals.mean():.4f}  std={cate_vals.std():.4f}")
print()

# ---------------------------------------------------------------------------
# 5. GATES / BLP inference (Chernozhukov et al. 2025)
# ---------------------------------------------------------------------------

print("=" * 70)
print("GATES / BLP inference (Chernozhukov et al. 2025)")
print("=" * 70)

t_inf = time.perf_counter()
inference = HeterogeneousInference(n_groups=5, n_splits=20, random_state=42)
inference_result = inference.fit(
    estimator=est,
    df=df,
    outcome="renewed",
    treatment="log_price_change",
    confounders=CONFOUNDERS,
)
print(f"Inference time: {time.perf_counter() - t_inf:.1f}s")
print()
print(inference_result.summary())
print()

# ---------------------------------------------------------------------------
# 6. RATE / AUTOC targeting evaluation
# ---------------------------------------------------------------------------

print("=" * 70)
print("RATE / AUTOC — does the CATE ranking add targeting value?")
print("=" * 70)

t_rate = time.perf_counter()
evaluator = TargetingEvaluator(method="autoc", n_bootstrap=200, random_state=42)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    rate_result = evaluator.fit(
        estimator=est,
        df=df,
        outcome="renewed",
        treatment="log_price_change",
        confounders=CONFOUNDERS,
    )
print(f"RATE time: {time.perf_counter() - t_rate:.1f}s")
print(f"AUTOC (RATE):  {rate_result.rate:.4f}")
print(f"Bootstrap SE:  {rate_result.se:.4f}")
print(f"p-value:       {rate_result.p_value:.4f}  {'(significant)' if rate_result.p_value < 0.05 else '(not significant)'}")
print()

# ---------------------------------------------------------------------------
# 7. Summary comparison
# ---------------------------------------------------------------------------

print("=" * 70)
print("SUMMARY COMPARISON")
print("=" * 70)
print()
print(f"{'Metric':<48} {'Uniform ATE':>12} {'GATE':>12}")
print("-" * 75)
print(f"{'ATE estimate':<48} {ate_val:>12.4f} {cate_vals.mean():>12.4f}")
print(f"{'True ATE (probability scale)':<48} {TRUE_ATE_PROB:>12.4f} {TRUE_ATE_PROB:>12.4f}")
print(f"{'ATE bias':<48} {ate_bias:>12.4f} {abs(cate_vals.mean()-TRUE_ATE_PROB):>12.4f}")
print(f"{'Segment RMSE vs true segment effects':<48} {uniform_rmse:>12.4f} {gate_rmse:>12.4f}")

# Most elastic segment (young urban)
yu_true = TRUE_GATE_PROB.get("young_urban", np.nan)
yu_row = gate_rows[gate_rows["segment"] == "young_urban"]
if len(yu_row) > 0 and not np.isnan(yu_true):
    print(f"{'Bias on most elastic (young_urban)':<48} {abs(ate_val-yu_true):>12.4f} {abs(yu_row['elasticity'].values[0]-yu_true):>12.4f}")

# Least elastic segment (senior rural)
sr_true = TRUE_GATE_PROB.get("senior_rural", np.nan)
sr_row = gate_rows[gate_rows["segment"] == "senior_rural"]
if len(sr_row) > 0 and not np.isnan(sr_true):
    print(f"{'Bias on least elastic (senior_rural)':<48} {abs(ate_val-sr_true):>12.4f} {abs(sr_row['elasticity'].values[0]-sr_true):>12.4f}")

sig_str = f"p={rate_result.p_value:.3f} {'sig' if rate_result.p_value < 0.05 else 'n.s.'}"
print(f"{'AUTOC (RATE): targeting value above random':<48} {'N/A':>12} {sig_str:>12}")
print(f"{'CI coverage rate (all 6 segments)':<48} {'N/A':>12} {coverage_rate:.0%}")
print(f"{'Corr(estimated CATE, true CATE)':<48} {'N/A':>12} {cate_corr:>12.3f}")
print(f"{'Fit time (s)':<48} {fit_time:>12.1f} {'(shared)':>12}")

print()
print("INTERPRETATION:")
print("-" * 70)
print()
print("Both estimators share the same causal forest fit. Uniform ATE = forest.ate().")
print("GATE = forest.gate(by='segment'). The comparison isolates the value of")
print("reporting segment-level heterogeneity vs a single portfolio average.")
print()
print("The true effects are probability-scale semi-elasticities (dY/dW where")
print("Y=renewal indicator, W=log_price_change). CausalForestDML estimates this")
print("directly via the linear DML moment condition.")
print()
print("If GATE RMSE < uniform segment RMSE AND AUTOC is significant:")
print("  -> Segment-level pricing based on GATE estimates adds value.")
print()
print("If AUTOC is not significant:")
print("  -> The CATE ranking is noise. Use the uniform ATE from forest.ate().")
print()
print("Honest caveats:")
print("  - This DGP has step-function heterogeneity (exactly 6 segments).")
print("    Real portfolios have smoother, harder-to-detect variation.")
print("  - nuisance_backend='sklearn' is used (catboost incompatible with")
print("    econml==0.15.1 score() API).")
print("  - GATE CIs use the forest's internal IJ variance estimate, which")
print("    tends to be anti-conservative for binary outcomes.")
