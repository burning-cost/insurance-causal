# Databricks notebook source
# MAGIC %md
# MAGIC # Benchmark: Causal Forest vs GLM Interaction Model — Heterogeneous Treatment Effects
# MAGIC
# MAGIC **Library:** `insurance-causal` — causal forest subpackage
# MAGIC
# MAGIC **Question:** When price elasticity varies by segment, does a causal forest recover
# MAGIC segment-level effects better than a Poisson GLM with interaction terms?
# MAGIC
# MAGIC The GLM with interactions is the natural baseline — it is what a pricing analyst
# MAGIC would already do when asked to estimate heterogeneous treatment effects. The causal
# MAGIC forest should beat the GLM on three things: confounding correction (DML partials
# MAGIC out the non-random treatment assignment), individual-level CATEs (GLM gives per-segment
# MAGIC constants), and detectability of heterogeneity not captured by pre-specified interactions.
# MAGIC
# MAGIC **DGP:** 20,000 synthetic UK motor policies, Poisson frequency outcome. Log-scale
# MAGIC semi-elasticities vary by age_band × urban: young urban −5.0, senior rural −0.8.
# MAGIC Treatment confounded by risk profile (technical rerate). Full methodology:
# MAGIC `benchmarks/benchmark_causal_forest.py`.
# MAGIC
# MAGIC **Date:** 2026-03-21
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %pip install "insurance-causal[all]" --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import warnings
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Data Generating Process

# COMMAND ----------

RNG = np.random.default_rng(42)
N = 20_000
N_REGIONS = 12

driver_age    = RNG.integers(17, 80, N).astype(float)
vehicle_value = RNG.uniform(5_000, 40_000, N)
ncd_years     = RNG.integers(0, 9, N).astype(float)
region        = RNG.integers(0, N_REGIONS, N)

urban_base = 0.6 + 0.3 * (driver_age < 35).astype(float) - 0.2 * (region >= 8).astype(float)
urban      = (RNG.uniform(size=N) < np.clip(urban_base, 0.05, 0.95)).astype(float)

age_band = np.select(
    [driver_age < 35, driver_age < 55],
    ["young", "mid"],
    default="senior",
)

# True log-scale semi-elasticities per segment
LOG_SCALE_ELASTICITY: Dict[Tuple[str, int], float] = {
    ("young", 1): -5.0, ("young", 0): -3.5,
    ("mid",   1): -3.0, ("mid",   0): -2.0,
    ("senior",1): -1.5, ("senior",0): -0.8,
}

log_scale_elasticity = np.array([
    LOG_SCALE_ELASTICITY[(b, int(u))]
    for b, u in zip(age_band, urban.astype(int))
])

# Confounded treatment: risky profiles get larger rerates
ncb_risk = np.exp(-0.10 * ncd_years)
age_risk = np.exp(0.015 * np.maximum(35 - driver_age, 0))
technical_rerate = 0.05 + 0.04 * ncb_risk + 0.03 * age_risk + 0.01 * urban + RNG.normal(0, 0.005, N)
exog = RNG.normal(0, 0.04, N)
log_price_change = technical_rerate + exog

# Poisson claim frequency outcome
BASE_FREQUENCY = 0.12
log_baseline = (
    np.log(BASE_FREQUENCY)
    + 0.01  * (vehicle_value / 10_000)
    - 0.008 * ncd_years
    + 0.10  * urban
    + 0.005 * np.maximum(35 - driver_age, 0)
    + RNG.normal(0, 0.05, N)
)
exposure    = RNG.uniform(0.5, 1.0, N)
log_mu      = log_baseline + log_scale_elasticity * log_price_change + np.log(exposure)
claim_count = RNG.poisson(np.exp(log_mu)).astype(float)

segment = (pd.Series(age_band) + "_" + pd.Series(urban.astype(int)).map({1: "urban", 0: "rural"})).values
TRUE_ATE = float(np.mean(log_scale_elasticity))
TRUE_GATE: Dict[str, float] = {
    f"{b}_{'urban' if u else 'rural'}": c
    for (b, u), c in LOG_SCALE_ELASTICITY.items()
}

df = pd.DataFrame({
    "driver_age": driver_age, "vehicle_value": vehicle_value,
    "ncd_years": ncd_years, "region": region.astype(float),
    "urban": urban, "log_price_change": log_price_change,
    "claim_count": claim_count, "exposure": exposure,
    "true_log_cate": log_scale_elasticity, "segment": segment,
    "age_band": age_band,
})
CONFOUNDERS = ["driver_age", "vehicle_value", "ncd_years", "region", "urban"]

print(f"N = {N:,}  |  claim rate = {claim_count.sum()/exposure.sum():.3f}/yr")
print(f"True ATE (log scale): {TRUE_ATE:.4f}")
print()
print("True segment log-scale elasticities:")
for seg, v in sorted(TRUE_GATE.items()):
    n_seg = int(np.sum(segment == seg))
    print(f"  {seg:<22} {v:>6.1f}  n={n_seg:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Baseline: Poisson GLM with Interaction Terms
# MAGIC
# MAGIC The pricing team approach. Fit log(mu) with treatment × age_band and treatment × urban
# MAGIC interactions. Extract the per-segment elasticity via the interaction coefficients.
# MAGIC This is the correct GLM specification for a step-function DGP on the log scale —
# MAGIC it gives the GLM its best chance.

# COMMAND ----------

# Dummy-encode age_band (reference = senior)
df["age_young"] = (df["age_band"] == "young").astype(float)
df["age_mid"]   = (df["age_band"] == "mid").astype(float)

# Interaction terms: treatment × segment indicators
df["T_young"]       = df["log_price_change"] * df["age_young"]
df["T_mid"]         = df["log_price_change"] * df["age_mid"]
df["T_urban"]       = df["log_price_change"] * df["urban"]
df["T_young_urban"] = df["log_price_change"] * df["age_young"] * df["urban"]
df["T_mid_urban"]   = df["log_price_change"] * df["age_mid"]   * df["urban"]

t_glm = time.perf_counter()
glm_model = smf.glm(
    "claim_count ~ log_price_change + age_young + age_mid + urban + ncd_years "
    "+ T_young + T_mid + T_urban + T_young_urban + T_mid_urban",
    data=df,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=np.log(df["exposure"].values),
).fit()
glm_time = time.perf_counter() - t_glm
print(f"GLM fit time: {glm_time:.3f}s")

# Extract per-segment elasticities
p = glm_model.params
glm_estimates = {
    "senior_rural":  p["log_price_change"],
    "senior_urban":  p["log_price_change"] + p["T_urban"],
    "mid_rural":     p["log_price_change"] + p["T_mid"],
    "mid_urban":     p["log_price_change"] + p["T_mid"] + p["T_urban"] + p["T_mid_urban"],
    "young_rural":   p["log_price_change"] + p["T_young"],
    "young_urban":   p["log_price_change"] + p["T_young"] + p["T_urban"] + p["T_young_urban"],
}

# Delta-method CIs for each segment
cov_arr = np.array(glm_model.cov_params())
param_names = list(glm_model.params.index)

def delta_ci(coeff_names):
    grad = np.zeros(len(param_names))
    for c in coeff_names:
        grad[param_names.index(c)] = 1.0
    se = float(np.sqrt(grad @ cov_arr @ grad))
    est = sum(float(p[c]) for c in coeff_names)
    return est - 1.96 * se, est + 1.96 * se

glm_ci_map = {
    "senior_rural":  delta_ci(["log_price_change"]),
    "senior_urban":  delta_ci(["log_price_change", "T_urban"]),
    "mid_rural":     delta_ci(["log_price_change", "T_mid"]),
    "mid_urban":     delta_ci(["log_price_change", "T_mid", "T_urban", "T_mid_urban"]),
    "young_rural":   delta_ci(["log_price_change", "T_young"]),
    "young_urban":   delta_ci(["log_price_change", "T_young", "T_urban", "T_young_urban"]),
}

print()
print(f"  {'Segment':<22} {'True':>7} {'GLM est':>8} {'CI lo':>8} {'CI hi':>8} {'Bias%':>7} {'Covers':>7}")
print("  " + "-" * 72)
glm_biases = []
glm_covered = 0
for seg in sorted(TRUE_GATE):
    true = TRUE_GATE[seg]
    est  = glm_estimates[seg]
    lo, hi = glm_ci_map[seg]
    bias_p = (est - true) / abs(true) * 100
    covers = lo <= true <= hi
    glm_biases.append(est - true)
    if covers:
        glm_covered += 1
    print(f"  {seg:<22} {true:>7.2f} {est:>8.3f} {lo:>8.3f} {hi:>8.3f} {bias_p:>+7.1f}% {'Y' if covers else 'N':>7}")

glm_rmse = float(np.sqrt(np.mean([b**2 for b in glm_biases])))
print(f"\n  GLM segment RMSE: {glm_rmse:.4f}  |  CI coverage: {glm_covered}/6")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Causal Forest GATE
# MAGIC
# MAGIC `HeterogeneousElasticityEstimator` with Poisson outcome family. CausalForestDML
# MAGIC with cross-fitted CatBoost nuisance models. GATE extracted by segment label.
# MAGIC
# MAGIC Unlike the GLM, the forest: (a) partials out the confounded treatment before
# MAGIC estimating effects, (b) produces per-policy CATEs, (c) can detect heterogeneity
# MAGIC not aligned with the pre-specified segments.

# COMMAND ----------

from insurance_causal.causal_forest.estimator import HeterogeneousElasticityEstimator
from insurance_causal.causal_forest.inference import HeterogeneousInference
from insurance_causal.causal_forest.targeting import TargetingEvaluator

t0 = time.perf_counter()
est = HeterogeneousElasticityEstimator(
    outcome_family="poisson",
    n_estimators=500,
    n_folds=5,
    nuisance_backend="sklearn",
    min_samples_leaf=20,
    random_state=42,
)
est.fit(
    df,
    outcome="claim_count",
    treatment="log_price_change",
    confounders=CONFOUNDERS,
    exposure="exposure",
)
forest_time = time.perf_counter() - t0
print(f"Forest fit time: {forest_time:.1f}s")

ate_val, ate_lb, ate_ub = est.ate()
print(f"\nForest ATE: {ate_val:.4f}  (true: {TRUE_ATE:.4f}  bias: {ate_val-TRUE_ATE:+.4f})")

# COMMAND ----------

gate_df   = est.gate(df, by="segment")
gate_rows = gate_df.to_pandas() if hasattr(gate_df, "to_pandas") else gate_df

print()
print(f"  {'Segment':<22} {'True':>7} {'Forest':>8} {'CI lo':>8} {'CI hi':>8} {'Bias%':>7} {'Covers':>7}")
print("  " + "-" * 72)
forest_biases = []
forest_covered = 0
for _, row in gate_rows.iterrows():
    seg   = row["segment"]
    true  = TRUE_GATE.get(seg, float("nan"))
    fest  = row["elasticity"]
    lo, hi = row["ci_lower"], row["ci_upper"]
    bias_p = (fest - true) / abs(true) * 100 if not np.isnan(true) else float("nan")
    covers = lo <= true <= hi
    if not np.isnan(fest - true):
        forest_biases.append(fest - true)
    if covers:
        forest_covered += 1
    print(f"  {seg:<22} {true:>7.2f} {fest:>8.3f} {lo:>8.3f} {hi:>8.3f} {bias_p:>+7.1f}% {'Y' if covers else 'N':>7}")

forest_rmse = float(np.sqrt(np.mean([b**2 for b in forest_biases])))
print(f"\n  Forest segment RMSE: {forest_rmse:.4f}  |  CI coverage: {forest_covered}/{len(gate_rows)}")

cate_vals = est.cate(df)
cate_corr = np.corrcoef(cate_vals, df["true_log_cate"].values)[0, 1]
print(f"  Corr(estimated CATE, true log-scale CATE): {cate_corr:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. BLP / GATES Inference

# COMMAND ----------

inference = HeterogeneousInference(n_groups=5, n_splits=20, random_state=42)
inference_result = inference.fit(
    estimator=est,
    df=df,
    outcome="claim_count",
    treatment="log_price_change",
    confounders=CONFOUNDERS,
)
print(inference_result.summary())

# COMMAND ----------

display(inference_result.gates)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. AUTOC — Targeting Evaluation

# COMMAND ----------

evaluator = TargetingEvaluator(method="autoc", n_bootstrap=200, random_state=42)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    rate_result = evaluator.fit(
        estimator=est,
        df=df,
        outcome="claim_count",
        treatment="log_price_change",
        confounders=CONFOUNDERS,
    )
print(f"AUTOC (RATE): {rate_result.rate:.4f}")
print(f"Bootstrap SE: {rate_result.se:.4f}")
print(f"p-value:      {rate_result.p_value:.4f}  {'(significant)' if rate_result.p_value < 0.05 else '(not significant)'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Summary Comparison

# COMMAND ----------

# Per-segment table
print("Per-segment comparison:")
print()
print(f"  {'Segment':<22} {'True':>7} {'GLM':>8} {'GLM bias%':>10} {'Forest':>8} {'Forest bias%':>13}")
print("  " + "-" * 74)
for seg in sorted(TRUE_GATE):
    true  = TRUE_GATE[seg]
    gest  = glm_estimates[seg]
    fbias = float("nan")
    frow  = gate_rows[gate_rows["segment"] == seg]
    if len(frow):
        fest  = float(frow["elasticity"].values[0])
        fbias = (fest - true) / abs(true) * 100
    else:
        fest = float("nan")
    gbias = (gest - true) / abs(true) * 100
    print(f"  {seg:<22} {true:>7.2f} {gest:>8.3f} {gbias:>+9.1f}% {fest:>8.3f} {fbias:>+12.1f}%")

print()
print(f"  {'Metric':<40} {'GLM interactions':>18} {'Causal forest':>14}")
print("  " + "-" * 75)
print(f"  {'Segment RMSE vs true effects':<40} {glm_rmse:>18.4f} {forest_rmse:>14.4f}")
print(f"  {'CI coverage (6 segments)':<40} {glm_covered}/6 {'':>13} {forest_covered}/{len(gate_rows)}")
sig = f"p={rate_result.p_value:.3f} {'sig' if rate_result.p_value < 0.05 else 'n.s.'}"
print(f"  {'AUTOC (targeting value)':<40} {'N/A':>18} {sig:>14}")
print(f"  {'Corr(CATE, true effect)':<40} {'N/A':>18} {cate_corr:>14.3f}")
print(f"  {'Fit time (s)':<40} {glm_time:>18.3f} {forest_time:>14.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Interpretation
# MAGIC
# MAGIC The GLM interaction model is well-matched to this DGP: the true DGP is log-linear
# MAGIC Poisson with step-function heterogeneity aligned to the same (age_band, urban)
# MAGIC segments. This is the GLM's best case. If the forest still beats it here on
# MAGIC confounding-adjusted estimates, it will beat it more decisively in real portfolios
# MAGIC where the DGP is not log-linear and segments are not pre-known.
# MAGIC
# MAGIC **The GLM interaction model:**
# MAGIC - Fast (<0.01s), interpretable, standard in pricing teams
# MAGIC - Correctly specified for this DGP
# MAGIC - Does not correct for confounding (treatment is a covariate, not residualised)
# MAGIC - No individual-level CATE, no targeting evaluation
# MAGIC
# MAGIC **The causal forest:**
# MAGIC - Slower (~40s), needs cross-fitting
# MAGIC - Corrects for the confounded treatment assignment via DML
# MAGIC - Provides per-policy CATEs and a valid targeting test (AUTOC)
# MAGIC - Can detect heterogeneity beyond pre-specified interactions
# MAGIC
# MAGIC **When to use the GLM interaction model:**
# MAGIC Use it when you need a fast segment-level number, the DGP is plausibly log-linear,
# MAGIC and you can assume the treatment assignment is ignorable given your controls. For
# MAGIC an annual rate review where you are adding a factor, the GLM is appropriate.
# MAGIC
# MAGIC **When to use the causal forest:**
# MAGIC Use it when you need to estimate treatment effects for commercial decisions
# MAGIC (discount targeting, campaign response measurement, NCD modelling) where the
# MAGIC treatment was assigned based on risk criteria. The confounding correction is the
# MAGIC primary benefit. Individual-level CATEs and AUTOC targeting are secondary benefits
# MAGIC that require the forest regardless of the GLM interaction model's performance.
