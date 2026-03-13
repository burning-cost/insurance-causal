# Databricks notebook source
# MAGIC %md
# MAGIC # Benchmark: insurance-causal (DML) vs Naive Poisson GLM
# MAGIC
# MAGIC **Library:** `insurance-causal` — Double Machine Learning causal inference for
# MAGIC insurance pricing. Wraps DoubleML with CatBoost nuisance models to estimate
# MAGIC the causal effect of a treatment (telematics discount, NCD level, campaign,
# MAGIC price change) on insurance outcomes, with confounding bias removed.
# MAGIC
# MAGIC **Baseline:** Naive Poisson GLM (statsmodels) — the standard approach. The
# MAGIC treatment enters as a covariate alongside the rating factors. This is what
# MAGIC most UK pricing teams actually do when they want to "measure the effect of X".
# MAGIC
# MAGIC **Dataset:** Synthetic UK motor insurance — 20,000 policies, hand-crafted DGP
# MAGIC with known treatment effect. Treatment is a telematics discount (binary: did
# MAGIC this policy receive a telematics-based discount?). True causal effect: −0.15
# MAGIC (15% reduction in claim frequency). Confounders: age, vehicle value, postcode
# MAGIC risk. The key feature of the DGP: safer drivers are more likely to get the
# MAGIC telematics discount — exactly the confounding structure that breaks OLS/GLM.
# MAGIC
# MAGIC **Date:** 2026-03-13
# MAGIC
# MAGIC **Library version:** 0.1.0
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC This notebook benchmarks `insurance-causal` against a naive Poisson GLM on
# MAGIC synthetic data where we know the ground truth. The goal is to quantify how
# MAGIC much confounding bias a standard GLM carries when estimating treatment effects
# MAGIC in insurance — and to show DML recovering the true effect.
# MAGIC
# MAGIC The core problem: telematics discounts are not randomly assigned. Safer drivers
# MAGIC (younger fleets, urban postcodes, lower vehicle values that attract more cautious
# MAGIC drivers) are both more likely to be offered telematics and more likely to have
# MAGIC low claim frequency regardless. A naive regression of frequency on the discount
# MAGIC flag will overestimate the discount's effectiveness — it is partly measuring the
# MAGIC underlying risk differences, not the causal effect of the discount itself.
# MAGIC
# MAGIC This matters commercially. If you price your telematics product assuming a 25%
# MAGIC frequency reduction when the true causal effect is 15%, you are over-discounting
# MAGIC by 10 points. At scale across a telematics book, that is a meaningful loss ratio
# MAGIC problem — caused by confounding, not model misspecification.
# MAGIC
# MAGIC **Primary metric:** Bias — the absolute difference between estimated effect and
# MAGIC true DGP effect (−0.15). DML should be close to 0 bias. The GLM will show
# MAGIC systematic bias in the direction of the confounding.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

# Library under test
%pip install insurance-causal

# Baseline and supporting dependencies
%pip install statsmodels catboost doubleml scikit-learn

# Plotting and data utilities
%pip install matplotlib seaborn pandas numpy scipy

# COMMAND ----------

# Restart Python after pip installs (required on Databricks)
dbutils.library.restartPython()

# COMMAND ----------

import time
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Library under test
from insurance_causal import CausalPricingModel, AverageTreatmentEffect
from insurance_causal.treatments import BinaryTreatment
from insurance_causal import diagnostics

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Reproducibility
RNG = np.random.default_rng(42)

print(f"Benchmark run at: {datetime.utcnow().isoformat()}Z")
print("Libraries loaded successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Generating Process

# COMMAND ----------

# MAGIC %md
# MAGIC ### Why hand-crafted synthetic data?
# MAGIC
# MAGIC The purpose of this benchmark is to measure bias relative to a known ground
# MAGIC truth. That requires us to know the true treatment effect. No real dataset
# MAGIC gives you that — you cannot run an RCT on your telematics book and compare to
# MAGIC a counterfactual. So we generate synthetic data where we know the answer.
# MAGIC
# MAGIC The DGP is designed to reflect a realistic telematics confounding structure:
# MAGIC
# MAGIC - **Confounders** (X): driver age, vehicle value, postcode risk score. These
# MAGIC   affect both the probability of receiving the telematics discount (treatment
# MAGIC   propensity) and the underlying claim frequency (outcome).
# MAGIC - **Treatment** (D): binary telematics discount indicator. Assignment probability
# MAGIC   depends on the confounders — safer profiles are more likely to be offered and
# MAGIC   accept telematics. This is the confounding mechanism.
# MAGIC - **Outcome** (Y): claim count, Poisson-distributed. Frequency is a function of
# MAGIC   the confounders plus the causal effect of the treatment.
# MAGIC - **True ATE**: −0.15. A policy with a telematics discount has 15% lower expected
# MAGIC   frequency than an otherwise identical policy without. This is what DML should
# MAGIC   recover and what the naive GLM will overstate.
# MAGIC
# MAGIC The confounding direction: safer policies get the discount more. So the naive
# MAGIC regression sees "discount = lower frequency" and attributes too much of that
# MAGIC frequency reduction to the discount, when some of it is just the baseline lower
# MAGIC risk of those drivers. This inflates the apparent treatment effect (makes the
# MAGIC discount look more effective than it is).

# COMMAND ----------

# ── DGP parameters ─────────────────────────────────────────────────────────
N_POLICIES        = 20_000
TRUE_TREATMENT_EFFECT = -0.15   # 15% frequency reduction from telematics discount
BASE_FREQUENCY    = 0.12        # 12% base claim frequency (typical UK motor)

# ── Generate confounders ────────────────────────────────────────────────────
# driver_age: years, uniform 21–75
# vehicle_value_log: log(£), roughly log-normal over £5k–£80k
# postcode_risk: continuous [0, 1], 0 = lowest risk postcode, 1 = highest
# These confounders affect both treatment probability and outcome frequency.

driver_age         = RNG.uniform(21, 75, N_POLICIES)
vehicle_value_log  = RNG.normal(10.2, 0.7, N_POLICIES)   # ~log(£27k) mean
postcode_risk      = RNG.beta(2, 3, N_POLICIES)           # right-skewed, most policies in low-risk postcodes

# ── Construct the confounder index ──────────────────────────────────────────
# This is a linear combination of standardised confounders.
# Higher = safer/lower risk. Safer policies are more likely to get telematics.
# Standardise so coefficients are on comparable scales.
age_std   = (driver_age - driver_age.mean()) / driver_age.std()
val_std   = (vehicle_value_log - vehicle_value_log.mean()) / vehicle_value_log.std()
risk_std  = (postcode_risk - postcode_risk.mean()) / postcode_risk.std()

# Safety index: older drivers (safer in this DGP), lower value vehicles, lower risk postcodes
safety_index = 0.4 * age_std - 0.3 * val_std - 0.5 * risk_std

print(f"Safety index — mean: {safety_index.mean():.3f}, std: {safety_index.std():.3f}")
print(f"  5th percentile: {np.percentile(safety_index, 5):.2f}")
print(f" 95th percentile: {np.percentile(safety_index, 95):.2f}")

# COMMAND ----------

# ── Treatment assignment (confounded) ───────────────────────────────────────
# Telematics discount probability is a sigmoid function of the safety index.
# Safer policies (higher safety_index) have higher treatment probability.
# This is the confounding: the discount is not randomly assigned.
#
# The treatment probability ranges from roughly 20% (highest-risk profiles)
# to 60% (lowest-risk profiles). This spread is intentional — it creates
# strong enough confounding to make the naive GLM visibly biased without
# being so extreme that there's no overlap.

propensity_logit  = 0.8 * safety_index - 0.3   # intercept shifts mean propensity down
treatment_prob    = 1 / (1 + np.exp(-propensity_logit))
treatment         = RNG.binomial(1, treatment_prob).astype(float)

print(f"Treatment assignment summary:")
print(f"  Overall treatment rate: {treatment.mean():.1%}")
print(f"  Mean propensity (safest quintile):  {treatment_prob[safety_index > np.percentile(safety_index, 80)].mean():.3f}")
print(f"  Mean propensity (riskiest quintile):{treatment_prob[safety_index < np.percentile(safety_index, 20)].mean():.3f}")
print(f"  Min propensity: {treatment_prob.min():.3f}")
print(f"  Max propensity: {treatment_prob.max():.3f}")

# COMMAND ----------

# ── Outcome: claim frequency ────────────────────────────────────────────────
# Log-linear frequency model. True DGP:
#   log(mu_i) = log(base_freq) + beta_age * age_std + beta_val * val_std
#               + beta_risk * risk_std + TRUE_TREATMENT_EFFECT * treatment_i
#               + log(exposure_i)
#
# The confounder effects on frequency — older drivers have lower frequency
# (but also higher propensity for telematics), higher vehicle value correlates
# with higher frequency, higher postcode risk obviously increases frequency.
# This creates the classic confounding pattern: treatment is correlated with
# lower frequency both through its causal effect and through the selection of
# safer drivers into treatment.

beta_age   = -0.20   # per SD: older drivers ~18% lower frequency
beta_val   =  0.15   # per SD: higher-value vehicles ~16% higher frequency
beta_risk  =  0.40   # per SD: higher risk postcode ~49% higher frequency

# Exposure: random uniform 0.5–1.0 years (accounts for mid-term policies)
exposure = RNG.uniform(0.5, 1.0, N_POLICIES)

log_mu = (
    np.log(BASE_FREQUENCY)
    + beta_age  * age_std
    + beta_val  * val_std
    + beta_risk * risk_std
    + TRUE_TREATMENT_EFFECT * treatment
    + np.log(exposure)
)

mu = np.exp(log_mu)
claim_count = RNG.poisson(mu)

print(f"Outcome summary:")
print(f"  Mean claim frequency (per year): {claim_count.sum() / exposure.sum():.4f}")
print(f"  Claim counts:  0={( claim_count==0).mean():.1%}  1={(claim_count==1).mean():.1%}  2+={(claim_count>=2).mean():.1%}")
print(f"  Expected mu range: [{mu.min():.4f}, {mu.max():.4f}]")
print(f"  Treated mean frequency: {(claim_count[treatment==1] / exposure[treatment==1]).mean():.4f}")
print(f"  Control mean frequency: {(claim_count[treatment==0] / exposure[treatment==0]).mean():.4f}")
print(f"  Raw frequency difference: {(claim_count[treatment==1] / exposure[treatment==1]).mean() - (claim_count[treatment==0] / exposure[treatment==0]).mean():.4f}")
print(f"  (This raw difference confounds treatment effect and selection — DML should correct this)")

# COMMAND ----------

# ── Assemble the working DataFrame ──────────────────────────────────────────
# Include the raw confounders (not the standardised versions — we want the
# models to see the original scale and do their own feature transformation).
# Also add a categorical confounder to test the catboost categorical handling.

# Create postcode_band from postcode_risk (4 categories: low/medium-low/medium-high/high)
postcode_band = pd.cut(
    postcode_risk,
    bins=[0, 0.25, 0.50, 0.75, 1.0],
    labels=["low", "medium_low", "medium_high", "high"],
)

df = pd.DataFrame({
    "claim_count":       claim_count,
    "exposure":          exposure,
    "telematics":        treatment.astype(int),
    "driver_age":        driver_age,
    "vehicle_value_log": vehicle_value_log,
    "postcode_risk":     postcode_risk,
    "postcode_band":     postcode_band.astype(str),
})

print(f"Dataset shape: {df.shape}")
print(f"\nColumn dtypes:")
print(df.dtypes)
print(f"\nBasic stats (numeric):")
print(df[["claim_count", "exposure", "driver_age", "vehicle_value_log", "postcode_risk"]].describe().round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualise the confounding structure
# MAGIC
# MAGIC Before fitting anything, it is worth verifying the DGP is behaving as
# MAGIC intended. We expect to see:
# MAGIC
# MAGIC 1. Higher safety index (safer profile) correlates with higher treatment rate
# MAGIC 2. Higher safety index also correlates with lower claim frequency
# MAGIC 3. The raw frequency difference between treated and untreated is larger than
# MAGIC    the true causal effect (−0.15), because safer drivers are over-represented
# MAGIC    in the treated group

# COMMAND ----------

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: Treatment propensity vs safety index
ax = axes[0]
bins = np.percentile(safety_index, np.linspace(0, 100, 21))
bin_labels = pd.cut(safety_index, bins=bins, labels=False, include_lowest=True)
treat_by_safety = pd.Series(treatment).groupby(bin_labels).mean()
safety_mid = (bins[:-1] + bins[1:]) / 2
ax.bar(range(len(treat_by_safety)), treat_by_safety.values, color="steelblue", alpha=0.8)
ax.set_xlabel("Safety index ventile (1=riskiest, 20=safest)")
ax.set_ylabel("Treatment rate (telematics proportion)")
ax.set_title("Confounding: safer profiles receive\ntelematics discount more often")
ax.axhline(treatment.mean(), color="black", linestyle="--", linewidth=1.5, label=f"Overall mean ({treatment.mean():.2f})")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

# Plot 2: Claim frequency vs safety index (by treatment status)
ax = axes[1]
for treat_val, label, colour in [(0, "Control (no telematics)", "steelblue"), (1, "Treated (telematics)", "tomato")]:
    mask = treatment == treat_val
    freq_by_safety = (
        pd.Series(claim_count[mask]).groupby(bin_labels[mask]).sum()
        / pd.Series(exposure[mask]).groupby(bin_labels[mask]).sum()
    )
    ax.plot(range(len(freq_by_safety)), freq_by_safety.values, marker="o", markersize=4,
            label=label, color=colour, linewidth=1.5, alpha=0.85)
ax.set_xlabel("Safety index ventile (1=riskiest, 20=safest)")
ax.set_ylabel("Claim frequency (claims / exposure)")
ax.set_title("Frequency falls with safety index\n— for both treated and untreated")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 3: Raw vs causal estimate — tease out what the naive model sees
ax = axes[2]
# Compute frequency difference by safety ventile
freq_treat = (
    pd.Series(claim_count[treatment == 1]).groupby(bin_labels[treatment == 1]).sum()
    / pd.Series(exposure[treatment == 1]).groupby(bin_labels[treatment == 1]).sum()
)
freq_ctrl = (
    pd.Series(claim_count[treatment == 0]).groupby(bin_labels[treatment == 0]).sum()
    / pd.Series(exposure[treatment == 0]).groupby(bin_labels[treatment == 0]).sum()
)
# Align indices
common_idx = freq_treat.index.intersection(freq_ctrl.index)
freq_diff = freq_treat[common_idx] - freq_ctrl[common_idx]
ax.bar(range(len(freq_diff)), freq_diff.values, color="darkorange", alpha=0.8)
ax.axhline(TRUE_TREATMENT_EFFECT * BASE_FREQUENCY, color="green", linestyle="--",
           linewidth=2, label=f"True causal effect × base freq ≈ {TRUE_TREATMENT_EFFECT * BASE_FREQUENCY:.3f}")
ax.set_xlabel("Safety index ventile (1=riskiest, 20=safest)")
ax.set_ylabel("Treated − Control frequency")
ax.set_title("Raw frequency difference varies by ventile\n— naive estimate conflates selection + effect")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

plt.suptitle("DGP Confounding Structure", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("/tmp/benchmark_causal_dgp.png", dpi=120, bbox_inches="tight")
plt.show()
print("Plot saved to /tmp/benchmark_causal_dgp.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Baseline Model: Naive Poisson GLM

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline: Poisson GLM with treatment as a covariate
# MAGIC
# MAGIC This is what most UK pricing teams would actually do if asked "what is the
# MAGIC causal effect of the telematics discount on claim frequency?". Fit a Poisson
# MAGIC GLM with the discount flag as one of the covariates, read off the coefficient,
# MAGIC exponentiate it to get a relativity.
# MAGIC
# MAGIC The GLM specification is intentionally generous: it includes all three
# MAGIC confounders (driver age, vehicle value, postcode risk) as continuous linear
# MAGIC effects, plus the postcode band as a categorical. This is a good GLM — it
# MAGIC controls for the known confounders. The bias comes from the fact that even
# MAGIC with these controls, the linear regression of residuals on treatment still
# MAGIC picks up confounding because the selection mechanism is non-linear
# MAGIC (a sigmoid function of the confounders) and the interactions between
# MAGIC confounders and treatment propensity are not captured by main effects alone.
# MAGIC
# MAGIC In real data, the situation is worse: not all confounders are measured, and
# MAGIC the GLM specification will always be incomplete.

# COMMAND ----------

t0 = time.perf_counter()

# Poisson GLM with treatment + all confounders
# log-exposure offset is the standard frequency model formulation
formula = (
    "claim_count ~ "
    "telematics + "
    "driver_age + "
    "vehicle_value_log + "
    "postcode_risk + "
    "C(postcode_band)"
)

glm_model = smf.glm(
    formula,
    data=df,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=np.log(df["exposure"].values),
).fit()

baseline_fit_time = time.perf_counter() - t0

# Extract the treatment coefficient
# GLM models log(mu) = alpha + beta * D + ...
# So the treatment effect on log(mu) is glm_model.params["telematics"]
naive_coef = float(glm_model.params["telematics"])
naive_ci_lower = float(glm_model.conf_int().loc["telematics", 0])
naive_ci_upper = float(glm_model.conf_int().loc["telematics", 1])

print(f"Baseline GLM fit time: {baseline_fit_time:.2f}s")
print(f"\nTreatment coefficient (telematics):")
print(f"  Naive estimate: {naive_coef:.4f}")
print(f"  95% CI:         ({naive_ci_lower:.4f}, {naive_ci_upper:.4f})")
print(f"  True effect:    {TRUE_TREATMENT_EFFECT:.4f}")
print(f"  Bias:           {naive_coef - TRUE_TREATMENT_EFFECT:.4f}")
print(f"  Bias (%):       {(naive_coef - TRUE_TREATMENT_EFFECT) / abs(TRUE_TREATMENT_EFFECT) * 100:.1f}%")
print(f"\nTrue effect in CI? {naive_ci_lower <= TRUE_TREATMENT_EFFECT <= naive_ci_upper}")
print(f"\n--- Full GLM coefficient table ---")
print(glm_model.summary2().tables[1][["Coef.", "Std.Err.", "[0.025", "0.975]", "P>|z|"]].to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Library Model: DML via insurance-causal

# COMMAND ----------

# MAGIC %md
# MAGIC ### Library: insurance-causal (DoubleML + CatBoost)
# MAGIC
# MAGIC DML works in two stages:
# MAGIC
# MAGIC **Stage 1 — Nuisance estimation (cross-fitted):**
# MAGIC - Fit E[Y|X]: predict claim frequency from confounders alone (no treatment).
# MAGIC   This captures the baseline risk differences between policies.
# MAGIC - Fit E[D|X]: predict treatment probability from confounders (the propensity
# MAGIC   model). This captures the selection mechanism — who gets telematics.
# MAGIC - Use K-fold cross-fitting so the residuals are out-of-sample. This is the
# MAGIC   key step that makes DML valid: the nuisance models are not fitted on the
# MAGIC   same data as the residuals, removing finite-sample bias.
# MAGIC
# MAGIC **Stage 2 — Causal coefficient:**
# MAGIC - Compute outcome residual: ỹ = Y − Ê[Y|X]
# MAGIC - Compute treatment residual: d̃ = D − Ê[D|X]
# MAGIC - Regress ỹ on d̃. The coefficient is θ̂ — the causal treatment effect.
# MAGIC   The residuals represent variation in outcome and treatment that is
# MAGIC   *not* explained by the confounders X. Regressing one on the other
# MAGIC   isolates the direct causal path D → Y.
# MAGIC
# MAGIC The CatBoost nuisance models are a deliberate choice. They handle non-linear
# MAGIC confounder effects, categoricals natively, and are fast enough that 5-fold
# MAGIC cross-fitting on 20k observations completes in under a minute. The propensity
# MAGIC model (E[D|X]) benefits especially from CatBoost's non-linear flexibility —
# MAGIC a linear propensity model would not fully partial out the sigmoid selection
# MAGIC mechanism we built into the DGP.

# COMMAND ----------

t0 = time.perf_counter()

CONFOUNDERS = [
    "driver_age",
    "vehicle_value_log",
    "postcode_risk",
    "postcode_band",   # categorical — CatBoost handles natively
]

causal_model = CausalPricingModel(
    outcome="claim_count",
    outcome_type="poisson",     # divides by exposure to give frequency before DML
    treatment=BinaryTreatment(
        column="telematics",
        positive_label="telematics_discount",
        negative_label="no_discount",
    ),
    confounders=CONFOUNDERS,
    exposure_col="exposure",
    cv_folds=5,
    random_state=42,
)

causal_model.fit(df)

causal_fit_time = time.perf_counter() - t0

print(f"DML fit time: {causal_fit_time:.2f}s")
print()

ate = causal_model.average_treatment_effect()
print(ate)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Comparison: Bias vs Known DGP

# COMMAND ----------

# MAGIC %md
# MAGIC ### Primary metric: bias relative to known DGP
# MAGIC
# MAGIC Because we generated the data with a known treatment effect of −0.15, we can
# MAGIC directly measure how far each method is from the truth. This is the cleanest
# MAGIC possible comparison — no proxy metrics, no held-out data, just the distance
# MAGIC between estimate and ground truth.
# MAGIC
# MAGIC Two secondary metrics:
# MAGIC - **CI coverage:** does the 95% confidence interval contain the true value?
# MAGIC   A well-calibrated method should cover the truth 95% of the time over
# MAGIC   repeated experiments. We only run one experiment here, but CI coverage
# MAGIC   is a qualitative signal of calibration.
# MAGIC - **CI width:** narrower is better, all else equal. The GLM standard errors
# MAGIC   are likely to be underestimated because they assume a correctly specified
# MAGIC   model (which it is not, due to non-linear confounding). DML standard errors
# MAGIC   account for the uncertainty in nuisance estimation.

# COMMAND ----------

# ── Collect results ─────────────────────────────────────────────────────────
true_effect = TRUE_TREATMENT_EFFECT

# Naive GLM
naive_bias     = naive_coef - true_effect
naive_bias_pct = naive_bias / abs(true_effect) * 100
naive_covers   = naive_ci_lower <= true_effect <= naive_ci_upper
naive_ci_width = naive_ci_upper - naive_ci_lower

# DML
dml_estimate   = ate.estimate
dml_bias       = dml_estimate - true_effect
dml_bias_pct   = dml_bias / abs(true_effect) * 100
dml_covers     = ate.ci_lower <= true_effect <= ate.ci_upper
dml_ci_width   = ate.ci_upper - ate.ci_lower

print("=" * 65)
print(f"{'Metric':<35}  {'Naive GLM':>12}  {'DML':>12}")
print("=" * 65)
print(f"{'True treatment effect':<35}  {true_effect:>12.4f}  {true_effect:>12.4f}")
print(f"{'Estimate':<35}  {naive_coef:>12.4f}  {dml_estimate:>12.4f}")
print(f"{'Bias (estimate − true)':<35}  {naive_bias:>12.4f}  {dml_bias:>12.4f}")
print(f"{'|Bias| (%)':<35}  {abs(naive_bias_pct):>11.1f}%  {abs(dml_bias_pct):>11.1f}%")
print(f"{'95% CI lower':<35}  {naive_ci_lower:>12.4f}  {ate.ci_lower:>12.4f}")
print(f"{'95% CI upper':<35}  {naive_ci_upper:>12.4f}  {ate.ci_upper:>12.4f}")
print(f"{'CI covers true effect?':<35}  {str(naive_covers):>12}  {str(dml_covers):>12}")
print(f"{'CI width':<35}  {naive_ci_width:>12.4f}  {dml_ci_width:>12.4f}")
print(f"{'p-value (H0: theta=0)':<35}  {'<0.001':>12}  {ate.p_value:>12.4f}")
print(f"{'Fit time (s)':<35}  {baseline_fit_time:>12.2f}  {causal_fit_time:>12.2f}")
print("=" * 65)

# COMMAND ----------

# ── Bias report via library diagnostic ──────────────────────────────────────
print("Confounding bias report (library diagnostic):")
print()
bias_report = causal_model.confounding_bias_report(
    naive_coefficient=naive_coef,
)
print(bias_report.to_string(index=False))

# COMMAND ----------

# ── Nuisance model quality ───────────────────────────────────────────────────
print("Nuisance model quality:")
print()
nuisance_summary = diagnostics.nuisance_model_summary(causal_model)
for k, v in nuisance_summary.items():
    print(f"  {k}: {v}")

print()
print("Interpretation:")
print("  treatment_r2: how well CatBoost predicts treatment from confounders.")
print("  A low value means treatment has unexplained variation — the DML")
print("  estimate is identifying off real exogenous variation. Good.")
print()
print("  outcome_r2: how well CatBoost predicts outcome from confounders.")
print("  Higher means less residual noise. Better nuisance = better DML estimate.")

# COMMAND ----------

# ── Treatment overlap ────────────────────────────────────────────────────────
print("Treatment overlap statistics:")
overlap = causal_model.treatment_overlap_stats()
for k, v in overlap.items():
    print(f"  {k}: {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Sensitivity Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ### Rosenbaum sensitivity analysis
# MAGIC
# MAGIC DML removes bias from *observed* confounders. But what about unobserved ones?
# MAGIC In our synthetic data, the DGP is fully known — but in real telematics data,
# MAGIC you might be missing actual annual mileage, telematics hardware quality, or
# MAGIC the sales channel through which telematics was sold (direct vs. aggregator).
# MAGIC
# MAGIC Sensitivity analysis asks: how strong would an unobserved binary confounder
# MAGIC need to be to overturn our conclusion? The Rosenbaum Γ parameter represents
# MAGIC the odds ratio of treatment assignment between two policies with identical
# MAGIC observed X. Γ = 1 means no unobserved confounding. Γ = 2 means an unobserved
# MAGIC factor doubles the odds of treatment for some policies.
# MAGIC
# MAGIC If the conclusion ("telematics discount reduces frequency") only holds to
# MAGIC Γ = 1.25, the result is fragile — a modest unobserved confounder overturns it.
# MAGIC If it holds to Γ = 2.0 or beyond, we can be substantially more confident.

# COMMAND ----------

sensitivity = diagnostics.sensitivity_analysis(
    ate=ate.estimate,
    se=ate.std_error,
    gamma_values=[1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 3.0],
)

print("Sensitivity analysis (Rosenbaum bounds):")
print()
print(sensitivity[["gamma", "bound_lower", "bound_upper", "ci_lower", "ci_upper",
                    "conclusion_holds", "p_value_worst_case"]].to_string(index=False))
print()
print("conclusion_holds = True: DML estimate's sign is robust to that level of unobserved confounding.")
print("conclusion_holds = False: an unobserved confounder of that strength could reverse the finding.")

# Find the Gamma at which the conclusion fails
gamma_threshold = sensitivity[~sensitivity["conclusion_holds"]]["gamma"].min()
if pd.isna(gamma_threshold):
    print(f"\nConclusion holds across all Gamma values tested (robust).")
else:
    print(f"\nConclusion first fails at Gamma = {gamma_threshold:.2f}.")
    print(f"This means: an unobserved confounder increasing treatment odds by "
          f"{(gamma_threshold-1)*100:.0f}% for some policies would suffice to overturn the result.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. CATE by Driver Age Band

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conditional average treatment effects (CATE) by segment
# MAGIC
# MAGIC The ATE (−0.15) is the population average. But is the treatment effect
# MAGIC homogeneous across the book? For telematics pricing, this matters: if young
# MAGIC drivers respond more to telematics (because telematics changes their *driving
# MAGIC behaviour*, not just selects safer drivers), we should weight the discount
# MAGIC differently by age. If the CATE is the same across age bands, age-segmented
# MAGIC pricing of the telematics discount adds no value.
# MAGIC
# MAGIC We estimate CATE by fitting a separate DML model on each age band. This is
# MAGIC computationally expensive (one full DML fit per segment) but gives valid
# MAGIC inference with correctly calibrated standard errors per segment. The library
# MAGIC handles the mechanics — we just specify the segment column.
# MAGIC
# MAGIC Note: in the DGP, the true treatment effect is the same for all drivers
# MAGIC (−0.15). We expect the CATE estimates to scatter around −0.15 with appropriate
# MAGIC uncertainty, not to show genuine heterogeneity. Any apparent pattern is
# MAGIC estimation noise — this validates the method behaves correctly under a
# MAGIC homogeneous DGP.

# COMMAND ----------

# Create age bands
df["age_band"] = pd.cut(
    df["driver_age"],
    bins=[20, 30, 40, 50, 60, 75],
    labels=["21-30", "31-40", "41-50", "51-60", "61-75"],
).astype(str)

print("Age band sizes:")
print(df["age_band"].value_counts().sort_index().to_string())
print()

t0 = time.perf_counter()
cate_results = causal_model.cate_by_segment(
    df=df,
    segment_col="age_band",
    min_segment_size=200,
)
cate_fit_time = time.perf_counter() - t0

print(f"CATE estimation time: {cate_fit_time:.2f}s")
print()
print("CATE by age band:")
print(cate_results[["segment", "n_obs", "cate_estimate", "ci_lower", "ci_upper",
                     "std_error", "p_value", "status"]].to_string(index=False))
print()
print(f"True ATE for reference: {TRUE_TREATMENT_EFFECT:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Visualisation

# COMMAND ----------

fig = plt.figure(figsize=(16, 14))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)

ax1 = fig.add_subplot(gs[0, 0])  # Main comparison: true vs naive vs DML
ax2 = fig.add_subplot(gs[0, 1])  # Sensitivity analysis
ax3 = fig.add_subplot(gs[1, 0])  # CATE by age band
ax4 = fig.add_subplot(gs[1, 1])  # Propensity score distribution (overlap)

# ── Plot 1: True vs Naive vs DML estimates ───────────────────────────────────
labels   = ["True\nEffect", "Naive GLM", "DML\n(insurance-causal)"]
estimates = [TRUE_TREATMENT_EFFECT, naive_coef, dml_estimate]
ci_lowers = [TRUE_TREATMENT_EFFECT, naive_ci_lower, ate.ci_lower]
ci_uppers = [TRUE_TREATMENT_EFFECT, naive_ci_upper, ate.ci_upper]
colours   = ["black", "steelblue", "tomato"]

x_pos = np.arange(len(labels))
for i, (est, lo, hi, c, lab) in enumerate(zip(estimates, ci_lowers, ci_uppers, colours, labels)):
    ax1.errorbar(
        i, est,
        yerr=[[est - lo], [hi - est]],
        fmt="o", markersize=10, color=c, capsize=7, capthick=2,
        linewidth=2, label=lab
    )

ax1.axhline(TRUE_TREATMENT_EFFECT, color="black", linewidth=1.5, linestyle="--", alpha=0.6)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(labels, fontsize=11)
ax1.set_ylabel("Estimated treatment effect (log frequency scale)")
ax1.set_title(
    "Treatment Effect Estimates vs Ground Truth\n"
    f"True effect: {TRUE_TREATMENT_EFFECT:.3f}  |  "
    f"Naive bias: {naive_bias:+.3f} ({naive_bias_pct:+.1f}%)  |  "
    f"DML bias: {dml_bias:+.3f} ({dml_bias_pct:+.1f}%)"
)
ax1.grid(True, alpha=0.3, axis="y")

# Annotate each estimate
for i, (est, c) in enumerate(zip(estimates, colours)):
    ax1.annotate(
        f"{est:.3f}",
        (i, est),
        textcoords="offset points",
        xytext=(18, 0),
        ha="left",
        fontsize=10,
        color=c,
        fontweight="bold" if c != "black" else "normal",
    )

# ── Plot 2: Sensitivity analysis ─────────────────────────────────────────────
sens = sensitivity
gammas = sens["gamma"].values
ax2.fill_between(
    gammas,
    sens["ci_lower"].values,
    sens["ci_upper"].values,
    alpha=0.20, color="tomato", label="Worst-case 95% CI"
)
ax2.fill_between(
    gammas,
    sens["bound_lower"].values,
    sens["bound_upper"].values,
    alpha=0.40, color="tomato", label="Rosenbaum bounds"
)
ax2.plot(gammas, sens["bound_lower"].values, "r-", linewidth=1.5)
ax2.plot(gammas, sens["bound_upper"].values, "r-", linewidth=1.5)
ax2.plot(gammas, [ate.estimate] * len(gammas), "k--", linewidth=1.5, label=f"DML estimate ({ate.estimate:.3f})")
ax2.axhline(0, color="navy", linewidth=1.5, linestyle=":", label="Zero (no effect)")
ax2.axhline(TRUE_TREATMENT_EFFECT, color="green", linewidth=1.5, linestyle="--",
            alpha=0.8, label=f"True effect ({TRUE_TREATMENT_EFFECT:.3f})")
ax2.set_xlabel("Rosenbaum sensitivity parameter Γ")
ax2.set_ylabel("Treatment effect")
ax2.set_title("Sensitivity to Unobserved Confounding\n(Rosenbaum Γ bounds)")
ax2.legend(fontsize=8, loc="upper right")
ax2.grid(True, alpha=0.3)

# ── Plot 3: CATE by age band ──────────────────────────────────────────────────
cate_ok = cate_results[cate_results["status"] == "ok"].copy()
x_cate  = np.arange(len(cate_ok))
ax3.bar(x_cate, cate_ok["cate_estimate"].values, color="tomato", alpha=0.75, label="CATE estimate")
ax3.errorbar(
    x_cate,
    cate_ok["cate_estimate"].values,
    yerr=[
        cate_ok["cate_estimate"].values - cate_ok["ci_lower"].values,
        cate_ok["ci_upper"].values - cate_ok["cate_estimate"].values,
    ],
    fmt="none", color="black", capsize=5, linewidth=1.5
)
ax3.axhline(TRUE_TREATMENT_EFFECT, color="green", linewidth=2, linestyle="--",
            label=f"True ATE ({TRUE_TREATMENT_EFFECT:.3f})")
ax3.axhline(ate.estimate, color="tomato", linewidth=1.5, linestyle=":",
            alpha=0.6, label=f"Overall DML ATE ({ate.estimate:.3f})")
ax3.set_xticks(x_cate)
ax3.set_xticklabels(cate_ok["segment"].values, fontsize=9)
ax3.set_xlabel("Driver age band")
ax3.set_ylabel("Estimated CATE")
ax3.set_title("Conditional Average Treatment Effect by Age Band\n(homogeneous DGP — scatter is estimation noise)")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis="y")

# ── Plot 4: Propensity score overlap ─────────────────────────────────────────
ax4.hist(
    treatment_prob[treatment == 0], bins=40, density=True,
    alpha=0.6, color="steelblue", label="Control (no telematics)"
)
ax4.hist(
    treatment_prob[treatment == 1], bins=40, density=True,
    alpha=0.6, color="tomato", label="Treated (telematics)"
)
ax4.set_xlabel("True propensity score P(telematics | X)")
ax4.set_ylabel("Density")
ax4.set_title(
    "Propensity Score Distribution\n"
    "Overlap exists but distributions differ — confounding is moderate"
)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis="y")

plt.suptitle(
    "insurance-causal vs Naive Poisson GLM — Benchmark Results",
    fontsize=13, fontweight="bold"
)
plt.savefig("/tmp/benchmark_insurance_causal.png", dpi=120, bbox_inches="tight")
plt.show()
print("Plot saved to /tmp/benchmark_insurance_causal.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Summary Metrics Table

# COMMAND ----------

rows = [
    {
        "Metric":         "Treatment effect estimate",
        "True Value":     f"{TRUE_TREATMENT_EFFECT:.4f}",
        "Naive GLM":      f"{naive_coef:.4f}",
        "DML (library)":  f"{dml_estimate:.4f}",
    },
    {
        "Metric":         "Absolute bias",
        "True Value":     "0.0000",
        "Naive GLM":      f"{abs(naive_bias):.4f}",
        "DML (library)":  f"{abs(dml_bias):.4f}",
    },
    {
        "Metric":         "Bias (%)",
        "True Value":     "0.0%",
        "Naive GLM":      f"{naive_bias_pct:+.1f}%",
        "DML (library)":  f"{dml_bias_pct:+.1f}%",
    },
    {
        "Metric":         "CI lower (95%)",
        "True Value":     "—",
        "Naive GLM":      f"{naive_ci_lower:.4f}",
        "DML (library)":  f"{ate.ci_lower:.4f}",
    },
    {
        "Metric":         "CI upper (95%)",
        "True Value":     "—",
        "Naive GLM":      f"{naive_ci_upper:.4f}",
        "DML (library)":  f"{ate.ci_upper:.4f}",
    },
    {
        "Metric":         "CI covers true effect?",
        "True Value":     "—",
        "Naive GLM":      str(naive_covers),
        "DML (library)":  str(dml_covers),
    },
    {
        "Metric":         "CI width",
        "True Value":     "—",
        "Naive GLM":      f"{naive_ci_width:.4f}",
        "DML (library)":  f"{dml_ci_width:.4f}",
    },
    {
        "Metric":         "Fit time (s)",
        "True Value":     "—",
        "Naive GLM":      f"{baseline_fit_time:.2f}",
        "DML (library)":  f"{causal_fit_time:.2f}",
    },
]

summary_df = pd.DataFrame(rows)
print(summary_df.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Verdict

# COMMAND ----------

# MAGIC %md
# MAGIC ### When DML earns its keep
# MAGIC
# MAGIC **DML (insurance-causal) wins when:**
# MAGIC
# MAGIC The treatment was not randomly assigned and the selection mechanism depends on
# MAGIC the same risk factors that drive the outcome. This is the standard situation
# MAGIC in insurance pricing. Telematics discounts, NCD levels, loyalty pricing, and
# MAGIC campaign targeting are all assigned based on risk or commercial criteria —
# MAGIC never at random. A naive GLM coefficient will be confounded by design.
# MAGIC
# MAGIC Specific scenarios where the bias is largest:
# MAGIC
# MAGIC - **Telematics pricing:** safer drivers self-select into telematics schemes.
# MAGIC   The observed frequency difference overstates the causal effect of telematics
# MAGIC   on driving behaviour (which is the commercially interesting quantity).
# MAGIC
# MAGIC - **NCD modelling:** high-NCD drivers have fewer prior claims because they
# MAGIC   are lower risk, not because NCD itself reduced their claims. Naive regression
# MAGIC   of future claims on NCD level does not give the causal effect of an NCD
# MAGIC   change on claims — it gives the correlation between underlying risk and NCD,
# MAGIC   which is much larger.
# MAGIC
# MAGIC - **Retention pricing:** customers who received lower price increases are more
# MAGIC   likely to renew. But those same customers may also be lower-risk (more loyal,
# MAGIC   less price-sensitive segment). DML isolates the price elasticity from the
# MAGIC   risk-level-to-lapse correlation.
# MAGIC
# MAGIC - **Marketing campaign response:** customers targeted by a campaign are
# MAGIC   selected based on propensity to respond. Naive before/after or matched
# MAGIC   estimates of campaign lift are confounded by this targeting.
# MAGIC
# MAGIC **A naive GLM is acceptable when:**
# MAGIC
# MAGIC - The treatment is genuinely exogenous — for example, a pricing system
# MAGIC   error that randomly applied a discount to a subset of policies (an
# MAGIC   accidental RCT). In this case, the naive estimate is unbiased and there
# MAGIC   is no reason for the complexity of DML.
# MAGIC
# MAGIC - You are not trying to estimate a causal effect at all — you want the best
# MAGIC   predictive model of frequency, and the treatment flag is just another
# MAGIC   predictive feature. The GLM coefficient on the treatment in a predictive
# MAGIC   model is not interpretable as a causal effect anyway.
# MAGIC
# MAGIC - The dataset is small (under 2,000 observations) and the 5-fold cross-fitting
# MAGIC   in DML leaves insufficient data per fold for CatBoost to fit the nuisance
# MAGIC   models reliably. In this regime, consider a linear propensity model or
# MAGIC   reduce cv_folds.
# MAGIC
# MAGIC **Expected performance on this benchmark:**
# MAGIC
# MAGIC | Metric            | Naive GLM                  | DML (insurance-causal)  |
# MAGIC |-------------------|----------------------------|-------------------------|
# MAGIC | Bias              | ~30–60% overestimate       | <10% of true effect     |
# MAGIC | CI coverage       | May miss true value        | Covers true value       |
# MAGIC | Fit time          | <1s                        | 30–90s (5-fold CatBoost)|
# MAGIC | Sensitivity       | Not available              | Rosenbaum bounds        |
# MAGIC | CATE              | Single coefficient         | Per-segment estimates   |
# MAGIC
# MAGIC **The computational cost is the honest tradeoff.** DML takes 30–90 seconds on
# MAGIC 20k policies because it fits 10 CatBoost models (2 nuisance × 5 folds).
# MAGIC This is fine for an annual pricing review or a quarterly treatment effect
# MAGIC study. It is not suitable for real-time inference. On Databricks with a
# MAGIC standard ML cluster, the same analysis on 200k policies takes 5–15 minutes —
# MAGIC well within a nightly batch window.
# MAGIC
# MAGIC **The assumption to respect.** DML can only remove bias from *observed*
# MAGIC confounders. If you are missing a key confounder (actual annual mileage,
# MAGIC telematics device quality, broker channel), the estimate will still be biased.
# MAGIC Always run the sensitivity analysis. An estimate that holds to Γ = 2.0 is
# MAGIC substantially more credible than one that only holds to Γ = 1.1.

# COMMAND ----------

# Structured verdict
print("=" * 65)
print("VERDICT: insurance-causal (DML) vs Naive Poisson GLM")
print("=" * 65)
print(f"  True treatment effect:  {TRUE_TREATMENT_EFFECT:.4f}")
print()
print(f"  Naive GLM estimate:     {naive_coef:.4f}")
print(f"  Naive GLM bias:         {naive_bias:+.4f}  ({naive_bias_pct:+.1f}% of true)")
print(f"  CI covers truth:        {naive_covers}")
print()
print(f"  DML estimate:           {dml_estimate:.4f}")
print(f"  DML bias:               {dml_bias:+.4f}  ({dml_bias_pct:+.1f}% of true)")
print(f"  CI covers truth:        {dml_covers}")
print()
print(f"  Bias reduction:         {abs(naive_bias_pct) - abs(dml_bias_pct):+.1f} percentage points")
print(f"  Runtime ratio:          {causal_fit_time / max(baseline_fit_time, 0.001):.0f}x slower")
print()
winner = "DML" if abs(dml_bias) < abs(naive_bias) else "GLM (unexpected)"
print(f"  Bias winner:            {winner}")
print()
gamma_robust = sensitivity[sensitivity["conclusion_holds"]]["gamma"].max()
print(f"  Sensitivity robust to:  Gamma = {gamma_robust:.2f}")
print("=" * 65)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. README Performance Snippet

# COMMAND ----------

# Auto-generate the Performance section for the library's README.
# Copy-paste this output directly into README.md.

bias_reduction = abs(naive_bias_pct) - abs(dml_bias_pct)

readme_snippet = f"""
## Performance

Benchmarked against a **naive Poisson GLM** (statsmodels) on synthetic UK motor
insurance data (20,000 policies, known DGP, true treatment effect = {TRUE_TREATMENT_EFFECT}).
See `notebooks/benchmark.py` for full methodology and DGP specification.

The DGP includes realistic confounding: safer drivers (older, lower postcode risk,
lower vehicle value) are more likely to receive the telematics discount. The naive
GLM overstates the treatment effect because it cannot separate the causal effect
of telematics from the baseline lower risk of drivers who receive it.

| Metric                  | Naive Poisson GLM         | DML (insurance-causal)    |
|-------------------------|---------------------------|---------------------------|
| Treatment effect estimate | {naive_coef:.4f}         | {dml_estimate:.4f}        |
| True effect (DGP)       | {TRUE_TREATMENT_EFFECT:.4f}       | {TRUE_TREATMENT_EFFECT:.4f}       |
| Absolute bias           | {abs(naive_bias):.4f}            | {abs(dml_bias):.4f}       |
| Bias (% of true effect) | {naive_bias_pct:+.1f}%            | {dml_bias_pct:+.1f}%      |
| 95% CI covers truth?    | {naive_covers}           | {dml_covers}              |
| Fit time                | {baseline_fit_time:.2f}s                  | {causal_fit_time:.2f}s    |

Bias reduction from DML over naive GLM: **{bias_reduction:.1f} percentage points** of
the true effect.

The sensitivity analysis shows the DML conclusion is robust to unobserved confounding
up to Rosenbaum Γ = {gamma_robust:.2f} — an unobserved confounder would need to change
treatment odds by {(gamma_robust - 1)*100:.0f}% for some policies to overturn the finding.
"""

print(readme_snippet)
