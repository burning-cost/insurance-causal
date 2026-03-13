# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-autodml: Automatic Debiased ML for Continuous Treatment
# MAGIC
# MAGIC This notebook demonstrates the full `insurance-autodml` workflow on synthetic data.
# MAGIC
# MAGIC **Estimands covered:**
# MAGIC 1. Average Marginal Effect (AME) — price elasticity
# MAGIC 2. Dose-response curve E[Y(d)]
# MAGIC 3. Policy shift effect E[Y(D*(1+delta))] - E[Y]
# MAGIC 4. Selection-corrected elasticity (renewal selection bias)
# MAGIC
# MAGIC **Data:** Synthetic UK motor insurance portfolio with known ground truth.

# COMMAND ----------

# MAGIC %pip install insurance-autodml matplotlib jinja2

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from insurance_causal.autodml import (
    PremiumElasticity,
    DoseResponseCurve,
    PolicyShiftEffect,
    SelectionCorrectedElasticity,
    ElasticityReport,
    SyntheticContinuousDGP,
    OutcomeFamily,
)

print("insurance-autodml loaded successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate Synthetic Portfolio Data
# MAGIC
# MAGIC The DGP models a UK motor renewal portfolio:
# MAGIC - 5,000 policyholders with 8 features (age, NCB, vehicle age, postcode risk, + noise)
# MAGIC - Premium set by a technical pricing model + random component (confounded with risk)
# MAGIC - Outcome: continuous approximation of pure premium (Gaussian for simplicity)
# MAGIC - Known true AME so we can validate our estimates

# COMMAND ----------

dgp = SyntheticContinuousDGP(
    n=5000,
    n_features=8,
    outcome_family="gaussian",
    beta_D=-0.002,           # True causal effect: -0.002 per £1 premium
    confounding_strength=0.5,
    sigma_D=30.0,
    base_premium=350.0,
    random_state=42,
)

X, D, Y, S = dgp.generate()
df = dgp.as_dataframe()

print(f"Observations: {len(df):,}")
print(f"Premium range: £{D.min():.0f} — £{D.max():.0f}")
print(f"Mean premium: £{D.mean():.0f}")
print(f"Outcome range: {Y.min():.3f} — {Y.max():.3f}")
print(f"\nTrue AME: {dgp.true_ame_:.6f}")
print(f"(Interpretation: each £1 increase in premium changes outcome by {dgp.true_ame_:.4f})")
display(df.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Why naive OLS is biased
# MAGIC
# MAGIC The premium is correlated with risk (by design). OLS will overstate or understate
# MAGIC the true causal effect depending on confounding direction.

# COMMAND ----------

from sklearn.linear_model import LinearRegression

# Naive: regress Y on D only
naive = LinearRegression()
naive.fit(D.reshape(-1, 1), Y)
print(f"Naive OLS (D only) estimate: {naive.coef_[0]:.6f}")

# Partially controlled: add X
controlled = LinearRegression()
controlled.fit(np.column_stack([D.reshape(-1, 1), X]), Y)
print(f"OLS with controls:           {controlled.coef_[0]:.6f}")

print(f"True AME:                    {dgp.true_ame_:.6f}")
print(f"\nBias of naive OLS: {abs(naive.coef_[0] - dgp.true_ame_):.6f}")
print(f"Bias of OLS+controls: {abs(controlled.coef_[0] - dgp.true_ame_):.6f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. AME Estimation with PremiumElasticity
# MAGIC
# MAGIC The debiased ML estimator using Riesz regression. No GPS estimation required.

# COMMAND ----------

model = PremiumElasticity(
    outcome_family="gaussian",
    n_folds=5,
    riesz_type="forest",
    riesz_kwargs={"n_estimators": 200, "max_depth": 6, "random_state": 0},
    inference="eif",
    ci_level=0.95,
    random_state=42,
)

model.fit(X, D, Y)
result = model.estimate()

print("=== AME Estimation Results ===")
print(result.summary())
print(f"\nTrue AME:   {dgp.true_ame_:.6f}")
print(f"Estimated:  {result.estimate:.6f}")
print(f"Bias:       {result.estimate - dgp.true_ame_:.6f}")
print(f"CI covers truth: {result.ci_low <= dgp.true_ame_ <= result.ci_high}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Riesz loss diagnostics
# MAGIC
# MAGIC Lower is better. Useful for comparing riesz_type or hyperparameter choices.

# COMMAND ----------

loss = model.riesz_loss()
print(f"Out-of-fold Riesz loss: {loss:.6f}")

# Compare with linear Riesz
model_linear = PremiumElasticity(
    outcome_family="gaussian",
    n_folds=5,
    riesz_type="linear",
    random_state=42,
)
model_linear.fit(X, D, Y)
result_linear = model_linear.estimate()
loss_linear = model_linear.riesz_loss()

print(f"\nForest Riesz loss:  {loss:.6f}  AME={result.estimate:.6f}")
print(f"Linear Riesz loss:  {loss_linear:.6f}  AME={result_linear.estimate:.6f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Segment-Level Effects
# MAGIC
# MAGIC No refitting required. The AME decomposes additively over subgroups via the EIF.

# COMMAND ----------

# Create age bands and NCB segments
age_bands = pd.cut(
    df["age_norm"],
    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    labels=["very_young", "young", "middle", "mature", "senior"]
)

segment_results = model.effect_by_segment(age_bands)

print("=== Segment-Level AME (by age band) ===")
print(f"{'Segment':<15} {'AME':>10} {'SE':>8} {'95% CI':>22} {'N':>7}")
print("-" * 65)
for sr in segment_results:
    r = sr.result
    print(
        f"{sr.segment_name:<15} {r.estimate:>+10.5f} {r.se:>8.5f} "
        f"[{r.ci_low:>+9.5f}, {r.ci_high:>+9.5f}] {sr.n_obs:>7,}"
    )

# COMMAND ----------

# Plot segment AMEs
fig, ax = plt.subplots(figsize=(9, 5))
names = [sr.segment_name for sr in segment_results]
ests = [sr.result.estimate for sr in segment_results]
lows = [sr.result.ci_low for sr in segment_results]
highs = [sr.result.ci_high for sr in segment_results]
yerr = np.array([
    [e - l for e, l in zip(ests, lows)],
    [h - e for e, h in zip(ests, highs)]
])
ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
ax.axhline(dgp.true_ame_, color="red", linestyle=":", label=f"True AME={dgp.true_ame_:.4f}")
ax.errorbar(names, ests, yerr=yerr, fmt="o", capsize=5, color="steelblue")
ax.set_xlabel("Age Band")
ax.set_ylabel("AME")
ax.set_title("Segment-Level Average Marginal Effects")
ax.legend()
plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Dose-Response Curve

# COMMAND ----------

dr_model = DoseResponseCurve(
    outcome_family="gaussian",
    n_folds=5,
    bandwidth="silverman",
    kernel="gaussian",
    random_state=42,
)
dr_model.fit(X, D, Y)

d_grid = np.linspace(np.percentile(D, 3), np.percentile(D, 97), 50)
dr_result = dr_model.predict(d_grid)

print(f"Bandwidth (Silverman): {dr_result.bandwidth:.2f}")
print(f"E[Y(d=300)] = {dr_model.predict(np.array([300.0])).ate[0]:.4f}")
print(f"E[Y(d=500)] = {dr_model.predict(np.array([500.0])).ate[0]:.4f}")

# True dose-response for comparison
true_dr = np.array([dgp.true_dose_response_(d) for d in d_grid])

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(d_grid, dr_result.ate, color="steelblue", label="Estimated E[Y(d)]")
ax.fill_between(d_grid, dr_result.ci_low, dr_result.ci_high, alpha=0.25, color="steelblue")
ax.plot(d_grid, true_dr, color="red", linestyle="--", label="True E[Y(d)]")
ax.plot(D, np.full_like(D, dr_result.ate.min() - 0.05), "|", color="grey", alpha=0.2, markersize=3)
ax.set_xlabel("Annual Premium (£)")
ax.set_ylabel("E[Y(d)]")
ax.set_title("Dose-Response Curve: Counterfactual Mean Outcome by Premium Level")
ax.legend()
plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Policy Shift Effect

# COMMAND ----------

ps_model = PolicyShiftEffect(
    outcome_family="gaussian",
    n_folds=5,
    random_state=42,
)
ps_model.fit(X, D, Y)

# Single delta estimate
result_5pct = ps_model.estimate(delta=0.05)
print("=== Policy Shift: +5% premium across portfolio ===")
print(result_5pct.summary())

# Curve across a range of deltas
delta_grid = np.linspace(-0.15, 0.15, 31)
effects_curve = ps_model.estimate_curve(delta_grid)

ests = [effects_curve[d].estimate for d in delta_grid]
lows = [effects_curve[d].ci_low for d in delta_grid]
highs = [effects_curve[d].ci_high for d in delta_grid]

fig, ax = plt.subplots(figsize=(9, 5))
ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
ax.axvline(0, color="grey", linestyle="--", alpha=0.5)
ax.plot(delta_grid * 100, ests, color="steelblue", label="Policy shift effect")
ax.fill_between(delta_grid * 100, lows, highs, alpha=0.25, color="steelblue")
ax.set_xlabel("Premium Change (%)")
ax.set_ylabel("E[Y(D*(1+delta))] - E[Y]")
ax.set_title("Policy Shift Effect: Counterfactual Impact of Uniform Premium Change")
ax.legend()
plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Selection-Corrected Elasticity
# MAGIC
# MAGIC Simulates the renewal selection bias problem: claims only observed for renewers.

# COMMAND ----------

# Generate data with selection
dgp_sel = SyntheticContinuousDGP(
    n=5000,
    n_features=8,
    outcome_family="gaussian",
    beta_D=-0.002,
    selection_strength=1.5,
    random_state=42,
)
X_sel, D_sel, Y_sel, S_sel = dgp_sel.generate(include_selection=True)
print(f"Renewal rate: {S_sel.mean():.1%}")

# Replace NaN (non-renewers) with 0
Y_obs = np.where(np.isnan(Y_sel), 0.0, Y_sel)

# Naive: ignore selection
naive_model = PremiumElasticity(outcome_family="gaussian", n_folds=5, random_state=42)
naive_model.fit(X_sel[S_sel == 1], D_sel[S_sel == 1], Y_obs[S_sel == 1])
naive_result = naive_model.estimate()

# Selection-corrected
sel_model = SelectionCorrectedElasticity(
    outcome_family="gaussian",
    n_folds=5,
    random_state=42,
)
sel_model.fit(X_sel, D_sel, Y_obs, S=S_sel)
sel_result = sel_model.estimate()

print(f"\nTrue AME:                 {dgp_sel.true_ame_:.6f}")
print(f"Naive (renewers only):    {naive_result.estimate:.6f}  se={naive_result.se:.6f}")
print(f"Selection-corrected:      {sel_result.estimate:.6f}  se={sel_result.se:.6f}")
print(f"\nRenewal rate: {sel_result.notes}")

# COMMAND ----------

# Sensitivity bounds for unobserved confounding of selection
bounds = sel_model.sensitivity_bounds(gamma_grid=np.array([1.0, 1.5, 2.0, 3.0]))
print("\n=== Sensitivity Bounds (Gamma = odds ratio for unobserved selection confounders) ===")
print(f"{'Gamma':<8} {'Lower AME':>12} {'Upper AME':>12}")
for gamma, b in bounds.items():
    print(f"{gamma:<8.1f} {b['lower']:>12.6f} {b['upper']:>12.6f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. FCA Evidence Report

# COMMAND ----------

report = ElasticityReport(
    estimator=model,
    segment_results=segment_results,
    analyst="Burning Cost Demo",
    title="Motor Renewal Price Elasticity Analysis",
)

html = report.to_html("/tmp/elasticity_report.html")
json_data = report.to_json("/tmp/elasticity_report.json")

print("Report generated: /tmp/elasticity_report.html")
print(f"JSON keys: {list(json_data.keys())}")
print(f"Overall AME: {json_data['overall_ame']['estimate']:.6f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Validation Summary
# MAGIC
# MAGIC Compare all estimates against ground truth.

# COMMAND ----------

print("=" * 65)
print("VALIDATION SUMMARY")
print("=" * 65)
print(f"{'Method':<35} {'Estimate':>10} {'SE':>8} {'Covers truth':>14}")
print("-" * 65)

true_ame = dgp.true_ame_

rows = [
    ("Naive OLS (no controls)", naive.coef_[0], None),
    ("OLS with controls", controlled.coef_[0], None),
    ("PremiumElasticity (ForestRiesz)", result.estimate, result),
    ("PremiumElasticity (LinearRiesz)", result_linear.estimate, result_linear),
    ("SelectionCorrected", sel_result.estimate, sel_result),
]

for name, est, r in rows:
    if r is not None:
        covers = "YES" if r.ci_low <= true_ame <= r.ci_high else "NO"
        print(f"{name:<35} {est:>+10.5f} {r.se:>8.5f} {covers:>14}")
    else:
        print(f"{name:<35} {est:>+10.5f} {'N/A':>8} {'N/A':>14}")

print(f"\nTrue AME: {true_ame:.6f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notes
# MAGIC
# MAGIC **Runtime on this dataset (n=5,000, p=8):** approximately 3-5 minutes on a single
# MAGIC Databricks CPU cluster (Standard_DS3_v2 or equivalent). The bottleneck is the
# MAGIC random forest fitting in each cross-fitting fold.
# MAGIC
# MAGIC **For production use:**
# MAGIC - Increase `n_estimators` to 500+ for more stable Riesz estimates
# MAGIC - Use `nuisance_backend="catboost"` for better-calibrated Poisson/Tweedie outcomes
# MAGIC - 5-fold cross-fitting is standard; use 10 folds for final published estimates
# MAGIC - The `cluster_ids` parameter on `PremiumElasticity` handles household-level clustering

# COMMAND ----------

print("Demo complete.")
print(f"insurance-autodml version: {__import__('insurance_autodml').__version__}")
