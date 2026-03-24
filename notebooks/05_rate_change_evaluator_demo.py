# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # RateChangeEvaluator Demo
# MAGIC
# MAGIC **insurance-causal v0.6.0**
# MAGIC
# MAGIC Every pricing team asks the same question after a rate change: *We raised rates 8%
# MAGIC on young drivers in January. What actually happened to conversion and loss ratio?*
# MAGIC
# MAGIC This notebook demonstrates the `RateChangeEvaluator` — a post-hoc causal evaluation
# MAGIC tool that uses Difference-in-Differences (DiD) or Interrupted Time Series (ITS)
# MAGIC depending on whether a control group exists.
# MAGIC
# MAGIC ---
# MAGIC ## Method selection logic
# MAGIC
# MAGIC | Situation | Method | Identification |
# MAGIC |-----------|--------|----------------|
# MAGIC | Segment-specific rate change (young drivers only) | DiD | Parallel trends |
# MAGIC | Whole-book rate change (no control group) | ITS | Pre-trend extrapolation |
# MAGIC
# MAGIC DiD is stronger. ITS is the fallback. Both are implemented here.

# COMMAND ----------
# MAGIC %pip install insurance-causal>=0.6.0 statsmodels>=0.14 matplotlib>=3.7

# COMMAND ----------
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from insurance_causal.rate_change import (
    RateChangeEvaluator,
    make_rate_change_data,
    UK_INSURANCE_SHOCKS,
    check_shock_proximity,
)

print("insurance-causal loaded successfully")

# COMMAND ----------
# MAGIC %md ## Part 1: DiD — Segment-Specific Rate Change

# COMMAND ----------
# Simulate a segment-specific rate change:
# - 40 rating segments, 16 quarters
# - Rate change applied to half the segments at quarter 9
# - True ATT = -5 percentage points on conversion rate
df_did = make_rate_change_data(
    n_segments=40,
    n_periods=16,
    treatment_period=9,
    true_att=-0.05,
    treated_fraction=0.5,
    base_outcome=0.08,  # 8% claim frequency baseline
    noise_scale=0.008,
    exposure_mean=500,
    seed=42,
)
print(f"Shape: {df_did.shape}")
print(f"Treated segments: {df_did.groupby('segment')['treated'].first().sum()}")
print(df_did.head())

# COMMAND ----------
# Fit the evaluator
evaluator_did = RateChangeEvaluator(
    outcome_col="outcome",
    treatment_period=9,
    unit_col="segment",
    weight_col="earned_exposure",
)
evaluator_did.fit(df_did)
print(evaluator_did.summary())

# COMMAND ----------
# Parallel trends test
pt = evaluator_did.parallel_trends_test()
print(f"Parallel trends: F={pt['f_stat']:.2f}, p={pt['p_value']:.3f}")
print(f"Pre-trend periods: {pt['periods']}")
print(f"Pre-trend coefficients: {[f'{c:.5f}' for c in pt['coefs']]}")
print(f"Passed: {pt['passed']}")

# COMMAND ----------
# Event study plot
fig, ax = plt.subplots(figsize=(8, 5))
evaluator_did.plot_event_study(ax=ax, outcome_label="Claim Frequency")
ax.set_title("Pre-Treatment Parallel Trends Check\nEvent study coefficients should be near zero")
plt.tight_layout()
plt.savefig("/tmp/event_study.png", dpi=120)
display(fig)

# COMMAND ----------
# Pre/post outcome plot
fig, ax = plt.subplots(figsize=(9, 5))
evaluator_did.plot_pre_post(
    df_did, ax=ax,
    title="Claim Frequency: Treated vs Control Groups"
)
plt.tight_layout()
plt.savefig("/tmp/pre_post.png", dpi=120)
display(fig)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Checking exposure weighting
# MAGIC
# MAGIC One of the key design decisions: always weight by earned exposure. An
# MAGIC unweighted estimate treats a 50-policy segment identically to a 5,000-policy
# MAGIC segment. That's wrong for any ratio outcome (loss ratio, frequency, conversion).

# COMMAND ----------
# Compare weighted vs unweighted
ev_wt = RateChangeEvaluator(
    outcome_col="outcome", treatment_period=9,
    unit_col="segment", weight_col="earned_exposure"
)
ev_unwt = RateChangeEvaluator(
    outcome_col="outcome", treatment_period=9,
    unit_col="segment", weight_col=None
)
ev_wt.fit(df_did)
ev_unwt.fit(df_did)

d_wt = ev_wt._result.did
d_unwt = ev_unwt._result.did
print(f"Weighted ATT:   {d_wt.att:+.4f} (SE={d_wt.se:.4f})")
print(f"Unweighted ATT: {d_unwt.att:+.4f} (SE={d_unwt.se:.4f})")
print(f"True ATT: -0.0500")
print(f"\nWeighted is closer to truth: {abs(d_wt.att - (-0.05)) < abs(d_unwt.att - (-0.05))}")

# COMMAND ----------
# MAGIC %md ## Part 2: ITS — Whole-Book Rate Change

# COMMAND ----------
# Simulate an ITS scenario: whole-book rate change, no control group
# True level shift = -2 percentage points, slope change = -0.3pp/quarter
df_its = make_rate_change_data(
    mode="its",
    n_periods=20,
    treatment_period=9,
    true_level_shift=-0.02,
    true_slope_change=-0.003,
    base_outcome=0.07,
    noise_scale=0.005,
    seed=42,
)
print(f"Shape: {df_its.shape}")
print(df_its.head())

# COMMAND ----------
# Fit ITS — will emit warning about no control group
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    evaluator_its = RateChangeEvaluator(
        outcome_col="outcome",
        treatment_period=9,
        weight_col="earned_exposure",
    )
    evaluator_its.fit(df_its)

print(evaluator_its.summary())

# COMMAND ----------
# ITS plot: observed vs counterfactual
fig, ax = plt.subplots(figsize=(9, 5))
evaluator_its.plot_its(
    df_its, ax=ax,
    title="ITS: Claim Frequency with Counterfactual Trend"
)
plt.tight_layout()
plt.savefig("/tmp/its_plot.png", dpi=120)
display(fig)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Effect over time (ITS)
# MAGIC
# MAGIC ITS gives a time-varying causal effect: `level_shift + slope_change * k`
# MAGIC where k is periods since intervention.

# COMMAND ----------
res_its = evaluator_its._result.its
print("Causal effect by periods post-intervention:")
for k in range(5):
    eff = res_its.effect_at_k(k)
    print(f"  k={k}: {eff:+.4f}")

# COMMAND ----------
# MAGIC %md ## Part 3: UK Insurance Shock Proximity

# COMMAND ----------
# The evaluator automatically checks whether your treatment period
# coincides with known UK insurance market shocks

print("Known UK insurance shocks:\n")
for period, desc in UK_INSURANCE_SHOCKS.items():
    print(f"  {period}: {desc}")

# COMMAND ----------
# Check GIPP period
msgs = check_shock_proximity("2022-Q1")
print("Warnings for Jan 2022 rate change:")
for m in msgs:
    print(f"\n  ! {m}")

# COMMAND ----------
# MAGIC %md ## Part 4: Staggered Adoption Warning

# COMMAND ----------
# Create a dataset with staggered adoption (two cohorts)
df_staggered = make_rate_change_data(n_segments=30, n_periods=16, seed=0)
segs = sorted(df_staggered["segment"].unique())
early = set(segs[:15])

df_staggered = df_staggered.copy()
df_staggered["treated"] = 0
df_staggered.loc[
    df_staggered["segment"].isin(early) & (df_staggered["period"] >= 7), "treated"
] = 1
df_staggered.loc[
    ~df_staggered["segment"].isin(early) & (df_staggered["period"] >= 11), "treated"
] = 1

ev_staggered = RateChangeEvaluator(
    outcome_col="outcome",
    treatment_period=7,
    unit_col="segment",
)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    ev_staggered.fit(df_staggered)

print("Staggered adoption warnings:")
for w in caught:
    if "staggered" in str(w.message).lower() or "cohort" in str(w.message).lower():
        print(f"\n  ! {w.message}")

# COMMAND ----------
# MAGIC %md ## Part 5: Integration — Realistic Motor Rate Change Scenario

# COMMAND ----------
# Realistic scenario: Q1 2023 motor rate change on under-25s segment
# Outcome: claim frequency
# Control: 25-40 age segment

np.random.seed(100)
n_periods = 20
treatment_period = 13  # Q1 2023 if we start from Q1 2020

rows = []
for period in range(1, n_periods + 1):
    quarter = ((period - 1) % 4) + 1
    for age_band, is_treated in [("under_25", 1), ("25_40", 0), ("over_40", 0)]:
        # Claim frequency varies by age
        base = {"under_25": 0.12, "25_40": 0.07, "over_40": 0.05}[age_band]
        seasonal = {1: 0.004, 2: -0.002, 3: -0.001, 4: 0.006}[quarter]
        trend = 0.0005 * period
        treatment_eff = -0.015 * is_treated * (period >= treatment_period)
        noise = np.random.normal(0, 0.003)
        exposure = max(50, np.random.lognormal(np.log(300), 0.4))

        rows.append({
            "age_band": age_band,
            "period": period,
            "quarter": quarter,
            "treated": is_treated,
            "claim_frequency": max(0.001, base + seasonal + trend + treatment_eff + noise),
            "earned_exposure": exposure,
        })

df_motor = pd.DataFrame(rows)

# Evaluate
ev_motor = RateChangeEvaluator(
    outcome_col="claim_frequency",
    treatment_period=treatment_period,
    unit_col="age_band",
    weight_col="earned_exposure",
    n_pre_periods=3,
)
ev_motor.fit(df_motor)
print(ev_motor.summary())

# COMMAND ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ev_motor.plot_event_study(ax=axes[0], outcome_label="Claim Frequency")
ev_motor.plot_pre_post(
    df_motor, ax=axes[1],
    title="Motor Claim Frequency by Age Band"
)
plt.tight_layout()
plt.savefig("/tmp/motor_scenario.png", dpi=120)
display(fig)

print("\nDemo complete. All scenarios ran successfully.")
