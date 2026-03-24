# Databricks notebook source
# MAGIC %md
# MAGIC # RateChangeEvaluator — Demo Notebook
# MAGIC
# MAGIC **insurance-causal v0.6.0**
# MAGIC
# MAGIC This notebook demonstrates the full workflow for post-hoc causal evaluation
# MAGIC of an insurance rate change. Two methods are covered:
# MAGIC
# MAGIC 1. **Difference-in-Differences (DiD)** — when you have a treated group (e.g.
# MAGIC    young drivers who received a rate increase) and a control group (other segments
# MAGIC    where rates were unchanged).
# MAGIC
# MAGIC 2. **Interrupted Time Series (ITS)** — when the entire book was treated and no
# MAGIC    control group exists. Uses the pre-intervention trend as the counterfactual.
# MAGIC
# MAGIC The canonical question: "We raised rates 8% on young drivers in Q3. Given our
# MAGIC before/after portfolio data, what was the causal effect on conversion rate? On
# MAGIC loss ratio?"
# MAGIC
# MAGIC **FCA Consumer Duty context:** PS21/5 and PRIN 12 require firms to evidence
# MAGIC that rate changes deliver fair value. Post-hoc causal attribution is the
# MAGIC correct evidencing framework — this module produces output suitable for a
# MAGIC regulatory evidence pack.

# COMMAND ----------

# MAGIC %pip install insurance-causal[rate_change] --quiet

# COMMAND ----------

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from insurance_causal.rate_change import (
    RateChangeEvaluator,
    make_rate_change_data,
    make_its_data,
    RateChangeResult,
    UK_INSURANCE_SHOCKS,
)

print("insurance-causal rate_change module loaded successfully")
print(f"\nKnown UK insurance shocks:")
for shock, quarter in UK_INSURANCE_SHOCKS.items():
    print(f"  {quarter}: {shock}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Difference-in-Differences (DiD)
# MAGIC
# MAGIC ### Generate synthetic panel data
# MAGIC
# MAGIC We simulate a motor book with 20 segments, 12 quarters, and a rate change
# MAGIC in Q7. 40% of segments (8 segments) are treated with a true ATT of -3pp on
# MAGIC loss ratio. The DGP satisfies parallel trends by construction.

# COMMAND ----------

df = make_rate_change_data(
    n_policies=15_000,
    n_segments=20,
    n_periods=12,
    change_period=7,
    treated_fraction=0.4,
    true_att=-0.03,         # 3pp reduction in loss ratio
    outcome="loss_ratio",
    random_state=42,
)

print(f"Dataset: {len(df):,} policy-periods")
print(f"Segments: {df['segment_id'].nunique()} total")
print(f"Treated segments: {df[df['treated']==1]['segment_id'].nunique()}")
print(f"Control segments: {df[df['treated']==0]['segment_id'].nunique()}")
print(f"\nSample:")
print(df.head(8).to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit the evaluator — auto method selection

# COMMAND ----------

evaluator = RateChangeEvaluator(
    method="auto",          # selects DiD because control group exists
    outcome_col="loss_ratio",
    period_col="period",
    treated_col="treated",
    change_period=7,
    exposure_col="exposure",
    unit_col="segment_id",
)

with warnings.catch_warnings(record=True) as caught_warnings:
    warnings.simplefilter("always")
    result = evaluator.fit(df).summary()

print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parallel trends diagnostic
# MAGIC
# MAGIC The event study shows pre-treatment coefficients (e = -4, -3, -2) should be
# MAGIC near zero if the parallel trends assumption holds. The joint F-test p-value
# MAGIC should be > 0.05.

# COMMAND ----------

pt = evaluator.parallel_trends_test()
print(f"\nParallel trends joint F-test: F={pt.joint_pt_fstat:.3f}, p={pt.joint_pt_pvalue:.3f}")
print(f"Passes (p > 0.05): {pt.passes}")
print(f"\nEvent study coefficients:")
print(pt.event_study_df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualisation

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

evaluator.plot_event_study(
    ax=axes[0],
    title="Event Study: Motor Young Driver Rate Change Q7",
)

evaluator.plot_pre_post(
    ax=axes[1],
    title="Pre-Post Loss Ratio: Treated vs Control",
)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Interrupted Time Series (ITS)
# MAGIC
# MAGIC When the entire book was repriced and no control group exists, ITS uses the
# MAGIC pre-intervention trend as the counterfactual.
# MAGIC
# MAGIC The model is:
# MAGIC `Y_t = beta_0 + beta_1*t + beta_2*D_t + beta_3*(t-T)*D_t + seasonal + epsilon`
# MAGIC
# MAGIC - `beta_2` = immediate level shift (the step change in loss ratio)
# MAGIC - `beta_3` = change in slope post-intervention (continuing trend change)

# COMMAND ----------

df_ts = make_its_data(
    n_periods=20,
    change_period=11,
    true_level_shift=-0.05,    # 5pp immediate reduction
    true_slope_change=-0.002,  # continuing 0.2pp/quarter improvement
    true_pre_trend=0.003,      # slight pre-intervention deterioration
    random_state=42,
)

print(f"Time series: {len(df_ts)} periods")
print(f"Pre-treatment: {(df_ts['period'] < 11).sum()} periods")
print(f"Post-treatment: {(df_ts['period'] >= 11).sum()} periods")
print(f"\nPre mean: {df_ts[df_ts['period'] < 11]['outcome'].mean():.4f}")
print(f"Post mean: {df_ts[df_ts['period'] >= 11]['outcome'].mean():.4f}")

# COMMAND ----------

its_evaluator = RateChangeEvaluator(
    method="its",
    outcome_col="outcome",
    period_col="period",
    change_period=11,
    exposure_col="exposure",
)

result_its = its_evaluator.fit(df_ts).summary()
print(result_its)

# COMMAND ----------

its_evaluator.plot_pre_post(title="ITS: Portfolio-Wide Rate Change — Observed vs Counterfactual")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. UK Shock Proximity Check
# MAGIC
# MAGIC When the intervention date is near a known UK insurance market shock,
# MAGIC RateChangeEvaluator warns automatically. This is particularly important when
# MAGIC the intervention coincided with GIPP implementation or a COVID lockdown.

# COMMAND ----------

# Build quarterly ITS data centred on 2022Q1 (GIPP implementation)
# In practice this would be real portfolio data
quarters = ["2021Q1","2021Q2","2021Q3","2021Q4","2022Q1","2022Q2","2022Q3","2022Q4",
            "2023Q1","2023Q2","2023Q3","2023Q4"]
np.random.seed(42)
df_gipp = pd.DataFrame({
    "period": quarters,
    "outcome": [0.68, 0.67, 0.66, 0.69, 0.63, 0.62, 0.61, 0.63,
                0.60, 0.59, 0.61, 0.58],
    "exposure": [50_000] * 12,
    "quarter": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
})

gipp_evaluator = RateChangeEvaluator(
    method="its",
    outcome_col="outcome",
    period_col="period",
    change_period="2022Q1",  # coincides with GIPP
    exposure_col="exposure",
)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    gipp_evaluator.fit(df_gipp)

print("Warnings raised:")
for w in caught:
    if issubclass(w.category, UserWarning):
        print(f"  [{w.category.__name__}] {w.message}")

print("\n")
result_gipp = gipp_evaluator.summary()
print(result_gipp)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Staggered Adoption Detection
# MAGIC
# MAGIC Standard TWFE DiD is biased when different segments receive treatment at
# MAGIC different times (Goodman-Bacon 2021). RateChangeEvaluator detects this
# MAGIC automatically and warns.

# COMMAND ----------

# Build staggered data: some segments treated from period 5, others from period 7
rng = np.random.default_rng(777)
staggered_rows = []
for seg in range(20):
    treated = 0 if seg < 8 else 1
    cohort = 5 if (treated == 1 and seg < 14) else (7 if treated == 1 else 99)
    for p in range(1, 13):
        effect = -0.03 if (treated == 1 and p >= cohort) else 0.0
        staggered_rows.append({
            "segment_id": seg,
            "period": p,
            "treated": treated,
            "loss_ratio": 0.65 + effect + rng.normal(0, 0.03),
            "exposure": rng.lognormal(0, 0.4) * 500,
        })

df_staggered = pd.DataFrame(staggered_rows)

staggered_ev = RateChangeEvaluator(
    method="did",
    outcome_col="loss_ratio",
    period_col="period",
    treated_col="treated",
    change_period=5,
    exposure_col="exposure",
    unit_col="segment_id",
)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    staggered_ev.fit(df_staggered)

for w in caught:
    if issubclass(w.category, UserWarning):
        print(f"Warning: {w.message}")

result_staggered = staggered_ev.summary()
print(f"\nstaggered_adoption_detected: {result_staggered.staggered_adoption_detected}")
print(f"Estimated ATT (biased): {result_staggered.att:.4f}")
print("\nFor valid staggered DiD, use: insurance_causal_policy.StaggeredEstimator")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. FCA Consumer Duty Evidence Pack Context
# MAGIC
# MAGIC The output of `RateChangeEvaluator.summary()` is designed to be drop-in
# MAGIC evidence for an FCA Consumer Duty review:
# MAGIC
# MAGIC - ATT with confidence interval: "The rate change caused a statistically
# MAGIC   significant -3.1pp reduction in loss ratio (95% CI: -5.1pp to -1.1pp)"
# MAGIC - Parallel trends test: validates the counterfactual assumption
# MAGIC - UK shock proximity warnings: flags concurrent market events
# MAGIC - Staggered adoption: warns if TWFE is potentially biased
# MAGIC
# MAGIC The `str(result)` output can be pasted directly into a review document.

# COMMAND ----------

print("=" * 60)
print("RATE CHANGE EVIDENCE PACK EXTRACT")
print("=" * 60)
print()
print(result)  # from the DiD run above
print()
print("Parallel Trends Validation:")
print(f"  Joint F-test p-value: {pt.joint_pt_pvalue:.3f}")
print(f"  Assessment: {'Pre-treatment trends are parallel (no violation detected)' if pt.passes else 'CAUTION: Parallel trends may be violated'}")

# COMMAND ----------

print("\nDemo complete.")
