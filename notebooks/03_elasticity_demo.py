# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-elasticity: Causal Price Elasticity for UK Renewal Pricing
# MAGIC
# MAGIC This notebook demonstrates the full workflow:
# MAGIC
# MAGIC 1. Generate synthetic UK motor renewal data with known true elasticity
# MAGIC 2. Run treatment variation diagnostics (pre-flight check for DML)
# MAGIC 3. Fit the CausalForestDML elasticity estimator
# MAGIC 4. Inspect ATE, CATE distribution, and GATEs by segment
# MAGIC 5. Plot the elasticity surface (NCD × age)
# MAGIC 6. Run the FCA PS21/5-compliant pricing optimiser
# MAGIC 7. Generate and plot the portfolio demand curve
# MAGIC
# MAGIC **Runtime**: approximately 8–12 minutes on Databricks serverless (ML runtime 14+).

# COMMAND ----------
# MAGIC %md ## 1. Install the library

# COMMAND ----------

# MAGIC %pip install insurance-elasticity[all] --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md ## 2. Generate synthetic renewal data

# COMMAND ----------

from insurance_causal.elasticity.data import make_renewal_data, true_gate_by_ncd, true_gate_by_age
import polars as pl

# 10,000 records: fast enough to run in a demo, large enough for stable estimates
df = make_renewal_data(n=10_000, seed=42, price_variation_sd=0.08)

print(f"Rows: {len(df):,}")
print(f"Renewal rate: {df['renewed'].mean():.1%}")
print(f"Mean log price change: {df['log_price_change'].mean():.4f}")
print(f"SD log price change: {df['log_price_change'].std():.4f}")
print()
print(df.head(5))

# COMMAND ----------
# MAGIC %md
# MAGIC ### True elasticities in the DGP (for validation later)
# MAGIC
# MAGIC The data generator has known, heterogeneous true elasticities by NCD band and age.
# MAGIC We use these to validate the estimator's accuracy.

# COMMAND ----------

print("True GATE by NCD band:")
print(true_gate_by_ncd(df))
print()
print("True GATE by age band:")
print(true_gate_by_age(df))

# COMMAND ----------
# MAGIC %md ## 3. Treatment variation diagnostic (pre-flight check)
# MAGIC
# MAGIC Before fitting DML, check whether there is enough residual price variation
# MAGIC after conditioning on observable risk factors. If less than 10% of the
# MAGIC price variation is "unexplained" by risk factors, DML results will be unreliable.

# COMMAND ----------

from insurance_causal.elasticity.diagnostics import ElasticityDiagnostics

confounders = ["age", "ncd_years", "vehicle_group", "region", "channel"]

diag = ElasticityDiagnostics(n_folds=3, random_state=42)
report = diag.treatment_variation_report(
    df,
    treatment="log_price_change",
    confounders=confounders,
)
print(report.summary())

# COMMAND ----------
# MAGIC %md ### Calibration check: does renewal rate fall with price?

# COMMAND ----------

calib = diag.calibration_summary(df, outcome="renewed", treatment="log_price_change", n_bins=10)
print(calib)

# COMMAND ----------
# MAGIC %md ## 4. Fit the elasticity estimator

# COMMAND ----------

from insurance_causal.elasticity.fit import RenewalElasticityEstimator

est = RenewalElasticityEstimator(
    cate_model="causal_forest",
    n_estimators=150,
    catboost_iterations=300,
    n_folds=5,
    random_state=42,
)

print("Fitting CausalForestDML with CatBoost nuisance models...")
est.fit(df, outcome="renewed", treatment="log_price_change", confounders=confounders)
print("Done.")

# COMMAND ----------
# MAGIC %md ## 5. Average treatment effect

# COMMAND ----------

ate, lb, ub = est.ate()
print(f"ATE:        {ate:.4f}")
print(f"95% CI:     [{lb:.4f}, {ub:.4f}]")
print()
print(f"Interpretation: a 10% price increase (log change ≈ 0.095) reduces")
print(f"renewal probability by approximately {ate * 0.095:.3f} percentage points")
print(f"on average across the portfolio.")

# COMMAND ----------
# MAGIC %md ## 6. CATE distribution

# COMMAND ----------

import numpy as np

cate_vals = est.cate(df)
lb_vals, ub_vals = est.cate_interval(df)

print(f"CATE distribution across {len(cate_vals):,} customers:")
print(f"  Mean:   {np.mean(cate_vals):.4f}  (should be close to ATE)")
print(f"  SD:     {np.std(cate_vals):.4f}  (heterogeneity)")
print(f"  P10:    {np.percentile(cate_vals, 10):.4f}")
print(f"  P50:    {np.percentile(cate_vals, 50):.4f}")
print(f"  P90:    {np.percentile(cate_vals, 90):.4f}")

# COMMAND ----------
# MAGIC %md ## 7. Group average treatment effects (GATEs)

# COMMAND ----------

gate_ncd = est.gate(df, by="ncd_years")
print("Estimated GATE by NCD band:")
print(gate_ncd)
print()
print("True GATE by NCD band:")
print(true_gate_by_ncd(df))

# COMMAND ----------

gate_channel = est.gate(df, by="channel")
print("Estimated GATE by channel:")
print(gate_channel)

# COMMAND ----------
# MAGIC %md ## 8. Elasticity surface plots

# COMMAND ----------

import matplotlib.pyplot as plt
from insurance_causal.elasticity.surface import ElasticitySurface

surface = ElasticitySurface(est)

# Segment summary by NCD
summary = surface.segment_summary(df, by="ncd_years")
print("Segment summary by NCD:")
print(summary)

# COMMAND ----------

# GATE bar chart: elasticity by NCD band
fig1 = surface.plot_gate(df, by="ncd_years", title="Elasticity by NCD band")
display(fig1)
plt.close("all")

# COMMAND ----------

# GATE by channel
fig2 = surface.plot_gate(df, by="channel", title="Elasticity by acquisition channel")
display(fig2)
plt.close("all")

# COMMAND ----------

# 2D heatmap: NCD × channel
fig3 = surface.plot_surface(df, dims=["ncd_years", "channel"],
                            title="Elasticity surface: NCD × channel")
display(fig3)
plt.close("all")

# COMMAND ----------
# MAGIC %md ## 9. FCA PS21/5-compliant renewal pricing optimisation

# COMMAND ----------

from insurance_causal.elasticity.optimise import RenewalPricingOptimiser

opt = RenewalPricingOptimiser(
    est,
    technical_premium_col="tech_prem",
    enbp_col="enbp",
    floor_loading=1.0,   # must at least break even on technical premium
)

priced_df = opt.optimise(df, objective="profit")

print("Sample output:")
print(priced_df.select([
    "policy_id", "tech_prem", "enbp", "offer_price",
    "optimal_price", "predicted_renewal_prob", "expected_profit", "enbp_headroom"
]).head(10))

# COMMAND ----------

# Compare mean prices
print(f"Current mean offer price:  £{df['offer_price'].mean():.2f}")
print(f"Optimal mean price:        £{priced_df['optimal_price'].mean():.2f}")
print(f"Mean ENBP:                 £{df['enbp'].mean():.2f}")
print()
print(f"Expected renewal rate at optimal pricing: {priced_df['predicted_renewal_prob'].mean():.1%}")

# COMMAND ----------
# MAGIC %md ## 10. ENBP compliance audit

# COMMAND ----------

audit = opt.enbp_audit(priced_df)

n_breaches = (audit["compliant"] == False).sum()
print(f"ENBP compliance audit:")
print(f"  Total policies:     {len(audit):,}")
print(f"  Compliant:          {audit['compliant'].sum():,}")
print(f"  Breaches:           {n_breaches:,}")
print(f"  Breach rate:        {100 * n_breaches / len(audit):.2f}%")

if n_breaches == 0:
    print("  Status: PASS — all optimal prices are at or below ENBP.")
else:
    print("  Status: FAIL — review breaches before production deployment.")
    print(audit.filter(pl.col("compliant") == False).head(10))

# COMMAND ----------
# MAGIC %md ## 11. Portfolio demand curve

# COMMAND ----------

from insurance_causal.elasticity.demand import demand_curve, plot_demand_curve

demand_df = demand_curve(est, df, price_range=(-0.25, 0.25, 50))
print(demand_df)

# COMMAND ----------

fig4 = plot_demand_curve(demand_df, show_profit=True)
display(fig4)
plt.close("all")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 12. Near-deterministic price problem demonstration
# MAGIC
# MAGIC This section shows what happens when you apply DML to a dataset where the
# MAGIC price change is nearly a deterministic function of observable risk factors —
# MAGIC the situation that occurs when an insurer uses a tight pricing grid with no
# MAGIC exogenous variation.

# COMMAND ----------

df_nd = make_renewal_data(n=10_000, seed=99, near_deterministic=True)

diag_nd = ElasticityDiagnostics(n_folds=3)
report_nd = diag_nd.treatment_variation_report(
    df_nd,
    treatment="log_price_change",
    confounders=confounders,
)
print(report_nd.summary())

# COMMAND ----------
# MAGIC %md
# MAGIC If `weak_treatment = True`, the library is telling you: do not trust DML results
# MAGIC on this data. The suggested remedies (A/B tests, panel data, bulk re-rate quasi-
# MAGIC experiments) are in the `suggestions` list above.

# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## Summary
# MAGIC
# MAGIC | Step | Method | Output |
# MAGIC |------|--------|--------|
# MAGIC | Pre-flight | `ElasticityDiagnostics.treatment_variation_report()` | `TreatmentVariationReport` |
# MAGIC | Fit | `RenewalElasticityEstimator.fit()` | Fitted estimator |
# MAGIC | ATE | `.ate()` | Float tuple |
# MAGIC | Per-customer | `.cate()`, `.cate_interval()` | numpy arrays |
# MAGIC | Segment | `.gate(by=...)` | polars DataFrame |
# MAGIC | Visualise | `ElasticitySurface.plot_gate()`, `.plot_surface()` | matplotlib Figure |
# MAGIC | Price | `RenewalPricingOptimiser.optimise()` | polars DataFrame with optimal_price |
# MAGIC | Audit | `.enbp_audit()` | polars DataFrame with compliant flag |
# MAGIC | Portfolio | `demand_curve()`, `plot_demand_curve()` | polars DataFrame + Figure |
