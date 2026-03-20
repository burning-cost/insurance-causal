# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-causal: Heterogeneous Treatment Effects with CausalForestDML
# MAGIC
# MAGIC This notebook demonstrates the full HTE workflow using the `causal_forest` subpackage:
# MAGIC
# MAGIC 1. Generate synthetic UK motor renewal data with known elasticity heterogeneity by NCD band
# MAGIC 2. Fit `HeterogeneousElasticityEstimator` (CausalForestDML with CatBoost nuisances)
# MAGIC 3. Run `HeterogeneousInference` (BLP test for heterogeneity, GATES, CLAN)
# MAGIC 4. Evaluate the targeting rule with `TargetingEvaluator` (RATE/AUTOC/QINI)
# MAGIC 5. Run `CausalForestDiagnostics` to validate data quality
# MAGIC
# MAGIC **What problem does this solve?**
# MAGIC
# MAGIC A portfolio-average price elasticity of -0.18 tells you very little about where to
# MAGIC set renewal prices. NCD=0 customers may be 3x more price-elastic than NCD=5 customers.
# MAGIC Using the population ATE to set discounts means over-discounting loyal customers who
# MAGIC would renew anyway, and under-retaining price-sensitive customers who would lapse.
# MAGIC
# MAGIC `causal_forest` gives you per-customer elasticity estimates with valid confidence
# MAGIC intervals, a formal test that the heterogeneity is real (not noise), and a measure
# MAGIC of how useful those estimates are for targeting discounts.
# MAGIC
# MAGIC **Runtime**: approximately 10–15 minutes on Databricks serverless (ML runtime 14+).

# COMMAND ----------
# MAGIC %md ## 1. Install dependencies

# COMMAND ----------

# MAGIC %pip install "insurance-causal[all]" --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md ## 2. Generate synthetic HTE renewal data

# COMMAND ----------

from insurance_causal.causal_forest.data import make_hte_renewal_data, true_cate_by_ncd
import polars as pl
import numpy as np

# 10,000 policies: fast demo, stable estimates
# The DGP has known heterogeneity: NCD=0 -> CATE=-0.30, NCD=5 -> CATE=-0.10
df = make_hte_renewal_data(n=10_000, seed=42, price_sd=0.10)

print(f"Rows: {len(df):,}")
print(f"Renewal rate: {df['renewed'].mean():.1%}")
print(f"Mean log price change: {df['log_price_change'].mean():.4f}")
print(f"SD log price change: {df['log_price_change'].std():.4f}")
print()
print("First 5 rows:")
print(df.head(5))

# COMMAND ----------
# MAGIC %md
# MAGIC ### True CATEs in the DGP (ground truth for validation)
# MAGIC
# MAGIC We will compare our estimated GATEs against these known true values.

# COMMAND ----------

print("True CATE by NCD band (ground truth):")
print(true_cate_by_ncd(df))

# COMMAND ----------
# MAGIC %md ## 3. Fit HeterogeneousElasticityEstimator

# COMMAND ----------

from insurance_causal.causal_forest.estimator import HeterogeneousElasticityEstimator

CONFOUNDERS = ["age", "ncd_years", "vehicle_group", "channel"]

est = HeterogeneousElasticityEstimator(
    binary_outcome=True,
    n_folds=5,
    n_estimators=200,   # must be divisible by n_folds*2=10; auto-rounded if not
    min_samples_leaf=20,
    catboost_iterations=300,
    random_state=42,
)

print("Fitting causal forest... (this takes 3-5 minutes on serverless)")
est.fit(df, outcome="renewed", treatment="log_price_change", confounders=CONFOUNDERS)
print("Done.")

# COMMAND ----------
# MAGIC %md
# MAGIC ### ATE with 95% confidence interval

# COMMAND ----------

ate, lb, ub = est.ate()
print(f"ATE (average treatment effect):  {ate:.4f}")
print(f"95% CI: [{lb:.4f}, {ub:.4f}]")
print()
print(f"Interpretation: a 10% price increase (log_price_change ≈ 0.095)")
print(f"changes renewal probability by approximately {ate * 0.095:.3f} percentage points")
print(f"(95% CI: [{lb * 0.095:.3f}, {ub * 0.095:.3f}])")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Per-customer CATE estimates

# COMMAND ----------

cates = est.cate(df)
lb_cate, ub_cate = est.cate_interval(df)

df_with_cate = df.with_columns([
    pl.Series("cate_estimate", cates),
    pl.Series("cate_lb", lb_cate),
    pl.Series("cate_ub", ub_cate),
])

print("CATE distribution:")
print(f"  Mean:     {np.mean(cates):.4f}")
print(f"  SD:       {np.std(cates):.4f}")
print(f"  Min:      {np.min(cates):.4f}")
print(f"  Max:      {np.max(cates):.4f}")
print(f"  P10:      {np.percentile(cates, 10):.4f}")
print(f"  P90:      {np.percentile(cates, 90):.4f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ### GATEs by NCD band

# COMMAND ----------

gates_by_ncd = est.gate(df, by="ncd_years")
true_by_ncd = true_cate_by_ncd(df)

# Join estimated vs true
comparison = gates_by_ncd.join(
    true_by_ncd.rename({"true_cate_mean": "true_cate"}),
    on="ncd_years",
    how="left"
)
print("Estimated vs True GATE by NCD band:")
print(comparison.select(["ncd_years", "cate", "ci_lower", "ci_upper", "true_cate", "n"]))

# COMMAND ----------
# MAGIC %md
# MAGIC ### GATEs by channel

# COMMAND ----------

gates_by_channel = est.gate(df, by="channel")
print("GATE by channel:")
print(gates_by_channel)

# COMMAND ----------
# MAGIC %md ## 4. HeterogeneousInference (BLP / GATES / CLAN)
# MAGIC
# MAGIC Now we run the formal Chernozhukov et al. (2020/2025) procedure to:
# MAGIC - **Test** whether the CATE proxy explains real heterogeneity (BLP)
# MAGIC - **Quantify** treatment effect variation across CATE quantile groups (GATES)
# MAGIC - **Identify** which risk factors drive the heterogeneity (CLAN)

# COMMAND ----------

from insurance_causal.causal_forest.inference import HeterogeneousInference

inf = HeterogeneousInference(
    n_splits=100,
    k_groups=5,
    random_state=42,
)

print("Running BLP / GATES / CLAN inference (100 data splits)...")
result = inf.run(df, estimator=est, cate_proxy=cates)

print(result.summary())

# COMMAND ----------
# MAGIC %md
# MAGIC ### Plot GATES

# COMMAND ----------

result.plot_gates()

# COMMAND ----------
# MAGIC %md
# MAGIC ### Plot CLAN (feature differences between extreme groups)

# COMMAND ----------

result.plot_clan()

# COMMAND ----------
# MAGIC %md ## 5. Targeting Evaluation (RATE / AUTOC / QINI)
# MAGIC
# MAGIC The RATE answers: "If we target the top q% of policies by estimated elasticity
# MAGIC for retention discounts, what is the average treatment effect in that group?"
# MAGIC
# MAGIC AUTOC > 0 means the targeting rule is better than random.
# MAGIC QINI > 1 means targeting outperforms treating everyone.

# COMMAND ----------

from insurance_causal.causal_forest.targeting import TargetingEvaluator

ev = TargetingEvaluator(n_bootstrap=200, n_toc_points=50, random_state=42)

print("Evaluating targeting rule...")
targeting = ev.evaluate(df, estimator=est, cate_proxy=cates)

print(targeting.summary())

# COMMAND ----------
# MAGIC %md
# MAGIC ### Plot TOC (Treatment on Classified) curve

# COMMAND ----------

targeting.plot_toc()

# COMMAND ----------
# MAGIC %md ## 6. Diagnostics
# MAGIC
# MAGIC Always run diagnostics before presenting CATE estimates to a pricing committee.
# MAGIC The three key checks: overlap, residual variation, and BLP calibration.

# COMMAND ----------

from insurance_causal.causal_forest.diagnostics import CausalForestDiagnostics

diag = CausalForestDiagnostics(n_splits=50, random_state=42)
report = diag.check(df, estimator=est, cates=cates)
print(report.summary())

# COMMAND ----------
# MAGIC %md
# MAGIC ### Contrast: near-deterministic price data (the problem we're diagnosing)

# COMMAND ----------

df_degen = make_hte_renewal_data(n=5000, seed=99, price_sd=0.005)
est_degen = HeterogeneousElasticityEstimator(
    n_folds=5, n_estimators=100, min_samples_leaf=20, catboost_iterations=100
)
est_degen.fit(df_degen, confounders=CONFOUNDERS)
cates_degen = est_degen.cate(df_degen)

report_degen = diag.check(df_degen, estimator=est_degen, cates=cates_degen)
print("Diagnostics on near-deterministic price data (price_sd=0.005):")
print(report_degen.summary())

# COMMAND ----------
# MAGIC %md ## 7. Exposure weighting for claim frequency outcomes
# MAGIC
# MAGIC For claim frequency models (Poisson / rate outcomes), use
# MAGIC `build_exposure_weighted_nuisances()` and `prepare_rate_outcome()`.

# COMMAND ----------

from insurance_causal.causal_forest.exposure import (
    build_exposure_weighted_nuisances,
    prepare_rate_outcome,
)
import numpy as np

# Simulated claims count data with varying exposure
rng = np.random.default_rng(42)
n = 1000
Y_counts = rng.poisson(0.05, size=n)
exposure = rng.uniform(0.5, 1.0, size=n)  # partial-year exposures

Y_rate, exposure_weights = prepare_rate_outcome(Y_counts, exposure)
print(f"Mean claim count: {np.mean(Y_counts):.4f}")
print(f"Mean claim rate (count/exposure): {np.mean(Y_rate):.4f}")
print(f"Mean exposure: {np.mean(exposure):.4f}")

model_y, model_t = build_exposure_weighted_nuisances(
    binary_outcome=False,
    catboost_iterations=50,
)
print(f"\nOutcome nuisance model: {type(model_y).__name__}")
print(f"Treatment nuisance model: {type(model_t).__name__}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC The `causal_forest` subpackage provides:
# MAGIC
# MAGIC | Component | Purpose |
# MAGIC |---|---|
# MAGIC | `HeterogeneousElasticityEstimator` | CausalForestDML with CatBoost, min_samples_leaf=20, honest splits |
# MAGIC | `HeterogeneousInference` | BLP test (is heterogeneity real?), GATES, CLAN |
# MAGIC | `TargetingEvaluator` | AUTOC/QINI: how good is the targeting rule? |
# MAGIC | `CausalForestDiagnostics` | Overlap, residual variation, calibration |
# MAGIC | `build_exposure_weighted_nuisances` | Poisson/rate outcome support |
# MAGIC | `make_hte_renewal_data` | Synthetic DGP with known NCD-driven heterogeneity |
# MAGIC
# MAGIC **For a UK pricing actuary:**
# MAGIC - Run the BLP test. If beta_2 > 0 is significant, the heterogeneity is real and you
# MAGIC   should not use a single elasticity across the book.
# MAGIC - Check GATES. If NCD=0 GATE is significantly more negative than NCD=5 GATE, build
# MAGIC   NCD-dependent discount structures.
# MAGIC - Check AUTOC. If it's positive and the CI excludes 0, targeting by CATE estimate
# MAGIC   is better than random — use the estimates to prioritise retention discounts.
# MAGIC - Run diagnostics. If residual variation < 10%, the estimates are unreliable.
# MAGIC   Consider A/B price tests to get clean identification.
