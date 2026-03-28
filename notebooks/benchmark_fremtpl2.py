# Databricks notebook source
# MAGIC %md
# MAGIC # Benchmark: DML on freMTPL2 — BonusMalus Causal Effect on Claim Frequency
# MAGIC
# MAGIC **Library:** `insurance-causal` — Double Machine Learning causal inference for
# MAGIC insurance pricing.
# MAGIC
# MAGIC **Dataset:** freMTPL2 (OpenML dataset 41214) — 677,991 French motor third-party
# MAGIC liability policies from a single insurer, compiled by Arthur Charpentier. A
# MAGIC standard actuarial benchmark dataset covering policy year, exposure, claim count,
# MAGIC BonusMalus score, driver age, vehicle age, vehicle power, vehicle brand, fuel
# MAGIC type, and geographic region.
# MAGIC
# MAGIC **Causal question:** What is the causal effect of a policyholder's BonusMalus
# MAGIC score on their claim frequency, after controlling for the observed rating factors
# MAGIC that also determine BonusMalus level?
# MAGIC
# MAGIC **Why this is interesting:** BonusMalus (the French equivalent of NCD in the UK)
# MAGIC is not randomly assigned — it accumulates based on past claims history. A
# MAGIC policyholder with BonusMalus=100 (neutral) is very different from one with
# MAGIC BonusMalus=150 (penalised). The raw correlation between BonusMalus and current
# MAGIC claim frequency is confounded by the underlying risk factors that drove both the
# MAGIC past claims (and thus the BonusMalus score) and the current claims.
# MAGIC
# MAGIC Specifically: older vehicles, certain regions, and younger/older driver ages all
# MAGIC drive higher claim rates. These same factors correlate with BonusMalus. The naive
# MAGIC GLM estimate of the BonusMalus-frequency relationship picks up this confounding.
# MAGIC DML isolates the causal signal by partialling out the observed rating factors from
# MAGIC both the treatment (BonusMalus) and the outcome (claim count).
# MAGIC
# MAGIC **This is a quasi-experiment, not a true RCT.** We cannot observe the
# MAGIC counterfactual: what would this policyholder's claim frequency have been at a
# MAGIC different BonusMalus level? DML gives us the best available observational estimate,
# MAGIC with valid confidence intervals conditional on the observed confounders. Residual
# MAGIC confounding from unobserved factors (driving behaviour, annual mileage, occupation)
# MAGIC remains. The sensitivity analysis section quantifies the robustness.
# MAGIC
# MAGIC **Date:** 2026-03-28
# MAGIC **Library version:** 0.6.2

# COMMAND ----------

import time
import warnings

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.datasets import fetch_openml

from insurance_causal import CausalPricingModel, diagnostics
from insurance_causal.treatments import ContinuousTreatment

warnings.filterwarnings("ignore", category=UserWarning, module="catboost")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load freMTPL2 from OpenML
# MAGIC
# MAGIC OpenML dataset 41214 is the exposure-weighted freMTPL2 dataset. It merges the
# MAGIC frequency table (freMTPL2freq) and severity table from Charpentier's CASdatasets
# MAGIC R package. We use only the frequency side here: claim count and exposure.
# MAGIC
# MAGIC The dataset has 677,991 rows. Each row is one policy-year. The key columns are:
# MAGIC
# MAGIC | Column | Description |
# MAGIC |--------|-------------|
# MAGIC | ClaimNb | Number of claims in the policy year |
# MAGIC | Exposure | Fraction of year the policy was active (0–1) |
# MAGIC | BonusMalus | Score 50–350; 100 = neutral, >100 = penalised for past claims |
# MAGIC | VehAge | Vehicle age in years |
# MAGIC | DrivAge | Driver age in years |
# MAGIC | VehPower | Vehicle power category (4–15) |
# MAGIC | VehBrand | Vehicle brand (B1–B14 categories) |
# MAGIC | VehGas | Fuel type (Diesel/Regular) |
# MAGIC | Area | Geographic area code (A–F) |
# MAGIC | Region | French region code (10 categories) |
# MAGIC | Density | Log population density of the municipality |

# COMMAND ----------

print("Loading freMTPL2 from OpenML (dataset 41214)...")
t0 = time.perf_counter()

raw = fetch_openml(data_id=41214, as_frame=True, parser="auto")
df_raw = raw.frame

load_time = time.perf_counter() - t0
print(f"Loaded in {load_time:.1f}s")
print(f"Shape: {df_raw.shape}")
print(f"\nColumns: {list(df_raw.columns)}")
print(f"\nData types:\n{df_raw.dtypes}")
print(f"\nFirst 3 rows:")
print(df_raw.head(3).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Preparation
# MAGIC
# MAGIC ### Treatment: BonusMalus
# MAGIC
# MAGIC BonusMalus is a continuous score ranging from 50 (maximum bonus, 50% discount) to
# MAGIC 350 (maximum malus, 3.5x base premium). We use it as-is on its natural scale —
# MAGIC a 1-unit increase in BonusMalus corresponds to moving one step up the penalty
# MAGIC schedule.
# MAGIC
# MAGIC For interpretability we also standardise it (mean 0, SD 1) so the DML coefficient
# MAGIC is the effect of a 1-SD change in BonusMalus on log claim frequency.
# MAGIC
# MAGIC ### Outcome: ClaimNb (Poisson with offset)
# MAGIC
# MAGIC Claim count with Exposure as offset. This is the standard actuarial Poisson
# MAGIC frequency model. We use `outcome_type="poisson"` and `exposure_col="Exposure"`.
# MAGIC
# MAGIC ### Confounders
# MAGIC
# MAGIC We control for the observed rating factors that the insurer would have used to
# MAGIC price the policy — and that also correlate with BonusMalus through the risk
# MAGIC selection mechanism:
# MAGIC
# MAGIC - **DrivAge**: driver age. Young and very old drivers have higher claim rates AND
# MAGIC   tend to accumulate higher BonusMalus over time (less driving experience / more
# MAGIC   years in the system respectively).
# MAGIC - **VehAge**: vehicle age. Older vehicles may attract less-experienced or
# MAGIC   higher-risk drivers, correlating with both claims and BonusMalus.
# MAGIC - **VehPower**: vehicle power category. High-power vehicles are associated with
# MAGIC   higher claim rates and with driver profiles that accumulate BonusMalus.
# MAGIC - **VehBrand**: vehicle brand categorical. Proxy for vehicle type / driver profile.
# MAGIC - **VehGas**: fuel type (Diesel/Regular). Correlated with use type.
# MAGIC - **Area**: geographic area (A–F). Captures urban/rural risk differential.
# MAGIC - **Region**: French region (10 categories). Regional risk variation.
# MAGIC - **Density**: log population density. Continuous geographic risk proxy.
# MAGIC
# MAGIC ### What we are NOT controlling for (and why it matters)
# MAGIC
# MAGIC We cannot observe: annual mileage, occupation, whether the policyholder is the
# MAGIC primary or secondary driver, and actual driving behaviour. These all drive claim
# MAGIC rates and correlate with BonusMalus. The DML estimate is therefore an
# MAGIC observational causal estimate conditional on the observed confounders — the
# MAGIC sensitivity analysis will quantify robustness to the unobserved component.

# COMMAND ----------

# ── Type coercion ────────────────────────────────────────────────────────────
# OpenML returns everything as object or category; we need clean numerics
# and string categoricals for CatBoost.

df = df_raw.copy()

# Numeric columns
for col in ["ClaimNb", "Exposure", "BonusMalus", "VehPower", "VehAge", "DrivAge", "Density"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# String categoricals (CatBoost handles these natively)
for col in ["VehBrand", "VehGas", "Area", "Region"]:
    df[col] = df[col].astype(str)

# Drop rows with missing values in any used column (extremely few in freMTPL2)
USED_COLS = ["ClaimNb", "Exposure", "BonusMalus", "VehPower", "VehAge",
             "DrivAge", "Density", "VehBrand", "VehGas", "Area", "Region"]
df = df[USED_COLS].dropna().reset_index(drop=True)

# Clip extreme BonusMalus (the distribution is heavily right-tailed above 150)
bm_p99 = df["BonusMalus"].quantile(0.99)
print(f"BonusMalus: min={df['BonusMalus'].min():.0f}, median={df['BonusMalus'].median():.0f}, "
      f"p99={bm_p99:.0f}, max={df['BonusMalus'].max():.0f}")
print(f"  Clipping at p99 = {bm_p99:.0f} for stability")
df["BonusMalus"] = df["BonusMalus"].clip(upper=bm_p99)

# Clip exposure to valid range (a handful of near-zero exposures cause issues)
df = df[df["Exposure"] > 0.01].reset_index(drop=True)

print(f"\nFinal dataset shape after cleaning: {df.shape}")
print(f"Total claims: {df['ClaimNb'].sum():,.0f}")
print(f"Total exposure (policy-years): {df['Exposure'].sum():,.0f}")
print(f"Claim frequency (per policy-year): {df['ClaimNb'].sum() / df['Exposure'].sum():.4f}")
print(f"Zero claims: {(df['ClaimNb'] == 0).mean():.1%}")

# COMMAND ----------

print("\nBonusMalus distribution (post-clip):")
print(df["BonusMalus"].describe().round(2))
print()
bm_bands = pd.cut(df["BonusMalus"], bins=[50, 80, 100, 120, 150, 350],
                  labels=["50-80 (bonus)", "81-100 (near neutral)", "101-120 (light malus)",
                          "121-150 (moderate malus)", "151+ (heavy malus)"])
print("BonusMalus band distribution:")
print(bm_bands.value_counts().sort_index().to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Visualise the Confounding Structure
# MAGIC
# MAGIC Before fitting anything, let's look at the raw association between BonusMalus
# MAGIC and claim frequency, and how the observed confounders relate to BonusMalus.
# MAGIC This motivates why a naive estimate is biased.

# COMMAND ----------

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Panel 1: BonusMalus distribution
axes[0].hist(df["BonusMalus"], bins=60, color="steelblue", alpha=0.7, edgecolor="white", linewidth=0.3)
axes[0].axvline(100, color="black", linestyle="--", linewidth=1.5, label="BonusMalus=100 (neutral)")
axes[0].set_xlabel("BonusMalus score")
axes[0].set_ylabel("Count")
axes[0].set_title(f"BonusMalus Distribution\n(n={len(df):,}, clipped at p99={bm_p99:.0f})")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Panel 2: Claim frequency by BonusMalus ventile
bm_ventile = pd.qcut(df["BonusMalus"], 20, labels=False)
freq_by_bm = (
    df.groupby(bm_ventile, observed=True)
      .apply(lambda g: g["ClaimNb"].sum() / g["Exposure"].sum())
)
bm_means = df.groupby(bm_ventile, observed=True)["BonusMalus"].mean()

axes[1].plot(bm_means.values, freq_by_bm.values, "o-", color="tomato", linewidth=2, markersize=6)
axes[1].axhline(df["ClaimNb"].sum() / df["Exposure"].sum(), color="black", linestyle="--",
                linewidth=1.5, label="Overall mean")
axes[1].set_xlabel("BonusMalus (ventile mean)")
axes[1].set_ylabel("Claim frequency (per policy-year)")
axes[1].set_title("Raw: Claim Frequency by BonusMalus\n(naive association — confounded)")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

# Panel 3: Driver age by BonusMalus ventile (showing one confounder)
age_by_bm = df.groupby(bm_ventile, observed=True)["DrivAge"].mean()
axes[2].plot(bm_means.values, age_by_bm.values, "o-", color="purple", linewidth=2, markersize=6)
axes[2].set_xlabel("BonusMalus (ventile mean)")
axes[2].set_ylabel("Mean driver age")
axes[2].set_title("Driver Age vs BonusMalus\n(older drivers have higher BM — this is confounding)")
axes[2].grid(True, alpha=0.3)

plt.suptitle("freMTPL2: Raw Data Structure Before Causal Adjustment", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("/tmp/fremtpl2_confounding.png", dpi=100, bbox_inches="tight")
plt.show()
print("Saved to /tmp/fremtpl2_confounding.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Baseline: Naive Poisson GLM
# MAGIC
# MAGIC The standard actuarial approach: Poisson GLM with BonusMalus as a continuous
# MAGIC covariate alongside the rating factors. This is what a pricing team would do
# MAGIC if asked "what is the effect of BonusMalus on claim frequency, controlling for
# MAGIC vehicle and driver characteristics?"
# MAGIC
# MAGIC We fit two GLM specifications:
# MAGIC - **GLM-simple**: BonusMalus + numeric confounders only (no categoricals)
# MAGIC - **GLM-full**: BonusMalus + all confounders including brand/gas/area/region
# MAGIC
# MAGIC Both are confounded. The question is how much the DML estimate differs, and
# MAGIC whether the DML confidence interval is meaningfully different.
# MAGIC
# MAGIC Note: on a dataset of 677k rows, both GLMs fit quickly. The DML model takes
# MAGIC substantially longer (5-fold CatBoost cross-fitting over 677k observations).

# COMMAND ----------

# Fit naive GLM — simple specification (continuous confounders only)
t0 = time.perf_counter()

formula_simple = (
    "ClaimNb ~ BonusMalus + DrivAge + VehAge + VehPower + np.log(Density)"
)

glm_simple = smf.glm(
    formula_simple,
    data=df,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=np.log(df["Exposure"].values),
).fit()

glm_simple_time = time.perf_counter() - t0
bm_simple = float(glm_simple.params["BonusMalus"])
bm_simple_lo = float(glm_simple.conf_int().loc["BonusMalus", 0])
bm_simple_hi = float(glm_simple.conf_int().loc["BonusMalus", 1])

print(f"GLM-simple fit time: {glm_simple_time:.2f}s")
print(f"BonusMalus coefficient: {bm_simple:.6f}")
print(f"95% CI: ({bm_simple_lo:.6f}, {bm_simple_hi:.6f})")
print(f"Relative effect per 10 BM units: {(np.exp(bm_simple * 10) - 1) * 100:.2f}%")

# COMMAND ----------

# Fit naive GLM — full specification (including categoricals)
t0 = time.perf_counter()

formula_full = (
    "ClaimNb ~ BonusMalus + DrivAge + VehAge + VehPower + np.log(Density) "
    "+ C(VehBrand) + C(VehGas) + C(Area) + C(Region)"
)

glm_full = smf.glm(
    formula_full,
    data=df,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=np.log(df["Exposure"].values),
).fit()

glm_full_time = time.perf_counter() - t0
bm_full = float(glm_full.params["BonusMalus"])
bm_full_lo = float(glm_full.conf_int().loc["BonusMalus", 0])
bm_full_hi = float(glm_full.conf_int().loc["BonusMalus", 1])

print(f"GLM-full fit time: {glm_full_time:.2f}s")
print(f"BonusMalus coefficient: {bm_full:.6f}")
print(f"95% CI: ({bm_full_lo:.6f}, {bm_full_hi:.6f})")
print(f"Relative effect per 10 BM units: {(np.exp(bm_full * 10) - 1) * 100:.2f}%")

print(f"\n--- GLM-full coefficient table (selected) ---")
coef_table = glm_full.summary2().tables[1][["Coef.", "Std.Err.", "[0.025", "0.975]", "P>|z|"]]
selected = coef_table[coef_table.index.str.startswith(("BonusMalus", "DrivAge", "VehAge", "VehPower"))]
print(selected.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. DML Estimate via insurance-causal
# MAGIC
# MAGIC DML partials out the confounders from both BonusMalus and ClaimNb, then
# MAGIC estimates the causal coefficient from the residual regression.
# MAGIC
# MAGIC **Nuisance models (CatBoost):**
# MAGIC - E[BonusMalus | X]: predicts BonusMalus from all observed confounders
# MAGIC - E[ClaimNb | X]: predicts claim count from all observed confounders
# MAGIC
# MAGIC **The DML coefficient** is the effect of a 1-unit increase in BonusMalus on
# MAGIC log claim frequency, after removing the variation in BonusMalus explained by
# MAGIC the observed rating factors. This is the causal effect under the assumption
# MAGIC that observed confounders are sufficient for identification.
# MAGIC
# MAGIC **Runtime note:** With 677k rows and 5-fold cross-fitting, this fits 10 CatBoost
# MAGIC models (2 nuisance x 5 folds). Expect 10–25 minutes on Databricks serverless.
# MAGIC The fitting is the cost of doing this properly.

# COMMAND ----------

CONFOUNDERS = [
    "DrivAge",
    "VehAge",
    "VehPower",
    "Density",   # log-transform will be applied inside CatBoost via its log1p feature
    "VehBrand",  # categorical — CatBoost handles natively
    "VehGas",    # categorical
    "Area",      # categorical
    "Region",    # categorical
]

# Log-transform Density before passing (CatBoost doesn't apply feature transforms)
df["log_Density"] = np.log(df["Density"].clip(lower=1.0))
CONFOUNDERS_FINAL = [c if c != "Density" else "log_Density" for c in CONFOUNDERS]

causal_model = CausalPricingModel(
    outcome="ClaimNb",
    outcome_type="poisson",
    treatment=ContinuousTreatment(
        column="BonusMalus",
        standardise=False,   # keep on natural scale (BM units) for interpretability
    ),
    confounders=CONFOUNDERS_FINAL,
    exposure_col="Exposure",
    cv_folds=5,
    random_state=42,
)

print("Fitting DML model on freMTPL2 (677k rows, 5-fold CV)...")
print("Expected runtime: 10–25 minutes on Databricks serverless.")
print()

t0 = time.perf_counter()
causal_model.fit(df)
dml_fit_time = time.perf_counter() - t0

print(f"\nDML fit time: {dml_fit_time:.1f}s ({dml_fit_time / 60:.1f} min)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Results

# COMMAND ----------

ate = causal_model.average_treatment_effect()
print(ate)

# COMMAND ----------

dml_estimate = ate.estimate
dml_ci_lo    = ate.ci_lower
dml_ci_hi    = ate.ci_upper
dml_se       = ate.std_error
dml_pvalue   = ate.p_value

# Relative effects (per 10 BM units)
def relative_effect_per_10(coef: float) -> float:
    return (np.exp(coef * 10) - 1) * 100

print("=" * 70)
print("PRIMARY COMPARISON: Naive GLM vs DML on freMTPL2")
print("=" * 70)
print()
print(f"{'Metric':<50} {'GLM-simple':>12} {'GLM-full':>12} {'DML':>12}")
print("-" * 88)
print(f"{'BonusMalus coefficient':<50} {bm_simple:>12.6f} {bm_full:>12.6f} {dml_estimate:>12.6f}")
print(f"{'95% CI lower':<50} {bm_simple_lo:>12.6f} {bm_full_lo:>12.6f} {dml_ci_lo:>12.6f}")
print(f"{'95% CI upper':<50} {bm_simple_hi:>12.6f} {bm_full_hi:>12.6f} {dml_ci_hi:>12.6f}")
print(f"{'Relative effect per +10 BM units':<50} "
      f"{relative_effect_per_10(bm_simple):>11.2f}% "
      f"{relative_effect_per_10(bm_full):>11.2f}% "
      f"{relative_effect_per_10(dml_estimate):>11.2f}%")
print(f"{'Fit time (s)':<50} {glm_simple_time:>12.1f} {glm_full_time:>12.1f} {dml_fit_time:>12.1f}")
print()

# Bias decomposition (DML vs GLM-full, which has the richest confounder specification)
bias_dml_vs_full = bm_full - dml_estimate
print(f"GLM-full vs DML bias: {bias_dml_vs_full:.6f}  "
      f"(relative: {bias_dml_vs_full / abs(dml_estimate) * 100:.1f}% of DML estimate)")
print()
print("Interpretation:")
print(f"  GLM-simple: each +10 BM units associates with "
      f"{relative_effect_per_10(bm_simple):+.2f}% frequency change (naive, confounded)")
print(f"  GLM-full:   each +10 BM units associates with "
      f"{relative_effect_per_10(bm_full):+.2f}% frequency change (richer controls, still confounded)")
print(f"  DML:        each +10 BM units causes (estimated) "
      f"{relative_effect_per_10(dml_estimate):+.2f}% frequency change (after partialling confounders)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Nuisance Model Diagnostics
# MAGIC
# MAGIC How well did the CatBoost nuisance models predict BonusMalus and ClaimNb
# MAGIC from the observed confounders? High R² on the BonusMalus model means the
# MAGIC confounders explain a lot of BonusMalus variation — which is good for DML
# MAGIC (more variation to partial out). High R² on the outcome model means the
# MAGIC confounders are strong predictors of frequency — which reduces noise in
# MAGIC the residual regression.

# COMMAND ----------

nuisance_summary = diagnostics.nuisance_model_summary(causal_model)
print("Nuisance model performance (out-of-fold R²):")
for k, v in nuisance_summary.items():
    print(f"  {k}: {v}")

print()
print("Interpretation:")
print("  treatment_r2: fraction of BonusMalus variance explained by observed rating factors.")
print("    High values mean confounders strongly predict treatment — more power to DML.")
print("    Low values mean BonusMalus is nearly exogenous conditional on confounders.")
print()
print("  outcome_r2: fraction of claim count variance explained by observed rating factors.")
print("    freMTPL2 has ~10-15% outcome R² for well-specified Poisson GLMs, which is")
print("    expected for insurance claim frequency data (high overdispersion).")

# COMMAND ----------

print("\nTreatment (BonusMalus) overlap statistics:")
overlap = causal_model.treatment_overlap_stats()
for k, v in overlap.items():
    print(f"  {k}: {v}")

print()
print("  BonusMalus is continuous so 'overlap' here refers to the distribution of")
print("  treatment residuals — we want these to have meaningful spread across the range.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Sensitivity Analysis
# MAGIC
# MAGIC The Rosenbaum Γ bounds ask: how large would an unobserved confounder need to
# MAGIC be to overturn our conclusion?
# MAGIC
# MAGIC In this dataset, we know there ARE unobserved confounders (annual mileage,
# MAGIC occupation, driving behaviour). The question is whether the DML estimate is
# MAGIC robust enough that these unobserved factors would need to be implausibly strong
# MAGIC to reverse the sign or significance.
# MAGIC
# MAGIC Γ = 1.5 means the unobserved confounder would need to change the odds of
# MAGIC observing a given BonusMalus level by 50% for two otherwise-identical policyholders.
# MAGIC Given that BonusMalus is largely mechanically determined by past claims (which
# MAGIC we do not control for directly), a Γ around 1.5–2.0 is plausible. We want
# MAGIC to see whether the conclusion survives to at least Γ = 1.5.

# COMMAND ----------

sensitivity = diagnostics.sensitivity_analysis(
    ate=ate.estimate,
    se=ate.std_error,
    gamma_values=[1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],
)

print("Sensitivity analysis (Rosenbaum bounds):")
print()
print(sensitivity[["gamma", "bound_lower", "bound_upper", "ci_lower", "ci_upper",
                   "conclusion_holds", "p_value_worst_case"]].to_string(index=False))
print()

gamma_threshold = sensitivity[~sensitivity["conclusion_holds"]]["gamma"].min()
if pd.isna(gamma_threshold):
    print(f"Conclusion holds across all Gamma values tested.")
else:
    print(f"Conclusion first fails at Gamma = {gamma_threshold:.2f}.")
    print(f"  This means an unobserved confounder would need to change treatment")
    print(f"  assignment odds by {gamma_threshold:.2f}x to overturn the result.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Heterogeneous Effects by Driver Age Band
# MAGIC
# MAGIC Does the causal effect of BonusMalus on claim frequency differ by driver age?
# MAGIC Young drivers start with BonusMalus=100 and accumulate a penalty schedule
# MAGIC faster if they have early accidents. Older drivers have had more time to build
# MAGIC up a bonus position. The dynamic may differ — a high BonusMalus might signal
# MAGIC different underlying risk for a 25-year-old vs a 55-year-old.
# MAGIC
# MAGIC Note: CATE by segment fits a separate DML model per segment. With 677k total rows,
# MAGIC most age bands will have 80k+ observations — well above the minimum for stable
# MAGIC DML estimation. This is the main advantage of a large dataset like freMTPL2
# MAGIC over the synthetic 20k-row benchmarks.

# COMMAND ----------

df["age_band"] = pd.cut(
    df["DrivAge"],
    bins=[17, 25, 35, 45, 55, 65, 100],
    labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"],
).astype(str)

print("Driver age band sizes:")
print(df["age_band"].value_counts().sort_index().to_string())
print()

t0 = time.perf_counter()
cate_results = causal_model.cate_by_segment(
    df=df,
    segment_col="age_band",
    min_segment_size=5_000,  # lower threshold since we have large n per segment
)
cate_fit_time = time.perf_counter() - t0

print(f"CATE estimation time: {cate_fit_time:.1f}s")
print()
print("Causal effect of BonusMalus by driver age band:")
print(cate_results[["segment", "n_obs", "cate_estimate", "ci_lower", "ci_upper",
                    "std_error", "p_value", "status"]].to_string(index=False))
print()
print(f"Population ATE for reference: {dml_estimate:.6f}")
print()

# Compute relative effects per 10 BM units for each segment
cate_ok = cate_results[cate_results["status"] == "ok"].copy()
cate_ok["rel_effect_per_10bm"] = cate_ok["cate_estimate"].apply(relative_effect_per_10)
print("Relative frequency change per +10 BM units, by age band:")
for _, row in cate_ok.iterrows():
    print(f"  {row['segment']:>8}: {row['rel_effect_per_10bm']:+.2f}%  "
          f"(n={int(row['n_obs']):,})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Comparison: Naive Correlation vs Causal Estimate
# MAGIC
# MAGIC This is the key question: does controlling for confounders change the estimated
# MAGIC relationship between BonusMalus and claim frequency, and by how much?
# MAGIC
# MAGIC The naive estimate (raw correlation or simple GLM) includes the systematic
# MAGIC relationship between BonusMalus and the unobserved risk factors that drove
# MAGIC past claims — and therefore drove the BonusMalus score to where it is. The
# MAGIC DML estimate strips out the component of BonusMalus variation that is explained
# MAGIC by the current-period observed rating factors.

# COMMAND ----------

# Raw correlation: mean claim frequency by BonusMalus decile
bm_decile = pd.qcut(df["BonusMalus"], 10, labels=False)
raw_by_decile = (
    df.groupby(bm_decile, observed=True)
      .apply(lambda g: pd.Series({
          "freq": g["ClaimNb"].sum() / g["Exposure"].sum(),
          "bm_mean": g["BonusMalus"].mean(),
          "n": len(g),
      }))
      .reset_index(drop=True)
)

print("Raw claim frequency by BonusMalus decile (naive):")
print(raw_by_decile.to_string(index=False))

# Compare magnitude of raw gradient to DML estimate
raw_slope_approx = (
    (raw_by_decile["freq"].iloc[-1] - raw_by_decile["freq"].iloc[0]) /
    (raw_by_decile["bm_mean"].iloc[-1] - raw_by_decile["bm_mean"].iloc[0])
)
print(f"\nApproximate raw frequency slope (per BM unit, linear): {raw_slope_approx:.6f}")
print(f"GLM-full coefficient (log-linear):                      {bm_full:.6f}")
print(f"DML coefficient (log-linear, causal):                   {dml_estimate:.6f}")
print()

ratio = abs(bm_full) / abs(dml_estimate) if dml_estimate != 0 else float("nan")
print(f"GLM / DML ratio: {ratio:.2f}x  "
      f"(if >1, GLM overstates the causal effect by this factor)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Visualisation

# COMMAND ----------

fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# ── Plot 1: Method comparison (coefficient and CI) ───────────────────────────
labels    = ["GLM-simple", "GLM-full", "DML\n(insurance-causal)"]
estimates = [bm_simple, bm_full, dml_estimate]
ci_lowers = [bm_simple_lo, bm_full_lo, dml_ci_lo]
ci_uppers = [bm_simple_hi, bm_full_hi, dml_ci_hi]
colours   = ["steelblue", "cornflowerblue", "tomato"]

x_pos = np.arange(len(labels))
for i, (est, lo, hi, c) in enumerate(zip(estimates, ci_lowers, ci_uppers, colours)):
    ax1.errorbar(
        i, est,
        yerr=[[est - lo], [hi - est]],
        fmt="o", markersize=10, color=c, capsize=7, capthick=2, linewidth=2,
    )
    ax1.annotate(
        f"{est:.5f}",
        (i, est),
        textcoords="offset points", xytext=(18, 0),
        ha="left", fontsize=9, color=c, fontweight="bold",
    )

ax1.axhline(0, color="black", linewidth=1.5, linestyle=":", alpha=0.5)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(labels, fontsize=10)
ax1.set_ylabel("BonusMalus coefficient (log-linear scale)")
ax1.set_title(
    "BonusMalus Effect Estimates with 95% CI\n"
    f"(per BM unit; relative: +10 BM ~ "
    f"GLM: {relative_effect_per_10(bm_full):+.1f}%, "
    f"DML: {relative_effect_per_10(dml_estimate):+.1f}%)"
)
ax1.grid(True, alpha=0.3, axis="y")

# ── Plot 2: Raw frequency vs BonusMalus (with DML line) ──────────────────────
bm_ventile_plot = pd.qcut(df["BonusMalus"], 30, labels=False)
freq_v = (
    df.groupby(bm_ventile_plot, observed=True)
      .apply(lambda g: g["ClaimNb"].sum() / g["Exposure"].sum())
)
bm_v = df.groupby(bm_ventile_plot, observed=True)["BonusMalus"].mean()

ax2.scatter(bm_v.values, freq_v.values, color="steelblue", alpha=0.7, s=50,
            label="Observed frequency (30 ventiles)")

# DML implied line (from mean BM, exp(coef * delta_BM))
bm_grid = np.linspace(df["BonusMalus"].min(), df["BonusMalus"].max(), 100)
mean_bm = df["BonusMalus"].mean()
mean_freq = df["ClaimNb"].sum() / df["Exposure"].sum()
dml_line = mean_freq * np.exp(dml_estimate * (bm_grid - mean_bm))
glm_line = mean_freq * np.exp(bm_full * (bm_grid - mean_bm))

ax2.plot(bm_grid, dml_line, "r-", linewidth=2.5, label=f"DML causal estimate ({dml_estimate:.5f}/BM)")
ax2.plot(bm_grid, glm_line, "b--", linewidth=1.5, alpha=0.6, label=f"GLM-full estimate ({bm_full:.5f}/BM)")
ax2.set_xlabel("BonusMalus score")
ax2.set_ylabel("Claim frequency (per policy-year)")
ax2.set_title("Observed Frequency vs DML Causal Estimate\n(DML line uses population mean as anchor)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# ── Plot 3: CATE by driver age band ──────────────────────────────────────────
if len(cate_ok) > 0:
    x_cate = np.arange(len(cate_ok))
    ax3.bar(x_cate, cate_ok["cate_estimate"].values * 10, color="tomato", alpha=0.75)
    ax3.errorbar(
        x_cate, cate_ok["cate_estimate"].values * 10,
        yerr=[(cate_ok["cate_estimate"].values - cate_ok["ci_lower"].values) * 10,
              (cate_ok["ci_upper"].values - cate_ok["cate_estimate"].values) * 10],
        fmt="none", color="black", capsize=5, linewidth=1.5
    )
    ax3.axhline(dml_estimate * 10, color="black", linewidth=2, linestyle="--",
                label=f"Population ATE ({dml_estimate * 10:.5f} per 10 BM)")
    ax3.set_xticks(x_cate)
    ax3.set_xticklabels(cate_ok["segment"].values, fontsize=9)
    ax3.set_xlabel("Driver age band")
    ax3.set_ylabel("CATE estimate (per 10 BM units)")
    ax3.set_title("Causal Effect of BonusMalus by Driver Age Band\n"
                  "(DML — 677k rows enables reliable per-segment estimation)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis="y")

# ── Plot 4: Sensitivity analysis ─────────────────────────────────────────────
sens   = sensitivity
gammas = sens["gamma"].values
ax4.fill_between(gammas, sens["ci_lower"].values, sens["ci_upper"].values,
                 alpha=0.20, color="tomato", label="Worst-case 95% CI")
ax4.fill_between(gammas, sens["bound_lower"].values, sens["bound_upper"].values,
                 alpha=0.40, color="tomato", label="Rosenbaum bounds")
ax4.plot(gammas, sens["bound_lower"].values, "r-", linewidth=1.5)
ax4.plot(gammas, sens["bound_upper"].values, "r-", linewidth=1.5)
ax4.plot(gammas, [dml_estimate] * len(gammas), "k--", linewidth=1.5,
         label=f"DML estimate ({dml_estimate:.5f})")
ax4.axhline(0, color="navy", linewidth=1.5, linestyle=":", label="Zero (no effect)")
ax4.set_xlabel("Rosenbaum Γ")
ax4.set_ylabel("BonusMalus effect")
ax4.set_title("Sensitivity to Unobserved Confounding\n"
              "(how strong must unobserved confounders be to flip the conclusion?)")
ax4.legend(fontsize=8, loc="upper right")
ax4.grid(True, alpha=0.3)

plt.suptitle(
    "freMTPL2 Benchmark — DML vs Naive GLM (n=677k)\n"
    "Causal effect of BonusMalus on claim frequency",
    fontsize=12, fontweight="bold"
)
plt.savefig("/tmp/fremtpl2_benchmark.png", dpi=120, bbox_inches="tight")
plt.show()
print("Plot saved to /tmp/fremtpl2_benchmark.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Confounding Bias Report

# COMMAND ----------

print("Confounding bias report (library diagnostic — DML vs GLM-full):")
print()
bias_report = causal_model.confounding_bias_report(
    naive_coefficient=bm_full,
)
print(bias_report.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Summary and Interpretation
# MAGIC
# MAGIC ### What did we find?
# MAGIC
# MAGIC The table below summarises the three estimates. Because freMTPL2 has no known
# MAGIC ground truth (unlike the synthetic benchmarks), we cannot measure bias directly.
# MAGIC Instead we interpret the pattern of estimates.

# COMMAND ----------

print("=" * 75)
print("SUMMARY: BonusMalus Causal Effect on Claim Frequency — freMTPL2")
print("=" * 75)
print()
print(f"Dataset: freMTPL2, n={len(df):,}, total claims={df['ClaimNb'].sum():,.0f}")
print(f"         total exposure={df['Exposure'].sum():,.0f} policy-years")
print()
print(f"{'Method':<35} {'Coef/BM unit':>14} {'Rel. per +10 BM':>18} {'95% CI':>24}")
print("-" * 93)
print(f"{'GLM-simple (no categoricals)':<35} {bm_simple:>14.6f} "
      f"{relative_effect_per_10(bm_simple):>+16.2f}%  "
      f"({bm_simple_lo:.6f}, {bm_simple_hi:.6f})")
print(f"{'GLM-full (all confounders)':<35} {bm_full:>14.6f} "
      f"{relative_effect_per_10(bm_full):>+16.2f}%  "
      f"({bm_full_lo:.6f}, {bm_full_hi:.6f})")
print(f"{'DML (insurance-causal)':<35} {dml_estimate:>14.6f} "
      f"{relative_effect_per_10(dml_estimate):>+16.2f}%  "
      f"({dml_ci_lo:.6f}, {dml_ci_hi:.6f})")
print()

glm_dml_ratio = abs(bm_full) / abs(dml_estimate) if dml_estimate != 0 else float("nan")
print(f"GLM-full / DML ratio: {glm_dml_ratio:.2f}x")
print(f"  If >1: GLM overstates the causal magnitude. The confounders not fully captured")
print(f"  by linear GLM terms are inflating the estimated BonusMalus-frequency relationship.")
print(f"  If <1: GLM understates — unlikely in this context.")
print()

gamma_robust = sensitivity[sensitivity["conclusion_holds"]]["gamma"].max()
print(f"Sensitivity: conclusion robust to Gamma = {gamma_robust:.2f}")
print(f"  (unobserved confounder would need to change treatment odds by {gamma_robust:.2f}x)")
print()
print(f"Fit times:")
print(f"  GLM-simple: {glm_simple_time:.1f}s")
print(f"  GLM-full:   {glm_full_time:.1f}s")
print(f"  DML:        {dml_fit_time:.1f}s ({dml_fit_time / 60:.1f} min)")
print()
print("=" * 75)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Verdict
# MAGIC
# MAGIC ### The causal question on freMTPL2
# MAGIC
# MAGIC **What we estimated:** the causal effect of BonusMalus score on current-year
# MAGIC claim frequency, after controlling for the observed rating factors (driver age,
# MAGIC vehicle age, vehicle power, brand, fuel type, area, region, density).
# MAGIC
# MAGIC **The confounding mechanism:** BonusMalus is not randomly assigned. It
# MAGIC accumulates based on past claims. A policyholder with high BonusMalus has
# MAGIC had more past claims, which is itself correlated with the unobserved latent
# MAGIC risk drivers (annual mileage, occupation, driving style). Even after controlling
# MAGIC for the observed rating factors, part of the BonusMalus-frequency correlation
# MAGIC reflects this shared underlying risk rather than a direct causal effect of the
# MAGIC tariff penalty on driving behaviour.
# MAGIC
# MAGIC **The DML estimate** strips out the component of BonusMalus variation that is
# MAGIC explained by the observed covariates, isolating the residual variation. This
# MAGIC residual correlation is closer to the true causal effect — but is still
# MAGIC confounded by the unobserved factors.
# MAGIC
# MAGIC **Practical implication for pricing:** If the GLM estimate substantially exceeds
# MAGIC the DML estimate, the insurer's Poisson GLM pricing model overstates the
# MAGIC causal role of BonusMalus in determining future claim risk. This matters if the
# MAGIC BonusMalus score is being used as a prospective pricing variable (not just as
# MAGIC a historical summary) — the GLM would overweight it relative to its actual
# MAGIC causal effect.
# MAGIC
# MAGIC **Caveat:** The freMTPL2 dataset does not contain annual mileage or driving
# MAGIC behaviour data. These are the most plausible residual confounders. The
# MAGIC sensitivity analysis (Γ threshold) tells us how strong these unobserved
# MAGIC confounders would need to be to materially change the conclusion.
# MAGIC
# MAGIC **This notebook is a quasi-experimental benchmark.** Unlike the synthetic
# MAGIC `benchmark.py` (which has a known ground truth), we cannot measure bias here —
# MAGIC only the gap between naive and causal estimates. The freMTPL2 benchmark is
# MAGIC valuable for demonstrating the method on a widely-used real dataset, for
# MAGIC validating that the library runs at scale (677k rows), and for showing that the
# MAGIC DML estimate is plausible given what we know about the data-generating process.
# MAGIC
# MAGIC **Large-dataset advantage:** With 677k rows, segment-level CATE estimates are
# MAGIC statistically reliable at a granularity that would be impossible on typical UK
# MAGIC small-book data. The per-age-band CATEs in Section 9 illustrate heterogeneity
# MAGIC that would be noise-dominated at n=20k.

# COMMAND ----------

# Final summary line for copy-paste into reports
print("\nOne-line result for reporting:")
print(f"  freMTPL2 (n={len(df):,}): DML estimates BonusMalus causal effect at "
      f"{dml_estimate:.5f}/BM unit (95% CI: {dml_ci_lo:.5f}–{dml_ci_hi:.5f}), "
      f"vs GLM-full {bm_full:.5f}/BM unit "
      f"(GLM/DML ratio {glm_dml_ratio:.2f}x, robust to Gamma={gamma_robust:.1f}).")
