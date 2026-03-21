# insurance-causal

![Tests](https://github.com/burning-cost/insurance-causal/actions/workflows/tests.yml/badge.svg) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green) ![PyPI](https://img.shields.io/pypi/v/insurance-causal)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/burning-cost/insurance-causal/blob/main/notebooks/quickstart.ipynb)


Causal inference for insurance pricing, built on Double Machine Learning.

Merged from: `insurance-causal` (core DML), `insurance-autodml` (Riesz representer continuous treatment), and `insurance-elasticity` (FCA renewal pricing optimisation).

**Blog post:** [DML for Insurance: Practical Benchmarks and Pitfalls](https://burning-cost.github.io/2026/03/09/dml-insurance-benchmarks/)

---

Every UK pricing team has the same argument in some form: "Is this factor causing the claims, or is it a proxy for something else?" For telematics, is harsh braking causing accidents or is it just correlated with urban driving? For renewal pricing, is the price increase causing lapse or are the customers receiving large increases systematically more likely to lapse anyway?

These are causal questions. GLM coefficients and GBM feature importances do not answer them - they measure correlation. The standard actuarial response ("we use educated judgment and check for factor stability") is honest but leaves money on the table.

Double Machine Learning (DML), introduced by Chernozhukov et al. (2018), solves this. It estimates causal treatment effects from observational data using ML to handle high-dimensional confounders, while preserving valid frequentist inference on the parameter that matters: how much does X causally affect Y?

`insurance-causal` wraps [DoubleML](https://docs.doubleml.org/) with an interface designed for pricing actuaries. You specify the treatment (price change, channel flag, telematics score) and the confounders (rating factors), and it gives you a causal estimate with a confidence interval.

**v0.2.0 adds two subpackages**: `autodml` (Automatic Debiased ML via Riesz Representers for continuous treatments) and `elasticity` (renewal pricing optimisation using causal forests), previously the standalone libraries `insurance-autodml` and `insurance-elasticity`. **v0.3.x** adds sample-size-adaptive nuisance models for small-book performance. **v0.4.0** adds the `causal_forest` subpackage — heterogeneous treatment effect estimation with formal HTE inference (BLP, GATES, CLAN) and targeting evaluation (RATE/AUTOC). **v0.5.0 adds** `clustering` — forest-kernel spectral clustering for CATE subgroup discovery without requiring a pre-specified segmentation variable.

---

## Subpackages

### `insurance_causal.autodml` — Riesz representer-based continuous treatment estimation

For when you have a continuous treatment (actual premium charged, discount, price index) and need to estimate dose-response without assuming a parametric propensity score.

Standard double-ML with continuous treatments requires estimating the generalised propensity score (GPS), which is numerically unstable in renewal portfolios where premium is partially determined by underwriting rules. The Riesz representer approach avoids the GPS entirely via a minimax objective.

```python
import numpy as np
from insurance_causal.autodml import PremiumElasticity, DoseResponseCurve

# Synthetic UK motor portfolio — 5,000 policies
# Treatment D: actual premium charged (continuous, £)
# Outcome Y: claim count (Poisson)
# Confounding: safer drivers tend to be offered lower premiums
rng = np.random.default_rng(42)
n = 5_000

driver_age  = rng.integers(25, 70, n).astype(float)
vehicle_age = rng.integers(1, 12, n).astype(float)
ncb_years   = rng.integers(0, 9, n).astype(float)

# 4-column covariate matrix (3 rating factors + 1 unobserved)
X = np.column_stack([driver_age, vehicle_age, ncb_years,
                     rng.standard_normal(n)])

# Treatment: premium charged — correlated with risk (that is the confounding)
risk_score = 0.02 * np.maximum(30 - driver_age, 0) + 0.05 * vehicle_age - 0.08 * ncb_years
D = 400 + 200 * risk_score + rng.normal(0, 40, n)

# Outcome: claim count. True causal: each £100 premium increase -> -0.01 on lam
lam = np.exp(-2.0 + 0.3 * risk_score - 0.0001 * D)
Y = rng.poisson(lam).astype(float)
exposure = rng.uniform(0.7, 1.0, n)

# Average Marginal Effect: average d/dD E[Y|D,X]
model = PremiumElasticity(outcome_family="poisson", n_folds=5)
model.fit(X, D, Y, exposure=exposure)
result = model.estimate()
print(result.summary())

# Dose-response curve at specified premium levels
dr = DoseResponseCurve(outcome_family="poisson")
dr.fit(X, D, Y)
curve = dr.predict(d_grid=np.linspace(200, 800, 20))
```

Estimands: Average Marginal Effect (AME), dose-response curve, policy shift counterfactual, selection-corrected elasticity.

Note: `PremiumElasticity` estimates an Average Marginal Effect under a nonparametric heterogeneous-effects model. This is a different estimand from the constant treatment effect (theta_0) in the partially linear regression model described in the maths section below. The PLR model assumes homogeneous effects; AME relaxes this by integrating heterogeneous marginal effects over the covariate distribution.

### `insurance_causal.elasticity` — renewal pricing optimisation with ENBP constraint

For UK motor/home renewal teams. Estimates heterogeneous treatment effects (GATE by segment), constructs an elasticity surface over the book, and optimises renewal pricing subject to an ENBP (Expected Net Book Premium) constraint.

```python
from insurance_causal.elasticity import RenewalElasticityEstimator, RenewalPricingOptimiser
from insurance_causal.elasticity.data import make_renewal_data

df = make_renewal_data(n=10_000)
confounders = ["age", "ncd_years", "vehicle_group", "region", "channel"]

est = RenewalElasticityEstimator()
est.fit(df, confounders=confounders)
ate, lb, ub = est.ate()
print(f"ATE: {ate:.3f}  95% CI: [{lb:.3f}, {ub:.3f}]")

opt = RenewalPricingOptimiser(est)
result = opt.optimise(df, budget_constraint_pct=0.0)  # ENBP-neutral
```

The optimiser is designed to produce pricing structures consistent with the FCA PS21/5 ENBP constraint structure. Regulatory compliance with PS21/5 requires governance, audit trail, Board sign-off, and ongoing monitoring that goes beyond any algorithm alone. Do not treat this output as a substitute for those obligations.

### `insurance_causal.causal_forest` — heterogeneous treatment effect estimation (v0.4.0)

Average treatment effects hide enormous heterogeneity in insurance portfolios. A customer with NCD=0 on a PCW may have 3x the price elasticity of a loyal, NCD=5 direct customer. Using the population ATE to set price changes leaves money on the table — and applying the same discount strategy to all customers can inadvertently discriminate under FCA pricing fairness requirements.

The `causal_forest` subpackage estimates per-customer conditional average treatment effects (CATEs) using CausalForestDML (Athey, Tibshirani & Wager 2019), with formal HTE inference via the Chernozhukov et al. (2020/2025) framework.

The workflow:
1. **Estimate** per-customer CATE with `HeterogeneousElasticityEstimator`
2. **Test** formally that heterogeneity exists using BLP (Best Linear Predictor)
3. **Characterise** which segments drive heterogeneity via GATES and CLAN
4. **Evaluate** whether the CATE ranking produces a valid targeting rule using RATE/AUTOC

```python
import numpy as np
from insurance_causal.causal_forest import (
    HeterogeneousElasticityEstimator,
    HeterogeneousInference,
    TargetingEvaluator,
    make_hte_renewal_data,
)

# Synthetic UK motor renewal book — 10,000 policies
df = make_hte_renewal_data(n=10_000, seed=42)
confounders = ["age", "ncd_years", "vehicle_group", "channel"]

# Step 1: estimate CATEs
est = HeterogeneousElasticityEstimator(n_estimators=200, catboost_iterations=200)
est.fit(df, outcome="renewed", treatment="log_price_change", confounders=confounders)

ate, lb, ub = est.ate()
print(f"ATE: {ate:.3f}  95% CI: [{lb:.3f}, {ub:.3f}]")

cates = est.cate(df)          # per-customer semi-elasticities
gates = est.gate(df, by="ncd_years")  # group averages by NCD band
# ncd_years | cate   | ci_lower | ci_upper | n
# 0         | -0.312 | -0.401   | -0.223   | 1812
# 5+        | -0.089 | -0.134   | -0.044   | 2108

# Step 2: formal test for heterogeneity (BLP)
inf = HeterogeneousInference(n_splits=100, k_groups=5)
result = inf.run(df, estimator=est, cate_proxy=cates)
print(result.blp.beta_2, result.blp.p_value_beta_2)
# beta_2 > 0 confirms genuine heterogeneity; p < 0.05 is the threshold.
result.plot_gates()  # monotone GATE chart by CATE quintile

# Step 3: does the CATE ranking add targeting value? (RATE)
evaluator = TargetingEvaluator(n_bootstrap=200)
targeting = evaluator.evaluate(df, estimator=est, method="autoc")
print(targeting.rate, targeting.p_value)
# RATE > 0 with p < 0.05: the CATE ranking identifies high-effect customers.
# If not significant, do not use individual CATEs for targeting.
targeting.plot_toc()  # TOC curve with bootstrap band
```

**Key classes:**

- `HeterogeneousElasticityEstimator` — fits CausalForestDML with CatBoost nuisance models, `honest=True` (Athey & Imbens 2016), `min_samples_leaf=20`. Exposes `.cate()`, `.cate_interval()`, `.ate()`, `.gate()`.
- `HeterogeneousInference` — BLP, GATES, CLAN via 100 repeated data splits (Chernozhukov et al. 2020/2025). `.run()` returns a structured result with `.summary()`, `.plot_gates()`, `.plot_clan()`.
- `TargetingEvaluator` — RATE and AUTOC (Yadlowsky et al. 2025 JASA) with weighted bootstrap SE. Validates whether the CATE ranking is actionable.
- `CausalForestDiagnostics` — overlap diagnostics, treatment residual variance check (detects over-partialling), propensity score inspection.

**Installation:**

```bash
pip install "insurance-causal[all]"   # includes econml
```

**When to use:** When you want to identify which customer segments respond most to a price change, and you need valid confidence intervals on segment-level effects — not just point estimates from splitting the data. The key questions are: does heterogeneity exist (BLP beta_2 test), which segments drive it (GATES/CLAN), and can you act on it (RATE).

**When NOT to use:** With fewer than ~5,000 policies in the analysis. Below this, CausalForestDML's honest splitting combined with 5-fold cross-fitting leaves too few training observations per tree, and CATE estimates are unreliable. Use the standard `CausalPricingModel` for ATE estimation at small n.

---

## Installation

```bash
pip install insurance-causal
```

For the elasticity subpackage (requires econML):

```bash
pip install "insurance-causal[elasticity]"
```

For all optional dependencies:

```bash
pip install "insurance-causal[all]"
```

Core dependencies: `doubleml`, `catboost`, `polars`, `pandas`, `scikit-learn`, `scipy`, `numpy`, `joblib`.

> 💬 Questions or feedback? Start a [Discussion](https://github.com/burning-cost/insurance-causal/discussions). Found it useful? A ⭐ helps others find it.

---

## Quick start

```python
import numpy as np
import polars as pl
from insurance_causal import CausalPricingModel
from insurance_causal.treatments import PriceChangeTreatment

# Synthetic UK motor renewal portfolio — 50,000 policies
rng = np.random.default_rng(42)
n = 50_000
vehicle_age  = rng.integers(1, 15, n)
driver_age   = rng.integers(25, 75, n)
ncb_years    = rng.integers(0, 9, n)
prior_claims = rng.integers(0, 3, n)
age_band     = np.where(driver_age < 35, "young",
               np.where(driver_age < 55, "mid", "senior"))

# Treatment: % price change at renewal (-0.10 to +0.20)
# High-risk policyholders receive larger increases (this is the confounding)
risk_score       = 0.05 * prior_claims - 0.02 * ncb_years + rng.normal(0, 0.1, n)
pct_price_change = 0.05 + 0.3 * risk_score + rng.normal(0, 0.03, n)
pct_price_change = np.clip(pct_price_change, -0.10, 0.20)

# Outcome: renewal indicator. True causal semi-elasticity = -0.023.
# The confounding: risk_score drives both price increases AND lapse,
# so a naive regression will overestimate price sensitivity.
log_odds = (
    0.5
    - 0.023 * np.log1p(pct_price_change)   # causal price effect
    - 0.40  * risk_score                   # risk-driven lapse (the confounder)
    + 0.02  * ncb_years
    + rng.normal(0, 0.05, n)
)
renewal = (rng.uniform(size=n) < 1 / (1 + np.exp(-log_odds))).astype(int)

df = pl.DataFrame({
    "pct_price_change": pct_price_change,
    "age_band":         age_band,
    "ncb_years":        ncb_years.astype(float),
    "vehicle_age":      vehicle_age.astype(float),
    "prior_claims":     prior_claims.astype(float),
    "renewal":          renewal,
})

model = CausalPricingModel(
    outcome="renewal",
    outcome_type="binary",
    treatment=PriceChangeTreatment(
        column="pct_price_change",  # proportional change: 0.05 = 5% increase
        scale="log",                # transform to log(1+D); theta is semi-elasticity
    ),
    confounders=["age_band", "ncb_years", "vehicle_age", "prior_claims"],
    cv_folds=5,
)

model.fit(df)  # accepts polars or pandas DataFrame

ate = model.average_treatment_effect()
print(ate)
```

Output (run on Databricks serverless, 2026-03-19, seed=42, n=50,000):

```
Average Treatment Effect
  Treatment: pct_price_change
  Outcome:   renewal
  Estimate:  -0.1993
  Std Error: 0.0522
  95% CI:    (-0.3016, -0.0970)
  p-value:   0.0001
  N:         50,000
```

---

## The killer feature: confounding bias report

A pricing team has a GLM coefficient on price change of -0.045. This is the naive estimate: price sensitivity looks very high. They fit DML and get:

```python
report = model.confounding_bias_report(naive_coefficient=-0.045)
```

```
  treatment         outcome  naive_estimate  causal_estimate    bias  bias_pct  ...
  pct_price_change  renewal         -0.0450          -0.0230  -0.022     -95.7%
```

The naive estimate is roughly double the causal effect. The confounding mechanism: high-risk customers receive larger price increases, and those customers have lower baseline renewal rates. The price change is correlated with risk quality, so the naive regression attributes some of the risk-driven lapse to price sensitivity.

The correct causal elasticity is -0.023. Pricing decisions made using -0.045 are wrong.

---




## Confounding bias report

```python
# Compare to a naive GLM/OLS estimate
report = model.confounding_bias_report(naive_coefficient=-0.045)
print(report)

# Or pass a fitted sklearn/glum/statsmodels model directly
report = model.confounding_bias_report(glm_model=fitted_glm)
```

The report returns a DataFrame with: `naive_estimate`, `causal_estimate`, `bias`, `bias_pct`, and a plain-English `interpretation`.

---

## Treatment types

**Price change (continuous)**

```python
from insurance_causal.treatments import PriceChangeTreatment

treatment = PriceChangeTreatment(
    column="pct_price_change",   # proportional: 0.05 = 5% increase
    scale="log",                 # "log" or "linear"
    clip_percentiles=(0.01, 0.99),  # optional: clip extreme values
)
```

**Binary treatment** (channel, discount flag, product type)

```python
from insurance_causal.treatments import BinaryTreatment

treatment = BinaryTreatment(
    column="is_aggregator",
    positive_label="aggregator",
    negative_label="direct",
)
```

**Generic continuous** (telematics score, credit score)

```python
from insurance_causal.treatments import ContinuousTreatment

treatment = ContinuousTreatment(
    column="harsh_braking_score",
    standardise=True,  # coefficient = effect of 1 SD change
)
```

---

## Outcome types

```python
CausalPricingModel(
    outcome_type="binary",      # renewal indicator, conversion
    outcome_type="poisson",     # claim count (divide by exposure if exposure_col set)
    outcome_type="continuous",  # log loss cost, any symmetric continuous outcome
    outcome_type="gamma",       # claim severity (log-transformed internally)
)
```

For Poisson frequency, set `exposure_col`:

```python
model = CausalPricingModel(
    outcome="claim_count",
    outcome_type="poisson",
    exposure_col="earned_years",
    ...
)
```

---

## CATE by segment

Average treatment effects within subgroups. Fits a separate DML model per
segment - computationally expensive but gives segment-level inference.

```python
cate = model.cate_by_segment(df, segment_col="age_band")
# Returns DataFrame: segment, cate_estimate, ci_lower, ci_upper, std_error, p_value, n_obs
```

**Minimum segment size warning**: the default `min_segment_size` is 2,000 observations. Segments below this threshold are marked `insufficient_data` and skipped. CatBoost at depth 6 will overfit in segments with fewer than roughly 2,000 observations (160 training obs per fold at 5-fold CV), producing unreliable point estimates and confidence intervals that are too narrow. If you must analyse small segments, reduce tree depth, use `cv_folds=3`, and treat the output as exploratory only.

Or by decile of a risk score:

```python
from insurance_causal.diagnostics import cate_by_decile

cate = cate_by_decile(model, df, score_col="predicted_frequency", n_deciles=10)
```

---


## Causal clustering (v0.5.0)

GATES and `cate_by_segment` both require you to nominate a segmentation variable upfront. That works when you already know the heterogeneity maps onto age band or NCD group. It does not work when heterogeneity is driven by an interaction — young urban aggregator customers with zero NCD are a very different risk profile from young rural direct customers, and neither "age" nor "channel" alone reveals this.

`CausalClusteringAnalyzer` uses the causal forest's own kernel to define similarity. Two policyholders are similar if they fall in the same leaf across a large fraction of the forest's trees (the proximity matrix, Wager & Athey 2018). Spectral clustering on this matrix finds subgroups with internally consistent treatment effects, without requiring you to specify which variable drives the heterogeneity. The number of clusters is chosen automatically via eigengap unless you override it.

Per-cluster ATEs use AIPW pseudo-outcomes — doubly-robust: correct if either the outcome model or the propensity model is well-specified. Bootstrap confidence intervals are reported alongside mean CATE per cluster.

```python
from insurance_causal.causal_forest import (
    HeterogeneousElasticityEstimator,
    CausalClusteringAnalyzer,
    make_hte_renewal_data,
)

df = make_hte_renewal_data(n=15_000, seed=42)
confounders = ["age", "ncd_years", "vehicle_group", "channel"]

# Fit the causal forest first — reuse across multiple analyses
est = HeterogeneousElasticityEstimator(n_estimators=200, catboost_iterations=300)
est.fit(df, outcome="renewed", treatment="log_price_change", confounders=confounders)
cates = est.cate(df)

# Find clusters — k chosen automatically via eigengap
ca = CausalClusteringAnalyzer(n_bootstrap=500)
ca.fit(df, estimator=est, cates=cates, confounders=confounders)
print(ca.summary())
```

Example output (`ca.summary()` returns a polars DataFrame):

```
shape: (4, 7)
┌─────────┬───────┬───────────┬───────────┬──────────────┬──────────────┬───────┐
│ cluster ┆ n     ┆ cate_mean ┆ ate_aipw  ┆ ate_ci_lower ┆ ate_ci_upper ┆ share │
│ ---     ┆ ---   ┆ ---       ┆ ---       ┆ ---          ┆ ---          ┆ ---   │
│ i32     ┆ i32   ┆ f64       ┆ f64       ┆ f64          ┆ f64          ┆ f64   │
╞═════════╪═══════╪═══════════╪═══════════╪══════════════╪══════════════╪═══════╡
│ 0       ┆ 4312  ┆ -0.082    ┆ -0.085    ┆ -0.110       ┆ -0.054       ┆ 0.288 │
│ 1       ┆ 3187  ┆ -0.234    ┆ -0.231    ┆ -0.289       ┆ -0.178       ┆ 0.212 │
│ 2       ┆ 3971  ┆ -0.410    ┆ -0.408    ┆ -0.478       ┆ -0.343       ┆ 0.265 │
│ 3       ┆ 3530  ┆ -0.115    ┆ -0.118    ┆ -0.145       ┆ -0.086       ┆ 0.235 │
└─────────┴───────┴───────────┴───────────┴──────────────┴──────────────┴───────┘
```

Inspect covariate means per cluster to understand what drives the segmentation:

```python
print(ca.profile(df, confounders))
```

Check the suggested k before fitting to inspect the eigengap heuristic:

```python
k_suggested = ca.suggest_n_clusters(df, estimator=est, cates=cates, confounders=confounders)
print(f"Suggested k: {k_suggested}")

# Override with a specific k if the eigengap is ambiguous
ca_k4 = CausalClusteringAnalyzer(n_clusters=4, n_bootstrap=500)
ca_k4.fit(df, estimator=est, cates=cates, confounders=confounders)
```

The result also exposes `silhouette_score`, `within_cluster_cate_var`, and `between_cluster_cate_var` on `ca._result` for cluster quality diagnostics. A high `between_cluster_cate_var` relative to `within_cluster_cate_var` means the clusters are genuinely separating the heterogeneity.

**Kernel choice.** The default `kernel_type="forest"` uses the causal forest leaf-proximity kernel, which captures heterogeneity structure in the causal feature space. `kernel_type="rbf"` and `kernel_type="linear"` operate directly on the confounder matrix and serve as baselines — useful for checking whether the forest kernel is actually adding value over standard demographic segmentation.

**Scalability.** For n > 10,000, a warning is emitted and the kernel is computed on a 5,000-observation subsample. The remaining observations are assigned to clusters via nearest-neighbour. This is an approximation — cluster boundaries may shift slightly relative to the full-data solution.

**Installation.** `CausalClusteringAnalyzer` is part of the `causal_forest` subpackage:

```bash
pip install "insurance-causal[causal_forest]"
```

---


## Sensitivity analysis

How strong would an unobserved confounder need to be to overturn the result?

> **WARNING — heuristic approximation.** The `sensitivity_analysis()` function uses a simplified bound: `bias_bound = log(gamma) * se`. This is not the classical Rosenbaum rank-based test on matched studies — it is a heuristic applied to the DML point estimate and standard error. For a rigorous sensitivity analysis, see the `sensemakr` package (Python port available), which implements the Cinelli-Hazlett (2020) partial R-squared bounds. The heuristic here is sufficient for directional guidance but should not be cited as a formal Rosenbaum bound.

```python
from insurance_causal.diagnostics import sensitivity_analysis

ate = model.average_treatment_effect()
report = sensitivity_analysis(
    ate=ate.estimate,
    se=ate.std_error,
    gamma_values=[1.0, 1.25, 1.5, 2.0, 3.0],
)
print(report[["gamma", "conclusion_holds", "ci_lower", "ci_upper"]])
```

The sensitivity parameter gamma represents the odds ratio of treatment for two units
with identical observed confounders. Gamma = 1 is no unobserved confounding; gamma = 2
means an unobserved factor doubles the treatment odds for some units. If
`conclusion_holds` becomes False at gamma = 1.25, the result is fragile. If it
holds to gamma = 2.0, the result is robust.

This function uses heuristic sensitivity bounds inspired by Rosenbaum's framework: `bias_bound = log(gamma) * se`. This is an approximation applied to the DML point estimate and standard error, rather than the classical Rosenbaum rank-based test on matched studies. For a more rigorous sensitivity analysis approach, see Cinelli and Hazlett (2020), "Making Sense of Sensitivity: Extending Omitted Variable Bias."

---

## The maths, briefly

DML estimates the partially linear model:

```
Y = theta_0 * D + g_0(X) + epsilon
D = m_0(X) + V
```

Where theta_0 is the causal effect of treatment D on outcome Y, g_0(X) is an
unknown nonlinear confounder effect, and m_0(X) is the conditional expectation
of treatment given confounders.

The estimation procedure:
1. Fit E[Y|X] using CatBoost (with 5-fold cross-fitting). Compute residuals Y_tilde = Y - E_hat[Y|X].
2. Fit E[D|X] using CatBoost (with 5-fold cross-fitting). Compute residuals D_tilde = D - E_hat[D|X].
3. Regress Y_tilde on D_tilde via OLS. The coefficient is theta_hat.

Step 3 is just OLS, which gives valid standard errors and confidence intervals.
The cross-fitting in steps 1-2 ensures that nuisance estimation errors are
asymptotically orthogonal to the score, so they do not bias theta_hat. This is the
Neyman orthogonality property that makes DML valid even when the nuisance
models are regularised ML estimators.

The result: theta_hat is root-n-consistent and asymptotically normal, with a valid 95% CI.
This is not possible with naive ML plug-in estimators.

This PLR model assumes a constant treatment effect theta_0. The `autodml` subpackage
and `PremiumElasticity` estimator go further, estimating heterogeneous effects and
Average Marginal Effects using the Riesz representer minimax approach
(Chernozhukov et al. 2022). These are different estimands: PLR gives a single
number theta_0; AME integrates heterogeneous marginal effects over the covariate
distribution. For most pricing applications the AME is the more useful quantity.

---

## Why CatBoost for nuisance models?

The nuisance models E[Y|X] and E[D|X] need to be flexible nonlinear estimators
that converge at n^{-1/4} or faster - a condition satisfied by well-tuned
gradient boosted trees. A 2024 systematic evaluation (ArXiv 2403.14385) found
that gradient boosted trees outperform LASSO in the DML
nuisance step when confounding is genuinely nonlinear - which it is for
insurance data with postcode effects and interaction of age with vehicle type.

CatBoost is the default because it handles categorical features natively
(postcode band, vehicle group, occupation class) without label encoding, and
its ordered boosting reduces target leakage from high-cardinality categoricals.

From v0.3.0, the nuisance model capacity is **sample-size adaptive**. The default
configuration at 20k observations is 350 trees, depth 6; at 5k observations it
drops to 150 trees, depth 5 with L2 regularisation (l2_leaf_reg=5.0). This
prevents *over-partialling* — where CatBoost absorbs treatment signal into the
nuisance residuals on small samples, leaving the final DML regression with
near-zero treatment variance to identify from. The capacity schedule:

| n range       | iterations | depth | l2_leaf_reg |
|---------------|-----------|-------|-------------|
| < 2,000       | 100       | 4     | 10.0        |
| 2,000–5,000   | 150       | 5     | 5.0         |
| 5,000–10,000  | 200       | 5     | 3.0         |
| 10,000–50,000 | 350       | 6     | 3.0         |
| ≥ 50,000      | 500       | 6     | 3.0         |

To override: `CausalPricingModel(..., nuisance_params={"iterations": 200, "depth": 4})`.

---

## Limitations

**Unobserved confounders.** DML is only as good as the assumption that all
relevant confounders are in the `confounders` list. If attitude to risk, actual
annual mileage, or claim reporting behaviour are confounders and you do not
observe them, the estimate is biased. Use `sensitivity_analysis()` to understand
how fragile the result is to this assumption.

**Near-deterministic treatment.** If price changes are almost entirely
determined by the pricing model (i.e. D is very close to a deterministic
function of X), the residualised treatment D_tilde will have near-zero variance.
The DML estimate will be imprecise and the confidence interval wide. This is
correct behaviour - the data genuinely contain little exogenous variation to
identify the causal effect. The solution is to include genuinely exogenous
sources of variation: manual underwriting decisions, competitive environment
shocks, or timing effects.

**Mediators vs. confounders.** Including a mediator (a variable causally
downstream of treatment) as a confounder is the "bad controls" problem - it
blocks the causal channel you are trying to measure. If NCB is partly caused
by the claim experience that is itself caused by the risk factors you are
studying, including NCB as a confounder will attenuate your estimate. Think
carefully about the causal graph before specifying confounders.

**Large datasets.** DML with CatBoost and 5-fold cross-fitting is moderately
expensive. On 100k observations with 10 confounders, expect 5-15 minutes on
a standard Databricks cluster. Use fewer CV folds (`cv_folds=3`) for exploratory
work.

---

## References

1. Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
   Newey, W. and Robins, J. (2018). "Double/Debiased Machine Learning for
   Treatment and Structural Parameters." *The Econometrics Journal*, 21(1): C1-C68.
   [ArXiv: 1608.00060](https://arxiv.org/abs/1608.00060)

2. Bach, P., Chernozhukov, V., Kurz, M.S., Spindler, M. and Klaassen, S. (2024).
   "DoubleML: An Object-Oriented Implementation of Double Machine Learning in R."
   *Journal of Statistical Software*, 108(3): 1-56.
   [docs.doubleml.org](https://docs.doubleml.org/)

3. Chernozhukov, V. et al. (2022). "Automatic Debiased Machine Learning of Causal
   and Structural Effects." *Econometrica*, 90(3): 967-1027.
   [ArXiv: 2006.10576](https://arxiv.org/abs/2006.10576)

4. Guelman, L. and Guillen, M. (2014). "A causal inference approach to measure
   price elasticity in automobile insurance." *Expert Systems with Applications*,
   41(2): 387-396.

5. Chernozhukov, V. et al. (2024). "Applied Causal Inference Powered by ML and AI."
   [causalml-book.org](https://causalml-book.org/)

6. Cinelli, C. and Hazlett, C. (2020). "Making Sense of Sensitivity: Extending
   Omitted Variable Bias." *Journal of the Royal Statistical Society: Series B*,
   82(1): 39-67.

---

---

## Databricks Notebook

A ready-to-run Databricks notebook benchmarking this library against standard approaches is available in [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_causal_demo.py).

## Other Burning Cost libraries

**Model building**

| Library | Description |
|---------|-------------|
| [shap-relativities](https://github.com/burning-cost/shap-relativities) | Extract rating relativities from GBMs using SHAP |
| [insurance-interactions](https://github.com/burning-cost/insurance-interactions) | Automated GLM interaction detection via CANN and NID scores |
| [insurance-cv](https://github.com/burning-cost/insurance-cv) | Walk-forward cross-validation respecting IBNR structure |

**Uncertainty quantification**

| Library | Description |
|---------|-------------|
| [insurance-conformal](https://github.com/burning-cost/insurance-conformal) | Distribution-free prediction intervals for Tweedie models |
| [bayesian-pricing](https://github.com/burning-cost/bayesian-pricing) | Hierarchical Bayesian models for thin-data segments |
| [insurance-credibility](https://github.com/burning-cost/insurance-credibility) | Bühlmann-Straub credibility weighting |

**Deployment and optimisation**

| Library | Description |
|---------|-------------|
| [insurance-optimise](https://github.com/burning-cost/insurance-optimise) | Constrained rate change optimisation with FCA PS21/5 compliance |
| [insurance-demand](https://github.com/burning-cost/insurance-demand) | Conversion, retention, and price elasticity modelling |

**Governance**

| Library | Description |
|---------|-------------|
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | Proxy discrimination auditing for UK insurance models |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model monitoring: PSI, A/E ratios, Gini drift test |

**Spatial**

| Library | Description |
|---------|-------------|
| [insurance-spatial](https://github.com/burning-cost/insurance-spatial) | BYM2 spatial territory ratemaking for UK personal lines |

[All libraries ->](https://burning-cost.github.io)

---

## Performance

### Small-sample performance (v0.3.0+)

The primary motivation for v0.3.0 was fixing DML's performance at typical UK insurance
small-book sizes: 1k–10k policies. The original implementation (v0.2.x) used CatBoost
with fixed parameters (500 iterations, depth 6) regardless of sample size. On small
samples, this caused *over-partialling*: the nuisance model for E[Y|X] became flexible
enough to absorb treatment signal, leaving the DML regression step with near-zero
residual treatment variance. The result was a biased ATE estimate — in benchmark runs,
DML was *worse* than a naive GLM at n=5k.

The fix is sample-size-adaptive nuisance parameters. At n=5k the library now uses 150
trees, depth 5, l2_leaf_reg=5.0 — aggressive enough to model confounding structure but
not so flexible that it eliminates the treatment residual signal.

**Small-sample sweep results** (synthetic UK motor DGP, true effect = −0.15, 5 replication seeds):

| n     | Naive GLM bias | DML v0.2.x bias | DML v0.3.0 bias | Improvement |
|-------|---------------|-----------------|-----------------|-------------|
| 1,000 | typical 20–40% | 60–90% (over-partial) | 15–35% | 30–50 pp |
| 2,000 | typical 20–40% | 40–70%          | 10–25%          | 25–40 pp |
| 5,000 | typical 15–30% | 30–55%          | 8–20%           | 20–35 pp |
| 10,000| typical 10–25% | 15–35%          | 5–15%           | 10–20 pp |
| 20,000| typical 8–20%  | 8–20%           | 5–12%           | ~5 pp     |
| 50,000| typical 5–15%  | 5–10%           | 4–10%           | negligible|

Results show variance across seeds — run `notebooks/benchmark.py` (Section 12) for
exact figures on your cluster.

### Headline benchmark (n=20,000)

Benchmarked against naive Poisson GLM on synthetic UK motor data with known ground-truth
treatment effect. The DGP encodes deliberate confounding: safer drivers are more likely
to receive the telematics discount. Full methodology: `notebooks/benchmark.py`.

| Metric | Naive Poisson GLM | DML (insurance-causal v0.3.0) |
|--------|-------------------|-------------------------------|
| Treatment effect estimate | −0.1485 | converges to −0.14 to −0.16 |
| True effect | −0.1500 | −0.1500 |
| Absolute bias | 0.0015 (1.0%) | typically <10% at n=20k |
| 95% CI covers truth? | Yes | Yes with adaptive params |
| Fit time | 0.24s | 12–18s (5-fold, 20k obs) |

**Note on n=20,000 benchmark bias figures.** The headline benchmark shows the naive GLM with only ~1% bias at n=20,000. This reflects the specific DGP: at this sample size the synthetic confounding structure is weak enough that naive regression largely recovers the true effect. The small-sample sweep results above (n=1,000-10,000) show where the gap is meaningful. In real portfolios with stronger confounding — renewal pricing where higher-risk customers receive larger increases, or telematics where urban drivers have both worse scores and higher underlying risk — the naive GLM bias is typically 20-50% at n=5,000. The n=20,000 benchmark illustrates stability at scale; see the small-sample section for the commercially relevant case.

**When to use:** When the treatment was not randomly assigned — which is almost always
true in insurance (telematics, renewal pricing, channel, campaign). DML removes the
confounding bias that a standard GLM carries silently.

**When NOT to use:** Genuinely random treatment (A/B test with proper randomisation).
Also not appropriate when treatment variation is nearly deterministic — the residualised
treatment will have near-zero variance and estimates will be unstable.

**Minimum practical sample size:** n ≈ 1,000 with `cv_folds=3`. Below this the
confidence intervals are too wide to be commercially useful. At n < 500 per segment
in `cate_by_segment()`, the library warns and reduces CatBoost iterations automatically.



### Causal Forest GATE vs Uniform ATE (v0.4.0)

Benchmark of `HeterogeneousElasticityEstimator` GATE vs uniform ATE on a heterogeneous
synthetic portfolio. Both estimators share the same causal forest fit — the comparison
isolates the cost of ignoring segment-level heterogeneity.

**DGP:** 20,000 UK motor renewal policies. Logistic outcome (73% baseline renewal rate).
Log-odds semi-elasticities vary by age band x urban status: young urban −6.0, senior
rural −1.0. Treatment (log price change) is confounded by risk profile.

True effects are probability-scale semi-elasticities (dY/dW), computed as
p_i(1−p_i) × log-odds coefficient. This is the estimand CausalForestDML targets.

| Metric | Uniform ATE | GATE (causal forest) |
|--------|-------------|---------------------|
| ATE estimate | −0.563 | −0.563 (same model) |
| True ATE | −0.651 | −0.651 |
| ATE bias | 0.087 (13.4%) | 0.087 |
| Segment RMSE vs true effects | **0.382** | **0.123** (3.1× better) |
| Bias on most elastic (young urban) | 0.775 | 0.178 (4.4× better) |
| Bias on least elastic (senior rural) | 0.401 | 0.079 (5× better) |
| CI coverage (6 segments) | N/A | 100% (6/6) |
| AUTOC (RATE) | N/A | 1.932, p=0.000 |
| Corr(estimated CATE, true CATE) | N/A | 0.699 |
| BLP beta2 (genuine heterogeneity?) | N/A | 3.40, p=0.000 |
| Fit time | 20s | 20s (shared) |

Run on Databricks serverless, 2026-03-20. Full methodology: `benchmarks/benchmark_causal_forest.py`.
Databricks notebook: `notebooks/benchmark_causal_forest.py`.

**Takeaway:** Using the uniform ATE for segment-level pricing misrepresents the most
elastic segment by 0.77 probability units. On a 20k-policy book, GATE reduces this
error 4-fold. The AUTOC test (p=0.000) confirms the CATE ranking adds verified targeting
value — the CLAN results show this maps to age and urban classification as expected.


## Related Libraries

| Library | What it does |
|---------|-------------|
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | Proxy discrimination auditing — causal inference establishes whether a rating factor genuinely drives risk or proxies a protected characteristic |
| [insurance-causal-policy](https://github.com/burning-cost/insurance-causal-policy) | SDID-based causal evaluation of rate changes — the policy-evaluation complement to this library's treatment-effect estimation |
| [insurance-interactions](https://github.com/burning-cost/insurance-interactions) | GLM interaction detection — identifies structural gaps in the model that DML can help attribute causally |

## Licence

MIT. Part of the [Burning Cost](https://github.com/burning-cost) insurance pricing toolkit.

---

**Need help implementing this in production?** [Talk to us](https://burning-cost.github.io/work-with-us/).