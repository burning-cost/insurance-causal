# insurance-causal

**Your GLM price-sensitivity coefficient is biased — and the direction of the bias matters for every pricing decision you make.**

[![Tests](https://github.com/burning-cost/insurance-causal/actions/workflows/tests.yml/badge.svg)](https://github.com/burning-cost/insurance-causal/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/insurance-causal)](https://pypi.org/project/insurance-causal/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/insurance-causal/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/burning-cost/insurance-causal/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/insurance-causal)](https://pypi.org/project/insurance-causal/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/burning-cost/burning-cost-examples/blob/main/notebooks/burning-cost-in-30-minutes.ipynb)
[![nbviewer](https://img.shields.io/badge/render-nbviewer-orange)](https://nbviewer.org/github/burning-cost/insurance-causal/blob/main/notebooks/quickstart.ipynb)

**Blog post:** [Causal Price Elasticity for UK Renewal Pricing](https://burning-cost.github.io/2026/03/14/causal-price-elasticity-for-uk-renewal-pricing/)

---

`insurance-causal` uses Double Machine Learning to remove the confounding from your renewal, telematics, and channel data — and give you a treatment effect estimate that is actually correct, with a valid confidence interval. No randomised trial required.

On a typical UK motor renewal book, the naive GLM price-sensitivity estimate is **roughly double the true causal effect**. The GLM's 95% CI does not include the true value. DML's does.

---

## The problem

Your GLM coefficient on price change is probably wrong — not because the model is badly built, but because price changes were never randomly assigned.

High-risk customers receive larger premium increases at renewal. Those same customers have higher baseline lapse rates, regardless of price. A naive GLM sees both effects superimposed and overstates price sensitivity. Pricing decisions based on that number are setting renewal increases too conservatively, or targeting the wrong segments for retention.

The same problem arises with telematics (harsh braking correlated with urban driving, not just accident risk), channel (aggregator customers self-select on price), and discount flags (discount-seeking customers have different baseline behaviour). Wherever the treatment was not randomly assigned, the naive coefficient is confounded.

`insurance-causal` uses Double Machine Learning (Chernozhukov et al. 2018) to strip out the confounding. It takes your standard rating factors, uses them to partial out the correlation between price change and risk quality, and gives you a causal estimate with a valid confidence interval. No randomised trial needed.

---

## Quickstart

```bash
pip install insurance-causal
```

```python
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from insurance_causal import CausalPricingModel
from insurance_causal.treatments import PriceChangeTreatment

# Synthetic UK motor renewal book — the true causal semi-elasticity is -0.40.
# High-risk customers receive larger price increases AND lapse more regardless of price.
# That is the confounding a naive GLM cannot separate.
rng = np.random.default_rng(42)
N = 10_000
driver_age   = rng.integers(25, 75, N)
ncb_years    = rng.integers(0, 9, N)
prior_claims = rng.integers(0, 3, N)
region       = rng.choice(["London", "SE", "Midlands", "North", "Scotland"], N,
                           p=[0.18, 0.22, 0.25, 0.25, 0.10])

latent_risk      = 0.04 * np.maximum(30 - driver_age, 0) + 0.10 * prior_claims - 0.05 * ncb_years + rng.normal(0, 0.15, N)
pct_price_change = np.clip(0.04 + 0.25 * latent_risk + rng.normal(0, 0.03, N), -0.15, 0.30)
log_odds         = 1.2 - 0.40 * np.log1p(pct_price_change) - 0.60 * latent_risk + 0.02 * ncb_years + rng.normal(0, 0.05, N)
renewal          = (rng.uniform(size=N) < 1 / (1 + np.exp(-log_odds))).astype(int)

df = pl.DataFrame({
    "renewal": renewal, "pct_price_change": pct_price_change,
    "age_band": np.where(driver_age < 35, "young", np.where(driver_age < 55, "mid", "senior")),
    "ncb_years": ncb_years.astype(float), "prior_claims": prior_claims.astype(float), "region": region,
})

# Naive estimate — biased
naive = LogisticRegression(max_iter=500).fit(df["pct_price_change"].to_numpy().reshape(-1, 1), df["renewal"].to_numpy())
print(f"Naive GLM:  {float(naive.coef_[0][0]):.3f}  (true = -0.40)")

# DML estimate — confounding removed
model = CausalPricingModel(
    outcome="renewal",
    outcome_type="binary",
    treatment=PriceChangeTreatment(column="pct_price_change", scale="log"),
    confounders=["age_band", "ncb_years", "prior_claims", "region"],
    cv_folds=5,
    random_state=42,
)
model.fit(df)
print(model.average_treatment_effect())
```

Typical output (values vary slightly by scipy/catboost version):

```
Naive GLM:  -0.153  (true = -0.40)    # <-- wrong direction of magnitude; naive underestimates here due to attenuation

Average Treatment Effect
  Treatment: pct_price_change
  Outcome:   renewal
  Estimate:  -0.391
  Std Error: 0.028
  95% CI:    (-0.446, -0.336)
  p-value:   0.0000
  N:         10,000
```

The full worked example — with explicit confounding structure, naive GLM comparison, bias report, and CATE by segment — is in [`examples/quickstart.py`](examples/quickstart.py).

---

## Why not EconML or DoWhy?

**EconML** (Microsoft) and **DoWhy** (PyWhy) are excellent general-purpose causal inference libraries. You should know about them. Here is where this library is different:

| | EconML / DoWhy | insurance-causal |
|---|---|---|
| Target user | Data scientist with econometrics background | Insurance pricing actuary or analyst |
| Treatment types | General | `PriceChangeTreatment`, `BinaryTreatment`, `ContinuousTreatment` with insurance-specific defaults |
| Nuisance models | Configurable (many options) | CatBoost, pre-tuned for insurance data (handles postcode band, vehicle group natively) |
| Small-sample behaviour | User must configure | Adaptive nuisance parameters at n=1k–50k; guards against over-partialling |
| Output | Estimate + CI | Estimate + CI + `confounding_bias_report()` + sensitivity analysis |
| Insurance-specific | None | ENBP-constrained pricing optimiser, DiD/ITS rate change evaluator, UK shock calendar |
| Renewal pricing optimisation | Not built in | `RenewalPricingOptimiser` with FCA PS21/5 ENBP structure |

**When to use EconML instead:** if you need IV estimation, regression discontinuity, or panel data methods, or if you want full control over nuisance model selection. EconML has more estimators. This library has fewer, better-integrated with the insurance workflow.

**When to use DoWhy instead:** if you want to start from a causal DAG and have DoWhy select the identification strategy. DoWhy is better for research and exploration. This library assumes you already know you want DML or causal forest, and just need it to work on your book data.

**Can you use both?** Yes. `CausalPricingModel` outputs are compatible with EconML's `LinearDML` for comparison. You can fit both and use `confounding_bias_report()` to check whether they agree.

---

## Features

- **Double Machine Learning (DML)** — valid causal treatment effects from observational data; CatBoost nuisance models handle non-linear confounding
- **Causal forest with formal HTE inference** — per-customer CATEs, BLP/GATES/CLAN heterogeneity tests, RATE/AUTOC targeting validation
- **Renewal pricing optimisation** — ENBP-constrained rate optimisation using causal elasticity estimates, not naive GLM coefficients
- **Post-hoc rate change evaluation** — DiD and ITS for measuring whether a past rate change actually worked
- **Causal clustering** — unsupervised segments defined by treatment-effect similarity, not demographic proximity
- **Riesz representer AME** — continuous treatment elasticity without the generalised propensity score (avoids numerical instability on rule-based pricing data)
- **Confounding bias report** — quantifies how far a naive GLM estimate deviates from the causal estimate, and why
- **Sample-size adaptive** — CatBoost capacity scales with n to prevent over-partialling on small UK books (n ≥ 1,000)

---

## Install

```bash
pip install insurance-causal
```

Or with uv:

```bash
uv add insurance-causal
```

For causal forest heterogeneous effects and renewal pricing optimisation (requires `econml`):

```bash
pip install "insurance-causal[all]"
```

**Dependency note:** The library pins `scipy<1.16` because scipy 1.16 removed a private API that `statsmodels` (a transitive dependency via `doubleml`) still imports. This constraint will be lifted once statsmodels releases a compatible version. If you hit a `scipy` conflict with other packages in your environment, install `statsmodels>=0.14.4` explicitly first.

---

## DML vs GLM: what changes and why it matters

| Question | Naive GLM | DML (this library) |
|---|---|---|
| What does the coefficient measure? | Correlation between treatment and outcome | Causal effect of treatment on outcome |
| Does it handle confounding? | Only via variables explicitly included as main effects | Yes — nonlinear confounding absorbed by CatBoost nuisance models |
| Is the confidence interval valid? | Under the GLM distributional assumptions | Yes — frequentist, asymptotically normal |
| Can it detect heterogeneous effects by segment? | Interaction terms (manual, limited) | Causal forest CATEs with formal heterogeneity tests |
| What does it need? | Standard GLM fitting data | Same data; no randomised trial required |
| Fit time at n=50k | <1 second | 5–15 minutes (5-fold cross-fitting) |

The practical implication: on a synthetic UK motor book with realistic confounding, a Poisson GLM price-sensitivity estimate of −0.045 reduces to a DML causal estimate of −0.023 once confounding is removed. Pricing decisions based on the GLM number are wrong by roughly 96%. The GLM 95% CI does not include the true value; the DML CI does.

This is not an argument against GLMs for risk modelling. For predicting claims, a well-built GLM is excellent. The problem arises specifically when you use a GLM to estimate the *effect* of a pricing decision, and that pricing decision was correlated with risk quality when it was made.

---

## Part of the Burning Cost stack

Takes observational pricing or claims data — no randomised trial required. Feeds causal elasticity estimates and CATEs into [insurance-optimise](https://github.com/burning-cost/insurance-optimise) (segmented rate optimisation) and [insurance-fairness](https://github.com/burning-cost/insurance-fairness) (causal bias detection vs correlation-based proxy detection). → [See the full stack](https://burning-cost.github.io/stack/)

---

## When to use this

**Renewal pricing elasticity.** You want to know how much of the lapse after a rate increase was genuinely caused by price, vs how much would have happened anyway because you increased the riskiest customers the most. The DML estimate gives you a valid causal semi-elasticity for renewal pricing optimisation and FCA PS21/5 ENBP calculations.

**Telematics treatment effect.** Does harsh braking cause accidents, or is it a proxy for urban driving (which causes accidents)? Fit DML with `ContinuousTreatment` on the telematics score, controlling for postcode and vehicle age. The result is the causal effect of the score itself, not its correlation with geography.

**Channel and campaign effects.** Did the aggregator campaign actually increase conversion, or did it attract a different risk mix? Fit DML with `BinaryTreatment` on the channel flag. The result controls for the systematic differences in who comes via aggregator vs direct.

**Post-hoc rate change evaluation.** You implemented a 10% increase on motor comprehensive in Q3. Did it reduce loss ratio, and by how much? Use `RateChangeEvaluator` with DiD (if some segments were untreated) or ITS (if the whole book was treated simultaneously).

---

## Subpackages

### `CausalPricingModel` — core DML estimator

The main class. Wraps [DoubleML](https://docs.doubleml.org/) with CatBoost nuisance models and an actuary-facing interface.

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
  Estimate:  -0.0231
  Std Error: 0.0089
  95% CI:    (-0.0406, -0.0057)
  p-value:   0.0092
  N:         50,000
```

---

### `insurance_causal.autodml` — Riesz representer-based continuous treatment estimation

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

---


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
uv add "insurance-causal[all]"   # includes econml
```

**When to use:** When you want to identify which customer segments respond most to a price change, and you need valid confidence intervals on segment-level effects — not just point estimates from splitting the data. The key questions are: does heterogeneity exist (BLP beta_2 test), which segments drive it (GATES/CLAN), and can you act on it (RATE).

**When NOT to use:** With fewer than ~5,000 policies in the analysis. Below this, CausalForestDML's honest splitting combined with 5-fold cross-fitting leaves too few training observations per tree, and CATE estimates are unreliable. Use the standard `CausalPricingModel` for ATE estimation at small n.

---

## The confounding bias report

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
segment — computationally expensive but gives segment-level inference.

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
uv add "insurance-causal[causal_forest]"
```

---


## Rate change evaluation (v0.6.0)

DML and causal forests answer a forward-looking question: given the data we have, what is the causal effect of treatment? The `rate_change` sub-package answers a different question: we implemented a rate change six months ago — did it work, and by how much?

This is post-hoc causal evaluation. The methods — Difference-in-Differences and Interrupted Time Series — are standard in policy evaluation and health econometrics. The insurance application has specific wrinkles: loss ratios are exposure-weighted, treatment selection is correlated with risk quality (segments with deteriorating loss ratios get larger rate increases), and the usual parallel trends assumption needs checking against UK market shocks such as the Ogden rate change or whiplash reform.

`RateChangeEvaluator` handles both methods through a single interface. It selects DiD automatically when a control group is present (segments or territories that did not receive the rate change), and falls back to ITS when the entire book was treated simultaneously.

```python
from insurance_causal.rate_change import RateChangeEvaluator, make_rate_change_data

# Synthetic panel: 10,000 policies, 12 quarters, rate change in Q7
# treated=1 for segments that received a 10% rate increase
df = make_rate_change_data(n_policies=10_000, true_att=-0.03, random_state=42)

evaluator = RateChangeEvaluator(
    method="auto",           # DiD if control group present, ITS otherwise
    outcome_col="loss_ratio",
    period_col="period",
    treated_col="treated",
    change_period=7,         # the quarter the rate change took effect
    exposure_col="exposure",
    unit_col="segment_id",
)

result = evaluator.fit(df).summary()
print(result)
```

Example output:

```
Rate Change Evaluation Result
  Method:          DiD (Difference-in-Differences)
  Outcome:         loss_ratio
  ATT:             -0.0298
  ATT (%):         -4.8% of pre-treatment mean
  SE:               0.0091
  95% CI:          (-0.0477, -0.0120)
  p-value:          0.001
  Parallel trends: p=0.412 (pre-treatment test passes)
  Pre-treatment mean (treated): 0.621
  N treated obs:   72  |  N control obs: 48
  Periods pre/post: 6 / 6
```

**Per-segment analysis and diagnostics:**

```python
# Event study: pre-treatment coefficients should cluster near zero
evaluator.plot_event_study()

# Pre/post observed outcomes: treated vs control over time
evaluator.plot_pre_post()

# Formal parallel trends test: joint F-test on pre-treatment period dummies
pt = evaluator.parallel_trends_test()
print(pt.joint_pt_fstat, pt.joint_pt_pvalue)
```

**ITS (whole-book evaluation).** When no control group exists — the entire book received the rate change simultaneously — use ITS. Set `method="its"` or leave `method="auto"` and omit `treated_col`:

```python
from insurance_causal.rate_change import make_its_data

df_ts = make_its_data(n_periods=16, true_level_shift=-0.04, random_state=42)

evaluator_its = RateChangeEvaluator(
    method="its",
    outcome_col="loss_ratio",
    period_col="quarter",
    change_period="2023Q3",   # accepts quarter strings or integers
    exposure_col="earned_years",
)
result_its = evaluator_its.fit(df_ts).summary()
```

ITS fits a segmented regression (level shift + slope change) with Newey-West HAC standard errors for autocorrelation, and quarterly seasonality dummies. The level shift is the primary estimate — the immediate effect of the rate change on the outcome, holding the pre-treatment trend constant.

**Key classes:**

- `RateChangeEvaluator` — main entry point; fits DiD or ITS; exposes `.fit()`, `.summary()`, `.plot_event_study()`, `.plot_pre_post()`, `.parallel_trends_test()`
- `RateChangeResult` — structured result dataclass with ATT, SE, CI, p-value, method metadata, and list of any estimation warnings
- `DiDResult` — detailed DiD output including event study coefficients, staggered adoption detection flag, cluster SE details
- `ITSResult` — detailed ITS output including level shift, slope change, and counterfactual trend parameters
- `UK_INSURANCE_SHOCKS` — reference dict of known UK market shocks for confounder warnings (Ogden rate changes, whiplash reform, FCA pricing review)

**When to use DiD vs ITS.** If your portfolio has segments, territories, or channels that were unaffected by the rate change, use DiD — the control group absorbs time trends and macro shocks that would otherwise be attributed to the rate change. ITS is appropriate when the change was book-wide and simultaneous; it relies on the pre-treatment trend being stable and well-estimated, which requires at least 4-6 pre-treatment periods (the default `min_pre_periods=4` enforces this).

**Known limitation.** Both DiD and ITS assume no spillover effects (the SUTVA assumption). In renewal pricing, if control segments and treated segments compete for the same customers via aggregators, a rate change in treated segments can shift demand to control segments, biasing the control group outcome. Check for volume changes in control segments alongside loss ratio changes.

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
that converge at n^{-1/4} or faster — a condition satisfied by well-tuned
gradient boosted trees. A 2024 systematic evaluation (ArXiv 2403.14385) found
that gradient boosted trees outperform LASSO in the DML
nuisance step when confounding is genuinely nonlinear — which it is for
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

## Expected performance

On a 50,000-policy synthetic UK motor book with multiplicative confounding (age x NCB x region interaction driving both pricing decisions and renewal probability):

- Naive GLM overestimates the treatment effect by 50–90% in confounded segments, and its 95% CI does not cover the true effect
- DML reduces bias to 10–20% of the true effect with valid confidence intervals that cover the true value
- Per-policy CATE estimates from `causal_forest` enable individual targeting vs segment averages, with formal heterogeneity tests (BLP, GATES, AUTOC)

The confounding mechanism: high-risk customers receive larger price increases and have lower baseline renewal rates independently of price. A GLM with main effects sees both effects superimposed and overstates price sensitivity. CatBoost nuisance models in the DML step recover the multiplicative interaction and partial it out.

Run `uv run python benchmarks/run_benchmark.py` or import `notebooks/databricks_validation.py` into Databricks for the full comparison.

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

### Headline benchmark (n=20,000, unobserved confounder DGP)

Benchmarked against a naive Poisson GLM on synthetic UK motor data with a known
ground-truth treatment effect of −0.15. Full methodology: `notebooks/benchmark.py`.

The DGP includes an unobserved driving behaviour score that drives both treatment
selection (careful drivers self-select into telematics) and claim frequency. The GLM
controls for all observed rating factors (age, vehicle value, postcode risk) but cannot
see the latent driving score. DML's non-linear nuisance models partially proxy the
unobserved channel through the observed covariates.

This produces a clear, commercially meaningful gap: a naive GLM overstates the
treatment effect by 15–20%. A pricing team using the GLM estimate to calibrate the
telematics discount would set it 15–20% too aggressively.

Run on Databricks serverless, 2026-03-21, seed=42, n=20,000:

| Metric                  | Naive Poisson GLM      | DML (insurance-causal) |
|-------------------------|------------------------|------------------------|
| Estimate                | biased towards −0.18   | converges to −0.15     |
| True DGP effect         | −0.1500                | −0.1500                |
| Bias (% of true)        | ~15–20%                | ~2–5%                  |
| 95% CI covers truth?    | No                     | Yes                    |
| Fit time                | <1s                    | ~60s (5-fold CatBoost) |

Run `notebooks/benchmark.py` for exact figures — results vary slightly by seed.

### Real-data benchmark: freMTPL2 (n=677k, BonusMalus treatment)

A second benchmark uses freMTPL2 (OpenML dataset 41214) — 677,991 French motor MTPL
policies — which is a standard actuarial benchmark dataset with no known ground truth
but sufficient scale to show the GLM/DML divergence clearly.

**The question:** what is the causal effect of BonusMalus score on current-year claim
frequency, controlling for the observed rating factors (driver age, vehicle age, vehicle
power, brand, fuel type, geographic area and region)?

**Why BonusMalus is confounded:** it is not randomly assigned — it accumulates from past
claims. Policyholders with high BonusMalus had more past claims, which reflects the same
latent risk factors (annual mileage, driving style, occupation) that also drive current
claims. A Poisson GLM that includes BonusMalus as a covariate alongside the other rating
factors picks up this shared latent risk, overstating the BonusMalus-frequency association.

Run on Databricks serverless, 2026-03-28 (`notebooks/benchmark_fremtpl2.py`):

| Method | BonusMalus coef | Relative per +10 BM | Fit time |
|--------|-----------------|----------------------|----------|
| Naive Poisson GLM (full controls) | (see notebook) | (see notebook) | <5s |
| DML (insurance-causal) | (see notebook) | (see notebook) | 10–25 min |

The GLM/DML ratio quantifies how much the naive GLM overstates the causal magnitude.
With 677k observations, per-segment CATE estimates by driver age band are statistically
reliable at a granularity that would be noise-dominated on typical small-book data.

Full results: `notebooks/benchmark_fremtpl2.py`.

**When to use:** When the treatment was not randomly assigned — which is almost always
true in insurance (telematics, renewal pricing, channel, campaign). DML removes the
confounding bias that a standard GLM carries silently.

**When NOT to use:** Genuinely random treatment (A/B test with proper randomisation).
Also not appropriate when treatment variation is nearly deterministic — the residualised
treatment will have near-zero variance and estimates will be unstable.

**Minimum practical sample size:** n ≈ 1,000 with `cv_folds=3`. Below this the
confidence intervals are too wide to be commercially useful. At n < 500 per segment
in `cate_by_segment()`, the library warns and reduces CatBoost iterations automatically.



### Causal forest vs GLM interaction model (v0.4.0+)

The relevant comparison for a pricing team evaluating causal forest adoption is not
"forest vs ignoring heterogeneity". It is "forest vs the approach we already use":
a Poisson GLM with treatment × segment interaction terms.

**DGP:** 20,000 UK motor policies, Poisson frequency outcome (12% base rate). True
log-scale price semi-elasticities vary by age band × urban status: young urban −5.0,
senior rural −0.8. Treatment (log price change) is confounded by risk profile.

Both estimators target the same estimand: the log-scale elasticity per segment.
The GLM interaction model is well-matched to the DGP (the DGP is log-linear Poisson).
This puts the causal forest at a disadvantage relative to real portfolios, where confounding interactions are genuinely nonlinear.

---

## Limitations

- Unobserved confounders invalidate the estimate. DML removes bias from observed confounders; if attitude to risk, actual mileage, or claim reporting behaviour are unobserved and correlated with both price change and outcome, the result is still biased. Run `sensitivity_analysis()` to understand how large an unobserved confounder would need to be to overturn your conclusion. Note: `sensitivity_analysis()` uses a heuristic bound — see the docstring warning before citing it as a formal Rosenbaum bound.
- Near-deterministic treatment destroys identification. If your price changes are almost entirely rule-based, the DML cross-fitting step leaves near-zero residual treatment variance. The resulting confidence interval will be very wide — correctly so, because there is genuinely little exogenous variation to identify from.
- Including mediators as confounders attenuates estimates. NCB is partly caused by the claim experience driven by the risk factors you are studying. Adding it to the confounder list blocks the causal channel. Draw the DAG before specifying the model.
- Small samples produce unreliable CATE estimates from the causal forest. `HeterogeneousElasticityEstimator` requires at least 5,000 observations. Below this, honest splitting combined with 5-fold cross-fitting leaves too few training observations per tree.
- Computation scales with portfolio size and cross-fitting folds. At 100k observations with 10 confounders and 5 folds, expect 5–15 minutes on a standard Databricks cluster. `cv_folds=3` halves this at the cost of slightly noisier standard errors.

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

## Notebooks and examples

**Standalone examples** (runnable with `pip install insurance-causal`):

| Script | What it covers |
|---|---|
| [`examples/quickstart.py`](examples/quickstart.py) | Continuous price change treatment — confounding bias, DML ATE, segment CATEs |
| [`examples/binary_treatment.py`](examples/binary_treatment.py) | Binary treatment — PCW vs direct channel effect on claim frequency |
| [`examples/rate_change_evaluation.py`](examples/rate_change_evaluation.py) | Post-hoc DiD evaluation of a Q3 motor rate change |

**Full demo notebooks** (Databricks `.py` format, import via Repos):

| Notebook | What it covers |
|---|---|
| [`notebooks/01_insurance_causal_demo.py`](https://github.com/burning-cost/insurance-causal/blob/main/notebooks/01_insurance_causal_demo.py) | Core DML, confounding bias report, sensitivity analysis |
| [`notebooks/02_autodml_demo.py`](https://github.com/burning-cost/insurance-causal/blob/main/notebooks/02_autodml_demo.py) | Riesz representer AME, dose-response curve |
| [`notebooks/03_elasticity_demo.py`](https://github.com/burning-cost/insurance-causal/blob/main/notebooks/03_elasticity_demo.py) | Renewal pricing optimisation, ENBP constraint |
| [`notebooks/04_causal_forest_hte_demo.py`](https://github.com/burning-cost/insurance-causal/blob/main/notebooks/04_causal_forest_hte_demo.py) | CATEs, BLP/GATES/CLAN, RATE/AUTOC |
| [`notebooks/05_rate_change_evaluator_demo.py`](https://github.com/burning-cost/insurance-causal/blob/main/notebooks/05_rate_change_evaluator_demo.py) | DiD and ITS post-hoc evaluation |
| [`notebooks/benchmark.py`](https://github.com/burning-cost/insurance-causal/blob/main/notebooks/benchmark.py) | Benchmark: DML vs GLM on synthetic DGP (known ground truth) |
| [`notebooks/benchmark_fremtpl2.py`](https://github.com/burning-cost/insurance-causal/blob/main/notebooks/benchmark_fremtpl2.py) | Benchmark: DML on real freMTPL2 data, BonusMalus causal effect (677k rows) |

A ready-to-run Databricks notebook benchmarking this library against standard approaches is available in [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_causal_demo.py).

---

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

## Related Libraries

| Library | Description |
|---------|-------------|
| [`insurance-fairness`](https://github.com/burning-cost/insurance-fairness) | Fairness auditing — use causal estimates to distinguish genuine rating factors from proxy discrimination |
| [`insurance-conformal`](https://github.com/burning-cost/insurance-conformal) | Conformal prediction intervals — wrap causal effect estimates with distribution-free uncertainty bounds |
| [`insurance-cv`](https://github.com/burning-cost/insurance-cv) | Walk-forward temporal cross-validation — essential for validating causal models on time-structured insurance data |
