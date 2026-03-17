# insurance-causal

![Tests](https://github.com/burning-cost/insurance-causal/actions/workflows/tests.yml/badge.svg) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green) ![PyPI](https://img.shields.io/pypi/v/insurance-causal)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/burning-cost/insurance-causal/blob/main/notebooks/quickstart.ipynb)

Causal inference for insurance pricing, built on Double Machine Learning.

Merged from: `insurance-causal` (core DML), `insurance-autodml` (Riesz representer continuous treatment), and `insurance-elasticity` (FCA renewal pricing optimisation).

---

Every UK pricing team has the same argument in some form: "Is this factor causing the claims, or is it a proxy for something else?" For telematics, is harsh braking causing accidents or is it just correlated with urban driving? For renewal pricing, is the price increase causing lapse or are the customers receiving large increases systematically more likely to lapse anyway?

These are causal questions. GLM coefficients and GBM feature importances do not answer them - they measure correlation. The standard actuarial response ("we use educated judgment and check for factor stability") is honest but leaves money on the table.

Double Machine Learning (DML), introduced by Chernozhukov et al. (2018), solves this. It estimates causal treatment effects from observational data using ML to handle high-dimensional confounders, while preserving valid frequentist inference on the parameter that matters: how much does X causally affect Y?

`insurance-causal` wraps [DoubleML](https://docs.doubleml.org/) with an interface designed for pricing actuaries. You specify the treatment (price change, channel flag, telematics score) and the confounders (rating factors), and it gives you a causal estimate with a confidence interval.

**v0.2.0 adds two subpackages**: `autodml` (Automatic Debiased ML via Riesz Representers for continuous treatments) and `elasticity` (renewal pricing optimisation using causal forests), previously the standalone libraries `insurance-autodml` and `insurance-elasticity`.

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

---

## Quick start

```python
import numpy as np
import polars as pl
from insurance_causal import CausalPricingModel
from insurance_causal.treatments import PriceChangeTreatment

# Synthetic UK motor renewal portfolio — 15,000 policies
rng = np.random.default_rng(42)
n = 15_000
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

Output:

```
Average Treatment Effect
  Treatment: pct_price_change
  Outcome:   renewal
  Estimate:  -0.0231
  Std Error: 0.0041
  95% CI:    (-0.0311, -0.0151)
  p-value:   0.0000
  N:         15,000
```

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

## Sensitivity analysis

How strong would an unobserved confounder need to be to overturn the result?

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
The nuisance model architecture: 500 trees, depth 6, learning rate 0.05. This
is more conservative than a typical predictive model but appropriate for the
debiasing goal.

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

## Performance

Benchmarked on Databricks serverless (Python 3.11, seed=42). Full script: `benchmarks/run_benchmark.py`.

**Setup:** 10,000 UK motor renewal policies. Treatment: continuous telematics score (standardised, mean 0 std 1). Outcome: binary renewal indicator (renewal rate 64%). True causal effect: +0.08 log-odds per SD of telematics score. Confounding mechanism: multiplicative risk score (age × NCB × region interaction) drives both telematics performance and renewal probability — the GLM models these as additive main effects and misses the interaction; CatBoost tree splits learn it naturally.

| Metric | Naive Logistic GLM | DML (insurance-causal) |
|--------|-------------------|------------------------|
| Treatment effect estimate | +0.0546 | +0.0077 |
| True effect | +0.0800 | +0.0800 |
| Bias (absolute) | 0.0254 (31.8%) | 0.0723 (90.4%) |
| 95% CI covers true effect | Yes | No |
| CI width | 0.090 | 0.019 |
| Fit time | 0.3s | 6.6s (5-fold CatBoost) |

**What these results show.** On this synthetic dataset both estimators are biased — neither sees the true DGP's multiplicative interaction directly. The GLM is biased upward by 32% because it models the nonlinear confounding as additive main effects. DML is biased downward by 90% because its CatBoost nuisance models absorb too much of the treatment variation in E[Y|X], leaving Y_tilde with insufficient signal for the final OLS regression.

This is the "over-partialling" problem: when the nuisance model for E[Y|X] fits the outcome too well in each cross-fitting fold, the residuals Y_tilde approach zero and the final regression is poorly conditioned. It occurs most acutely at n=10,000 with a binary outcome and a small true effect — the CatBoost model has enough power to explain most of the variation in Y from X, leaving little room for the treatment coefficient.

**When DML wins in practice.** The over-partialling problem shrinks with larger n (DML is root-n-consistent; at n=100,000 the nuisance estimation error is small relative to the signal), stronger true effects, and more outcome variation that genuinely depends on treatment. The library's design is optimised for real insurance datasets where these conditions typically hold: a renewal book of 100k+ policies, price effects on the order of 2–5% per 10% price increase, and rich confounders that a GLM genuinely misspecifies.

On the synthetic renewal datasets used in the quick-start example (n=15,000, `make_renewal_data()`), DML recovers the semi-elasticity of −0.023 accurately within its confidence interval. The benchmark script uses a harder DGP (small effect, nonlinear confounding, moderate n) that exposes the limits of the estimator.

**The `confounding_bias_report()` use case.** DML's primary value in insurance is not always the absolute estimate — it's the direction and magnitude of the confounding correction. When a GLM gives −0.045 and DML gives −0.023, the team learns that observed confounders explain roughly half the naive estimate. Even if the DML estimate has some regularisation bias, the signal is: "your naive estimate is inflated." The sensitivity analysis (`sensitivity_analysis()`) quantifies how much an unobserved confounder would need to be to overturn that conclusion.

**When to use:** When the treatment is not randomly assigned — telematics discounts, renewal price changes, channel, campaign flags. At n ≥ 50,000. The `confounding_bias_report()` is useful even when the absolute estimate is uncertain.

**When NOT to use:** n < 10,000 (DML's asymptotic properties don't kick in at small n). When the treatment is genuinely random (A/B test). When outcome variance is very low (sparse Poisson claim counts at low frequency). When treatment variation is near-deterministic — confidence intervals will be wide and the estimate will be attenuated.


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
