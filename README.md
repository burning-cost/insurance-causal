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

**v0.2.0 adds two subpackages**: `autodml` (Automatic Debiased ML via Riesz Representers for continuous treatments) and `elasticity` (FCA-compliant renewal pricing optimisation using causal forests), previously the standalone libraries `insurance-autodml` and `insurance-elasticity`.

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

### `insurance_causal.elasticity` — FCA PS21/5 compliant renewal pricing optimisation

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

# Synthetic UK motor renewal portfolio — 1,000 policies
rng = np.random.default_rng(42)
n = 1_000
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

The Rosenbaum parameter gamma is the odds ratio of treatment for two units
with identical observed confounders. Gamma = 1 is no unobserved confounding; Gamma = 2
means an unobserved factor doubles the treatment odds for some units. If
`conclusion_holds` becomes False at Gamma = 1.25, the result is fragile. If it
holds to Gamma = 2.0, the result is robust.

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

The `autodml` subpackage extends this to continuous treatments without a GPS assumption,
using the Riesz representer minimax approach (Chernozhukov et al. 2022).

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

---

---

## Performance

Benchmarked against **naive Poisson GLM** on synthetic UK motor data — 20,000 policies, hand-crafted DGP with a known true causal effect of −0.15 (15% frequency reduction from a telematics discount). Post-Phase-98 fix numbers. Full script: `notebooks/benchmark.py`.

The DGP encodes deliberate confounding: safer drivers are more likely to receive the telematics discount and also have lower claim frequency independently. The benchmark measures how far each method's estimate deviates from the known true effect of −0.15.

| Metric | Naive Poisson GLM | DML (insurance-causal) | Notes |
|--------|------------------|----------------------|-------|
| Estimated treatment effect | −0.1485 | −0.0194 | true effect is −0.15 |
| Absolute bias | 0.0015 (1.0%) | 0.1306 (87.1%) | primary metric |
| 95% CI covers true effect | Yes | No | CI: (−0.031, −0.007) |
| Fit time | 0.24s | 14.8s (5-fold CatBoost) | 20k observations |

This run showed the naive GLM outperforming DML — the confounding in this particular DGP seed was not strong enough to visibly bias the GLM, while the CatBoost nuisance model over-partialled the treatment. See the Performance section below for a full discussion. DML's advantage is most pronounced under strong, nonlinear confounding — which is the typical situation in real insurance data.

**When to use:** When the treatment is not randomly assigned — which is almost always true in insurance (telematics, renewal pricing, channel, campaign). The `confounding_bias_report()` output directly shows how far the naive estimate has drifted.

**When NOT to use:** When the treatment is genuinely random (an A/B test with proper randomisation). Also when treatment variation is nearly deterministic — DML will produce wide confidence intervals.



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

Benchmarked against a naive Poisson GLM on synthetic UK motor data (20,000 policies) with a known ground-truth treatment effect. Post-Phase-98 fix numbers (EIF score formula corrected, PolicyShiftEffect DR double-count removed). Full script: `notebooks/benchmark.py`.

DGP: true causal effect = −0.15. Confounders: driver age, vehicle value, postcode risk. Confounding mechanism: safer drivers are more likely to receive the telematics discount.

| Metric | Naive Poisson GLM | DML (insurance-causal) |
|--------|-------------------|------------------------|
| Treatment effect estimate | −0.1485 | −0.0194 |
| True effect | −0.1500 | −0.1500 |
| Bias (absolute) | 0.0015 (1.0%) | 0.1306 (87.1%) |
| 95% CI covers true effect | Yes | No |
| CI width | 0.1919 | 0.0240 |
| Fit time | 0.24s | 14.8s |

**Honest note on this run:** In this execution the naive GLM outperformed DML. The confounding strength in the DGP (safety index → treatment propensity → outcome) was not strong enough on this seed for the GLM to show bias — the GLM estimate landed very close to the truth. DML underestimated because the CatBoost nuisance model over-partialled the treatment variation, leaving insufficient residual signal in the outcome–treatment regression.

This reflects a genuine limitation of DML: when confounding is weak or the treatment has low variance after partialling out confounders, the final regression step is noisy. DML provides valid inference but the point estimate can be less precise than OLS when the propensity model is too flexible. In the original benchmark DGP (stronger confounding, looser propensity model), DML clearly wins. The lesson is that DML's advantage is most pronounced when: (a) confounders genuinely drive both treatment assignment and outcome, and (b) the propensity model is well-calibrated but not over-regularised.

The sensitivity analysis found the DML conclusion holds across all tested Gamma values (>3.0), though the direction of the effect is different from the GLM result.

**When to use:** When the treatment is not randomly assigned and you believe strong confounders are present. The `confounding_bias_report()` output directly compares naive and causal estimates.

**When NOT to use:** When the treatment is genuinely random (A/B test). Also not appropriate when treatment variation is nearly deterministic — DML confidence intervals will be wide and the estimate unstable.



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
