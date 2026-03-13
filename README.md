# insurance-causal

![Tests](https://github.com/burning-cost/insurance-causal/actions/workflows/tests.yml/badge.svg) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green) ![PyPI](https://img.shields.io/pypi/v/insurance-causal)

Causal inference for insurance pricing, built on Double Machine Learning.

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
from insurance_causal.autodml import PremiumElasticity, DoseResponseCurve

# Average Marginal Effect: average d/dD E[Y|D,X]
model = PremiumElasticity(outcome_family="poisson", n_folds=5)
model.fit(X, D, Y, exposure=exposure)
result = model.estimate()
print(result.summary())

# Dose-response curve at specified premium levels
dr = DoseResponseCurve(outcome_family="poisson")
dr.fit(X, D, Y)
curve = dr.evaluate(D_grid=np.linspace(200, 800, 20))
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
uv add insurance-causal
```

For the elasticity subpackage (requires econML):

```bash
uv add "insurance-causal[elasticity]"
```

For all optional dependencies:

```bash
uv add "insurance-causal[all]"
```

Core dependencies: `doubleml`, `catboost`, `polars`, `pandas`, `scikit-learn`, `scipy`, `numpy`, `joblib`.

---

## Quick start

```python
from insurance_causal import CausalPricingModel
from insurance_causal.treatments import PriceChangeTreatment

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
| [credibility](https://github.com/burning-cost/credibility) | Bühlmann-Straub credibility weighting |

**Deployment and optimisation**

| Library | Description |
|---------|-------------|
| [rate-optimiser](https://github.com/burning-cost/rate-optimiser) | Constrained rate change optimisation with FCA PS21/5 compliance |
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

## Licence

MIT. Part of the [Burning Cost](https://github.com/burning-cost) insurance pricing toolkit.
