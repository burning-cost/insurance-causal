# insurance-causal

Double Machine Learning for causal treatment effect estimation from observational insurance data — price elasticity, telematics effects, and post-hoc rate change evaluation.

[![Tests](https://github.com/burning-cost/insurance-causal/actions/workflows/tests.yml/badge.svg)](https://github.com/burning-cost/insurance-causal/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/insurance-causal)](https://pypi.org/project/insurance-causal/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/insurance-causal/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/burning-cost/insurance-causal/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/insurance-causal)](https://pypi.org/project/insurance-causal/)

**Blog post:** [Causal Price Elasticity for UK Renewal Pricing](https://burning-cost.github.io/2026/03/14/causal-price-elasticity-for-uk-renewal-pricing/)

---

## The problem

Your GLM coefficient on price change is probably wrong — not because the model is badly built, but because price changes were never randomly assigned.

High-risk customers receive larger premium increases at renewal. Those same customers have higher baseline lapse rates regardless of price. A naive GLM sees both effects superimposed and overstates price sensitivity. Pricing decisions based on that number are setting renewal increases too conservatively, or targeting the wrong segments for retention.

The same problem appears with telematics (harsh braking correlated with urban driving), channel (aggregator customers self-select on price), and discount flags (discount-seeking customers have different baseline behaviour). Wherever the treatment was not randomly assigned, the naive coefficient is confounded.

`insurance-causal` uses Double Machine Learning (Chernozhukov et al. 2018) to strip out the confounding. No randomised trial required.

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

# Synthetic UK motor renewal book. True causal semi-elasticity = -0.40.
# High-risk customers receive larger price increases AND lapse more regardless of price.
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

# Naive estimate — biased by confounding
naive = LogisticRegression(max_iter=500).fit(df["pct_price_change"].to_numpy().reshape(-1, 1), df["renewal"].to_numpy())
print(f"Naive GLM:  {float(naive.coef_[0][0]):.3f}  (true = -0.40)")

# DML — confounding removed
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

Typical output:

```
Naive GLM:  -0.153  (true = -0.40)

Average Treatment Effect
  Treatment: pct_price_change
  Outcome:   renewal
  Estimate:  -0.391
  Std Error: 0.028
  95% CI:    (-0.446, -0.336)
  p-value:   0.0000
  N:         10,000
```

The naive GLM gets the magnitude substantially wrong. DML recovers the true effect. The full example with confounding bias report and segment CATEs is in [`examples/quickstart.py`](examples/quickstart.py).

---

## Why this library?

EconML (Microsoft) and DoWhy (PyWhy) are good general-purpose causal inference libraries. This one is different:

| | EconML / DoWhy | insurance-causal |
|---|---|---|
| Target user | Data scientist with econometrics background | Insurance pricing actuary or analyst |
| Treatment types | General | `PriceChangeTreatment`, `BinaryTreatment`, `ContinuousTreatment` with insurance-specific defaults |
| Nuisance models | Configurable | CatBoost, pre-tuned for insurance data (handles postcode band, vehicle group natively) |
| Small-sample behaviour | User must configure | Adaptive nuisance parameters at n=1k–50k; guards against over-partialling |
| Output | Estimate + CI | Estimate + CI + `confounding_bias_report()` + sensitivity analysis |
| Insurance-specific | None | ENBP-constrained pricing optimiser, DiD/ITS rate change evaluator, UK shock calendar |

Use EconML if you need IV estimation, regression discontinuity, or panel data methods. Use DoWhy if you want to start from a causal DAG. Use this library if you want DML or causal forest working on your book data with minimal configuration.

---

## Features

- **Double Machine Learning** — valid causal treatment effects from observational renewal data; CatBoost nuisance models handle non-linear confounding from postcode, vehicle group, and occupation
- **Causal forest with formal HTE inference** — per-customer CATEs, BLP/GATES/CLAN heterogeneity tests, RATE/AUTOC targeting validation
- **Renewal pricing optimisation** — ENBP-constrained rate optimisation using causal elasticity estimates, not naive GLM coefficients
- **Rate change evaluation** — DiD and ITS for measuring whether a past rate change actually shifted loss ratio; handles staggered adoption and parallel trends testing
- **Causal clustering** — segments defined by treatment-effect similarity using the causal forest proximity kernel, without requiring you to nominate the segmentation variable
- **Riesz representer AME** — continuous treatment elasticity without the generalised propensity score (avoids numerical instability on rule-based pricing data)
- **Confounding bias report** — quantifies how far a naive GLM estimate deviates from the causal estimate
- **Sample-size adaptive** — CatBoost capacity scales with n to prevent over-partialling on small books (n ≥ 1,000)

---

## Installation

```bash
pip install insurance-causal
# or
uv add insurance-causal
```

For causal forest heterogeneous effects and renewal pricing optimisation (requires `econml`):

```bash
pip install "insurance-causal[all]"
```

**Dependency note:** The library pins `scipy<1.16` due to a `statsmodels` compatibility issue with scipy 1.16. This constraint will be lifted once statsmodels releases a compatible version. If you hit a `scipy` conflict, install `statsmodels>=0.14.4` first.

---

## DML vs GLM: what changes

| Question | Naive GLM | DML |
|---|---|---|
| What does the coefficient measure? | Correlation between treatment and outcome | Causal effect of treatment on outcome |
| Does it handle confounding? | Only via variables explicitly included as main effects | Yes — nonlinear confounding absorbed by CatBoost nuisance models |
| Is the confidence interval valid? | Under GLM distributional assumptions | Yes — frequentist, asymptotically normal |
| Can it detect heterogeneous effects by segment? | Interaction terms (manual, limited) | Causal forest CATEs with formal heterogeneity tests |
| Fit time at n=50k | <1 second | 5–15 minutes (5-fold cross-fitting) |

On a synthetic UK motor book with realistic confounding, a Poisson GLM price-sensitivity estimate of −0.045 reduces to a DML causal estimate of −0.023 once confounding is removed. The GLM 95% CI does not include the true value; the DML CI does.

This is not an argument against GLMs for risk modelling. For predicting claims, a well-built GLM is appropriate. The problem is using a GLM to estimate the *effect* of a pricing decision that was correlated with risk quality when it was made.

---

## When to use this

**Renewal pricing elasticity.** You want to know how much of the lapse after a rate increase was genuinely caused by price, vs how much would have happened anyway because you increased the riskiest customers the most.

**Telematics treatment effect.** Does harsh braking cause accidents, or is it a proxy for urban driving? Fit DML with `ContinuousTreatment` on the telematics score, controlling for postcode and vehicle age.

**Channel and campaign effects.** Did the aggregator campaign increase conversion, or did it attract a different risk mix? Fit DML with `BinaryTreatment` on the channel flag.

**Post-hoc rate change evaluation.** You implemented a 10% increase on motor comprehensive in Q3. Did it reduce loss ratio, and by how much? Use `RateChangeEvaluator` with DiD (if some segments were untreated) or ITS (if the whole book was treated simultaneously).

---

## Core API

### `CausalPricingModel`

```python
from insurance_causal import CausalPricingModel
from insurance_causal.treatments import PriceChangeTreatment, BinaryTreatment, ContinuousTreatment

model = CausalPricingModel(
    outcome="renewal",
    outcome_type="binary",       # "binary", "poisson", "continuous", "gamma"
    treatment=PriceChangeTreatment(column="pct_price_change", scale="log"),
    confounders=["age_band", "ncb_years", "vehicle_age", "prior_claims"],
    cv_folds=5,
    exposure_col="earned_years", # for Poisson frequency models
)
model.fit(df)

ate = model.average_treatment_effect()
report = model.confounding_bias_report(naive_coefficient=-0.045)
cate = model.cate_by_segment(df, segment_col="age_band")
```

### `causal_forest` subpackage — heterogeneous effects

```python
from insurance_causal.causal_forest import HeterogeneousElasticityEstimator, HeterogeneousInference

est = HeterogeneousElasticityEstimator(n_estimators=200)
est.fit(df, outcome="renewed", treatment="log_price_change", confounders=confounders)

cates = est.cate(df)
gates = est.gate(df, by="ncd_years")   # group ATEs by NCD band

inf = HeterogeneousInference(n_splits=100, k_groups=5)
result = inf.run(df, estimator=est, cate_proxy=cates)
result.plot_gates()
```

Requires `pip install "insurance-causal[all]"`.

### `rate_change` subpackage — post-hoc evaluation

```python
from insurance_causal.rate_change import RateChangeEvaluator

evaluator = RateChangeEvaluator(
    method="auto",           # DiD if control group present, ITS otherwise
    outcome_col="loss_ratio",
    period_col="period",
    treated_col="treated",
    change_period=7,
    exposure_col="exposure",
    unit_col="segment_id",
)
result = evaluator.fit(df).summary()
evaluator.plot_event_study()
```

---

## Expected performance

On a 50,000-policy synthetic UK motor book with multiplicative confounding:

| Metric | Naive GLM | DML |
|--------|-----------|-----|
| Bias (% of true effect) | 50–90% | 2–5% |
| 95% CI covers true value? | No | Yes |
| Fit time (5-fold, CatBoost) | <1s | ~60–300s |

Small-sample performance: at n=1,000–5,000, the library uses adaptive CatBoost parameters (fewer iterations, shallower depth, stronger regularisation) to prevent over-partialling — where a too-flexible nuisance model absorbs the treatment signal before the DML regression step can see it. The minimum practical sample size is n ≈ 1,000 with `cv_folds=3`.

---

## Limitations

- **Unobserved confounders invalidate the estimate.** DML removes bias from observed confounders only. Run `sensitivity_analysis()` to understand how large an unobserved confounder would need to be to overturn the conclusion.
- **Near-deterministic treatment destroys identification.** If price changes are almost entirely rule-based, residual treatment variance is near zero and confidence intervals will be very wide — correctly so.
- **Including mediators as confounders attenuates estimates.** NCB is partly caused by the claim experience you are studying. Draw the DAG before specifying the model.
- **CATE estimates from causal forest are unreliable below n ≈ 5,000.** Below this, honest splitting combined with 5-fold cross-fitting leaves too few training observations per tree.
- **Fit time scales with n and cv_folds.** At 100k observations with 5 folds, expect 5–15 minutes on a Databricks cluster.

---

## Notebooks and examples

| Script | What it covers |
|---|---|
| [`examples/quickstart.py`](examples/quickstart.py) | Continuous price change treatment — confounding bias, ATE, segment CATEs |
| [`examples/binary_treatment.py`](examples/binary_treatment.py) | Binary treatment — PCW vs direct channel effect on claim frequency |
| [`examples/rate_change_evaluation.py`](examples/rate_change_evaluation.py) | Post-hoc DiD evaluation of a Q3 motor rate change |

Databricks notebooks:

| Notebook | What it covers |
|---|---|
| [`notebooks/01_insurance_causal_demo.py`](notebooks/01_insurance_causal_demo.py) | Core DML, confounding bias report, sensitivity analysis |
| [`notebooks/02_autodml_demo.py`](notebooks/02_autodml_demo.py) | Riesz representer AME, dose-response curve |
| [`notebooks/03_elasticity_demo.py`](notebooks/03_elasticity_demo.py) | Renewal pricing optimisation, ENBP constraint |
| [`notebooks/04_causal_forest_hte_demo.py`](notebooks/04_causal_forest_hte_demo.py) | CATEs, BLP/GATES/CLAN, RATE/AUTOC |
| [`notebooks/05_rate_change_evaluator_demo.py`](notebooks/05_rate_change_evaluator_demo.py) | DiD and ITS post-hoc evaluation |

---

## References

1. Chernozhukov, V. et al. (2018). "Double/Debiased Machine Learning for Treatment and Structural Parameters." *The Econometrics Journal*, 21(1): C1-C68. [arXiv:1608.00060](https://arxiv.org/abs/1608.00060)
2. Bach, P. et al. (2024). "DoubleML: An Object-Oriented Implementation of Double Machine Learning." *Journal of Statistical Software*, 108(3). [docs.doubleml.org](https://docs.doubleml.org/)
3. Guelman, L. & Guillen, M. (2014). "A causal inference approach to measure price elasticity in automobile insurance." *Expert Systems with Applications*, 41(2): 387-396.
4. Cinelli, C. & Hazlett, C. (2020). "Making Sense of Sensitivity: Extending Omitted Variable Bias." *Journal of the Royal Statistical Society: Series B*, 82(1): 39-67.

---

## Part of the Burning Cost stack

Takes observational pricing or claims data. Feeds causal elasticity estimates into [insurance-optimise](https://github.com/burning-cost/insurance-optimise) (segmented rate optimisation) and [insurance-fairness](https://github.com/burning-cost/insurance-fairness) (causal bias detection). → [See the full stack](https://burning-cost.github.io/stack/)

---

## Related libraries

| Library | Description |
|---------|-------------|
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | Proxy discrimination auditing — use causal estimates to distinguish genuine rating factors from proxy discrimination |
| [insurance-conformal](https://github.com/burning-cost/insurance-conformal) | Distribution-free prediction intervals for Tweedie pricing models |
| [insurance-causal-policy](https://github.com/burning-cost/insurance-causal-policy) | SDID and Doubly Robust Synthetic Controls for rate change evaluation |
| [insurance-optimise](https://github.com/burning-cost/insurance-optimise) | Constrained rate change optimisation with FCA PS21/5 ENBP structure |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Post-bind drift detection and A/E ratio monitoring |

[All libraries](https://burning-cost.github.io) | [Discussions](https://github.com/burning-cost/insurance-causal/discussions) | [Issues](https://github.com/burning-cost/insurance-causal/issues)
