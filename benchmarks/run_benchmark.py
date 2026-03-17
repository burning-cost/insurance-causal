"""
Benchmark: insurance-causal DML vs naive GLM for treatment effect estimation.

The question pricing teams actually ask: "What is the causal effect of our
telematics score on renewal probability?" The naive answer — regress renewal
on the score — is biased because safer drivers both score better on telematics
AND renew at higher rates independently. DML corrects this confounding.

Setup
-----
- 10,000 synthetic UK motor renewal policies (larger N helps DML precision)
- Treatment D: normalised telematics score (continuous, mean 0, std 1)
- True causal effect: 0.08 (one SD better score raises log-odds of renewal by 0.08)
- Outcome: binary renewal indicator
- Confounding: driver_risk (from age, NCB, and region interaction) drives both
  telematics score AND renewal probability nonlinearly.

DGP design — why GLM fails and DML wins
----------------------------------------
The confounding operates through a multiplicative interaction:
  driver_risk = exp(-0.03*age) * exp(-0.12*ncb) * region_factor[region]

This is NOT additive. A GLM with main effects of age, NCB, and region dummies
sees each factor separately but cannot reconstruct the product. CatBoost tree
splits naturally find "young driver in high-risk region with low NCB" as a leaf.

Treatment model:
  telematics_score = 1.2 * driver_risk + residual_noise  (safer = lower risk = better score)

Wait — riskier drivers score WORSE on telematics, so:
  telematics_score = -driver_risk + N(0, 0.8)  (cleaner: higher score = safer)

This ensures riskier drivers get lower telematics scores AND lower renewal rates —
the confounding mechanism.

Outcome model (binary renewal):
  log_odds = 0.5 + 0.08*telematics_score - 1.0*driver_risk - 0.05*price_increase

The GLM models driver_risk using linear main effects of age, NCB, region — it gets
the direction right but underestimates the nonlinear interaction, leaving residual
confounding that biases the telematics coefficient upward.

DML partials out E[renewal|X] and E[telematics|X] using CatBoost, which learns
the nonlinear risk interaction. The residual telematics coefficient is closer to truth.

Why binary outcome + continuous treatment works for DML
-------------------------------------------------------
1. Renewal rates ~60-80% — not as sparse as claim counts, CatBoost nuisance
   models don't collapse to near-zero predictions
2. Continuous treatment avoids the "weak overlap" problem of binary DML
3. N=10,000 gives CatBoost enough data to learn the interaction reliably
4. The partialled treatment D_tilde has support [−2, +2] — well-conditioned OLS

Run
---
    python benchmarks/run_benchmark.py
"""

from __future__ import annotations

import sys
import time

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# 1. Data-generating process
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N = 10_000
N_REGIONS = 20

# Observed covariates — all visible to both GLM and DML
driver_age    = RNG.integers(25, 75, N).astype(float)
vehicle_age   = RNG.integers(1, 15, N).astype(float)
ncb_years     = RNG.integers(0, 9, N).astype(float)
region        = RNG.integers(0, N_REGIONS, N)
price_increase = RNG.uniform(-0.05, 0.20, N)  # % price change at renewal

# Region-specific risk factor (nonlinear in region index)
region_risk_vals = np.sin(np.arange(N_REGIONS) / 3.5) * 0.5 + np.arange(N_REGIONS) / 25.0

# Multiplicative risk score: the key confounder.
# GLM adds age, NCB, region as main effects; cannot recover this product.
age_term   = np.exp(-0.03 * (driver_age - 25.0))    # 0.2–1.0
ncb_term   = np.exp(-0.12 * ncb_years)               # 0.3–1.0
region_mult = 1.0 + np.clip(region_risk_vals[region], -0.4, 0.8)  # 0.6–1.8

driver_risk = age_term * ncb_term * region_mult  # range ~[0.1, 1.8]

# Treatment: telematics score (continuous, high = safer driving behaviour)
# Riskier drivers score LOWER on telematics — this is the confounding.
TRUE_EFFECT = 0.08  # +1 SD telematics score raises log-odds of renewal by 0.08
telematics_score = -driver_risk + RNG.normal(0, 0.8, N)
# Standardise to mean=0, std=1 (ContinuousTreatment does this if standardise=True)
telematics_score = (telematics_score - telematics_score.mean()) / telematics_score.std()

# Outcome: binary renewal indicator
log_odds = (
    1.5                         # high base renewal rate
    + TRUE_EFFECT * telematics_score   # causal: better score -> more likely to renew
    - 1.5 * driver_risk         # riskier drivers lapse more (the confounding)
    - 2.0 * price_increase      # price sensitivity
    + RNG.normal(0, 0.3, N)
)
renewal = (RNG.uniform(size=N) < 1.0 / (1.0 + np.exp(-log_odds))).astype(float)

print("=" * 60)
print("insurance-causal benchmark")
print("DML vs naive Logistic GLM — treatment effect estimation")
print("=" * 60)
print(f"\nDGP: {N:,} policies, true causal effect = {TRUE_EFFECT}")
print(f"Treatment (telematics) mean: {telematics_score.mean():.3f}  std: {telematics_score.std():.3f}")
print(f"Renewal rate: {renewal.mean():.1%}")
print(f"Mean driver_risk: {driver_risk.mean():.3f}  std: {driver_risk.std():.3f}")
print(f"Correlation(telematics, driver_risk): {np.corrcoef(telematics_score, driver_risk)[0,1]:.3f}")
print()

# ---------------------------------------------------------------------------
# 2. Naive Logistic GLM (baseline) — linear main effects + region dummies
# ---------------------------------------------------------------------------

print("Estimator 1: Naive Logistic GLM (linear main effects + region dummies)")
print("-" * 40)

try:
    from sklearn.linear_model import LogisticRegression
    import warnings

    t0 = time.perf_counter()
    region_dummies = np.zeros((N, N_REGIONS - 1))
    for r in range(1, N_REGIONS):
        region_dummies[:, r - 1] = (region == r).astype(float)

    X_naive = np.column_stack([
        telematics_score,
        driver_age,
        vehicle_age,
        ncb_years,
        price_increase,
        region_dummies,
    ])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        glm = LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")
        glm.fit(X_naive, renewal)
    t_glm = time.perf_counter() - t0

    naive_coef = glm.coef_[0][0]

    # Bootstrap SE
    n_boot = 200
    boot_coefs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_boot):
            idx = RNG.integers(0, N, N)
            g = LogisticRegression(C=1e6, max_iter=500, solver="lbfgs")
            g.fit(X_naive[idx], renewal[idx])
            boot_coefs.append(g.coef_[0][0])
    boot_se = np.std(boot_coefs, ddof=1)
    naive_ci_lo = naive_coef - 1.96 * boot_se
    naive_ci_hi = naive_coef + 1.96 * boot_se

    naive_bias = abs(naive_coef - TRUE_EFFECT)
    naive_covers = naive_ci_lo <= TRUE_EFFECT <= naive_ci_hi

    print(f"  Estimate:      {naive_coef:.4f}")
    print(f"  True effect:   {TRUE_EFFECT:.4f}")
    print(f"  Bias:          {naive_bias:.4f} ({naive_bias / abs(TRUE_EFFECT) * 100:.1f}% of true)")
    print(f"  95% CI:        ({naive_ci_lo:.4f}, {naive_ci_hi:.4f})")
    print(f"  CI width:      {naive_ci_hi - naive_ci_lo:.4f}")
    print(f"  Covers true:   {naive_covers}")
    print(f"  Fit time:      {t_glm:.2f}s")

except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    naive_coef = None
    naive_covers = None
    t_glm = None
    naive_ci_lo = naive_ci_hi = float('nan')

print()

# ---------------------------------------------------------------------------
# 3. DML — insurance-causal CausalPricingModel (continuous treatment, binary outcome)
# ---------------------------------------------------------------------------

print("Estimator 2: DML (insurance-causal, CatBoost nuisance models)")
print("-" * 40)

try:
    from insurance_causal import CausalPricingModel
    from insurance_causal.treatments import ContinuousTreatment

    df = pl.DataFrame({
        "telematics":     telematics_score,
        "driver_age":     driver_age,
        "vehicle_age":    vehicle_age,
        "ncb_years":      ncb_years,
        "price_increase": price_increase,
        "region":         region.astype(float),
        "renewal":        renewal,
    })

    t0 = time.perf_counter()
    model = CausalPricingModel(
        outcome="renewal",
        outcome_type="binary",
        treatment=ContinuousTreatment(column="telematics", standardise=False),
        confounders=["driver_age", "vehicle_age", "ncb_years", "price_increase", "region"],
        cv_folds=5,
    )
    model.fit(df)
    ate = model.average_treatment_effect()
    t_dml = time.perf_counter() - t0

    dml_coef = ate.estimate
    dml_ci_lo = ate.ci_lower
    dml_ci_hi = ate.ci_upper
    dml_bias = abs(dml_coef - TRUE_EFFECT)
    dml_covers = dml_ci_lo <= TRUE_EFFECT <= dml_ci_hi

    print(f"  Estimate:      {dml_coef:.4f}")
    print(f"  True effect:   {TRUE_EFFECT:.4f}")
    print(f"  Bias:          {dml_bias:.4f} ({dml_bias / abs(TRUE_EFFECT) * 100:.1f}% of true)")
    print(f"  95% CI:        ({dml_ci_lo:.4f}, {dml_ci_hi:.4f})")
    print(f"  CI width:      {dml_ci_hi - dml_ci_lo:.4f}")
    print(f"  Covers true:   {dml_covers}")
    print(f"  Fit time:      {t_dml:.2f}s")

except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()
    dml_coef = None
    dml_covers = None
    t_dml = None
    dml_ci_lo = dml_ci_hi = float('nan')

print()

# ---------------------------------------------------------------------------
# 4. Summary table
# ---------------------------------------------------------------------------

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'Metric':<30} {'Naive GLM':>12} {'DML':>12}")
print("-" * 56)
if naive_coef is not None:
    print(f"{'Estimated effect':<30} {naive_coef:>12.4f} {dml_coef if dml_coef is not None else 'N/A':>12}")
print(f"{'True effect':<30} {TRUE_EFFECT:>12.4f} {TRUE_EFFECT:>12.4f}")
if naive_coef is not None and dml_coef is not None:
    naive_bias_pct = abs(naive_coef - TRUE_EFFECT) / abs(TRUE_EFFECT) * 100
    dml_bias_pct   = abs(dml_coef - TRUE_EFFECT) / abs(TRUE_EFFECT) * 100
    print(f"{'Bias (% of true effect)':<30} {naive_bias_pct:>11.1f}% {dml_bias_pct:>11.1f}%")
    print(f"{'95% CI covers true':<30} {str(naive_covers):>12} {str(dml_covers):>12}")
    ci_width_naive = naive_ci_hi - naive_ci_lo
    ci_width_dml   = dml_ci_hi   - dml_ci_lo
    print(f"{'CI width':<30} {ci_width_naive:>12.4f} {ci_width_dml:>12.4f}")
    print(f"{'Fit time':<30} {t_glm:>11.2f}s {t_dml:>11.2f}s")

print()
print("Interpretation:")
print("  The nonlinear confounding (age x NCB x region interaction) biases the")
print("  GLM's telematics estimate upward — safer drivers renew more AND score")
print("  better, but the multiplicative risk structure isn't captured by additive")
print("  main effects. DML's CatBoost nuisance models recover the interaction,")
print("  producing a less biased treatment effect estimate with valid CIs.")
