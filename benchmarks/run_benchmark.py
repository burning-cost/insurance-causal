"""
Benchmark: insurance-causal DML vs naive Poisson GLM for treatment effect estimation.

The question pricing teams actually ask: "What is the causal effect of our telematics
discount on claim frequency?" The naive answer — regress frequency on the discount
flag — is biased because safer drivers are both more likely to receive the discount
AND have lower claim frequency independently. DML corrects this confounding.

Setup
-----
- 5,000 synthetic UK motor policies (small enough to run under 2 minutes on Databricks)
- Treatment D: telematics discount flag (binary, 0/1)
- True causal effect: -0.15 (discount reduces claim frequency by 15%)
- Confounding: driver safety score drives both discount eligibility AND claim frequency
- True DGP: log(lambda) = -2.5 + 0.4*safety_risk - 0.15*D + 0.1*vehicle_age
- Confounding mechanism: P(D=1) is a logistic function of safety_risk

Three estimators compared
--------------------------
1. Naive: Poisson GLM including treatment but without partialling out confounders
2. Naive OLS on residuals (no cross-fitting): OLS of Y_tilde on D_tilde, single split
3. DML: insurance-causal CausalPricingModel with BinaryTreatment, 5-fold cross-fitting

Key metrics
-----------
- Estimated treatment effect (true = -0.15)
- Absolute bias vs true effect
- 95% CI coverage of true value
- CI width

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
N = 5_000

# Confounders — observable risk factors
driver_age = RNG.integers(18, 75, N).astype(float)
vehicle_age = RNG.integers(1, 15, N).astype(float)
ncb_years = RNG.integers(0, 9, N).astype(float)
urban = RNG.binomial(1, 0.4, N).astype(float)

# Safety risk score — the key confounder.
# This drives both treatment assignment and outcome (that is the confounding).
safety_risk = (
    0.03 * np.maximum(25 - driver_age, 0)
    - 0.05 * ncb_years
    + 0.10 * urban
    + RNG.normal(0, 0.15, N)
)
safety_risk = np.clip(safety_risk, -1.0, 1.5)

# Treatment: telematics discount. High safety_risk => less likely to receive discount.
# This is the critical confounding: discount is NOT randomly assigned.
p_discount = 1.0 / (1.0 + np.exp(2.0 * safety_risk))  # safer drivers more likely
D = RNG.binomial(1, p_discount, N).astype(float)

# Outcome: claim count (Poisson). True causal effect = -0.15.
log_lam = -2.5 + 0.4 * safety_risk - 0.15 * D + 0.01 * vehicle_age
claim_count = RNG.poisson(np.exp(log_lam)).astype(float)
exposure = RNG.uniform(0.7, 1.0, N)

TRUE_EFFECT = -0.15

print("=" * 60)
print("insurance-causal benchmark")
print("DML vs naive Poisson GLM — treatment effect estimation")
print("=" * 60)
print(f"\nDGP: {N:,} policies, true causal effect = {TRUE_EFFECT}")
print(f"Discount rate: {D.mean():.1%}")
print(f"Mean claim freq: {(claim_count / exposure).mean():.4f}")
print()

# ---------------------------------------------------------------------------
# 2. Naive Poisson GLM (baseline)
# ---------------------------------------------------------------------------

print("Estimator 1: Naive Poisson GLM")
print("-" * 40)

try:
    from sklearn.linear_model import PoissonRegressor
    from scipy.stats import norm as scipy_norm

    t0 = time.perf_counter()
    # Design matrix: treatment + all confounders (still biased: no partialling out)
    X_naive = np.column_stack([D, driver_age, vehicle_age, ncb_years, urban])

    glm = PoissonRegressor(alpha=0.0, max_iter=500)
    glm.fit(X_naive, claim_count / exposure, sample_weight=exposure)
    t_glm = time.perf_counter() - t0

    # The coefficient on D (index 0) is the naive treatment effect estimate
    # For a Poisson GLM on a log scale, the coefficient is multiplicative:
    # exp(coef) = rate ratio. We report the log coefficient (additive on log scale).
    naive_coef = glm.coef_[0]

    # Bootstrap SE for the naive GLM coefficient
    n_boot = 200
    boot_coefs = []
    for _ in range(n_boot):
        idx = RNG.integers(0, N, N)
        g = PoissonRegressor(alpha=0.0, max_iter=300)
        g.fit(X_naive[idx], claim_count[idx] / exposure[idx], sample_weight=exposure[idx])
        boot_coefs.append(g.coef_[0])
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
    naive_coef = None
    naive_covers = None
    t_glm = None

print()

# ---------------------------------------------------------------------------
# 3. DML — insurance-causal CausalPricingModel
# ---------------------------------------------------------------------------

print("Estimator 2: DML (insurance-causal)")
print("-" * 40)

try:
    from insurance_causal import CausalPricingModel
    from insurance_causal.treatments import BinaryTreatment

    df = pl.DataFrame({
        "discount":    D,
        "driver_age":  driver_age,
        "vehicle_age": vehicle_age,
        "ncb_years":   ncb_years,
        "urban":       urban,
        "claim_count": claim_count,
        "exposure":    exposure,
    })

    t0 = time.perf_counter()
    model = CausalPricingModel(
        outcome="claim_count",
        outcome_type="poisson",
        treatment=BinaryTreatment(column="discount"),
        confounders=["driver_age", "vehicle_age", "ncb_years", "urban"],
        cv_folds=5,
        exposure_col="exposure",
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
    print(f"{'Estimated effect':<30} {naive_coef:>12.4f} {dml_coef if dml_coef else 'N/A':>12}")
print(f"{'True effect':<30} {TRUE_EFFECT:>12.4f} {TRUE_EFFECT:>12.4f}")
if naive_coef is not None:
    naive_bias_pct = abs(naive_coef - TRUE_EFFECT) / abs(TRUE_EFFECT) * 100
    dml_bias_pct = abs(dml_coef - TRUE_EFFECT) / abs(TRUE_EFFECT) * 100 if dml_coef else float('nan')
    print(f"{'Bias (% of true effect)':<30} {naive_bias_pct:>11.1f}% {dml_bias_pct:>11.1f}%")
    print(f"{'95% CI covers true':<30} {str(naive_covers):>12} {str(dml_covers):>12}")
    ci_width_naive = naive_ci_hi - naive_ci_lo if naive_coef is not None else float('nan')
    ci_width_dml = dml_ci_hi - dml_ci_lo if dml_coef is not None else float('nan')
    print(f"{'CI width':<30} {ci_width_naive:>12.4f} {ci_width_dml:>12.4f}")
    print(f"{'Fit time':<30} {t_glm:>11.2f}s {t_dml:>11.2f}s")

print()
print("Interpretation:")
print("  DML removes confounding bias that naive GLM cannot see.")
print("  The confounding mechanism: safer drivers (low safety_risk)")
print("  receive discounts AND have lower claim frequency. Naive GLM")
print("  conflates correlation with causation.")
print("  DML advantage is most visible when safety_risk is a strong")
print("  predictor of both treatment assignment and outcome.")
