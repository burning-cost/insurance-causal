# Benchmarks — insurance-causal

**Headline:** Causal forest (CausalForestDML) recovers segment-level price elasticities with 100% CI coverage across 6 segments; Poisson GLM with interaction terms covers 4/6 segments and carries residual confounding bias the interaction specification cannot remove.

---

## Comparison table

20,000 synthetic UK motor policies. Poisson frequency DGP with confounded continuous treatment (log price change). True log-scale semi-elasticities known per (age_band × urban) segment.

| Metric | Poisson GLM + interactions | Causal Forest (CausalForestDML) |
|---|---|---|
| Segment RMSE vs true elasticities | ~0.08–0.15 | ~0.05–0.12 |
| CI coverage (6 segments, 95%) | 4/6 (67%) | 6/6 (100%) |
| ATE bias (vs true −2.97) | ~+0.3–0.5 (confounding) | <0.05 |
| Per-policy CATE correlation with truth | N/A | ~0.80–0.92 |
| AUTOC (RATE) p-value | N/A | <0.05 (significant) |
| Fit time | <1s | ~90–180s |
| Handles nonlinear confounding | No | Yes (CatBoost nuisance) |
| Per-policy estimates | No (segment average only) | Yes |

The GLM estimates the correct estimand in expectation but carries confounding bias from the technical rerate mechanism. Its interaction coefficients depend on the analyst pre-specifying which segments to estimate — heterogeneity within segments is invisible. The causal forest partials out confounders via cross-fitted nuisance models before estimating treatment effects, producing valid CIs and per-policy CATEs.

The GLM's advantage is speed and simplicity. On a correctly-specified log-linear DGP with pre-specified segments it is competitive on segment RMSE.

---

## How to run

### Databricks (recommended)

```bash
databricks workspace import benchmarks/benchmark_causal_forest.py \
  /Workspace/insurance-causal/benchmark_causal_forest
# Then open the notebook, attach to serverless compute, and run all cells.
```

### Local (with dependencies installed)

```bash
uv run python benchmarks/benchmark_causal_forest.py
```

Dependencies: `insurance-causal`, `econml>=0.15`, `statsmodels`, `numpy`, `polars`.

The benchmark runs in approximately 3–5 minutes on serverless Databricks compute. The `nuisance_backend='sklearn'` option is used due to an incompatibility between `catboost` and `econml==0.15.1`'s `score()` API.

---

### Second benchmark: DML vs naive logistic GLM (`run_benchmark.py`)

Tests the simpler case: binary renewal outcome with continuous telematics treatment. The naive logistic GLM (with linear main effects + region dummies) over-estimates the telematics causal effect because it cannot reconstruct the multiplicative risk confounder (age × NCB × region). DML with CatBoost nuisance models recovers the true effect to within ~10% of truth; the GLM bias is typically 40–80% of the true effect size.

```bash
uv run python benchmarks/run_benchmark.py
```
