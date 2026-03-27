# Changelog

## v0.6.2 (2026-03-27)

- fix: re-pin `scipy>=1.12,<1.16` — the upper bound was dropped in v0.6.0 when the rate_change subpackage was added. scipy 1.16 removed `_lazywhere` which statsmodels (a doubleml dependency) still imports, causing `ImportError` on fresh install. Constraint remains until statsmodels releases a fix.
- fix: bump `statsmodels>=0.14.4` (was `>=0.14`) — 0.14.4 removes the `_lazywhere` import internally; the floor clarifies which statsmodels versions are known-safe.
- fix: `examples/quickstart.py` line 91: `RNG.uniform(N)` -> `RNG.uniform(size=N)`. The positional-argument form passes N as the upper bound of a scalar draw, producing a single float instead of an array — which crashes with numpy>=2.0.
- docs: add one-liner hook to README for Python/insurance audience.
- docs: clarify README quickstart output is deterministic with seed=42 but may vary by scipy/catboost version.
- docs: add dependency note about scipy<1.16 constraint to Install section.

## v0.6.0 (2026-03-25)

- feat: add `rate_change` sub-package — post-hoc causal evaluation of historical insurance rate changes
- `RateChangeEvaluator`: unified API for DiD (difference-in-differences) and ITS (interrupted time series) estimation
  - `method="auto"` selects DiD when a control group is present, ITS otherwise
  - `fit(df)` accepts policy-level or segment-period data; aggregates automatically
  - `summary()` returns `RateChangeResult` with ATT, SE, 95% CI, p-value, and ATT as % of pre-treatment mean
  - `plot_event_study()` — event study chart with pre/post CIs (DiD)
  - `plot_pre_post()` — observed vs counterfactual trend (both methods)
  - `parallel_trends_test()` — pre-treatment event study coefficients and joint F-test
- `DiDEstimator`: two-way fixed effects with clustered SEs (Cameron & Miller 2015); staggered adoption detection (Goodman-Bacon 2021); parallel trends pre-test
- `ITSEstimator`: segmented regression with level shift + slope change; quarterly seasonality; Newey-West HAC SEs (Kontopantelis et al. 2015)
- `make_rate_change_data` and `make_its_data`: synthetic panel generators for testing and demos
- `UK_INSURANCE_SHOCKS`: reference dict of known UK market shocks (Ogden rate changes, whiplash reform, FCA pricing review) for confounder warnings
- 84 tests passing

## v0.5.3 (2026-03-23)
- fix: pin scipy<1.16 — scipy 1.16 removed `_lazywhere` which statsmodels still imports via doubleml, causing ImportError on fresh install

## v0.5.1 (2026-03-23)
- fix: bump doubleml floor to >=0.10.0 — older versions used check_X_y(force_all_finite=...) which was removed in scikit-learn 1.8; doubleml 0.10.0+ is compatible

## v0.5.0 (2026-03-22) [unreleased]
- Remove emoji from discussion CTA
- Rebuild benchmarks: stronger confounding in DML, GLM interaction baseline for causal forest
- docs: fix README review issues
- fix: bump development status classifier from Alpha to Beta

## v0.5.0 (2026-03-21)
- fix: remove duplicate [dependency-groups] dev section, update version pins
- fix: update stale docstring claiming CausalForest is unimplemented
- Add cross-links to related libraries in README
- docs: replace pip install with uv add in README
- Add blog post link and community CTA to README
- docs(v0.5.0): add CausalClusteringAnalyzer to README and implement clustering module
- Add CausalClusteringAnalyzer: forest-kernel spectral clustering for CATE subgroup discovery
- docs: add causal_forest subpackage documentation and update version history (v0.4.0)
- fix: replace np.trapz with compat shim for numpy 2.0+ (v0.4.3)
- fix: resolve 33 CI failures in causal_forest subpackage on Python 3.11 (v0.4.2)
- fix: resolve 33 CI errors and 1 flaky test for Python 3.11 (v0.4.1)
- Add causal forest GATE vs uniform ATE benchmark with full HTE workflow
- Add causal_forest subpackage v0.4.0: HTE estimation for insurance pricing
- Add DualSelectionDML: ATE under multivariate ordinal sample selection
- fix: repair broken test import of adaptive_dml_catboost_params
- fix: update quickstart to n=50000 and show actual DML output (Estimate: -0.1993)
- fix(v0.3.3): P1 small-segment overfit, DRLearner error msg, NaN input validation
- Fix: label-encode string categorical confounders in fit (v0.3.2)
- Add discussions link and star CTA
- feat(v0.3.0): adaptive CatBoost regularisation to fix DML over-partialling at small n
- v0.3.0: sample-size-adaptive nuisance model parameters
- Rebuild benchmark DGP with nonlinear confounding; honest Performance section
- Add benchmarks/run_benchmark.py and update Performance section
- Fix five reviewer-identified issues in README and source
- Add Google Colab quickstart notebook and Open-in-Colab badge
- Add CONTRIBUTING.md with bug reporting, feature request, and dev setup guidance
- Update Performance section with post-Phase-98 benchmark results
- fix: resolve 3 P0 and 4 P1 critical bugs in autodml subpackage (v0.2.2)
- Fix docs workflow: use pdoc not pdoc3 syntax (no --html flag)
- Add pdoc API documentation workflow with GitHub Pages deployment
- Add consulting CTA to README
- fix: add econml to dependency-groups, add importorskip guards in elasticity tests
- docs: add Databricks notebook link
- fix(docs): correct DoseResponseCurve API in README quick-start
- Add Related Libraries section to README
- fix: update cross-references to consolidated repos
- fix(docs): add synthetic data to autodml quick-start
- Add CITATION.cff for academic and software citation
- fix: make quick-start self-contained with inline synthetic data
- fix: update polars floor to >=1.0 and fix project URLs
- docs: add Performance section with benchmark results to README
- Add Performance section to README
- docs: fix installation instructions and add merged-from provenance
- Add Databricks benchmark notebook: DML vs naive Poisson GLM
- fix: add __version__ to autodml subpackage and fix np.ptp deprecation
- feat: absorb insurance-autodml and insurance-elasticity as subpackages (v0.2.0)
- Fix sensitivity_analysis: bias bound scales with SE not ATE (Imbens 2003)

## v0.1.0 (2026-03-09)
- Add GitHub Actions CI workflow and test badge
- fix: update org URLs from burningcost to burning-cost
- Add badges and cross-links to README
- Replace pip install with uv add in README installation instructions
- Clean up README: remove em dashes and drop LightGBM/XGBoost mentions
- Fix test compatibility issues found during Databricks CI
- Initial release of insurance-causal v0.1.0
