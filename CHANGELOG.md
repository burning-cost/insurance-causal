# Changelog

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
