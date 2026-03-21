"""
insurance_causal.causal_forest
==============================
Heterogeneous treatment effect estimation for insurance pricing via
CausalForestDML (Athey, Tibshirani & Wager 2019) with formal HTE inference
(Chernozhukov et al. 2020/2025).

The core problem this subpackage addresses: average treatment effects hide
enormous heterogeneity in insurance portfolios. A customer with NCD=0 and
PCW channel may have 3x the price elasticity of a loyal, NCD=5 direct
customer. Using the population ATE to set price changes or discount strategies
leaves money on the table — and may inadvertently discriminate under FCA
pricing fairness requirements.

The workflow:

1. Estimate per-customer CATE with HeterogeneousElasticityEstimator.
2. Test formally that heterogeneity exists (BLP) with HeterogeneousInference.
3. Characterise which segments drive heterogeneity (GATES, CLAN).
4. Evaluate whether the CATE estimates produce a good targeting rule (RATE/AUTOC).
5. Diagnose data quality issues (overlap, residual variation) with CausalForestDiagnostics.
6. Discover data-driven subgroups via CausalClusteringAnalyzer.

Quick start
-----------
>>> from insurance_causal.causal_forest import (
...     HeterogeneousElasticityEstimator,
...     HeterogeneousInference,
...     TargetingEvaluator,
...     CausalForestDiagnostics,
...     CausalClusteringAnalyzer,
...     make_hte_renewal_data,
... )
>>> df = make_hte_renewal_data(n=10_000)
>>> confounders = ["age", "ncd_years", "vehicle_group", "channel"]
>>> est = HeterogeneousElasticityEstimator(n_estimators=200, catboost_iterations=200)
>>> est.fit(df, outcome="renewed", treatment="log_price_change",
...         confounders=confounders)
>>> cates = est.cate(df)
>>> inf = HeterogeneousInference(n_splits=100, k_groups=5)
>>> result = inf.run(df, estimator=est, cate_proxy=cates)
>>> print(result.summary())
>>> result.plot_gates()

References
----------
Athey, Tibshirani & Wager (2019). "Generalized Random Forests."
    Annals of Statistics 47(2): 1148-1178.
Chernozhukov, Demirer, Duflo & Fernandez-Val (2020/2025). "Generic Machine
    Learning Inference on Heterogeneous Treatment Effects."
Yadlowsky, Fleming, Shah, Brunskill & Wager (2025). "Evaluating Treatment
    Prioritization Rules via Rank-Weighted Average Treatment Effects."
    JASA 120(549).
"""

from insurance_causal.causal_forest.estimator import HeterogeneousElasticityEstimator
from insurance_causal.causal_forest.inference import (
    HeterogeneousInference,
    HeterogeneousInferenceResult,
    BLPResult,
    GATESResult,
    CLANResult,
)
from insurance_causal.causal_forest.targeting import (
    TargetingEvaluator,
    TargetingResult,
)
from insurance_causal.causal_forest.diagnostics import (
    CausalForestDiagnostics,
    DiagnosticsReport,
)
from insurance_causal.causal_forest.exposure import (
    build_exposure_weighted_nuisances,
    prepare_rate_outcome,
)
from insurance_causal.causal_forest.data import (
    make_hte_renewal_data,
    true_cate_by_ncd,
)
from insurance_causal.causal_forest.clustering import (
    CausalClusteringAnalyzer,
    ClusteringResult,
)

__all__ = [
    # Core estimator
    "HeterogeneousElasticityEstimator",
    # HTE inference
    "HeterogeneousInference",
    "HeterogeneousInferenceResult",
    "BLPResult",
    "GATESResult",
    "CLANResult",
    # Targeting evaluation
    "TargetingEvaluator",
    "TargetingResult",
    # Diagnostics
    "CausalForestDiagnostics",
    "DiagnosticsReport",
    # Clustering
    "CausalClusteringAnalyzer",
    "ClusteringResult",
    # Exposure weighting
    "build_exposure_weighted_nuisances",
    "prepare_rate_outcome",
    # Synthetic data
    "make_hte_renewal_data",
    "true_cate_by_ncd",
]
