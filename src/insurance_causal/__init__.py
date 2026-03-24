"""
insurance-causal: causal inference for insurance pricing.

A thin, opinionated wrapper over DoubleML that gives pricing actuaries a clean
interface to Double Machine Learning (Chernozhukov et al., 2018) without
requiring knowledge of econometrics.

The core use case: you have observational insurance data where the treatment
(price change, channel, discount flag) was not randomly assigned — it was
determined by risk factors and commercial decisions. Naive regression gives
biased estimates of the treatment effect. DML removes this confounding bias
by partialling out the influence of observed confounders from both the outcome
and the treatment before regressing one residual on the other.

Subpackages
-----------
autodml
    Automatic Debiased ML via Riesz Representers for continuous treatment
    causal inference. Avoids GPS estimation via minimax Riesz regression.
    Key classes: PremiumElasticity, DoseResponseCurve, PolicyShiftEffect,
    SelectionCorrectedElasticity.

elasticity
    Causal price elasticity estimation and FCA-compliant renewal pricing
    optimisation. Uses econML CausalForest for heterogeneous treatment
    effects. Key classes: RenewalElasticityEstimator, RenewalPricingOptimiser,
    ElasticitySurface.

causal_forest
    Heterogeneous treatment effect estimation via CausalForestDML with formal
    BLP/GATES/CLAN inference (Chernozhukov et al. 2020/2025) and RATE/AUTOC
    targeting evaluation (Yadlowsky et al. 2025).
    Key classes: HeterogeneousElasticityEstimator, HeterogeneousInference,
    TargetingEvaluator, CausalForestDiagnostics.

rate_change
    Post-hoc causal evaluation of insurance rate changes using DiD and ITS.
    Answers: 'We changed rates on segment X in January. What actually happened
    to conversion and loss ratio?'
    Key classes: RateChangeEvaluator.
    Key functions: make_rate_change_data.

Quick start
-----------
>>> from insurance_causal import CausalPricingModel
>>> from insurance_causal.treatments import PriceChangeTreatment
>>>
>>> model = CausalPricingModel(
...     outcome="renewal",
...     outcome_type="binary",
...     treatment=PriceChangeTreatment(column="pct_price_change"),
...     confounders=["age", "vehicle_age", "postcode_band", "ncb"],
... )
>>> model.fit(df)
>>> ate = model.average_treatment_effect()
>>> print(ate)

# AutoDML subpackage
>>> from insurance_causal.autodml import PremiumElasticity
>>> model = PremiumElasticity(outcome_family="poisson", n_folds=5)
>>> model.fit(X, D, Y)
>>> result = model.estimate()

# Elasticity subpackage
>>> from insurance_causal.elasticity import RenewalElasticityEstimator
>>> est = RenewalElasticityEstimator()
>>> est.fit(df, confounders=confounders)
>>> ate, lb, ub = est.ate()

# Causal forest HTE subpackage
>>> from insurance_causal.causal_forest import (
...     HeterogeneousElasticityEstimator,
...     HeterogeneousInference,
...     make_hte_renewal_data,
... )
>>> df = make_hte_renewal_data(n=10_000)
>>> est = HeterogeneousElasticityEstimator(n_estimators=200)
>>> est.fit(df, confounders=["age", "ncd_years", "channel"])
>>> cates = est.cate(df)

# Rate change evaluation subpackage
>>> from insurance_causal.rate_change import RateChangeEvaluator, make_rate_change_data
>>> df = make_rate_change_data(n_segments=40, true_att=-0.05, seed=0)
>>> evaluator = RateChangeEvaluator(
...     outcome_col='outcome',
...     treatment_period=9,
...     unit_col='segment',
...     weight_col='earned_exposure',
... )
>>> result = evaluator.fit(df)
>>> print(result.summary())

References
----------
Chernozhukov, V. et al. (2018). "Double/Debiased Machine Learning for
Treatment and Structural Parameters." The Econometrics Journal, 21(1): C1-C68.

Bach, P. et al. (2024). "DoubleML: An Object-Oriented Implementation of
Double Machine Learning in R." Journal of Statistical Software, 108(3): 1-56.

Chernozhukov et al. (2022) "Automatic Debiased Machine Learning of Causal
and Structural Effects" Econometrica 90(3):967-1027.

Athey, Tibshirani & Wager (2019). "Generalized Random Forests."
    Annals of Statistics 47(2): 1148-1178.

Callaway, B. & Sant'Anna, P.H.C. (2021). "Difference-in-Differences with
    Multiple Time Periods." Journal of Econometrics 225(2): 200-230.

Goodman-Bacon, A. (2021). "Difference-in-differences with variation in
    treatment timing." Journal of Econometrics 225(2): 254-277.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("insurance-causal")
except PackageNotFoundError:
    __version__ = "0.0.0"  # not installed

from ._model import CausalPricingModel, AverageTreatmentEffect
from . import treatments
from . import diagnostics
from . import autodml
from . import elasticity
from . import causal_forest
from . import rate_change
from .rate_change import RateChangeEvaluator, make_rate_change_data

__all__ = [
    "CausalPricingModel",
    "AverageTreatmentEffect",
    "treatments",
    "diagnostics",
    "autodml",
    "elasticity",
    "causal_forest",
    "rate_change",
    "RateChangeEvaluator",
    "make_rate_change_data",
]
