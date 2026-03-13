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

References
----------
Chernozhukov, V. et al. (2018). "Double/Debiased Machine Learning for
Treatment and Structural Parameters." The Econometrics Journal, 21(1): C1-C68.

Bach, P. et al. (2024). "DoubleML: An Object-Oriented Implementation of
Double Machine Learning in R." Journal of Statistical Software, 108(3): 1-56.

Chernozhukov et al. (2022) "Automatic Debiased Machine Learning of Causal
and Structural Effects" Econometrica 90(3):967-1027.
"""

from ._model import CausalPricingModel, AverageTreatmentEffect
from . import treatments
from . import diagnostics
from . import autodml
from . import elasticity

__all__ = [
    "CausalPricingModel",
    "AverageTreatmentEffect",
    "treatments",
    "diagnostics",
    "autodml",
    "elasticity",
]

__version__ = "0.2.0"
