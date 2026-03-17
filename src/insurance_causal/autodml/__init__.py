"""
insurance_causal.autodml
========================
Automatic Debiased ML via Riesz Representers for continuous treatment causal
inference in UK personal lines insurance pricing.

The central problem this subpackage solves: you have a continuous treatment
(actual premium charged, discount applied, or price index) and you want to
estimate its causal effect on claims or retention — without assuming a
parametric model for how the treatment was assigned.

Standard double-ML with continuous treatments requires estimating the
generalised propensity score (GPS), which is ill-posed and numerically
unstable in the renewal portfolios typical of UK motor and home insurance.
The Riesz representer approach avoids the GPS entirely by directly learning
the reweighting functional from data via a minimax objective.

Estimands provided
------------------
- Average Marginal Effect (AME): E[dE[Y|D,X]/dD] — price elasticity
- Dose-response curve: E[Y(d)] at specified premium levels
- Policy shift effect: counterfactual impact of raising all premiums by delta%
- Selection-corrected elasticity: handles renewal selection bias

References
----------
Chernozhukov et al. (2022) "Automatic Debiased Machine Learning of Causal
  and Structural Effects" Econometrica 90(3):967-1027.
Colangelo & Lee (2020) "Double Debiased Machine Learning Nonparametric
  Inference with Continuous Treatments" arXiv:2004.03036.
Hirshberg & Wager (2021) "Augmented minimax linear estimation"
  Annals of Statistics 49(6):3206-3227.
arXiv:2601.08643 "Automatic debiased machine learning and sensitivity
  analysis for sample selection models."
"""

from insurance_causal.autodml.dgp import SyntheticContinuousDGP
from insurance_causal.autodml._nuisance import adaptive_catboost_params
from insurance_causal.autodml.elasticity import PremiumElasticity
from insurance_causal.autodml.dose_response import DoseResponseCurve
from insurance_causal.autodml.policy_shift import PolicyShiftEffect
from insurance_causal.autodml.selection import SelectionCorrectedElasticity
from insurance_causal.autodml.report import ElasticityReport
from insurance_causal.autodml.riesz import ForestRiesz, LinearRiesz
from insurance_causal.autodml._types import (
    EstimationResult,
    SegmentResult,
    DoseResponseResult,
    OutcomeFamily,
)

__version__ = "0.3.0"

__all__ = [
    # Main estimators
    "PremiumElasticity",
    "DoseResponseCurve",
    "PolicyShiftEffect",
    "SelectionCorrectedElasticity",
    # Reporting
    "ElasticityReport",
    # Riesz regressors (exposed for custom use)
    "ForestRiesz",
    "LinearRiesz",
    # Data generation
    "SyntheticContinuousDGP",
    # Result types
    "EstimationResult",
    "SegmentResult",
    "DoseResponseResult",
    "OutcomeFamily",
    # Utilities
    "adaptive_catboost_params",
    # Version
    "__version__",
]
