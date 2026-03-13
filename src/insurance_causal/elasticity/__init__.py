"""
insurance_causal.elasticity
===========================
Causal price elasticity estimation and FCA-compliant renewal pricing
optimisation for UK personal lines insurance.

Quick start
-----------
>>> from insurance_causal.elasticity.data import make_renewal_data
>>> from insurance_causal.elasticity.fit import RenewalElasticityEstimator
>>> from insurance_causal.elasticity.optimise import RenewalPricingOptimiser
>>> from insurance_causal.elasticity.surface import ElasticitySurface
>>> from insurance_causal.elasticity.demand import demand_curve, plot_demand_curve
>>> from insurance_causal.elasticity.diagnostics import ElasticityDiagnostics

>>> df = make_renewal_data(n=10_000)
>>> confounders = ["age", "ncd_years", "vehicle_group", "region", "channel"]
>>> est = RenewalElasticityEstimator()
>>> est.fit(df, confounders=confounders)
>>> ate, lb, ub = est.ate()
>>> print(f"ATE: {ate:.3f}  95% CI: [{lb:.3f}, {ub:.3f}]")
"""

from insurance_causal.elasticity.fit import RenewalElasticityEstimator
from insurance_causal.elasticity.surface import ElasticitySurface
from insurance_causal.elasticity.optimise import RenewalPricingOptimiser
from insurance_causal.elasticity.diagnostics import ElasticityDiagnostics, TreatmentVariationReport
from insurance_causal.elasticity.demand import demand_curve, plot_demand_curve
from insurance_causal.elasticity.data import make_renewal_data, true_gate_by_ncd, true_gate_by_age

__all__ = [
    "RenewalElasticityEstimator",
    "ElasticitySurface",
    "RenewalPricingOptimiser",
    "ElasticityDiagnostics",
    "TreatmentVariationReport",
    "demand_curve",
    "plot_demand_curve",
    "make_renewal_data",
    "true_gate_by_ncd",
    "true_gate_by_age",
]
