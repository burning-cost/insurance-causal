"""
Rate change evaluation module for insurance-causal.

Post-hoc causal attribution of insurance rate changes using
Difference-in-Differences or Interrupted Time Series.

Public API
----------
RateChangeEvaluator : Main class for estimating rate change effects.
make_rate_change_data : Synthetic panel data generator (for testing/demos).
make_its_data : Synthetic time-series data generator (for ITS testing/demos).
RateChangeResult : Result dataclass returned by RateChangeEvaluator.summary().
DiDResult : Detailed DiD results (in RateChangeResult.method_detail).
ITSResult : Detailed ITS results (in RateChangeResult.method_detail).
UK_INSURANCE_SHOCKS : Dict of known UK insurance market shocks for confounder warnings.
"""

from ._evaluator import RateChangeEvaluator
from ._data import make_rate_change_data, make_its_data
from ._result import RateChangeResult, DiDResult, ITSResult
from ._shocks import UK_INSURANCE_SHOCKS

__all__ = [
    "RateChangeEvaluator",
    "make_rate_change_data",
    "make_its_data",
    "RateChangeResult",
    "DiDResult",
    "ITSResult",
    "UK_INSURANCE_SHOCKS",
]
