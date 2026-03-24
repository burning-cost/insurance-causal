"""rate_change: post-hoc causal evaluation of insurance rate changes.

Provides DiD (Difference-in-Differences) and ITS (Interrupted Time Series)
estimation to answer: 'We changed rates on segment X. What actually happened?'

DiD is used when a control group exists (segment-specific rate change).
ITS is the fallback when the entire book was treated.

Public API
----------
RateChangeEvaluator
    Main class. Fit with .fit(df), inspect with .summary(), .parallel_trends_test(),
    .plot_event_study(), .plot_pre_post(), .plot_its().

make_rate_change_data
    Synthetic insurance panel data generator for testing and demos.

UK_INSURANCE_SHOCKS
    Dictionary of known UK market events that can confound rate change estimates.

Examples
--------
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
"""

from ._evaluator import RateChangeEvaluator
from ._data import make_rate_change_data
from ._result import RateChangeResult, DiDResult, ITSResult
from ._shocks import UK_INSURANCE_SHOCKS, check_shock_proximity

__all__ = [
    "RateChangeEvaluator",
    "make_rate_change_data",
    "RateChangeResult",
    "DiDResult",
    "ITSResult",
    "UK_INSURANCE_SHOCKS",
    "check_shock_proximity",
]
