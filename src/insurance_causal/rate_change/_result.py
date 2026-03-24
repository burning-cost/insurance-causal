"""Result dataclasses for RateChangeEvaluator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DiDResult:
    """Results from a Difference-in-Differences estimation.

    Attributes
    ----------
    att : float
        Average Treatment Effect on the Treated.
    se : float
        Standard error of the ATT estimate.
    ci_lower : float
        Lower bound of 95% confidence interval.
    ci_upper : float
        Upper bound of 95% confidence interval.
    p_value : float
        Two-sided p-value for the ATT.
    n_obs : int
        Number of observations used in estimation.
    n_units : int
        Number of unique units (segments/groups).
    n_treated : int
        Number of treated units.
    n_periods : int
        Number of time periods.
    se_type : str
        Standard error type used ('cluster' or 'HC3').
    parallel_trends_f_stat : float or None
        F-statistic from joint pre-trend test.
    parallel_trends_p_value : float or None
        P-value from joint pre-trend test.
    event_study_coefs : list of float or None
        Event study coefficients for pre- and post-treatment periods.
    event_study_se : list of float or None
        Standard errors for event study coefficients.
    event_study_periods : list of int or None
        Period indices for event study plot (relative to treatment).
    """

    att: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_obs: int
    n_units: int
    n_treated: int
    n_periods: int
    se_type: str
    parallel_trends_f_stat: Optional[float] = None
    parallel_trends_p_value: Optional[float] = None
    event_study_coefs: Optional[list] = None
    event_study_se: Optional[list] = None
    event_study_periods: Optional[list] = None

    def __repr__(self) -> str:
        sig = "***" if self.p_value < 0.01 else ("**" if self.p_value < 0.05 else ("*" if self.p_value < 0.1 else ""))
        return (
            f"DiDResult(ATT={self.att:.4f}{sig}, SE={self.se:.4f}, "
            f"95% CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}], "
            f"p={self.p_value:.4f}, SE type={self.se_type}, "
            f"n_obs={self.n_obs})"
        )


@dataclass
class ITSResult:
    """Results from an Interrupted Time Series estimation.

    Attributes
    ----------
    level_shift : float
        Estimated immediate level change at intervention (beta_2).
    level_shift_se : float
        Standard error of the level shift.
    level_shift_ci_lower : float
        Lower bound of 95% CI for level shift.
    level_shift_ci_upper : float
        Upper bound of 95% CI for level shift.
    level_shift_p_value : float
        P-value for the level shift.
    slope_change : float
        Estimated change in trend after intervention (beta_3).
    slope_change_se : float
        Standard error of the slope change.
    slope_change_ci_lower : float
        Lower bound of 95% CI for slope change.
    slope_change_ci_upper : float
        Upper bound of 95% CI for slope change.
    slope_change_p_value : float
        P-value for the slope change.
    pre_trend : float
        Pre-intervention slope (beta_1).
    n_pre : int
        Number of pre-intervention periods.
    n_post : int
        Number of post-intervention periods.
    r_squared : float
        R-squared of the segmented regression.
    hac_lags : int
        Number of lags used in Newey-West HAC standard errors.
    has_seasonality : bool
        Whether seasonal quarter dummies were included.
    """

    level_shift: float
    level_shift_se: float
    level_shift_ci_lower: float
    level_shift_ci_upper: float
    level_shift_p_value: float
    slope_change: float
    slope_change_se: float
    slope_change_ci_lower: float
    slope_change_ci_upper: float
    slope_change_p_value: float
    pre_trend: float
    n_pre: int
    n_post: int
    r_squared: float
    hac_lags: int
    has_seasonality: bool

    def effect_at_k(self, k: int) -> float:
        """Causal effect k periods after intervention.

        Parameters
        ----------
        k : int
            Number of periods after the intervention (0 = intervention period).

        Returns
        -------
        float
            Estimated causal effect = level_shift + slope_change * k.
        """
        return self.level_shift + self.slope_change * k

    def __repr__(self) -> str:
        sig_l = "***" if self.level_shift_p_value < 0.01 else ("**" if self.level_shift_p_value < 0.05 else ("*" if self.level_shift_p_value < 0.1 else ""))
        sig_s = "***" if self.slope_change_p_value < 0.01 else ("**" if self.slope_change_p_value < 0.05 else ("*" if self.slope_change_p_value < 0.1 else ""))
        return (
            f"ITSResult(level_shift={self.level_shift:.4f}{sig_l}, "
            f"slope_change={self.slope_change:.4f}{sig_s}, "
            f"HAC lags={self.hac_lags}, "
            f"n_pre={self.n_pre}, n_post={self.n_post}, "
            f"R²={self.r_squared:.3f})"
        )


@dataclass
class RateChangeResult:
    """Top-level result from RateChangeEvaluator.fit().

    Attributes
    ----------
    method : str
        Estimation method used: 'DiD' or 'ITS'.
    outcome : str
        Name of the outcome variable.
    treatment_period : int or str
        The period identified as treatment onset.
    did : DiDResult or None
        DiD result (populated when method='DiD').
    its : ITSResult or None
        ITS result (populated when method='ITS').
    warnings : list of str
        Any warnings raised during estimation (staggered adoption, UK shocks, etc.).
    notes : list of str
        Methodological notes.
    """

    method: str
    outcome: str
    treatment_period: object
    did: Optional[DiDResult] = None
    its: Optional[ITSResult] = None
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def __repr__(self) -> str:
        result = self.did if self.method == "DiD" else self.its
        return (
            f"RateChangeResult(method={self.method}, outcome={self.outcome!r}, "
            f"treatment_period={self.treatment_period!r}, result={result})"
        )
