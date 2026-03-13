"""
Shared result types and enumerations for insurance-autodml.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class OutcomeFamily(str, Enum):
    """
    Distribution family for the outcome model.

    Gaussian : continuous outcomes (log-loss ratio, pure premium on log scale).
    Poisson  : claim counts with exposure offset.
    Gamma    : severity or pure premium (always positive).
    Tweedie  : combined frequency-severity (compound Poisson-Gamma); most
               common for pure premium modelling in UK personal lines.
    """

    GAUSSIAN = "gaussian"
    POISSON = "poisson"
    GAMMA = "gamma"
    TWEEDIE = "tweedie"


@dataclass
class EstimationResult:
    """
    Result from a scalar causal effect estimator (AME, policy shift, etc.).

    Parameters
    ----------
    estimate : float
        Point estimate of the causal functional.
    se : float
        Standard error derived from the efficient influence function (EIF)
        or bootstrap, depending on the ``inference`` argument passed to the
        estimator.
    ci_low : float
        Lower bound of the (1 - alpha) confidence interval.
    ci_high : float
        Upper bound of the (1 - alpha) confidence interval.
    ci_level : float
        Nominal coverage of the confidence interval, e.g. 0.95.
    n_obs : int
        Number of observations used in estimation.
    n_folds : int
        Number of cross-fitting folds used.
    psi : np.ndarray
        EIF scores (influence function values), length n_obs. Useful for
        further inference (e.g. clustering standard errors).
    notes : str
        Any warnings or diagnostics from the fitting procedure.
    """

    estimate: float
    se: float
    ci_low: float
    ci_high: float
    ci_level: float = 0.95
    n_obs: int = 0
    n_folds: int = 5
    psi: np.ndarray = field(default_factory=lambda: np.array([]))
    notes: str = ""

    def __repr__(self) -> str:
        return (
            f"EstimationResult(estimate={self.estimate:.4f}, "
            f"se={self.se:.4f}, "
            f"ci=[{self.ci_low:.4f}, {self.ci_high:.4f}], "
            f"n={self.n_obs})"
        )

    @property
    def pvalue(self) -> float:
        """Two-sided p-value under the null hypothesis estimate == 0."""
        from scipy import stats

        if self.se <= 0:
            return float("nan")
        z = abs(self.estimate / self.se)
        return float(2 * (1 - stats.norm.cdf(z)))

    def summary(self) -> str:
        """Return a formatted one-line summary suitable for logging."""
        stars = ""
        p = self.pvalue
        if not np.isnan(p):
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"
        return (
            f"estimate={self.estimate:+.4f}  "
            f"se={self.se:.4f}  "
            f"{int(self.ci_level*100)}% CI=[{self.ci_low:+.4f}, {self.ci_high:+.4f}]  "
            f"p={self.pvalue:.4f}{stars}"
        )


@dataclass
class SegmentResult:
    """
    Segment-level average marginal effect.

    Parameters
    ----------
    segment_name : str
        Label for this segment (e.g. ``"age_band=25-34"``).
    result : EstimationResult
        Full estimation result for this segment.
    n_obs : int
        Number of observations in this segment.
    """

    segment_name: str
    result: EstimationResult
    n_obs: int

    def __repr__(self) -> str:
        return (
            f"SegmentResult({self.segment_name!r}, "
            f"estimate={self.result.estimate:.4f}, "
            f"n={self.n_obs})"
        )


@dataclass
class DoseResponseResult:
    """
    Dose-response curve E[Y(d)] evaluated at a grid of treatment values.

    Parameters
    ----------
    d_grid : np.ndarray
        Treatment grid points (e.g. premium levels in pounds).
    ate : np.ndarray
        Estimated E[Y(d)] at each grid point.
    se : np.ndarray
        Standard error of E[Y(d)] at each grid point.
    ci_low : np.ndarray
        Lower confidence band.
    ci_high : np.ndarray
        Upper confidence band.
    ci_level : float
        Nominal coverage.
    bandwidth : float
        Kernel bandwidth used.
    n_obs : int
        Total observations used.
    """

    d_grid: np.ndarray
    ate: np.ndarray
    se: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    ci_level: float = 0.95
    bandwidth: float = 0.0
    n_obs: int = 0

    def __repr__(self) -> str:
        return (
            f"DoseResponseResult(n_grid={len(self.d_grid)}, "
            f"d=[{self.d_grid.min():.1f}, {self.d_grid.max():.1f}], "
            f"bandwidth={self.bandwidth:.3f})"
        )
