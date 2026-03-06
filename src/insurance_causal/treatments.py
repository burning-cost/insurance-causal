"""
Treatment specifications for causal pricing models.

A treatment is the variable whose causal effect we want to estimate. In insurance
pricing, treatments are almost always one of three things: a continuous price
change, a binary flag (channel, discount, product type), or a generic continuous
variable (telematics score, credit score).

Treatment classes validate the treatment column, apply any necessary
transformations, and carry metadata that the main estimator uses to configure
the DoubleML problem correctly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class PriceChangeTreatment:
    """
    Continuous price change treatment, specified as a percentage.

    This is the most common treatment in insurance causal inference: we want
    the causal effect of raising or lowering price on some outcome (renewal
    conversion, claim frequency, exposure growth).

    The fundamental confounding problem: price changes are not random. High-risk
    customers receive larger premium increases. Those same customers have higher
    baseline lapse rates. Naive regression of renewal on price change gives a
    biased elasticity — DML isolates the exogenous variation in price (the part
    not explained by risk factors) to identify the causal effect.

    Parameters
    ----------
    column : str
        Name of the treatment column. Should contain percentage price changes
        (e.g. 0.05 = 5% increase, -0.10 = 10% reduction). Not log-transformed.
    scale : {"log", "linear"}
        How to transform the column before estimation.
        "log" applies log(1 + D) — appropriate when the percentage changes span
        a wide range and you want to interpret the coefficient as elasticity in
        log-log space. "linear" uses the column as-is.
    clip_percentiles : tuple[float, float]
        Clip extreme treatment values to these percentiles before fitting.
        Insurance renewal data often has extreme outliers from administrative
        corrections and re-rates applied to mid-term policies. Default: no clipping.

    Notes
    -----
    The DML coefficient θ from a PriceChangeTreatment with scale="log" is a
    semi-elasticity when the outcome is a binary renewal indicator: a 1-unit
    increase in log(1 + price_change) is associated with a θ change in the
    renewal probability. For a log outcome (log loss cost), θ is a full
    price-to-cost elasticity.
    """

    column: str
    scale: Literal["log", "linear"] = "log"
    clip_percentiles: tuple[float, float] | None = None

    @property
    def treatment_type(self) -> Literal["continuous"]:
        return "continuous"

    def transform(self, series: pd.Series) -> pd.Series:
        """Apply scale transformation to the raw treatment column."""
        values = series.copy()
        if self.clip_percentiles is not None:
            lo, hi = self.clip_percentiles
            lower = np.percentile(values.dropna(), lo * 100)
            upper = np.percentile(values.dropna(), hi * 100)
            values = values.clip(lower, upper)
        if self.scale == "log":
            values = np.log1p(values)
        return values

    def validate(self, series: pd.Series) -> None:
        """Check the treatment column looks like price changes."""
        if series.isnull().any():
            raise ValueError(
                f"Treatment column '{self.column}' contains nulls. "
                "Fill or drop them before fitting."
            )
        pct_extreme = (series.abs() > 1.0).mean()
        if pct_extreme > 0.05:
            raise ValueError(
                f"Treatment column '{self.column}': {pct_extreme:.1%} of values exceed "
                "±100%. PriceChangeTreatment expects proportional changes "
                "(0.05 = 5% increase). If your column is in basis points or percentage "
                "points, divide by 100 first."
            )


@dataclass
class BinaryTreatment:
    """
    Binary treatment: a 0/1 indicator.

    Use this for channel (aggregator vs. direct), discount flags, product type
    changes, or any intervention that was either applied or not.

    The DML estimate is the average treatment effect (ATE): the average causal
    difference in outcome between treated and untreated observations, after
    controlling for confounders. This is comparable to a GLM coefficient on
    the binary variable, but with confounding bias removed.

    Parameters
    ----------
    column : str
        Name of the binary column. Must contain only 0 and 1 (or True/False).
        The coefficient is interpreted as the effect of moving from 0 to 1.
    positive_label : str
        Human-readable label for the treated group (value=1). Used in
        diagnostics output only. Default: "treated".
    negative_label : str
        Human-readable label for the control group (value=0). Default: "control".
    """

    column: str
    positive_label: str = "treated"
    negative_label: str = "control"

    @property
    def treatment_type(self) -> Literal["binary"]:
        return "binary"

    def transform(self, series: pd.Series) -> pd.Series:
        return series.astype(float)

    def validate(self, series: pd.Series) -> None:
        if series.isnull().any():
            raise ValueError(
                f"Treatment column '{self.column}' contains nulls."
            )
        unique_vals = set(series.unique())
        if not unique_vals.issubset({0, 1, True, False, 0.0, 1.0}):
            raise ValueError(
                f"Treatment column '{self.column}' contains values other than 0/1. "
                f"Found: {unique_vals}. BinaryTreatment requires a binary indicator."
            )
        n_treated = series.sum()
        n_total = len(series)
        if n_treated < 50 or (n_total - n_treated) < 50:
            raise ValueError(
                f"Treatment column '{self.column}' has fewer than 50 observations "
                f"in one group (treated={int(n_treated)}, control={int(n_total - n_treated)}). "
                "DML requires sufficient overlap between groups."
            )


@dataclass
class ContinuousTreatment:
    """
    Generic continuous treatment.

    Use for telematics scores, credit scores, vehicle engine size, or any
    continuous variable whose causal effect you want to estimate.

    Parameters
    ----------
    column : str
        Name of the treatment column.
    standardise : bool
        If True, standardise the treatment to mean 0, SD 1 before fitting.
        Useful when the treatment is on an arbitrary scale (e.g. a telematics
        composite score from 0 to 1000). Makes the coefficient interpretable as
        "effect of a 1 standard deviation change in treatment". Default: False.
    """

    column: str
    standardise: bool = False
    _mean: float = field(default=0.0, init=False, repr=False)
    _std: float = field(default=1.0, init=False, repr=False)

    @property
    def treatment_type(self) -> Literal["continuous"]:
        return "continuous"

    def transform(self, series: pd.Series) -> pd.Series:
        if self.standardise:
            self._mean = series.mean()
            self._std = series.std()
            if self._std == 0:
                raise ValueError(
                    f"Treatment column '{self.column}' has zero variance. "
                    "Cannot standardise."
                )
            return (series - self._mean) / self._std
        return series.astype(float)

    def validate(self, series: pd.Series) -> None:
        if series.isnull().any():
            raise ValueError(
                f"Treatment column '{self.column}' contains nulls."
            )
        if series.nunique() < 10:
            import warnings
            warnings.warn(
                f"Treatment column '{self.column}' has only {series.nunique()} unique "
                "values. If this is actually a binary or ordinal variable, consider "
                "BinaryTreatment instead.",
                UserWarning,
                stacklevel=3,
            )


# Type alias for type annotations
AnyTreatment = PriceChangeTreatment | BinaryTreatment | ContinuousTreatment
