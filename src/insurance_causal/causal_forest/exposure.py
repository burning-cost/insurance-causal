"""
Exposure-weighted nuisance model helpers for Poisson/rate outcomes.

Insurance claims data is almost always rate data: we observe the number of
claims for a policy over an exposure period (typically 1 year, but can be
fractional for mid-term adjustments, cancellations, or new business with
partial-year exposure).

The standard approach for causal forests on rate outcomes:
1. Transform Y -> Y/exposure (claims per unit exposure).
2. Fit all nuisance models (outcome, treatment) with sample_weight=exposure.
3. Fit the causal forest on Y/exposure with sample_weight=exposure.

This is the exposure-weighted offset approach. It gives consistent estimates
of the causal effect on the log-rate, which is the standard actuarial estimand
for frequency models.

References
----------
Neyman (1923) / Rubin (1974) potential outcomes framework.
Wooldridge (2010) Chapter 18 — Poisson with exposure.
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


def build_exposure_weighted_nuisances(
    outcome_model: str = "catboost",
    binary_outcome: bool = False,
    catboost_iterations: int = 500,
    random_state: int = 42,
) -> tuple[object, object]:
    """Build CatBoost nuisance models for Poisson/rate outcomes.

    Returns outcome and treatment nuisance models suitable for use with
    CausalForestDML when the outcome is a claims rate (count / exposure).

    For Poisson outcomes, both nuisance models are regressors. The caller
    is responsible for:
    - Dividing Y by exposure before passing to estimator.fit().
    - Passing exposure as sample_weight to estimator.fit().

    Parameters
    ----------
    outcome_model:
        ``"catboost"`` (default) uses CatBoostRegressor. Pass any
        sklearn-compatible regressor to override.
    binary_outcome:
        If True, use CatBoostClassifier for the outcome nuisance model.
        Not typical for Poisson rate outcomes; included for completeness.
    catboost_iterations:
        Training iterations for CatBoost nuisance models.
    random_state:
        Random seed.

    Returns
    -------
    tuple of (model_y, model_t)
        Outcome and treatment nuisance models, ready to pass to
        CausalForestDML(model_y=model_y, model_t=model_t).

    Examples
    --------
    >>> from insurance_causal.causal_forest.exposure import (
    ...     build_exposure_weighted_nuisances
    ... )
    >>> model_y, model_t = build_exposure_weighted_nuisances(
    ...     binary_outcome=False, catboost_iterations=200
    ... )
    """
    model_y = _build_outcome_nuisance(outcome_model, binary_outcome, catboost_iterations, random_state)
    model_t = _build_treatment_nuisance(catboost_iterations, random_state)
    return model_y, model_t


def prepare_rate_outcome(
    Y: np.ndarray,
    exposure: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Divide count outcome by exposure and validate.

    Parameters
    ----------
    Y:
        Claim counts (non-negative integers or floats).
    exposure:
        Policy exposure in years (strictly positive).

    Returns
    -------
    tuple of (Y_rate, exposure)
        Y_rate = Y / exposure, and the exposure array (for use as
        sample_weight in model fitting).

    Raises
    ------
    ValueError
        If any exposure value is non-positive.
    """
    Y = np.asarray(Y, dtype=float)
    exposure = np.asarray(exposure, dtype=float)

    if np.any(exposure <= 0):
        raise ValueError(
            "All exposure values must be strictly positive. "
            "Found non-positive values. Check for mid-term cancellations "
            "with zero or negative exposure; exclude these rows."
        )

    if np.any(Y < 0):
        raise ValueError(
            "Outcome Y contains negative values. "
            "For claim counts, Y must be non-negative."
        )

    # Warn if exposure contains very small values (< 0.01 years ~ 4 days)
    # — these are likely data issues that inflate the rate
    tiny_exposure = np.sum(exposure < 0.01)
    if tiny_exposure > 0:
        warnings.warn(
            f"{tiny_exposure} rows have exposure < 0.01 years (< 4 days). "
            "These very short exposures will produce extremely high claim rates "
            "and may dominate the weighted nuisance fit. Consider excluding or "
            "setting a minimum exposure floor (e.g. 30 days = 0.082 years).",
            UserWarning,
            stacklevel=2,
        )

    Y_rate = Y / exposure
    return Y_rate, exposure


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_outcome_nuisance(
    model_spec: Union[str, object],
    binary_outcome: bool,
    catboost_iterations: int,
    random_state: int,
) -> object:
    """Build outcome nuisance model."""
    if not (isinstance(model_spec, str) and model_spec == "catboost"):
        return model_spec

    try:
        if binary_outcome:
            from catboost import CatBoostClassifier
            return CatBoostClassifier(
                iterations=catboost_iterations,
                verbose=0,
                random_seed=random_state,
                eval_metric="Logloss",
            )
        else:
            from catboost import CatBoostRegressor
            return CatBoostRegressor(
                iterations=catboost_iterations,
                verbose=0,
                random_seed=random_state,
                loss_function="RMSE",
            )
    except ImportError:
        warnings.warn(
            "CatBoost not installed. Falling back to GradientBoosting.",
            stacklevel=3,
        )
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        if binary_outcome:
            return GradientBoostingClassifier(n_estimators=100, random_state=random_state)
        else:
            return GradientBoostingRegressor(n_estimators=100, random_state=random_state)


def _build_treatment_nuisance(catboost_iterations: int, random_state: int) -> object:
    """Build treatment nuisance model (always a regressor)."""
    try:
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            iterations=catboost_iterations,
            verbose=0,
            random_seed=random_state,
            loss_function="RMSE",
        )
    except ImportError:
        warnings.warn(
            "CatBoost not installed. Falling back to GradientBoostingRegressor.",
            stacklevel=3,
        )
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(n_estimators=100, random_state=random_state)
