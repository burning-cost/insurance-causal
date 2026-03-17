"""
Internal utilities for insurance-causal.

Not part of the public API. Subject to change between versions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import polars as pl


def to_pandas(df: "pd.DataFrame | pl.DataFrame") -> pd.DataFrame:
    """
    Convert polars or pandas DataFrame to pandas.

    DoubleML operates on pandas DataFrames and numpy arrays. This function
    handles the conversion cleanly so the public API can accept either.

    If polars is installed but the conversion fails (e.g. due to a pyarrow ABI
    conflict in some environments), the error is re-raised with context.
    """
    if isinstance(df, pd.DataFrame):
        return df

    try:
        import polars as _polars
        if isinstance(df, _polars.DataFrame):
            return df.to_pandas()
    except ImportError:
        pass
    except Exception as exc:
        raise RuntimeError(
            f"Failed to convert polars DataFrame to pandas: {exc}. "
            "This can happen when polars and pyarrow have ABI conflicts. "
            "Try passing a pandas DataFrame directly."
        ) from exc

    raise TypeError(
        f"Expected a pandas or polars DataFrame, got {type(df).__name__}."
    )


def adaptive_catboost_params(n_samples: int) -> dict:
    """
    Compute sample-size-adaptive CatBoost hyperparameters for DML nuisance models.

    The problem this solves: CatBoost at 500 iterations / depth 6 is powerful
    enough that on small samples (n ≤ 10k) it absorbs treatment signal into the
    nuisance residuals — "over-partialling". The treatment residual D̃ = D - Ê[D|X]
    ends up near-zero variance, leaving the final DML regression step with almost
    nothing to identify θ from. This produces biased, imprecise ATE estimates.

    The fix is standard regularisation: reduce model capacity (iterations, depth)
    as n shrinks. We also apply L2 leaf regularisation and subsampling to prevent
    individual trees from memorising the training fold.

    The thresholds are chosen based on the degrees of freedom available per fold
    in 5-fold cross-fitting:
        - n < 2,000  → ~1,600 training obs per fold, very small
        - 2,000–10,000 → 1,600–8,000 training obs, typical insurance small-book
        - 10,000–50,000 → standard medium insurance book
        - ≥ 50,000 → large book, full CatBoost capacity appropriate

    Parameters
    ----------
    n_samples : int
        Total number of observations in the training set.

    Returns
    -------
    dict
        CatBoost kwargs suitable for CatBoostRegressor or CatBoostClassifier.
        Always excludes loss_function and random_seed (caller's responsibility).
    """
    if n_samples < 2_000:
        # Very small: shallow trees, few iterations, aggressive regularisation.
        # At this size, CatBoost is acting more like a regularised linear model.
        return {
            "iterations": 100,
            "learning_rate": 0.10,
            "depth": 4,
            "l2_leaf_reg": 10.0,
            "subsample": 0.8,
            "colsample_bylevel": 0.8,
            "min_data_in_leaf": 5,
        }
    elif n_samples < 5_000:
        # Small: moderate regularisation. Typical for single-product analysis on
        # a small book or a sub-portfolio (e.g., young drivers only).
        return {
            "iterations": 150,
            "learning_rate": 0.08,
            "depth": 5,
            "l2_leaf_reg": 5.0,
            "subsample": 0.8,
            "colsample_bylevel": 0.9,
            "min_data_in_leaf": 5,
        }
    elif n_samples < 10_000:
        # Small-medium: light regularisation. This is the sweet spot for most
        # UK pricing teams running causal analysis on a quarterly cohort.
        return {
            "iterations": 200,
            "learning_rate": 0.07,
            "depth": 5,
            "l2_leaf_reg": 3.0,
            "subsample": 0.9,
            "colsample_bylevel": 1.0,
            "min_data_in_leaf": 5,
        }
    elif n_samples < 50_000:
        # Medium: standard settings with mild regularisation.
        return {
            "iterations": 350,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "subsample": 1.0,
            "colsample_bylevel": 1.0,
            "min_data_in_leaf": 1,
        }
    else:
        # Large: full CatBoost capacity. The original default settings.
        return {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "subsample": 1.0,
            "colsample_bylevel": 1.0,
            "min_data_in_leaf": 1,
        }


def build_catboost_regressor(
    random_state: int = 42,
    n_samples: int | None = None,
    override_params: dict | None = None,
) -> object:
    """
    Build a CatBoost regressor suitable for DML nuisance estimation.

    When ``n_samples`` is provided, applies sample-size-adaptive regularisation
    to prevent over-partialling on small datasets. This is the key fix for the
    documented issue where DML underperformed naive GLM at n ≤ 10k.

    Parameters
    ----------
    random_state : int
        Random seed for CatBoost.
    n_samples : int | None
        Number of training observations. If provided, uses adaptive params
        from ``adaptive_catboost_params()``. If None, uses the original
        large-sample defaults (500 iterations, depth 6) — preserved for
        backward compatibility when callers do not pass sample size.
    override_params : dict | None
        Any CatBoost params that explicitly override the adaptive defaults.
        Useful for power users who want to tune specific settings.

    Returns a fitted-ready CatBoostRegressor with sklearn API.
    """
    from catboost import CatBoostRegressor

    if n_samples is not None:
        params = adaptive_catboost_params(n_samples)
    else:
        # Backward-compatible defaults: matches pre-0.3.0 behaviour.
        # These are appropriate for large samples (n ≥ 50k).
        params = {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
        }

    if override_params:
        params.update(override_params)

    return CatBoostRegressor(
        loss_function="RMSE",
        random_seed=random_state,
        verbose=0,
        allow_writing_files=False,
        **params,
    )


def build_catboost_classifier(
    random_state: int = 42,
    n_samples: int | None = None,
    override_params: dict | None = None,
) -> object:
    """
    Build a CatBoost classifier for binary nuisance models (propensity).

    Used when the treatment is binary (BinaryTreatment). The propensity
    model E[D|X] is a classification problem, and CatBoostClassifier's
    Logloss objective is correct for this.

    Parameters
    ----------
    random_state : int
        Random seed for CatBoost.
    n_samples : int | None
        Number of training observations. Adaptive regularisation applied
        when provided — same thresholds as ``build_catboost_regressor``.
    override_params : dict | None
        Any CatBoost params that explicitly override the adaptive defaults.
    """
    from catboost import CatBoostClassifier

    if n_samples is not None:
        params = adaptive_catboost_params(n_samples)
    else:
        params = {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
        }

    if override_params:
        params.update(override_params)

    return CatBoostClassifier(
        loss_function="Logloss",
        random_seed=random_state,
        verbose=0,
        allow_writing_files=False,
        **params,
    )


def make_doubleml_data(
    df_pandas: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    confounder_cols: list[str],
) -> object:
    """
    Construct a DoubleMLData object from prepared pandas data.

    DoubleML's data container keeps the outcome, treatment, and features
    bundled together and validates that column names are consistent.
    """
    import doubleml as dml
    return dml.DoubleMLData(
        df_pandas[confounder_cols + [treatment_col, outcome_col]],
        y_col=outcome_col,
        d_cols=treatment_col,
        x_cols=confounder_cols,
    )


def poisson_outcome_transform(
    y: np.ndarray,
    exposure: np.ndarray | None,
) -> np.ndarray:
    """
    Transform a Poisson claim count outcome for DML nuisance estimation.

    Standard DML assumes E[Y|X] is estimated via OLS / regression. For Poisson
    claim counts with exposure, we transform to an approximately continuous
    outcome by working with the claim rate:

        y_transformed = claim_count / exposure

    This is the frequency (claims per unit exposure). The nuisance model then
    estimates E[frequency | X], and the residual frequency is the DML outcome
    residual.

    If no exposure is provided, claim counts are used directly — appropriate
    when all policies have the same exposure period.
    """
    if exposure is not None:
        if np.any(exposure <= 0):
            raise ValueError(
                "Exposure must be strictly positive for all observations."
            )
        return y / exposure
    return y.astype(float)


def gamma_outcome_transform(
    y: np.ndarray,
    exposure: np.ndarray | None,
) -> np.ndarray:
    """
    Transform a Gamma claim severity outcome.

    Severity is claim amount conditional on a claim occurring. No exposure
    adjustment is needed, but we log-transform to work with approximately
    symmetric residuals for the nuisance regression.
    """
    if np.any(y <= 0):
        raise ValueError(
            "Gamma outcome must be strictly positive. "
            "Ensure you are passing claim amounts conditional on a claim occurring."
        )
    return np.log(y)


def check_overlap(treatment_values: np.ndarray, n_bins: int = 10) -> dict:
    """
    Check treatment overlap: verify that the treatment has sufficient variation
    not explained by the confounder space.

    This is a pre-fit sanity check on the raw treatment distribution. After
    fitting, the key check is whether the residualised treatment D̃ has
    sufficient variance (low R² from the nuisance model for D).

    Returns a dict with summary statistics.
    """
    stats = {
        "n_obs": len(treatment_values),
        "mean": float(np.mean(treatment_values)),
        "std": float(np.std(treatment_values)),
        "min": float(np.min(treatment_values)),
        "p5": float(np.percentile(treatment_values, 5)),
        "p25": float(np.percentile(treatment_values, 25)),
        "p75": float(np.percentile(treatment_values, 75)),
        "p95": float(np.percentile(treatment_values, 95)),
        "max": float(np.max(treatment_values)),
    }
    return stats
