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


def build_catboost_regressor(random_state: int = 42) -> object:
    """
    Build a CatBoost regressor suitable for DML nuisance estimation.

    These settings prioritise bias reduction over speed:
    - Many iterations with low learning rate
    - Moderate depth to capture nonlinear confounding without overfitting
    - verbose=0 to suppress training output

    Returns a fitted-ready CatBoostRegressor with sklearn API.
    """
    from catboost import CatBoostRegressor
    return CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        random_seed=random_state,
        verbose=0,
        allow_writing_files=False,
    )


def build_catboost_classifier(random_state: int = 42) -> object:
    """
    Build a CatBoost classifier for binary nuisance models (propensity).

    Used when the treatment is binary (BinaryTreatment). The propensity
    model E[D|X] is a classification problem, and CatBoostClassifier's
    Logloss objective is correct for this.
    """
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        random_seed=random_state,
        verbose=0,
        allow_writing_files=False,
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
