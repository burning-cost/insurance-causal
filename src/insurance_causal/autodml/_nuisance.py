"""
First-stage nuisance models for debiased ML.

These are the outcome regression E[Y | D, X] (and optionally E[D | X]) used
in the first stage of cross-fitted double-ML.  The library supports three
backends:

- sklearn (default): GradientBoostingRegressor or RandomForestRegressor
- catboost: CatBoostRegressor with Poisson/Gamma/Tweedie losses
- linear: Ridge regression for fast baselines and tests

All nuisance models are wrapped to expose a unified API:
    fit(D, X, Y, sample_weight) -> self
    predict(D, X) -> array of shape (n,)
"""
from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, PoissonRegressor, GammaRegressor
from sklearn.preprocessing import StandardScaler

from insurance_causal.autodml._types import OutcomeFamily


class _NuisanceWrapper:
    """Internal base class for nuisance outcome models."""

    def fit(
        self,
        D: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "_NuisanceWrapper":
        raise NotImplementedError

    def predict(self, D: np.ndarray, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, D: np.ndarray, X: np.ndarray) -> np.ndarray:
        return self.predict(D, X)


def _build_DX(D: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Stack [D, X] into a single float feature matrix for sklearn models."""
    D = np.asarray(D, dtype=float).reshape(-1, 1)
    X = np.asarray(X, dtype=float)
    return np.hstack([D, X])


def _build_DX_with_cats(
    D: np.ndarray, X
) -> "tuple":
    """
    Stack [D, X] preserving object/categorical columns for CatBoost.

    Returns (DX_array, cat_feature_indices) where DX_array is a numpy
    object array (for string categoricals) or float array (for purely numeric
    data), and cat_feature_indices is a list of column indices that CatBoost
    should treat as categorical.

    When X is a plain numpy float array, this is equivalent to _build_DX()
    with no categorical features identified.

    When X is a pandas DataFrame or contains object/category dtype columns,
    the categorical column indices are identified and returned so that
    CatBoost can handle them natively without label encoding.
    """
    import pandas as pd

    D_arr = np.asarray(D, dtype=float).reshape(-1, 1)
    cat_feature_indices: list = []

    if isinstance(X, pd.DataFrame):
        cat_cols = [
            i for i, dtype in enumerate(X.dtypes)
            if dtype == object or str(dtype) == "category"
        ]
        # Categorical columns shifted by 1 because D is prepended
        cat_feature_indices = [c + 1 for c in cat_cols]

        # Build a mixed-type matrix: use object dtype to preserve string cats
        if cat_cols:
            X_arr = X.to_numpy(dtype=object)
            DX = np.hstack([D_arr.astype(object), X_arr])
        else:
            X_arr = X.to_numpy(dtype=float)
            DX = np.hstack([D_arr, X_arr])
    else:
        # Plain numpy array — no categoricals available
        X_arr = np.asarray(X, dtype=float)
        DX = np.hstack([D_arr, X_arr])

    return DX, cat_feature_indices


class SklearnNuisance(_NuisanceWrapper):
    """
    Sklearn-based nuisance model wrapping a regressor that takes [D, X].

    Parameters
    ----------
    estimator : sklearn estimator, optional
        Base learner. Defaults to GradientBoostingRegressor.
    outcome_family : OutcomeFamily
        Used to choose the right sklearn GLM when estimator is None.
    scale_D : bool
        Whether to include D on its natural scale (False) or normalised.
        For GLMs we always include D on natural scale; for tree models
        it does not matter.
    """

    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        outcome_family: OutcomeFamily = OutcomeFamily.GAUSSIAN,
        scale_D: bool = False,
    ) -> None:
        self.outcome_family = outcome_family
        self.scale_D = scale_D

        if estimator is not None:
            self._base = clone(estimator)
        elif outcome_family == OutcomeFamily.POISSON:
            self._base = GradientBoostingRegressor(
                loss="squared_error", n_estimators=200, max_depth=4, random_state=0
            )
        elif outcome_family == OutcomeFamily.GAMMA:
            self._base = GradientBoostingRegressor(
                loss="squared_error", n_estimators=200, max_depth=4, random_state=0
            )
        else:
            self._base = GradientBoostingRegressor(
                n_estimators=200, max_depth=4, random_state=0
            )

        self._scaler: Optional[StandardScaler] = None
        self._is_fitted: bool = False

    def fit(
        self,
        D: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "SklearnNuisance":
        DX = _build_DX(D, X)
        Y = np.asarray(Y, dtype=float).ravel()

        # For Poisson/Gamma, clip Y to positive
        if self.outcome_family in (OutcomeFamily.POISSON, OutcomeFamily.GAMMA, OutcomeFamily.TWEEDIE):
            Y = np.clip(Y, 1e-8, None)

        kw: dict = {}
        if sample_weight is not None:
            kw["sample_weight"] = sample_weight

        self._base.fit(DX, Y, **kw)
        self._is_fitted = True
        return self

    def predict(self, D: np.ndarray, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("fit() must be called before predict().")
        DX = _build_DX(D, X)
        pred = self._base.predict(DX)

        # Ensure positive predictions for count/severity models
        if self.outcome_family in (OutcomeFamily.POISSON, OutcomeFamily.GAMMA, OutcomeFamily.TWEEDIE):
            pred = np.clip(pred, 1e-8, None)
        return pred


class CatBoostNuisance(_NuisanceWrapper):
    """
    CatBoost-based nuisance model with native Poisson/Tweedie support.

    Requires the ``catboost`` extra: ``pip install insurance-autodml[catboost]``.

    Parameters
    ----------
    outcome_family : OutcomeFamily
        Loss function: Gaussian -> RMSE, Poisson -> Poisson,
        Gamma -> not natively supported (uses RMSE), Tweedie -> Tweedie.
    iterations : int
        Number of boosting rounds.
    depth : int
        Tree depth.
    learning_rate : float
        Boosting learning rate.
    random_state : int or None
        Random seed.
    """

    def __init__(
        self,
        outcome_family: OutcomeFamily = OutcomeFamily.GAUSSIAN,
        iterations: int = 300,
        depth: int = 6,
        learning_rate: float = 0.05,
        random_state: Optional[int] = None,
    ) -> None:
        self.outcome_family = outcome_family
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self._model = None
        self._is_fitted = False

    def _build_model(self):
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError(
                "CatBoost is required for CatBoostNuisance. "
                "Install with: pip install insurance-autodml[catboost]"
            ) from exc

        loss_map = {
            OutcomeFamily.GAUSSIAN: "RMSE",
            OutcomeFamily.POISSON: "Poisson",
            OutcomeFamily.GAMMA: "RMSE",  # CatBoost has no Gamma loss
            OutcomeFamily.TWEEDIE: "Tweedie:variance_power=1.5",
        }
        loss = loss_map[self.outcome_family]
        return CatBoostRegressor(
            loss_function=loss,
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            random_seed=self.random_state,
            verbose=0,
        )

    def fit(
        self,
        D: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "CatBoostNuisance":
        DX, cat_feature_indices = _build_DX_with_cats(D, X)
        Y = np.asarray(Y, dtype=float).ravel()
        if self.outcome_family in (OutcomeFamily.POISSON, OutcomeFamily.GAMMA, OutcomeFamily.TWEEDIE):
            Y = np.clip(Y, 1e-8, None)

        self._cat_feature_indices = cat_feature_indices
        self._model = self._build_model()

        if cat_feature_indices:
            try:
                from catboost import Pool
            except ImportError as exc:
                raise ImportError(
                    "CatBoost is required for CatBoostNuisance. "
                    "Install with: pip install insurance-autodml[catboost]"
                ) from exc
            pool = Pool(
                DX, Y,
                cat_features=cat_feature_indices,
                weight=sample_weight,
            )
            self._model.fit(pool)
        else:
            kw: dict = {}
            if sample_weight is not None:
                kw["sample_weight"] = sample_weight
            self._model.fit(DX, Y, **kw)

        self._is_fitted = True
        return self

    def predict(self, D: np.ndarray, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("fit() must be called before predict().")
        cat_feature_indices = getattr(self, "_cat_feature_indices", [])
        DX, _ = _build_DX_with_cats(D, X)
        if cat_feature_indices:
            try:
                from catboost import Pool
            except ImportError as exc:
                raise ImportError(
                    "CatBoost is required for CatBoostNuisance. "
                    "Install with: pip install insurance-autodml[catboost]"
                ) from exc
            pool = Pool(DX, cat_features=cat_feature_indices)
            pred = self._model.predict(pool)
        else:
            pred = self._model.predict(DX)
        if self.outcome_family in (OutcomeFamily.POISSON, OutcomeFamily.GAMMA, OutcomeFamily.TWEEDIE):
            pred = np.clip(pred, 1e-8, None)
        return pred


def build_nuisance_model(
    outcome_family: OutcomeFamily = OutcomeFamily.GAUSSIAN,
    backend: str = "sklearn",
    estimator: Optional[BaseEstimator] = None,
    **kwargs,
) -> _NuisanceWrapper:
    """
    Factory function for nuisance outcome models.

    Parameters
    ----------
    outcome_family : OutcomeFamily
        Distribution family for the outcome.
    backend : {"sklearn", "catboost", "linear"}
        Which backend to use.
    estimator : sklearn estimator, optional
        Custom base learner (only used when backend="sklearn").
    **kwargs
        Passed to the nuisance model constructor.

    Returns
    -------
    nuisance : _NuisanceWrapper
        A fitted-ready nuisance model.
    """
    if backend == "catboost":
        return CatBoostNuisance(outcome_family=outcome_family, **kwargs)
    elif backend == "linear":
        return SklearnNuisance(
            estimator=Ridge(alpha=kwargs.get("alpha_reg", 1.0)),
            outcome_family=outcome_family,
        )
    else:
        return SklearnNuisance(
            estimator=estimator,
            outcome_family=outcome_family,
        )
