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

Adaptive regularisation
-----------------------
CatBoost's default hyperparameters (depth=6, l2_leaf_reg=3) overfit
severely at small sample sizes (n ≤ 10,000), causing the nuisance model
to absorb too much treatment variation — the "over-partialling" problem.
The function ``adaptive_catboost_params()`` selects regularisation tier
based on sample size.  It is called automatically in ``build_nuisance_model``
when backend="catboost" and no explicit ``nuisance_params`` override is
provided.
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, PoissonRegressor, GammaRegressor
from sklearn.preprocessing import StandardScaler

from insurance_causal.autodml._types import OutcomeFamily


# ---------------------------------------------------------------------------
# Adaptive CatBoost regularisation tiers
# ---------------------------------------------------------------------------

# Each tier is a dict of CatBoost constructor kwargs.
# Tiers are listed from smallest n (most regularisation) to largest (least).
_CATBOOST_TIERS: list[tuple[int, dict]] = [
    # (min_n, params) — params apply when n < next tier's min_n
    (50_000, {
        # n >= 50k: default CatBoost params — full model capacity
        "depth": 6,
        "l2_leaf_reg": 3,
        "learning_rate": 0.05,
        "iterations": 300,
    }),
    (10_000, {
        # 10k <= n < 50k: moderate regularisation
        "depth": 4,
        "l2_leaf_reg": 5,
        "learning_rate": 0.05,
        "iterations": 300,
    }),
    (5_000, {
        # 5k <= n < 10k: stronger regularisation
        "depth": 3,
        "l2_leaf_reg": 10,
        "learning_rate": 0.03,
        "iterations": 300,
    }),
    (1_000, {
        # 1k <= n < 5k: heavy regularisation
        "depth": 2,
        "l2_leaf_reg": 20,
        "learning_rate": 0.01,
        "iterations": 200,
    }),
    (0, {
        # n < 1k: very heavy regularisation
        "depth": 2,
        "l2_leaf_reg": 50,
        "learning_rate": 0.005,
        "iterations": 100,
    }),
]


def adaptive_catboost_params(n: int) -> dict:
    """
    Return CatBoost hyperparameters appropriate for a dataset of size ``n``.

    The default CatBoost settings (depth=6, l2_leaf_reg=3) overfit at small
    sample sizes during DML cross-fitting, causing over-partialling: the
    nuisance model absorbs the treatment variation it should leave for the
    second stage.  Stronger regularisation at small n keeps the nuisance
    models from memorising the training fold.

    Parameters
    ----------
    n : int
        Total sample size (number of rows in the full dataset, before
        cross-fitting splits).

    Returns
    -------
    params : dict
        CatBoost kwargs (depth, l2_leaf_reg, learning_rate, iterations).

    Examples
    --------
    >>> adaptive_catboost_params(500)
    {'depth': 2, 'l2_leaf_reg': 50, 'learning_rate': 0.005, 'iterations': 100}
    >>> adaptive_catboost_params(20_000)
    {'depth': 4, 'l2_leaf_reg': 5, 'learning_rate': 0.05, 'iterations': 300}
    >>> adaptive_catboost_params(100_000)
    {'depth': 6, 'l2_leaf_reg': 3, 'learning_rate': 0.05, 'iterations': 300}
    """
    for min_n, params in _CATBOOST_TIERS:
        if n >= min_n:
            return dict(params)
    # Fallback — should not be reached since the last tier has min_n=0
    return dict(_CATBOOST_TIERS[-1][1])


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
    l2_leaf_reg : float
        L2 regularisation coefficient on leaf weights.  Higher values
        prevent overfitting on small samples.  The default (3) is
        CatBoost's factory default.  Use ``adaptive_catboost_params()`` to
        select an appropriate value based on sample size.
    random_state : int or None
        Random seed.
    """

    def __init__(
        self,
        outcome_family: OutcomeFamily = OutcomeFamily.GAUSSIAN,
        iterations: int = 300,
        depth: int = 6,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3.0,
        random_state: Optional[int] = None,
    ) -> None:
        self.outcome_family = outcome_family
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
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
            l2_leaf_reg=self.l2_leaf_reg,
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
    nuisance_params: Optional[Dict] = None,
    n_samples: Optional[int] = None,
    **kwargs,
) -> _NuisanceWrapper:
    """
    Factory function for nuisance outcome models.

    When ``backend="catboost"`` and neither ``nuisance_params`` nor relevant
    ``kwargs`` override the CatBoost hyperparameters, this function
    automatically applies ``adaptive_catboost_params(n_samples)`` to select
    regularisation appropriate for the dataset size.  Pass
    ``nuisance_params`` explicitly to override the adaptive defaults.

    Parameters
    ----------
    outcome_family : OutcomeFamily
        Distribution family for the outcome.
    backend : {"sklearn", "catboost", "linear"}
        Which backend to use.
    estimator : sklearn estimator, optional
        Custom base learner (only used when backend="sklearn").
    nuisance_params : dict, optional
        Explicit hyperparameter overrides for the nuisance model.  For
        the CatBoost backend, recognised keys are: ``depth``,
        ``l2_leaf_reg``, ``learning_rate``, ``iterations``.  When
        provided, these take precedence over adaptive defaults.
    n_samples : int, optional
        Total dataset size.  Used to select the adaptive regularisation
        tier when ``backend="catboost"`` and ``nuisance_params`` is None.
        If not provided, the large-sample (n >= 50k) defaults are used.
    **kwargs
        Passed to the nuisance model constructor.  Deprecated in favour
        of ``nuisance_params``; kwargs are merged in with lower priority
        than ``nuisance_params``.

    Returns
    -------
    nuisance : _NuisanceWrapper
        A fitted-ready nuisance model.
    """
    if backend == "catboost":
        # Determine effective params: adaptive defaults -> kwargs -> nuisance_params
        if n_samples is not None:
            base_params = adaptive_catboost_params(n_samples)
        else:
            # No sample size info — use large-sample defaults (depth=6, etc.)
            base_params = dict(_CATBOOST_TIERS[0][1])

        # kwargs may contain legacy overrides; nuisance_params wins over both
        effective_params = {**base_params, **kwargs}
        if nuisance_params is not None:
            effective_params.update(nuisance_params)

        return CatBoostNuisance(outcome_family=outcome_family, **effective_params)
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
