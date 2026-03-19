"""
Cross-fitting infrastructure for debiased ML.

Cross-fitting splits the data into K folds, fits nuisance models on the
training folds, and generates out-of-fold predictions for the evaluation
fold.  This breaks the regularisation bias that arises when nuisance models
are fitted and evaluated on the same data.

The function ``cross_fit_nuisance`` returns:
  - g_hat : out-of-fold E[Y | D, X] predictions
  - alpha_hat : out-of-fold Riesz representer predictions
  - fold_indices : list of (train_idx, eval_idx) tuples for downstream use

Adaptive regularisation
-----------------------
When ``nuisance_backend="catboost"``, the CatBoost hyperparameters are
selected automatically based on the total sample size ``n`` via
``adaptive_catboost_params()``.  This prevents over-partialling at small
sample sizes (n <= 10,000) where default CatBoost settings overfit the
training fold.  Pass ``nuisance_params`` to override the adaptive defaults.
"""
from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Optional, Tuple, Type

import numpy as np
from sklearn.model_selection import KFold

from insurance_causal.autodml._nuisance import _NuisanceWrapper, build_nuisance_model
from insurance_causal.autodml._types import OutcomeFamily
from insurance_causal.autodml.riesz import ForestRiesz, LinearRiesz

def validate_inputs(
    X: np.ndarray,
    D: np.ndarray,
    Y: np.ndarray,
    *,
    allow_nan_Y: bool = False,
) -> None:
    """
    Validate primary inputs to autodml entry points.

    Raises ValueError on NaN in X or D (always illegal), and on NaN in Y
    unless allow_nan_Y=True (SelectionCorrectedElasticity handles NaN Y
    by coercing to 0 with a warning — all other estimators require clean Y).

    Parameters
    ----------
    X : array of shape (n, p)
        Covariates.
    D : array of shape (n,)
        Treatment.
    Y : array of shape (n,)
        Outcome.
    allow_nan_Y : bool
        If True, skip the NaN check for Y (caller handles it downstream).

    Raises
    ------
    ValueError
        If any NaN values are found in the inputs.
    """
    if np.any(np.isnan(X)):
        nan_count = int(np.isnan(X).sum())
        raise ValueError(
            f"X contains {nan_count} NaN value(s). Impute or drop rows with "
            "missing covariates before calling fit()."
        )
    if np.any(np.isnan(D)):
        nan_count = int(np.isnan(D).sum())
        raise ValueError(
            f"D (treatment) contains {nan_count} NaN value(s). The treatment "
            "must be fully observed for all rows. Drop or impute before calling fit()."
        )
    if not allow_nan_Y and np.any(np.isnan(Y)):
        nan_count = int(np.isnan(Y).sum())
        raise ValueError(
            f"Y (outcome) contains {nan_count} NaN value(s). The outcome must be "
            "fully observed. For selection models where Y is unobserved for "
            "non-renewers, use SelectionCorrectedElasticity and set Y=0 for "
            "non-renewers (not NaN)."
        )


def cross_fit_nuisance(
    X: np.ndarray,
    D: np.ndarray,
    Y: np.ndarray,
    outcome_family: OutcomeFamily = OutcomeFamily.GAUSSIAN,
    n_folds: int = 5,
    nuisance_backend: str = "sklearn",
    riesz_class: type = ForestRiesz,
    riesz_kwargs: Optional[dict] = None,
    nuisance_estimator=None,
    sample_weight: Optional[np.ndarray] = None,
    exposure: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
    nuisance_params: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray]], List[_NuisanceWrapper]]:
    """
    K-fold cross-fitting of nuisance models and Riesz representer.

    Parameters
    ----------
    X : array of shape (n, p)
        Covariates.
    D : array of shape (n,)
        Continuous treatment (e.g. premium).
    Y : array of shape (n,)
        Outcome (e.g. claim indicator, pure premium).
    outcome_family : OutcomeFamily
        Distribution family for outcome regression.
    n_folds : int
        Number of cross-fitting folds (typically 5).
    nuisance_backend : {"sklearn", "catboost", "linear"}
        Backend for the outcome nuisance model.
    riesz_class : class
        Riesz regressor class to instantiate (ForestRiesz or LinearRiesz).
    riesz_kwargs : dict, optional
        Keyword arguments passed to riesz_class().
    nuisance_estimator : sklearn estimator, optional
        Custom base learner for the nuisance model.
    sample_weight : array of shape (n,), optional
        Per-observation weights (e.g. policy count).
    exposure : array of shape (n,), optional
        Exposure offset for Poisson/Tweedie models (e.g. years at risk).
        Passed as a weight proxy: Y is divided by exposure for fitting and
        multiplied back in predictions.
    random_state : int or None
        Random seed for fold splits.
    nuisance_params : dict, optional
        Explicit hyperparameter overrides for the nuisance model.  For the
        CatBoost backend, recognised keys are: ``depth``, ``l2_leaf_reg``,
        ``learning_rate``, ``iterations``.  When provided, these override
        the adaptive defaults selected by ``adaptive_catboost_params()``.

        Example — force strong regularisation regardless of n::

            cross_fit_nuisance(..., nuisance_params={"depth": 2, "l2_leaf_reg": 50})

    Returns
    -------
    g_hat : array of shape (n,)
        Out-of-fold nuisance outcome predictions E[Y | D, X].
    alpha_hat : array of shape (n,)
        Out-of-fold Riesz representer predictions alpha(X).
    fold_indices : list of (train_idx, eval_idx)
        Fold split indices.
    nuisance_models : list of fitted _NuisanceWrapper
        One per fold (useful for dose-response prediction at new D values).
    """
    X = np.asarray(X, dtype=float)
    D = np.asarray(D, dtype=float).ravel()
    Y = np.asarray(Y, dtype=float).ravel()
    n = len(Y)

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float).ravel()
    if exposure is not None:
        exposure = np.asarray(exposure, dtype=float).ravel()
        exposure = np.clip(exposure, 1e-8, None)

    if riesz_kwargs is None:
        riesz_kwargs = {}

    g_hat = np.full(n, np.nan)
    alpha_hat = np.full(n, np.nan)
    fold_indices: List[Tuple[np.ndarray, np.ndarray]] = []
    nuisance_models: List[_NuisanceWrapper] = []

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for fold_idx, (train_idx, eval_idx) in enumerate(kf.split(X)):
        X_tr, X_ev = X[train_idx], X[eval_idx]
        D_tr, D_ev = D[train_idx], D[eval_idx]
        Y_tr, Y_ev = Y[train_idx], Y[eval_idx]

        sw_tr = sample_weight[train_idx] if sample_weight is not None else None
        exp_tr = exposure[train_idx] if exposure is not None else None
        exp_ev = exposure[eval_idx] if exposure is not None else None

        # Adjust Y for exposure: model Y/exposure, then rescale predictions
        if exp_tr is not None:
            Y_tr_fit = Y_tr / exp_tr
        else:
            Y_tr_fit = Y_tr

        # Fit nuisance outcome model.
        # Pass n (full dataset size) so adaptive_catboost_params() can select
        # the correct regularisation tier.  Per-fold sample size would give a
        # misleading picture — the tier should reflect the total portfolio size.
        nuisance = build_nuisance_model(
            outcome_family=outcome_family,
            backend=nuisance_backend,
            estimator=nuisance_estimator,
            nuisance_params=nuisance_params,
            n_samples=n,
        )
        nuisance.fit(D_tr, X_tr, Y_tr_fit, sample_weight=sw_tr)
        nuisance_models.append(nuisance)

        # Out-of-fold predictions
        g_ev = nuisance.predict(D_ev, X_ev)
        if exp_ev is not None:
            g_ev = g_ev * exp_ev
        g_hat[eval_idx] = g_ev

        # Fit Riesz representer on training fold
        # nuisance_fn for Riesz is always the rate-scale nuisance model;
        # when exposure is present we use the rate model (Y/exposure) so
        # the Riesz targets z = dg/dD are on the same scale as g_hat / exposure.
        def nuisance_fn(D_in, X_in, _n=nuisance):
            return _n.predict(D_in, X_in)

        riesz = riesz_class(**riesz_kwargs)
        sw_full_tr = sw_tr
        riesz.fit(X_tr, D_tr, nuisance_fn, sample_weight=sw_full_tr)

        alpha_hat[eval_idx] = riesz.predict(X_ev)
        fold_indices.append((train_idx, eval_idx))

    if np.any(np.isnan(g_hat)) or np.any(np.isnan(alpha_hat)):
        warnings.warn(
            "NaN values found in cross-fitted predictions. Check your data for "
            "extreme values or consider adjusting hyperparameters.",
            RuntimeWarning,
            stacklevel=2,
        )

    return g_hat, alpha_hat, fold_indices, nuisance_models


def compute_dg_dD(
    D: np.ndarray,
    X: np.ndarray,
    fold_indices: List[Tuple[np.ndarray, np.ndarray]],
    nuisance_models: List[_NuisanceWrapper],
    exposure: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute out-of-fold finite-difference derivative dg/dD at observed (D, X).

    Uses a central finite-difference approximation with step size
    eps = 1e-3 * (D.max() - D.min()).  The derivative is computed from the
    nuisance model fitted on the training fold for each evaluation observation,
    maintaining the out-of-fold property required for debiased ML validity.

    When exposure is provided the nuisance model returns the rate g(D,X)/exp,
    so the derivative is also on the rate scale.  The EIF uses dg/dD on
    the same scale as g_hat, so we rescale by exposure when needed.

    Parameters
    ----------
    D : array of shape (n,)
        Observed treatment values.
    X : array of shape (n, p)
        Covariates.
    fold_indices : list of (train_idx, eval_idx)
        Fold splits from cross_fit_nuisance.
    nuisance_models : list of _NuisanceWrapper
        One fitted model per fold.
    exposure : array of shape (n,), optional
        Exposure offset. When provided the derivative is rescaled from rate
        to count scale per the convention used in g_hat.

    Returns
    -------
    dg_dD : array of shape (n,)
        Finite-difference derivative of E[Y | D, X] with respect to D.
    """
    D = np.asarray(D, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    n = len(D)

    d_range = D.max() - D.min()
    eps = 1e-3 * d_range if d_range > 0 else 1e-3

    dg_dD = np.full(n, np.nan)

    for (train_idx, eval_idx), nuisance in zip(fold_indices, nuisance_models):
        D_ev = D[eval_idx]
        X_ev = X[eval_idx]

        g_plus = nuisance.predict(D_ev + eps, X_ev)
        g_minus = nuisance.predict(D_ev - eps, X_ev)
        deriv = (g_plus - g_minus) / (2.0 * eps)

        if exposure is not None:
            exp_ev = np.asarray(exposure, dtype=float)[eval_idx]
            deriv = deriv * exp_ev

        dg_dD[eval_idx] = deriv

    return dg_dD


def compute_ame_scores(
    Y: np.ndarray,
    g_hat: np.ndarray,
    alpha_hat: np.ndarray,
    dg_dD: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """
    Compute the AME point estimate and EIF scores from cross-fitted outputs.

    The Neyman orthogonal score for the AME is (Chernozhukov et al. 2022, eq. 2.7):

        psi_i = alpha_hat_i * (Y_i - g_hat_i) + (dg/dD)_i

    The first term debiases for regularisation error in the outcome nuisance.
    The second term is the plug-in gradient — the derivative of the conditional
    mean outcome with respect to treatment, evaluated at the observed (D_i, X_i).

    This is computed by finite difference from the nuisance model via
    ``compute_dg_dD()``. Pass the result as ``dg_dD``.

    Parameters
    ----------
    Y : array of shape (n,)
        Observed outcomes.
    g_hat : array of shape (n,)
        Out-of-fold nuisance outcome predictions.
    alpha_hat : array of shape (n,)
        Out-of-fold Riesz representer predictions.
    dg_dD : array of shape (n,), optional
        Finite-difference derivative dg/dD at observed (D, X).
        Should be computed via ``compute_dg_dD()``.  If None, a warning
        is raised and alpha_hat is used as a (biased) proxy — this is
        incorrect and retained only for backward compatibility.
    sample_weight : array of shape (n,), optional
        Observation weights.

    Returns
    -------
    ame : float
        Debiased AME estimate.
    psi : array of shape (n,)
        EIF scores.
    """
    Y = np.asarray(Y, dtype=float).ravel()
    g_hat = np.asarray(g_hat, dtype=float).ravel()
    alpha_hat = np.asarray(alpha_hat, dtype=float).ravel()

    if dg_dD is None:
        warnings.warn(
            "dg_dD not provided to compute_ame_scores. "
            "Using alpha_hat as a proxy for the plug-in gradient — this is "
            "incorrect and will produce biased AME estimates. "
            "Compute dg_dD via compute_dg_dD() and pass it explicitly.",
            UserWarning,
            stacklevel=2,
        )
        plug_in = alpha_hat
    else:
        plug_in = np.asarray(dg_dD, dtype=float).ravel()

    # EIF score: alpha(X) * (Y - g(D,X)) + dg/dD(D,X)
    # First term debiases for nuisance regularisation error.
    # Second term is the plug-in derivative of the conditional mean.
    psi = alpha_hat * (Y - g_hat) + plug_in

    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float)
        w = w / w.mean()
        ame = float(np.average(psi, weights=w))
    else:
        ame = float(np.mean(psi))

    return ame, psi
