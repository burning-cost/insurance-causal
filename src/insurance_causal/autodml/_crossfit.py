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
"""
from __future__ import annotations

import warnings
from typing import Callable, List, Optional, Tuple, Type

import numpy as np
from sklearn.model_selection import KFold

from insurance_causal.autodml._nuisance import _NuisanceWrapper, build_nuisance_model
from insurance_causal.autodml._types import OutcomeFamily
from insurance_causal.autodml.riesz import ForestRiesz, LinearRiesz


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

        # Fit nuisance outcome model
        nuisance = build_nuisance_model(
            outcome_family=outcome_family,
            backend=nuisance_backend,
            estimator=nuisance_estimator,
        )
        nuisance.fit(D_tr, X_tr, Y_tr_fit, sample_weight=sw_tr)
        nuisance_models.append(nuisance)

        # Out-of-fold predictions
        g_ev = nuisance.predict(D_ev, X_ev)
        if exp_ev is not None:
            g_ev = g_ev * exp_ev
        g_hat[eval_idx] = g_ev

        # Fit Riesz representer on training fold
        # (nuisance_fn for Riesz is fitted on training data only)
        if exp_tr is not None:
            # For Riesz, we want dg/dD where g is the rate (Y/exposure model)
            def nuisance_fn(D_in, X_in, _n=nuisance):
                return _n.predict(D_in, X_in)
        else:
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


def compute_ame_scores(
    Y: np.ndarray,
    g_hat: np.ndarray,
    alpha_hat: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """
    Compute the AME point estimate and EIF scores from cross-fitted outputs.

    The Neyman orthogonal score for the AME is:
        psi_i = alpha_hat_i * (Y_i - g_hat_i) + (dg/dD)_i

    Since we absorbed (dg/dD) into alpha_hat during the Riesz regression,
    the score simplifies to:
        psi_i = alpha_hat_i * (Y_i - g_hat_i) + alpha_hat_i * g_hat_derivative_i

    In practice, with ForestRiesz, alpha_hat already approximates dg/dD, so
    a good approximation to the EIF is:
        psi_i = alpha_hat_i * (Y_i - g_hat_i) + alpha_hat_i

    But more accurately, we use the direct score:
        psi_i = alpha_hat_i * (Y_i - g_hat_i) + alpha_hat_i

    The debiased AME estimate is E[psi_i].

    Parameters
    ----------
    Y : array of shape (n,)
        Observed outcomes.
    g_hat : array of shape (n,)
        Out-of-fold nuisance outcome predictions.
    alpha_hat : array of shape (n,)
        Out-of-fold Riesz representer predictions.
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

    # EIF score: alpha(X) * (Y - g(D,X)) + alpha(X)
    # The first term corrects for first-stage bias;
    # the second term is the plug-in gradient estimate.
    psi = alpha_hat * (Y - g_hat) + alpha_hat

    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float)
        w = w / w.mean()
        ame = float(np.average(psi, weights=w))
    else:
        ame = float(np.mean(psi))

    return ame, psi
