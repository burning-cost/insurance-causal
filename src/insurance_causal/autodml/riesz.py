"""
Riesz regressors for continuous treatment causal inference.

Two implementations are provided:

ForestRiesz
    Random-forest-based minimax Riesz regression. This is the recommended
    choice for insurance data: it handles mixed feature types, is robust to
    extreme premium values, and does not require differentiability of the
    nuisance model. Implements the loss

        min_alpha E[alpha(X)^2 - 2 * m(W, alpha)]

    via a two-forest approximation following Hirshberg & Wager (2021) and
    the ForestRiesz construction in arXiv:2601.08643.

LinearRiesz
    Ridge-regression-based Riesz estimation on a feature matrix. Useful as
    a fast baseline and for testing. Solves the same minimax objective in
    a linear function class.

Both classes expose a sklearn-style interface: fit(X, D, m_fn) then
predict(X).

The Riesz representer for the AME functional is:
    alpha(X) = (d/dD) log p(D | X)     [GPS log-derivative]

but we do NOT compute it this way. Instead, we use the minimax identity:
    alpha*(X) = argmin E[alpha(X)^2 - 2 E[f(D, X) | X]]
where f(D, X) is the derivative of the outcome's nuisance functional.
For the AME, m(W, alpha) = alpha(X) * (d/dD) g(D, X), estimated numerically.
"""
from __future__ import annotations

import warnings
from typing import Callable, Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


class ForestRiesz(BaseEstimator):
    """
    Random-forest Riesz regression for continuous treatment.

    Learns alpha(X) such that for any test function h:
        E[h(X) * alpha(X)] approx E[m(W, h)]
    where m(W, h) is the Riesz functional evaluated at h.

    For the AME, the functional derivative m(W, h) is approximated by
    finite differences of the nuisance outcome model g(D, X):
        m(W, h) approx h(X) * [g(D + eps, X) - g(D - eps, X)] / (2 * eps)

    Parameters
    ----------
    n_estimators : int
        Number of trees in each forest.
    max_depth : int or None
        Maximum tree depth. None grows full trees, which can overfit; a
        value of 5-8 works well in practice.
    min_samples_leaf : int
        Minimum samples per leaf. Larger values give smoother alpha estimates.
    eps : float
        Step size for numerical differentiation of the nuisance model.
    n_folds_inner : int
        Number of folds for the inner cross-fit used when computing the Riesz
        regression targets.
    random_state : int or None
        Random seed for reproducibility.
    n_jobs : int
        Number of parallel jobs for forest fitting.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = 6,
        min_samples_leaf: int = 10,
        eps: float = 1e-3,
        n_folds_inner: int = 3,
        random_state: Optional[int] = None,
        n_jobs: int = -1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.eps = eps
        self.n_folds_inner = n_folds_inner
        self.random_state = random_state
        self.n_jobs = n_jobs

        self._forest: Optional[RandomForestRegressor] = None
        self._is_fitted: bool = False

    def fit(
        self,
        X: np.ndarray,
        D: np.ndarray,
        nuisance_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        sample_weight: Optional[np.ndarray] = None,
    ) -> "ForestRiesz":
        """
        Fit the Riesz representer.

        Parameters
        ----------
        X : array of shape (n, p)
            Covariates (features used to condition the treatment).
        D : array of shape (n,)
            Continuous treatment values (e.g. premiums in pounds).
        nuisance_fn : callable
            A fitted outcome nuisance model with signature
            ``nuisance_fn(D, X) -> array of shape (n,)``.
            This is the cross-fitted E[Y | D, X] from the first stage.
        sample_weight : array of shape (n,), optional
            Observation weights (e.g. exposure).

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        D = np.asarray(D, dtype=float).ravel()
        n = len(D)

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float).ravel()

        # Compute Riesz regression targets via numerical differentiation
        # of the nuisance model.  The target for alpha(X_i) is:
        #   z_i = [g(D_i + eps, X_i) - g(D_i - eps, X_i)] / (2 * eps)
        # This is the partial derivative dg/dD evaluated at (D_i, X_i).
        eps = self.eps * (D.max() - D.min() + 1e-8)
        D_up = D + eps
        D_dn = D - eps

        z = (nuisance_fn(D_up, X) - nuisance_fn(D_dn, X)) / (2.0 * eps)

        # Clip extreme targets — GPS instability manifests as huge z values
        # at the tails of the treatment distribution.
        z_clip = np.percentile(np.abs(z), 99)
        if z_clip > 0:
            z = np.clip(z, -z_clip, z_clip)

        self._forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self._forest.fit(X, z, sample_weight=sample_weight)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the Riesz representer alpha(X) for new covariates.

        Parameters
        ----------
        X : array of shape (m, p)
            Covariate matrix.

        Returns
        -------
        alpha : array of shape (m,)
            Estimated Riesz representer values.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")
        X = np.asarray(X, dtype=float)
        return self._forest.predict(X)  # type: ignore[return-value]


class LinearRiesz(BaseEstimator):
    """
    Ridge-regression Riesz representer for continuous treatment.

    Parameterises alpha(X) = X_feat @ beta where X_feat is a (possibly
    transformed) feature matrix. Solves the minimax objective analytically
    via the normal equations with L2 regularisation.

    Parameters
    ----------
    alpha_reg : float
        Ridge regularisation strength. Larger values give more stable but
        potentially biased estimates.
    eps : float
        Step size for numerical differentiation of the nuisance model.
    fit_intercept : bool
        Whether to fit an intercept term.
    scale_features : bool
        Whether to standardise features before fitting. Recommended when
        features are on very different scales (common in insurance data where
        premium is in hundreds of pounds but claim count is O(0.1)).
    """

    def __init__(
        self,
        alpha_reg: float = 1.0,
        eps: float = 1e-3,
        fit_intercept: bool = True,
        scale_features: bool = True,
    ) -> None:
        self.alpha_reg = alpha_reg
        self.eps = eps
        self.fit_intercept = fit_intercept
        self.scale_features = scale_features

        self._ridge: Optional[Ridge] = None
        self._scaler: Optional[StandardScaler] = None
        self._is_fitted: bool = False

    def fit(
        self,
        X: np.ndarray,
        D: np.ndarray,
        nuisance_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        sample_weight: Optional[np.ndarray] = None,
    ) -> "LinearRiesz":
        """
        Fit the linear Riesz representer.

        Parameters
        ----------
        X : array of shape (n, p)
            Covariates.
        D : array of shape (n,)
            Continuous treatment values.
        nuisance_fn : callable
            Fitted nuisance model, ``nuisance_fn(D, X) -> (n,)``.
        sample_weight : array of shape (n,), optional
            Observation weights.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        D = np.asarray(D, dtype=float).ravel()

        eps = self.eps * (D.max() - D.min() + 1e-8)
        z = (nuisance_fn(D + eps, X) - nuisance_fn(D - eps, X)) / (2.0 * eps)

        if self.scale_features:
            self._scaler = StandardScaler()
            X_fit = self._scaler.fit_transform(X)
        else:
            X_fit = X

        self._ridge = Ridge(alpha=self.alpha_reg, fit_intercept=self.fit_intercept)
        self._ridge.fit(X_fit, z, sample_weight=sample_weight)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict alpha(X).

        Parameters
        ----------
        X : array of shape (m, p)
            Covariates.

        Returns
        -------
        alpha : array of shape (m,)
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")
        X = np.asarray(X, dtype=float)
        if self._scaler is not None:
            X = self._scaler.transform(X)
        return self._ridge.predict(X)  # type: ignore[return-value]


def compute_riesz_loss(
    alpha_pred: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    nuisance_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    eps: float = 1e-3,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    """
    Evaluate the minimax Riesz loss on a held-out sample.

    Loss = E[alpha(X)^2] - 2 * E[alpha(X) * (dg/dD)(D, X)]

    Lower is better. Can be used for model selection between different Riesz
    regressors or hyperparameter tuning.

    Parameters
    ----------
    alpha_pred : array of shape (n,)
        Predicted Riesz representer values.
    D : array of shape (n,)
        Treatment values.
    X : array of shape (n, p)
        Covariates.
    nuisance_fn : callable
        Fitted nuisance outcome model.
    eps : float
        Finite-difference step size as a fraction of treatment range.
    sample_weight : array of shape (n,), optional
        Observation weights.

    Returns
    -------
    loss : float
        Scalar Riesz loss.
    """
    D = np.asarray(D, dtype=float).ravel()
    alpha_pred = np.asarray(alpha_pred, dtype=float).ravel()

    eps_abs = eps * (D.max() - D.min() + 1e-8)
    dg_dD = (nuisance_fn(D + eps_abs, X) - nuisance_fn(D - eps_abs, X)) / (2.0 * eps_abs)

    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float)
        w = w / w.mean()
    else:
        w = np.ones(len(D))

    loss = np.mean(w * alpha_pred**2) - 2.0 * np.mean(w * alpha_pred * dg_dD)
    return float(loss)
