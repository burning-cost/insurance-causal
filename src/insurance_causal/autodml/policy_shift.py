"""
PolicyShiftEffect: counterfactual impact of a uniform premium shift.

This estimand answers the question: "If we increased all premiums by 5%,
what would the average outcome be?"  This is a policy-relevant question for
UK insurers preparing for regulatory pricing review or optimising portfolio
pricing.

Formally, for a shift delta (e.g. 0.05 for +5%), the estimand is:
    theta(delta) = E[Y(D * (1 + delta))] - E[Y]

which compares the counterfactual mean under the shifted premium distribution
to the observed mean.

The doubly-robust score for this functional is derived from the general
Riesz representer framework.  For a multiplicative shift D -> D*(1+delta),
the Riesz representer is:
    alpha_delta(X) = E[dg(D*(1+delta), X)/dD | X]
estimated via the same ForestRiesz approach as the AME, but with the
nuisance evaluated at the shifted treatment values.

We also provide a direct plug-in estimator:
    theta_plugin(delta) = (1/n) sum_i g_hat(D_i*(1+delta), X_i) - mean(Y)

The debiased estimator adds the first-order bias correction:
    theta_dml(delta) = theta_plugin(delta)
                       + (1/n) sum_i alpha_hat_i * (Y_i - g_hat_i)

where alpha_hat_i is the cross-fitted Riesz representer for the shift
functional at delta.
"""
from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from insurance_causal.autodml._crossfit import cross_fit_nuisance
from insurance_causal.autodml._inference import run_inference
from insurance_causal.autodml._types import EstimationResult, OutcomeFamily
from insurance_causal.autodml.riesz import ForestRiesz


class PolicyShiftEffect:
    """
    Counterfactual effect of a uniform proportional premium shift.

    Estimates E[Y(D*(1+delta))] - E[Y] for a specified delta, using a
    debiased ML approach that corrects for the regularisation bias of the
    plug-in estimator.

    Parameters
    ----------
    outcome_family : OutcomeFamily or str
        Distribution family for outcome regression.
    n_folds : int
        Cross-fitting folds.
    nuisance_backend : str
        Outcome nuisance backend.
    inference : {"eif", "bootstrap"}
        Inference method.
    ci_level : float
        Confidence level.
    random_state : int or None
        Random seed.
    """

    def __init__(
        self,
        outcome_family: Union[str, OutcomeFamily] = OutcomeFamily.GAUSSIAN,
        n_folds: int = 5,
        nuisance_backend: str = "sklearn",
        inference: str = "eif",
        ci_level: float = 0.95,
        random_state: Optional[int] = None,
    ) -> None:
        if isinstance(outcome_family, str):
            outcome_family = OutcomeFamily(outcome_family)
        self.outcome_family = outcome_family
        self.n_folds = n_folds
        self.nuisance_backend = nuisance_backend
        self.inference = inference
        self.ci_level = ci_level
        self.random_state = random_state

        self.g_hat_: Optional[np.ndarray] = None
        self.alpha_hat_: Optional[np.ndarray] = None
        self._Y: Optional[np.ndarray] = None
        self._D: Optional[np.ndarray] = None
        self._X: Optional[np.ndarray] = None
        self._sample_weight: Optional[np.ndarray] = None
        self._fold_indices = None
        self._nuisance_models = None
        self._is_fitted: bool = False

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        D: Union[np.ndarray, pd.Series],
        Y: Union[np.ndarray, pd.Series],
        exposure: Optional[Union[np.ndarray, pd.Series]] = None,
        sample_weight: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> "PolicyShiftEffect":
        """
        Fit the cross-fitted nuisance models.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Covariates.
        D : array-like of shape (n,)
            Observed treatment (premium).
        Y : array-like of shape (n,)
            Observed outcome.
        exposure : array-like of shape (n,), optional
            Exposure offset.
        sample_weight : array-like of shape (n,), optional
            Observation weights.

        Returns
        -------
        self
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        X = np.asarray(X, dtype=float)
        D = np.asarray(D, dtype=float).ravel()
        Y = np.asarray(Y, dtype=float).ravel()

        if exposure is not None:
            exposure = np.asarray(exposure, dtype=float).ravel()
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float).ravel()

        g_hat, alpha_hat, fold_indices, nuisance_models = cross_fit_nuisance(
            X=X,
            D=D,
            Y=Y,
            outcome_family=self.outcome_family,
            n_folds=self.n_folds,
            nuisance_backend=self.nuisance_backend,
            riesz_class=ForestRiesz,
            sample_weight=sample_weight,
            exposure=exposure,
            random_state=self.random_state,
        )

        self.g_hat_ = g_hat
        self.alpha_hat_ = alpha_hat
        self._Y = Y
        self._D = D
        self._X = X
        self._sample_weight = sample_weight
        self._exposure = exposure
        self._fold_indices = fold_indices
        self._nuisance_models = nuisance_models
        self._is_fitted = True
        return self

    def estimate(
        self,
        delta: float = 0.05,
    ) -> EstimationResult:
        """
        Estimate the policy shift effect for a proportional premium change.

        Parameters
        ----------
        delta : float
            Proportional premium shift, e.g. 0.05 for a 5% increase or
            -0.03 for a 3% reduction.

        Returns
        -------
        result : EstimationResult
            Debiased estimate of E[Y(D*(1+delta))] - E[Y].
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before estimate().")

        D_shifted = self._D * (1.0 + delta)

        # Predict g_hat at shifted treatment for each fold's eval observations
        g_hat_shifted = np.full(len(self._Y), np.nan)
        for (train_idx, eval_idx), nuisance in zip(self._fold_indices, self._nuisance_models):
            pred = nuisance.predict(D_shifted[eval_idx], self._X[eval_idx])
            if self._exposure is not None:
                pred = pred * self._exposure[eval_idx]
            g_hat_shifted[eval_idx] = pred

        # Doubly-robust estimator for E[Y(D*(1+delta))] - E[Y]:
        #
        #   theta_DML = mean(g_hat_shifted) + mean(alpha_hat * (Y - g_hat))
        #               - mean(Y)
        #
        # Written as a single EIF score:
        #   psi_i = g_hat_shifted_i + alpha_hat_i * (Y_i - g_hat_i) - Y_i
        #
        # Taking the mean of psi gives theta_DML directly.  Inference on
        # the mean of psi via the delta method gives the correct SE.
        psi = g_hat_shifted + self.alpha_hat_ * (self._Y - self.g_hat_) - self._Y

        est, se, ci_low, ci_high = run_inference(
            psi=psi,
            level=self.ci_level,
            inference=self.inference,
            sample_weight=self._sample_weight,
            random_state=self.random_state,
        )

        return EstimationResult(
            estimate=est,
            se=se,
            ci_low=ci_low,
            ci_high=ci_high,
            ci_level=self.ci_level,
            n_obs=len(self._Y),
            n_folds=self.n_folds,
            psi=psi,
            notes=f"delta={delta:+.3f}",
        )

    def estimate_curve(
        self,
        delta_grid: Union[np.ndarray, list],
    ) -> Dict[float, EstimationResult]:
        """
        Estimate the policy shift effect across a grid of delta values.

        Efficient: nuisance models are fitted once in ``fit()`` and reused.

        Parameters
        ----------
        delta_grid : array-like
            Grid of shift values, e.g. np.linspace(-0.10, 0.10, 21).

        Returns
        -------
        results : dict of {delta: EstimationResult}
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before estimate_curve().")

        delta_grid = np.asarray(delta_grid, dtype=float)
        return {float(d): self.estimate(delta=float(d)) for d in delta_grid}
