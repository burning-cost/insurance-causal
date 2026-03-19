"""
SelectionCorrectedElasticity: AME with renewal selection bias correction.

The fundamental problem in UK motor/home renewal pricing:

- At renewal, insurers observe outcomes (claims) only for policies that
  renew.  Non-renewers lapse and their outcomes are unobserved.
- Higher premiums cause higher lapse rates.
- The observed claim experience is therefore a selected sample, biased
  towards lower-risk retained policyholders.

If we naively estimate E[Y | D, X] from the observed renewals, we understate
the causal effect of premium increases because the retained sample is
increasingly adversarial-risk-selected as premiums rise.

Formally, let S_i = 1 if policy i renews (outcome observed), 0 otherwise.
We observe (X_i, D_i, Y_i * S_i, S_i) but not Y_i when S_i = 0.

The identification strategy follows arXiv:2601.08643:
1. The selection probability pi(X, D) = P(S=1 | X, D) is modelled.
2. The Riesz representer is extended to jointly account for treatment
   and selection.
3. Sensitivity bounds assess robustness to unobserved selection confounders.

The corrected EIF score is:
    psi_i = S_i / pi_hat_i * alpha_hat_i * (Y_i - g_hat_i)
            + alpha_hat_i

where g_hat_i = E[Y | D_i, X_i, S=1] (outcome model fitted on renewals only).

Note: this requires the assumption of no unobserved confounders of the
selection (conditional on X and D).  The sensitivity analysis below
tests robustness to violations of this assumption.
"""
from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold

from insurance_causal.autodml._crossfit import cross_fit_nuisance, validate_inputs
from insurance_causal.autodml._inference import run_inference
from insurance_causal.autodml._types import EstimationResult, OutcomeFamily
from insurance_causal.autodml.riesz import ForestRiesz


class SelectionCorrectedElasticity:
    """
    Price elasticity estimator corrected for renewal selection bias.

    Handles the missing-outcome problem in UK motor/home renewal portfolios:
    claim outcomes are only observed for policies that renew.

    Parameters
    ----------
    outcome_family : OutcomeFamily or str
        Distribution family for outcome regression (fitted on renewals only).
    n_folds : int
        Cross-fitting folds.
    nuisance_backend : str
        Backend for outcome nuisance model.
    selection_estimator : sklearn estimator, optional
        Model for P(S=1 | X, D).  Defaults to GradientBoostingClassifier.
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
        selection_estimator=None,
        inference: str = "eif",
        ci_level: float = 0.95,
        random_state: Optional[int] = None,
    ) -> None:
        if isinstance(outcome_family, str):
            outcome_family = OutcomeFamily(outcome_family)
        self.outcome_family = outcome_family
        self.n_folds = n_folds
        self.nuisance_backend = nuisance_backend
        self.selection_estimator = selection_estimator
        self.inference = inference
        self.ci_level = ci_level
        self.random_state = random_state

        self.g_hat_: Optional[np.ndarray] = None
        self.alpha_hat_: Optional[np.ndarray] = None
        self.pi_hat_: Optional[np.ndarray] = None
        self._Y: Optional[np.ndarray] = None
        self._S: Optional[np.ndarray] = None
        self._D: Optional[np.ndarray] = None
        self._X: Optional[np.ndarray] = None
        self._sample_weight: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    def _build_selection_model(self):
        if self.selection_estimator is not None:
            from sklearn.base import clone
            return clone(self.selection_estimator)
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=4, random_state=self.random_state
        )

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        D: Union[np.ndarray, pd.Series],
        Y: Union[np.ndarray, pd.Series],
        S: Union[np.ndarray, pd.Series],
        exposure: Optional[Union[np.ndarray, pd.Series]] = None,
        sample_weight: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> "SelectionCorrectedElasticity":
        """
        Fit nuisance models with selection correction.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Covariates.
        D : array-like of shape (n,)
            Continuous treatment (premium).
        Y : array-like of shape (n,)
            Outcome.  For policies with S=0 (non-renewers), Y must be set to 0,
            not NaN.  Setting Y=NaN for non-renewers will produce NaN estimates
            because numpy evaluates NaN * 0 = NaN in the score computation.
            Use ``np.nan_to_num(Y, nan=0.0)`` before calling fit() if your
            data uses NaN to indicate missing outcomes.
        S : array-like of shape (n,)
            Selection indicator: 1 if renewed/outcome observed, 0 otherwise.
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
        S = np.asarray(S, dtype=float).ravel()

        if exposure is not None:
            exposure = np.asarray(exposure, dtype=float).ravel()
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float).ravel()

        # allow_nan_Y=True: NaN outcomes for non-renewers are coerced to 0
        # in estimate() via np.where.  NaN in X or D is always illegal.
        validate_inputs(X, D, Y, allow_nan_Y=True)

        n = len(Y)
        g_hat = np.full(n, np.nan)
        alpha_hat = np.full(n, np.nan)
        pi_hat = np.full(n, np.nan)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        fold_data = []

        from insurance_causal.autodml._nuisance import build_nuisance_model, _build_DX

        for fold_idx, (train_idx, eval_idx) in enumerate(kf.split(X)):
            X_tr, X_ev = X[train_idx], X[eval_idx]
            D_tr, D_ev = D[train_idx], D[eval_idx]
            Y_tr, Y_ev = Y[train_idx], Y[eval_idx]
            S_tr, S_ev = S[train_idx], S[eval_idx]
            sw_tr = sample_weight[train_idx] if sample_weight is not None else None
            exp_tr = exposure[train_idx] if exposure is not None else None
            exp_ev = exposure[eval_idx] if exposure is not None else None

            # 1. Fit selection model P(S=1 | X, D)
            DX_tr = _build_DX(D_tr, X_tr)
            DX_ev = _build_DX(D_ev, X_ev)

            sel_model = self._build_selection_model()
            if sw_tr is not None:
                sel_model.fit(DX_tr, S_tr, sample_weight=sw_tr)
            else:
                sel_model.fit(DX_tr, S_tr)

            pi_ev = sel_model.predict_proba(DX_ev)[:, 1]
            pi_ev = np.clip(pi_ev, 0.05, 0.95)  # Overlap trimming
            pi_hat[eval_idx] = pi_ev

            # 2. Fit outcome nuisance on selected (renewed) training obs
            sel_mask_tr = S_tr == 1
            if sel_mask_tr.sum() < 20:
                warnings.warn(
                    f"Fold {fold_idx}: fewer than 20 selected observations in training set.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            Y_tr_fit = Y_tr[sel_mask_tr]
            if exp_tr is not None:
                Y_tr_fit = Y_tr_fit / exp_tr[sel_mask_tr]

            nuisance = build_nuisance_model(
                outcome_family=self.outcome_family,
                backend=self.nuisance_backend,
            )
            sw_sel = sw_tr[sel_mask_tr] if sw_tr is not None else None
            nuisance.fit(
                D_tr[sel_mask_tr],
                X_tr[sel_mask_tr],
                Y_tr_fit,
                sample_weight=sw_sel,
            )

            g_ev = nuisance.predict(D_ev, X_ev)
            if exp_ev is not None:
                g_ev = g_ev * exp_ev
            g_hat[eval_idx] = g_ev

            # 3. Fit Riesz representer on selected training obs
            def nuisance_fn(D_in, X_in, _n=nuisance):
                return _n.predict(D_in, X_in)

            riesz = ForestRiesz(random_state=self.random_state)
            riesz.fit(
                X_tr[sel_mask_tr],
                D_tr[sel_mask_tr],
                nuisance_fn,
                sample_weight=sw_sel,
            )
            alpha_hat[eval_idx] = riesz.predict(X_ev)
            fold_data.append((train_idx, eval_idx))

        self.g_hat_ = g_hat
        self.alpha_hat_ = alpha_hat
        self.pi_hat_ = pi_hat
        self._Y = Y
        self._S = S
        self._D = D
        self._X = X
        self._sample_weight = sample_weight
        self._fold_data = fold_data
        self._is_fitted = True
        return self

    def estimate(self) -> EstimationResult:
        """
        Compute the selection-corrected AME.

        Returns
        -------
        result : EstimationResult
            Debiased AME with selection correction.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before estimate().")

        # IPW-corrected EIF score:
        # psi_i = S_i / pi_hat_i * alpha_hat_i * (Y_i - g_hat_i) + alpha_hat_i
        #
        # For non-renewers (S_i = 0), Y_i should be 0 (not NaN).
        # We defensively replace any NaN in Y with 0 here to avoid propagating
        # NaN through the score, but callers should ensure Y=0 for non-renewers.
        S = self._S
        pi = self.pi_hat_
        alpha = self.alpha_hat_
        Y = np.where(np.isnan(self._Y), 0.0, self._Y)
        g = self.g_hat_

        if np.any(np.isnan(self._Y) & (S == 1)):
            import warnings
            warnings.warn(
                "NaN values found in Y for selected (S=1) observations. "
                "These indicate genuinely missing outcomes for renewers and will "
                "corrupt the estimate. Check your data.",
                RuntimeWarning,
                stacklevel=2,
            )

        psi = (S / pi) * alpha * (Y - g) + alpha

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
            notes=f"selection_rate={self._S.mean():.3f}, mean_pi={pi.mean():.3f}",
        )

    def sensitivity_bounds(
        self,
        gamma_grid: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Manski-style sensitivity bounds for unobserved selection confounding.

        Gamma is the odds ratio bound on how much the selection odds could
        differ from what our model predicts, due to unobserved confounders.
        Gamma=1 means no unobserved confounding (point identified).
        Gamma=2 means unobserved confounders could double or halve the
        selection odds.

        Parameters
        ----------
        gamma_grid : array-like, optional
            Grid of Gamma values to assess.  Defaults to [1, 1.5, 2, 3].

        Returns
        -------
        bounds : dict
            Keys are Gamma values; values are dicts with "lower" and "upper"
            partial identification bounds on the AME.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before sensitivity_bounds().")

        if gamma_grid is None:
            gamma_grid = np.array([1.0, 1.5, 2.0, 3.0])

        bounds = {}
        for gamma in gamma_grid:
            # Under Gamma-bounded confounding, the selection probability can
            # range from pi/(pi + (1-pi)*Gamma) to pi*Gamma/(pi*Gamma + (1-pi))
            pi = self.pi_hat_
            pi_lo = pi / (pi + (1.0 - pi) * gamma)
            pi_hi = pi * gamma / (pi * gamma + (1.0 - pi))
            pi_lo = np.clip(pi_lo, 0.01, 0.99)
            pi_hi = np.clip(pi_hi, 0.01, 0.99)

            alpha = self.alpha_hat_
            Y = self._Y
            g = self.g_hat_
            S = self._S

            psi_lo = (S / pi_hi) * alpha * (Y - g) + alpha
            psi_hi = (S / pi_lo) * alpha * (Y - g) + alpha

            bounds[float(gamma)] = {
                "lower": float(np.mean(psi_lo)),
                "upper": float(np.mean(psi_hi)),
                "gamma": float(gamma),
            }

        return bounds
