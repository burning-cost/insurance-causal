"""
PremiumElasticity: Average Marginal Effect (AME) via Riesz regression.

This is the main entry point for estimating price elasticity of demand in
UK personal lines insurance.  The treatment D is the actual premium charged
(or a log-premium, or a discount); the outcome Y is typically claim count
(Poisson) or pure premium (Tweedie).

The AME is defined as:
    theta_0 = E[dE[Y | D, X] / dD]

which represents the average derivative of the conditional mean outcome with
respect to the treatment, integrated over the covariate distribution.  For
a retention model, AME gives the average change in lapse probability per £1
increase in premium.

The estimator is Neyman-orthogonal (double-ML), meaning it achieves sqrt(n)
rates even when the nuisance models (E[Y|D,X] and the Riesz representer) are
estimated at slower rates.

Usage
-----
>>> from insurance_causal.autodml import PremiumElasticity
>>> model = PremiumElasticity(outcome_family="poisson", n_folds=5)
>>> model.fit(X, D, Y, exposure=exposure)
>>> result = model.estimate()
>>> print(result.summary())
>>> segment_results = model.effect_by_segment(segments)
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from insurance_causal.autodml._crossfit import cross_fit_nuisance, compute_ame_scores
from insurance_causal.autodml._inference import run_inference
from insurance_causal.autodml._types import EstimationResult, OutcomeFamily, SegmentResult
from insurance_causal.autodml.riesz import ForestRiesz, LinearRiesz


class PremiumElasticity:
    """
    Average Marginal Effect estimator for continuous premium treatments.

    Implements the debiased ML AME estimator using cross-fitted Riesz
    regression.  Avoids estimation of the generalised propensity score,
    which is numerically unstable in renewal portfolios where high-premium
    policies have renewal rates of 20-30%.

    Parameters
    ----------
    outcome_family : {"gaussian", "poisson", "gamma", "tweedie"} or OutcomeFamily
        Distribution family for the outcome regression nuisance model.
        Use "poisson" for claim counts with exposure offset, "tweedie" for
        pure premium, "gaussian" for log-transformed outcomes.
    n_folds : int
        Number of cross-fitting folds. 5 is standard; use 3 for small samples
        (n < 2000) or fast experimentation.
    nuisance_backend : {"sklearn", "catboost", "linear"}
        Backend for the outcome regression nuisance model.
    riesz_type : {"forest", "linear"}
        Type of Riesz regressor.  "forest" is recommended for most insurance
        data.  "linear" is useful for fast baselines.
    riesz_kwargs : dict, optional
        Keyword arguments passed to the Riesz regressor constructor.
    nuisance_estimator : sklearn estimator, optional
        Custom base learner for outcome nuisance.  Overrides nuisance_backend.
    inference : {"eif", "bootstrap"}
        Inference method.  "eif" is faster; "bootstrap" can give better
        finite-sample coverage in small datasets.
    ci_level : float
        Confidence interval coverage (default 0.95).
    cluster_ids : array of shape (n,), optional
        Cluster identifiers for cluster-robust standard errors.
    random_state : int or None
        Random seed for reproducibility.

    Attributes
    ----------
    result_ : EstimationResult or None
        Fitted estimation result. Available after calling ``estimate()``.
    g_hat_ : np.ndarray or None
        Out-of-fold outcome nuisance predictions. Available after ``fit()``.
    alpha_hat_ : np.ndarray or None
        Out-of-fold Riesz representer predictions. Available after ``fit()``.
    """

    def __init__(
        self,
        outcome_family: Union[str, OutcomeFamily] = OutcomeFamily.GAUSSIAN,
        n_folds: int = 5,
        nuisance_backend: str = "sklearn",
        riesz_type: str = "forest",
        riesz_kwargs: Optional[dict] = None,
        nuisance_estimator: Optional[BaseEstimator] = None,
        inference: str = "eif",
        ci_level: float = 0.95,
        cluster_ids: Optional[np.ndarray] = None,
        random_state: Optional[int] = None,
    ) -> None:
        if isinstance(outcome_family, str):
            outcome_family = OutcomeFamily(outcome_family)
        self.outcome_family = outcome_family
        self.n_folds = n_folds
        self.nuisance_backend = nuisance_backend
        self.riesz_type = riesz_type
        self.riesz_kwargs = riesz_kwargs or {}
        self.nuisance_estimator = nuisance_estimator
        self.inference = inference
        self.ci_level = ci_level
        self.cluster_ids = cluster_ids
        self.random_state = random_state

        # Set after fit()
        self.g_hat_: Optional[np.ndarray] = None
        self.alpha_hat_: Optional[np.ndarray] = None
        self._Y: Optional[np.ndarray] = None
        self._X: Optional[np.ndarray] = None
        self._D: Optional[np.ndarray] = None
        self._sample_weight: Optional[np.ndarray] = None
        self._fold_indices = None
        self._nuisance_models = None
        self._is_fitted: bool = False

        # Set after estimate()
        self.result_: Optional[EstimationResult] = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        D: Union[np.ndarray, pd.Series],
        Y: Union[np.ndarray, pd.Series],
        exposure: Optional[Union[np.ndarray, pd.Series]] = None,
        sample_weight: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> "PremiumElasticity":
        """
        Fit the cross-fitted nuisance models and Riesz representer.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Covariates: policyholder characteristics used to model selection
            into treatment.  Should include all variables that jointly affect
            the treatment (premium) and the outcome (claims/retention).
        D : array-like of shape (n,)
            Continuous treatment: actual premium charged, log-premium, or
            discount percentage.
        Y : array-like of shape (n,)
            Outcome: claim count, lapse indicator, or pure premium.
        exposure : array-like of shape (n,), optional
            Exposure offset for count models (years at risk, policy count).
            For Poisson models the nuisance fits Y/exposure and the AME is
            on the rate scale.
        sample_weight : array-like of shape (n,), optional
            Observation weights.  Useful when sampling from a larger
            portfolio with known sampling probabilities.

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

        riesz_cls = ForestRiesz if self.riesz_type == "forest" else LinearRiesz

        g_hat, alpha_hat, fold_indices, nuisance_models = cross_fit_nuisance(
            X=X,
            D=D,
            Y=Y,
            outcome_family=self.outcome_family,
            n_folds=self.n_folds,
            nuisance_backend=self.nuisance_backend,
            riesz_class=riesz_cls,
            riesz_kwargs=self.riesz_kwargs,
            nuisance_estimator=self.nuisance_estimator,
            sample_weight=sample_weight,
            exposure=exposure,
            random_state=self.random_state,
        )

        self.g_hat_ = g_hat
        self.alpha_hat_ = alpha_hat
        self._Y = Y
        self._X = X
        self._D = D
        self._sample_weight = sample_weight
        self._fold_indices = fold_indices
        self._nuisance_models = nuisance_models
        self._is_fitted = True
        return self

    def estimate(self) -> EstimationResult:
        """
        Compute the Average Marginal Effect with confidence intervals.

        Must be called after ``fit()``.

        Returns
        -------
        result : EstimationResult
            Debiased AME estimate with standard error and confidence interval.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before estimate().")

        ame, psi = compute_ame_scores(
            Y=self._Y,
            g_hat=self.g_hat_,
            alpha_hat=self.alpha_hat_,
            sample_weight=self._sample_weight,
        )

        est, se, ci_low, ci_high = run_inference(
            psi=psi,
            level=self.ci_level,
            inference=self.inference,
            cluster_ids=self.cluster_ids,
            sample_weight=self._sample_weight,
            random_state=self.random_state,
        )

        self.result_ = EstimationResult(
            estimate=est,
            se=se,
            ci_low=ci_low,
            ci_high=ci_high,
            ci_level=self.ci_level,
            n_obs=len(self._Y),
            n_folds=self.n_folds,
            psi=psi,
        )
        return self.result_

    def effect_by_segment(
        self,
        segments: Union[np.ndarray, pd.Series, pd.DataFrame],
    ) -> List[SegmentResult]:
        """
        Estimate segment-level average marginal effects.

        Splits the EIF scores by segment and computes a separate AME for
        each.  This is valid because the EIF decomposes additively over
        subgroups.  No refitting is required.

        Parameters
        ----------
        segments : array-like of shape (n,) or DataFrame
            Segment labels.  If a 1D array, each unique value defines a
            segment.  If a DataFrame with multiple columns, segments are
            defined by the Cartesian product of column values.

        Returns
        -------
        results : list of SegmentResult
            One entry per unique segment.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before effect_by_segment().")

        _, psi = compute_ame_scores(
            Y=self._Y,
            g_hat=self.g_hat_,
            alpha_hat=self.alpha_hat_,
            sample_weight=self._sample_weight,
        )

        if isinstance(segments, pd.DataFrame):
            seg_labels = segments.apply(
                lambda row: " & ".join(f"{c}={v}" for c, v in row.items()), axis=1
            ).to_numpy()
        else:
            seg_labels = np.asarray(segments).ravel()

        results: List[SegmentResult] = []
        for seg in np.unique(seg_labels):
            mask = seg_labels == seg
            psi_seg = psi[mask]
            sw_seg = self._sample_weight[mask] if self._sample_weight is not None else None

            est, se, ci_low, ci_high = run_inference(
                psi=psi_seg,
                level=self.ci_level,
                inference=self.inference,
                sample_weight=sw_seg,
                random_state=self.random_state,
            )
            seg_result = EstimationResult(
                estimate=est,
                se=se,
                ci_low=ci_low,
                ci_high=ci_high,
                ci_level=self.ci_level,
                n_obs=int(mask.sum()),
                n_folds=self.n_folds,
                psi=psi_seg,
            )
            results.append(SegmentResult(
                segment_name=str(seg),
                result=seg_result,
                n_obs=int(mask.sum()),
            ))

        return results

    def riesz_loss(self) -> float:
        """
        Compute the out-of-fold Riesz loss on the full sample.

        Useful for assessing Riesz representer quality.  Lower values
        indicate better estimation of the reweighting functional.

        Returns
        -------
        loss : float
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before riesz_loss().")

        from insurance_causal.autodml.riesz import compute_riesz_loss

        # Use last fitted nuisance model as proxy for full-sample estimate
        loss_vals = []
        for (train_idx, eval_idx), nuisance in zip(self._fold_indices, self._nuisance_models):
            loss = compute_riesz_loss(
                alpha_pred=self.alpha_hat_[eval_idx],
                D=self._D[eval_idx],
                X=self._X[eval_idx],
                nuisance_fn=nuisance,
                sample_weight=self._sample_weight[eval_idx] if self._sample_weight is not None else None,
            )
            loss_vals.append(loss)
        return float(np.mean(loss_vals))
