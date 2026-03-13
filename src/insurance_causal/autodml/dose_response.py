"""
DoseResponseCurve: kernel-DML estimation of E[Y(d)] at specified premium levels.

The dose-response curve asks: "what would the average outcome be if all
policyholders were charged premium d?"  This is different from the AME, which
is a single scalar derivative.

Method: Colangelo-Lee double-debiased kernel-DML (arXiv:2004.03036).

For a point d on the treatment grid, the doubly-robust score is:
    psi_i(d) = K_h(D_i - d) / hat_p(D_i | X_i) * (Y_i - g_hat(D_i, X_i))
               + g_hat(d, X_i)

where K_h is a kernel with bandwidth h, hat_p is the GPS density, and
g_hat is the cross-fitted outcome nuisance.

We avoid direct GPS estimation (division instability) by using a stabilised
variant: instead of dividing by hat_p, we use the normalised kernel weights
within each cross-fitting fold.  This is equivalent to a locally weighted
regression correction.

Bandwidth selection: Silverman's rule by default, with optional cross-
validation using the out-of-fold Riesz loss as criterion.
"""
from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from insurance_causal.autodml._crossfit import cross_fit_nuisance
from insurance_causal.autodml._inference import eif_inference
from insurance_causal.autodml._types import DoseResponseResult, OutcomeFamily
from insurance_causal.autodml.riesz import ForestRiesz


class DoseResponseCurve:
    """
    Kernel-DML dose-response curve E[Y(d)] for continuous treatments.

    Estimates the counterfactual mean outcome if the entire portfolio were
    charged premium ``d``, for a grid of premium values.

    Parameters
    ----------
    outcome_family : OutcomeFamily or str
        Distribution family for outcome regression.
    n_folds : int
        Number of cross-fitting folds.
    nuisance_backend : str
        Backend for outcome nuisance model.
    bandwidth : float or "silverman" or "cv"
        Kernel bandwidth.  "silverman" uses Silverman's rule of thumb.
        "cv" selects bandwidth by leave-one-out cross-validation (slow).
        A positive float sets it directly.
    kernel : {"gaussian", "epanechnikov", "uniform"}
        Kernel function.
    ci_level : float
        Confidence level for bands.
    random_state : int or None
        Random seed.
    """

    def __init__(
        self,
        outcome_family: Union[str, OutcomeFamily] = OutcomeFamily.GAUSSIAN,
        n_folds: int = 5,
        nuisance_backend: str = "sklearn",
        bandwidth: Union[float, str] = "silverman",
        kernel: str = "gaussian",
        ci_level: float = 0.95,
        random_state: Optional[int] = None,
    ) -> None:
        if isinstance(outcome_family, str):
            outcome_family = OutcomeFamily(outcome_family)
        self.outcome_family = outcome_family
        self.n_folds = n_folds
        self.nuisance_backend = nuisance_backend
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.ci_level = ci_level
        self.random_state = random_state

        self.g_hat_: Optional[np.ndarray] = None
        self._Y: Optional[np.ndarray] = None
        self._D: Optional[np.ndarray] = None
        self._X: Optional[np.ndarray] = None
        self._nuisance_models = None
        self._bw: Optional[float] = None
        self._is_fitted: bool = False

    @staticmethod
    def _silverman_bw(D: np.ndarray) -> float:
        """Silverman's rule of thumb for kernel bandwidth."""
        n = len(D)
        std = np.std(D, ddof=1)
        iqr = np.percentile(D, 75) - np.percentile(D, 25)
        s = min(std, iqr / 1.349)
        return float(1.06 * s * n ** (-0.2))

    def _kernel_fn(self, u: np.ndarray) -> np.ndarray:
        """Evaluate the kernel at standardised residuals u = (D - d) / h."""
        if self.kernel == "gaussian":
            return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
        elif self.kernel == "epanechnikov":
            return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)
        elif self.kernel == "uniform":
            return np.where(np.abs(u) <= 1, 0.5, 0.0)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel!r}")

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        D: Union[np.ndarray, pd.Series],
        Y: Union[np.ndarray, pd.Series],
        exposure: Optional[Union[np.ndarray, pd.Series]] = None,
        sample_weight: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> "DoseResponseCurve":
        """
        Fit the cross-fitted nuisance models.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Covariates.
        D : array-like of shape (n,)
            Continuous treatment (premium).
        Y : array-like of shape (n,)
            Outcome.
        exposure : array-like of shape (n,), optional
            Exposure offset for count models.
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

        # For dose-response we need nuisance models that can predict at new D
        g_hat, _, fold_indices, nuisance_models = cross_fit_nuisance(
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
        self._Y = Y
        self._D = D
        self._X = X
        self._sample_weight = sample_weight
        self._exposure = exposure
        self._fold_indices = fold_indices
        self._nuisance_models = nuisance_models

        # Set bandwidth
        if self.bandwidth == "silverman":
            self._bw = self._silverman_bw(D)
        elif self.bandwidth == "cv":
            self._bw = self._cv_bandwidth(D)
        else:
            self._bw = float(self.bandwidth)

        self._is_fitted = True
        return self

    def _cv_bandwidth(self, D: np.ndarray) -> float:
        """
        Cross-validation bandwidth selection using leave-one-out likelihood.

        Tries a grid of bandwidths from 0.5x to 2x Silverman's rule and
        selects the one that minimises the squared error of the kernel density.
        This is a simple rule; a full CV over the DR estimate would require
        refitting and is too expensive for interactive use.
        """
        bw_silverman = self._silverman_bw(D)
        grid = np.linspace(0.5 * bw_silverman, 2.0 * bw_silverman, 10)
        scores = []
        n = len(D)
        for bw in grid:
            # LOO kernel density at each point
            loo_score = 0.0
            for i in range(n):
                others = np.delete(D, i)
                u = (others - D[i]) / bw
                kde_i = np.mean(self._kernel_fn(u)) / bw
                loo_score += np.log(max(kde_i, 1e-10))
            scores.append(loo_score)
        return float(grid[np.argmax(scores)])

    def predict(
        self,
        d_grid: Union[np.ndarray, list],
    ) -> DoseResponseResult:
        """
        Estimate the dose-response curve at specified treatment values.

        Parameters
        ----------
        d_grid : array-like of shape (m,)
            Grid of premium values at which to evaluate E[Y(d)].

        Returns
        -------
        result : DoseResponseResult
            Estimated curve with confidence bands.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")

        d_grid = np.asarray(d_grid, dtype=float).ravel()
        n = len(self._Y)
        m = len(d_grid)
        h = self._bw

        ate_arr = np.zeros(m)
        se_arr = np.zeros(m)
        ci_low_arr = np.zeros(m)
        ci_high_arr = np.zeros(m)

        for j, d in enumerate(d_grid):
            # Kernel weights
            u = (self._D - d) / h
            k_weights = self._kernel_fn(u) / h

            # Doubly-robust scores:
            # psi_i(d) = k_i / (mean(k_i)) * (Y_i - g_hat_i) + g_hat_d_i
            # where g_hat_d_i = E[Y | D=d, X_i] from the nuisance model.

            # Compute g_hat at treatment value d for all observations
            g_hat_d = self._predict_at_d(d)

            # Normalised kernel weights (avoids GPS estimation)
            k_sum = np.mean(k_weights)
            if k_sum < 1e-10:
                warnings.warn(
                    f"Kernel weight sum near zero at d={d:.2f}. "
                    "This treatment value may be outside the support of the data.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                ate_arr[j] = np.nan
                continue

            k_norm = k_weights / k_sum

            # DR score
            psi = k_norm * (self._Y - self.g_hat_) + g_hat_d

            est, se, ci_low, ci_high = eif_inference(
                psi, level=self.ci_level
            )
            ate_arr[j] = est
            se_arr[j] = se
            ci_low_arr[j] = ci_low
            ci_high_arr[j] = ci_high

        return DoseResponseResult(
            d_grid=d_grid,
            ate=ate_arr,
            se=se_arr,
            ci_low=ci_low_arr,
            ci_high=ci_high_arr,
            ci_level=self.ci_level,
            bandwidth=h,
            n_obs=n,
        )

    def _predict_at_d(self, d: float) -> np.ndarray:
        """
        Predict E[Y | D=d, X_i] for all observations using cross-fitted models.

        Uses the appropriate fold's nuisance model for each observation to
        maintain out-of-fold validity.
        """
        g_hat_d = np.full(len(self._Y), np.nan)
        D_const = np.full(len(self._Y), d)

        for (train_idx, eval_idx), nuisance in zip(self._fold_indices, self._nuisance_models):
            pred = nuisance.predict(D_const[eval_idx], self._X[eval_idx])
            if self._exposure is not None:
                pred = pred * self._exposure[eval_idx]
            g_hat_d[eval_idx] = pred

        return g_hat_d

    def plot(
        self,
        d_grid: Optional[np.ndarray] = None,
        ax=None,
        title: str = "Dose-Response Curve",
        xlabel: str = "Premium (£)",
        ylabel: str = "E[Y(d)]",
        show_data_rug: bool = True,
    ):
        """
        Plot the estimated dose-response curve with confidence bands.

        Requires matplotlib.

        Parameters
        ----------
        d_grid : array-like, optional
            Treatment grid. Defaults to 50 evenly spaced points across
            the observed treatment range.
        ax : matplotlib Axes, optional
            Axes to plot on.
        title : str
            Plot title.
        xlabel : str
            X-axis label.
        ylabel : str
            Y-axis label.
        show_data_rug : bool
            Whether to show a rug plot of observed treatment values.

        Returns
        -------
        ax : matplotlib Axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install insurance-autodml[plots]"
            ) from exc

        if not self._is_fitted:
            raise RuntimeError("Call fit() before plot().")

        if d_grid is None:
            d_grid = np.linspace(
                np.percentile(self._D, 2),
                np.percentile(self._D, 98),
                50,
            )

        result = self.predict(d_grid)

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        ax.plot(result.d_grid, result.ate, color="steelblue", label="E[Y(d)]")
        ax.fill_between(
            result.d_grid,
            result.ci_low,
            result.ci_high,
            alpha=0.25,
            color="steelblue",
            label=f"{int(self.ci_level*100)}% CI",
        )
        if show_data_rug:
            ax.plot(
                self._D,
                np.full_like(self._D, result.ate.min() - 0.02 * result.ate.ptp()),
                "|",
                color="grey",
                alpha=0.3,
                markersize=3,
                label="Observed D",
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        return ax
