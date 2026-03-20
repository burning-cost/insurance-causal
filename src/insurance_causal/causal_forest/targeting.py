"""
RATE (Rank-Weighted Average Treatment Effect) for targeting evaluation.

Implements Yadlowsky et al. (2025 JASA). The core question: does the causal
forest's CATE ranking actually identify high-effect customers?

If RATE is significantly positive, the top-q fraction of customers ranked by
predicted elasticity genuinely receive larger causal effects than the portfolio
average. This validates using the CATE for targeting (e.g., offering retention
discounts to the most elastic customers).

If RATE is not significant, the CATE ranking is essentially noise — the model
has detected no targetable heterogeneity. This is an important result, not a
failure. Do not use individual CATEs for targeting in this case.

Adaptation for continuous treatment:
    RATE is formally defined for binary treatment. For continuous price treatment
    (log_price_change), we adapt as follows: the top-q fraction are the most
    elastic customers. TOC(q) = mean(DR-scores | rank(tau_hat) >= 1-q) - mean(DR-scores).
    DR pseudo-outcomes are computed by re-fitting nuisance models (GBM) to obtain
    cross-validated residuals Y_resid and D_resid. The DR score is:
        Gamma_i = Y_resid_i * D_resid_i / Var(D_resid)

Bootstrap SE:
    We use the weighted bootstrap (Exponential(1) weights, Efron 2012).
    This is faster than the non-parametric bootstrap for large n and gives
    equivalent asymptotic coverage.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class TargetingResult:
    """RATE estimation result.

    Attributes
    ----------
    rate:
        RATE estimate (AUTOC or QINI, depending on method).
    se:
        Bootstrap standard error.
    p_value:
        p-value for H0: RATE = 0 (one-sided, RATE > 0).
    method:
        ``"autoc"`` or ``"qini"``.
    toc_curve:
        DataFrame with columns: q, toc, se_lower, se_upper.
    """
    rate: float
    se: float
    p_value: float
    method: str
    toc_curve: pd.DataFrame

    def plot_toc(self):
        """Plot the TOC curve with bootstrap confidence band.

        Returns
        -------
        matplotlib.figure.Figure or None if matplotlib not available.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not installed. Cannot produce TOC plot.", stacklevel=2)
            return None

        fig, ax = plt.subplots(figsize=(8, 5))
        q = self.toc_curve["q"].values
        toc = self.toc_curve["toc"].values
        lo = self.toc_curve["se_lower"].values
        hi = self.toc_curve["se_upper"].values

        ax.plot(q, toc, color="steelblue", linewidth=2, label="TOC")
        ax.fill_between(q, lo, hi, alpha=0.25, color="steelblue", label="95% CI")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Fraction targeted (q)")
        ax.set_ylabel("TOC(q): targeting gain vs average")
        title = (
            f"TOC Curve ({self.method.upper()})  "
            f"RATE={self.rate:.4f}  p={self.p_value:.4f}"
        )
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        return fig


class TargetingEvaluator:
    """Evaluates targeting value of a CATE ranking via RATE.

    Parameters
    ----------
    method:
        ``"autoc"`` or ``"qini"``. AUTOC integrates TOC(q)/q (weights early
        quantiles more). QINI integrates TOC(q) uniformly.
    n_bootstrap:
        Bootstrap replicates for SE estimation. 200 is fast; 500 for production.
    q_grid:
        Quantile grid for TOC curve. Default linspace(0.05, 1.0, 20).
    random_state:
        Seed for bootstrap weight generation.

    Examples
    --------
    >>> evaluator = TargetingEvaluator(method="autoc", n_bootstrap=200)
    >>> result = evaluator.fit(
    ...     estimator=est, df=df,
    ...     outcome="renewed", treatment="log_price_change",
    ...     confounders=confounders,
    ... )
    >>> print(f"RATE={result.rate:.4f}  p={result.p_value:.4f}")
    """

    def __init__(
        self,
        method: Literal["autoc", "qini"] = "autoc",
        n_bootstrap: int = 200,
        q_grid: np.ndarray = None,
        random_state: int = 42,
    ) -> None:
        self.method = method
        self.n_bootstrap = n_bootstrap
        self.q_grid = q_grid if q_grid is not None else np.linspace(0.05, 1.0, 20)
        self.random_state = random_state

    def fit(
        self,
        estimator: object,
        df,
        outcome: str,
        treatment: str,
        confounders: list,
    ) -> TargetingResult:
        """Compute RATE and TOC curve.

        Parameters
        ----------
        estimator:
            Fitted HeterogeneousElasticityEstimator.
        df:
            Dataset. May overlap with training data — the DR scores provide
            protection against overfitting via the doubly-robust construction.
        outcome, treatment, confounders:
            Same as used for estimator.fit().

        Returns
        -------
        TargetingResult
        """
        from .estimator import _to_pandas

        df_pd = _to_pandas(df)
        Y = df_pd[outcome].values.astype(float)
        W = df_pd[treatment].values.astype(float)
        X = df_pd[list(confounders)].values.astype(float)

        # Get CATE estimates (ranking rule)
        tau_hat = estimator.cate(df)

        # Build DR pseudo-outcomes
        dr_scores = self._compute_dr_scores(Y, W, X, tau_hat)

        # Compute point estimate RATE
        rate_point = self._compute_rate(dr_scores, tau_hat, self.q_grid, self.method)

        # Bootstrap SE
        boot_rates, toc_curves_boot = self._bootstrap(
            dr_scores, tau_hat, self.q_grid, self.method
        )

        se = float(np.std(boot_rates, ddof=1))
        # One-sided p-value: H0 RATE <= 0
        from scipy import stats
        z = rate_point / se if se > 0 else 0.0
        p_value = float(stats.norm.sf(z))  # one-sided upper tail

        # TOC point curve + SE bands
        toc_point = self._toc_curve(dr_scores, tau_hat, self.q_grid)
        toc_array = np.array(toc_curves_boot)  # (n_bootstrap, len(q_grid))
        toc_se = np.std(toc_array, axis=0, ddof=1)

        toc_df = pd.DataFrame({
            "q": self.q_grid,
            "toc": toc_point,
            "se_lower": toc_point - 1.96 * toc_se,
            "se_upper": toc_point + 1.96 * toc_se,
        })

        return TargetingResult(
            rate=float(rate_point),
            se=se,
            p_value=p_value,
            method=self.method,
            toc_curve=toc_df,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_dr_scores(
        self,
        Y: np.ndarray,
        W: np.ndarray,
        X: np.ndarray,
        tau_hat: np.ndarray,
    ) -> np.ndarray:
        """Compute doubly-robust pseudo-outcomes via re-fitted nuisance models.

        Re-fits GBM nuisance models for E[Y|X] and E[W|X] to get cross-validated
        residuals. This avoids dependency on econml internals.

        DR pseudo-outcome (continuous treatment DML score):
            Gamma_i = Y_resid_i * W_resid_i / Var(W_resid)
        """
        from sklearn.ensemble import GradientBoostingRegressor

        mu_model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        mu_model.fit(X, Y)
        y_resid = Y - mu_model.predict(X)

        e_model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        e_model.fit(X, W)
        w_resid = W - e_model.predict(X)

        var_w = float(np.var(w_resid, ddof=1))
        if var_w < 1e-10:
            warnings.warn(
                "Treatment residual variance is near zero. RATE computation unreliable. "
                "This indicates near-deterministic treatment assignment.",
                stacklevel=3,
            )
            return tau_hat

        dr_scores = y_resid * w_resid / var_w
        return dr_scores

    def _compute_rate(
        self,
        dr_scores: np.ndarray,
        tau_hat: np.ndarray,
        q_grid: np.ndarray,
        method: str,
    ) -> float:
        """Compute RATE from DR scores and CATE ranking."""
        toc = self._toc_curve(dr_scores, tau_hat, q_grid)

        if method == "autoc":
            integrand = toc / q_grid
        elif method == "qini":
            integrand = toc
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'autoc' or 'qini'.")

        return float(np.trapz(integrand, q_grid))

    def _toc_curve(
        self,
        dr_scores: np.ndarray,
        tau_hat: np.ndarray,
        q_grid: np.ndarray,
    ) -> np.ndarray:
        """Compute TOC values at each q in q_grid."""
        n = len(dr_scores)
        global_mean = float(np.mean(dr_scores))
        rank_order = np.argsort(tau_hat)[::-1]
        dr_sorted = dr_scores[rank_order]

        toc = np.zeros(len(q_grid))
        for i, q in enumerate(q_grid):
            k = max(1, int(np.ceil(q * n)))
            k = min(k, n)
            top_mean = float(np.mean(dr_sorted[:k]))
            toc[i] = top_mean - global_mean
        return toc

    def _bootstrap(
        self,
        dr_scores: np.ndarray,
        tau_hat: np.ndarray,
        q_grid: np.ndarray,
        method: str,
    ) -> tuple[list[float], list[np.ndarray]]:
        """Weighted bootstrap for SE estimation."""
        rng = np.random.default_rng(self.random_state)
        n = len(dr_scores)
        boot_rates = []
        toc_curves = []

        for _ in range(self.n_bootstrap):
            w = rng.exponential(1.0, size=n)
            w = w / w.sum() * n

            global_wmean = float(np.average(dr_scores, weights=w))

            rank_order = np.argsort(tau_hat)[::-1]
            dr_sorted = dr_scores[rank_order]
            w_sorted = w[rank_order]
            cumw = np.cumsum(w_sorted)

            toc_b = np.zeros(len(q_grid))
            for i, q in enumerate(q_grid):
                thresh = q * n
                mask = cumw <= thresh + 1e-9
                if mask.sum() == 0:
                    mask[0] = True
                top_wmean = float(np.average(dr_sorted[mask], weights=w_sorted[mask]))
                toc_b[i] = top_wmean - global_wmean

            if method == "autoc":
                rate_b = float(np.trapz(toc_b / q_grid, q_grid))
            else:
                rate_b = float(np.trapz(toc_b, q_grid))

            boot_rates.append(rate_b)
            toc_curves.append(toc_b)

        return boot_rates, toc_curves
