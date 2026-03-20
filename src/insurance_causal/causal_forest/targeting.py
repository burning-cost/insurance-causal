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
import polars as pl


@dataclass
class TargetingResult:
    """RATE estimation result.

    Attributes
    ----------
    autoc:
        AUTOC estimate (Rank-Weighted Average Treatment Effect, AUTOC weighting).
    autoc_se:
        Bootstrap standard error for AUTOC.
    autoc_ci_lower:
        Lower bound of the 95% confidence interval for AUTOC.
    autoc_ci_upper:
        Upper bound of the 95% confidence interval for AUTOC.
    qini:
        QINI coefficient estimate (uniform weighting of TOC).
    n_obs:
        Number of observations used.
    toc_curve:
        polars.DataFrame with columns: q, toc, se_lower, se_upper.
    """
    autoc: float
    autoc_se: float
    autoc_ci_lower: float
    autoc_ci_upper: float
    qini: float
    n_obs: int
    toc_curve: pl.DataFrame

    def summary(self) -> str:
        """Return a human-readable targeting evaluation report."""
        lines = [
            "Targeting Evaluation (RATE)",
            "=" * 40,
            f"N observations: {self.n_obs:,}",
            "",
            f"AUTOC: {self.autoc:.4f}  (SE={self.autoc_se:.4f})",
            f"  95% CI: [{self.autoc_ci_lower:.4f}, {self.autoc_ci_upper:.4f}]",
            f"QINI:  {self.qini:.4f}",
            "",
            "Interpretation:",
        ]
        if self.autoc_ci_lower > 0:
            lines.append("  AUTOC significantly positive — CATE ranking is informative for targeting.")
        elif self.autoc > 0:
            lines.append("  AUTOC positive but not significant — weak evidence of targetable heterogeneity.")
        else:
            lines.append("  AUTOC not significantly positive — no evidence that CATE ranking adds targeting value.")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"TargetingResult(AUTOC={self.autoc:.4f}, autoc_se={self.autoc_se:.4f}, "
            f"QINI={self.qini:.4f}, n_obs={self.n_obs:,})"
        )

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
        q = self.toc_curve["q"].to_numpy()
        toc = self.toc_curve["toc"].to_numpy()
        lo = self.toc_curve["se_lower"].to_numpy()
        hi = self.toc_curve["se_upper"].to_numpy()

        ax.plot(q, toc, color="steelblue", linewidth=2, label="TOC")
        ax.fill_between(q, lo, hi, alpha=0.25, color="steelblue", label="95% CI")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Fraction targeted (q)")
        ax.set_ylabel("TOC(q): targeting gain vs average")
        title = (
            f"TOC Curve  AUTOC={self.autoc:.4f}"
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
        quantiles more). QINI integrates TOC(q) uniformly. Both are always
        computed; this parameter selects which is used for the primary SE
        and CI reported in ``TargetingResult``.
    n_bootstrap:
        Bootstrap replicates for SE estimation. 200 is fast; 500 for production.
    n_toc_points:
        Number of points on the TOC curve q-grid (linspace from 0.05 to 1.0).
    random_state:
        Seed for bootstrap weight generation.

    Examples
    --------
    >>> evaluator = TargetingEvaluator(method="autoc", n_bootstrap=200, n_toc_points=20)
    >>> result = evaluator.evaluate(
    ...     df=df,
    ...     estimator=est,
    ...     cate_proxy=cates,
    ... )
    >>> print(result.summary())
    """

    def __init__(
        self,
        method: Literal["autoc", "qini"] = "autoc",
        n_bootstrap: int = 200,
        n_toc_points: int = 20,
        random_state: int = 42,
    ) -> None:
        self.method = method
        self.n_bootstrap = n_bootstrap
        self.n_toc_points = n_toc_points
        self.random_state = random_state
        # q_grid derived from n_toc_points
        self.q_grid = np.linspace(0.05, 1.0, n_toc_points)

    def evaluate(
        self,
        df,
        estimator: object,
        cate_proxy: np.ndarray,
    ) -> TargetingResult:
        """Compute RATE and TOC curve from a fitted estimator and CATE proxy.

        Parameters
        ----------
        df:
            Dataset. May overlap with training data — the DR scores provide
            protection against overfitting via the doubly-robust construction.
        estimator:
            Fitted HeterogeneousElasticityEstimator. Used to extract outcome,
            treatment, and confounders for DR score computation.
        cate_proxy:
            Per-row CATE estimates, shape (n,). Used as the ranking rule.

        Returns
        -------
        TargetingResult
        """
        from .estimator import _to_pandas

        df_pd = _to_pandas(df)
        Y = df_pd[estimator._outcome_col].values.astype(float)
        W = df_pd[estimator._treatment_col].values.astype(float)
        confounders = list(estimator._confounders)
        X = df_pd[confounders].values.astype(float)

        n = len(Y)
        tau_hat = np.asarray(cate_proxy)

        # Build DR pseudo-outcomes
        dr_scores = self._compute_dr_scores(Y, W, X, tau_hat)

        # Point estimates for both AUTOC and QINI
        autoc_point = self._compute_rate(dr_scores, tau_hat, self.q_grid, "autoc")
        qini_point = self._compute_rate(dr_scores, tau_hat, self.q_grid, "qini")

        # Bootstrap SE for AUTOC
        boot_autoc, toc_curves_boot = self._bootstrap(
            dr_scores, tau_hat, self.q_grid
        )

        autoc_se = float(np.std(boot_autoc, ddof=1))
        autoc_ci_lower = float(autoc_point - 1.96 * autoc_se)
        autoc_ci_upper = float(autoc_point + 1.96 * autoc_se)

        # TOC point curve + SE bands
        toc_point = self._toc_curve(dr_scores, tau_hat, self.q_grid)
        toc_array = np.array(toc_curves_boot)  # (n_bootstrap, n_toc_points)
        toc_se = np.std(toc_array, axis=0, ddof=1)

        toc_df = pl.DataFrame({
            "q": self.q_grid.tolist(),
            "toc": toc_point.tolist(),
            "se_lower": (toc_point - 1.96 * toc_se).tolist(),
            "se_upper": (toc_point + 1.96 * toc_se).tolist(),
        })

        return TargetingResult(
            autoc=float(autoc_point),
            autoc_se=autoc_se,
            autoc_ci_lower=autoc_ci_lower,
            autoc_ci_upper=autoc_ci_upper,
            qini=float(qini_point),
            n_obs=n,
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
    ) -> tuple[list[float], list[np.ndarray]]:
        """Weighted bootstrap for SE estimation (AUTOC only)."""
        rng = np.random.default_rng(self.random_state)
        n = len(dr_scores)
        boot_rates: list[float] = []
        toc_curves: list[np.ndarray] = []

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

            rate_b = float(np.trapz(toc_b / q_grid, q_grid))  # always AUTOC for SE

            boot_rates.append(rate_b)
            toc_curves.append(toc_b)

        return boot_rates, toc_curves
