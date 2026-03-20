"""
RATE (Rank-Weighted Average Treatment Effect) for targeting policy evaluation.

RATE answers the question: "If I use the causal forest CATE estimates to target
the top q fraction of policies (those estimated as most responsive), what is
the average treatment effect in that group?"

The TOC (Treatment on Classified) curve plots E[Y(1) - Y(0) | tau_hat >= q]
against q. A good targeting rule produces a curve well above zero — the most
responsive policies (by CATE estimate) should have high actual treatment
benefit. AUTOC is the area under this curve.

Estimation uses doubly-robust (Augmented IPW) pseudo-outcomes to control for
confounding:

    Gamma_i = mu(1, x) - mu(0, x) + W*(Y - mu(1,x))/e - (1-W)*(Y - mu(0,x))/(1-e)

where mu(w, x) = E[Y(w)|X=x] and e = P(W=1|X). For continuous treatments,
this extends to the partial-out residual formulation.

The AUTOC is estimated as the weighted average of the pseudo-outcomes, ranked
by the CATE proxy. Bootstrap standard errors are computed by resampling the
pseudo-outcomes.

References
----------
Yadlowsky, S., Fleming, S., Shah, N., Brunskill, E., & Wager, S. (2025).
    "Evaluating Treatment Prioritization Rules via Rank-Weighted Average
    Treatment Effects." JASA 120(549). arXiv:2111.07966.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import polars as pl

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


DataFrameLike = Union[pl.DataFrame, "pd.DataFrame"]

_N_BOOTSTRAP: int = 200
_MIN_GROUP_SIZE: int = 100


@dataclass
class TargetingResult:
    """Results from RATE / TOC evaluation.

    Attributes
    ----------
    autoc:
        Area Under the TOC curve (AUTOC). Positive = good targeting rule.
        Zero = random targeting. Negative = anti-targeted (inverted).
    autoc_se:
        Bootstrap standard error of AUTOC.
    autoc_ci_lower, autoc_ci_upper:
        95% confidence interval for AUTOC.
    qini:
        QINI coefficient: AUTOC normalised by the overall ATE (AUTOC / |ATE|).
        > 1.0 means targeting outperforms treating everyone.
    toc_curve:
        polars.DataFrame with columns: q (quantile fraction, 0 to 1),
        toc (Treatment on Classified value), toc_se.
        Use plot_toc() to visualise.
    n_obs:
        Number of observations.
    n_bootstrap:
        Number of bootstrap replications.
    """

    autoc: float
    autoc_se: float
    autoc_ci_lower: float
    autoc_ci_upper: float
    qini: float
    toc_curve: pl.DataFrame
    n_obs: int
    n_bootstrap: int

    def __repr__(self) -> str:
        sig = " [significant]" if self.autoc_ci_lower > 0 else ""
        return (
            f"TargetingResult(AUTOC={self.autoc:.4f}±{self.autoc_se:.4f}{sig}, "
            f"QINI={self.qini:.3f})"
        )

    def summary(self) -> str:
        """Return a human-readable targeting evaluation summary."""
        lines = [
            "Targeting Evaluation (RATE/AUTOC)",
            "=" * 45,
            f"N observations:    {self.n_obs:,}",
            f"N bootstrap reps:  {self.n_bootstrap}",
            "",
            f"AUTOC:    {self.autoc:.4f}  (SE={self.autoc_se:.4f})",
            f"  95% CI: [{self.autoc_ci_lower:.4f}, {self.autoc_ci_upper:.4f}]",
            f"  {'TARGETING RULE IS INFORMATIVE (AUTOC > 0)' if self.autoc_ci_lower > 0 else 'No significant targeting benefit detected'}",
            "",
            f"QINI:     {self.qini:.4f}",
            f"  {'Targeting outperforms random assignment' if self.qini > 1 else 'Targeting does not outperform random assignment'}",
        ]
        return "\n".join(lines)

    def plot_toc(self, ax: Optional[object] = None) -> None:
        """Plot the Treatment on Classified (TOC) curve.

        The x-axis is the fraction of policies treated (ordered by CATE
        estimate). The y-axis is the average treatment effect in the treated
        fraction. A curve above the dashed line (uniform ATE) means the
        targeting rule correctly identifies the most responsive policies.

        Parameters
        ----------
        ax:
            Optional matplotlib Axes. If None, creates a new figure.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not installed. Cannot produce plot.", stacklevel=2)
            return

        q_vals = self.toc_curve["q"].to_list()
        toc_vals = self.toc_curve["toc"].to_list()

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(q_vals, toc_vals, color="steelblue", linewidth=2, label="TOC curve")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", label="No effect")

        # Shade area under TOC (AUTOC) — positive area is green, negative red
        q_arr = np.array(q_vals)
        toc_arr = np.array(toc_vals)
        ax.fill_between(q_arr, toc_arr, 0,
                        where=toc_arr >= 0, alpha=0.2, color="green", label="Positive AUTOC")
        ax.fill_between(q_arr, toc_arr, 0,
                        where=toc_arr < 0, alpha=0.2, color="red", label="Negative AUTOC")

        ax.set_xlabel("Fraction treated (ordered by CATE estimate)")
        ax.set_ylabel("Average treatment effect in treated fraction")
        ax.set_title(f"TOC Curve  |  AUTOC={self.autoc:.4f}  QINI={self.qini:.3f}")
        ax.legend(loc="upper right")
        plt.tight_layout()
        if ax is None:
            plt.show()


class TargetingEvaluator:
    """Evaluate the quality of a causal forest targeting rule via RATE/AUTOC/QINI.

    RATE (Yadlowsky et al. 2025) provides a formal test for whether the CATE
    proxy correctly identifies which policies benefit most from treatment.

    This is the key question for UK pricing teams: "Which customers should
    receive a retention discount?" The AUTOC measures whether sorting by
    estimated elasticity produces better-than-random targeting.

    Parameters
    ----------
    n_bootstrap:
        Bootstrap replications for AUTOC standard errors.
    n_toc_points:
        Number of quantile points for the TOC curve.
    random_state:
        Random seed.

    Examples
    --------
    >>> from insurance_causal.causal_forest.data import make_hte_renewal_data
    >>> from insurance_causal.causal_forest.estimator import HeterogeneousElasticityEstimator
    >>> from insurance_causal.causal_forest.targeting import TargetingEvaluator
    >>> df = make_hte_renewal_data(n=5000)
    >>> confounders = ["age", "ncd_years", "vehicle_group", "channel"]
    >>> est = HeterogeneousElasticityEstimator(n_estimators=100, catboost_iterations=100)
    >>> est.fit(df, outcome="renewed", treatment="log_price_change",
    ...         confounders=confounders)
    >>> cates = est.cate(df)
    >>> ev = TargetingEvaluator()
    >>> result = ev.evaluate(df, estimator=est, cate_proxy=cates)
    >>> print(result.summary())
    >>> result.plot_toc()
    """

    def __init__(
        self,
        n_bootstrap: int = _N_BOOTSTRAP,
        n_toc_points: int = 50,
        random_state: int = 42,
    ) -> None:
        self.n_bootstrap = n_bootstrap
        self.n_toc_points = n_toc_points
        self.random_state = random_state

    def evaluate(
        self,
        df: DataFrameLike,
        estimator: object,
        cate_proxy: np.ndarray,
    ) -> TargetingResult:
        """Evaluate the targeting rule via AUTOC and QINI.

        Parameters
        ----------
        df:
            Dataset used to fit the estimator.
        estimator:
            Fitted HeterogeneousElasticityEstimator.
        cate_proxy:
            Per-row CATE estimates (the targeting rule to evaluate).

        Returns
        -------
        TargetingResult
        """
        from insurance_causal.causal_forest.estimator import _to_pandas

        df_pd = _to_pandas(df)
        Y = estimator._Y_train
        D = estimator._D_train
        X = estimator._X_train
        n = len(Y)

        # Compute doubly-robust pseudo-outcomes
        gamma = self._compute_dr_pseudo_outcomes(Y, D, X)

        # TOC curve
        toc_curve = self._compute_toc_curve(gamma, cate_proxy)

        # AUTOC = integral of TOC(q)/q dq (trapezoid rule)
        autoc = self._compute_autoc(gamma, cate_proxy)

        # Bootstrap SE
        rng = np.random.default_rng(self.random_state)
        autoc_boot = np.array([
            self._compute_autoc(
                gamma[boot_idx := rng.integers(0, n, size=n)],
                cate_proxy[boot_idx],
            )
            for _ in range(self.n_bootstrap)
        ])
        autoc_se = float(np.std(autoc_boot, ddof=1))
        autoc_ci_lower = float(np.percentile(autoc_boot, 2.5))
        autoc_ci_upper = float(np.percentile(autoc_boot, 97.5))

        # QINI = AUTOC / |ATE|
        ate = float(np.mean(gamma))
        qini = autoc / (abs(ate) + 1e-12)

        return TargetingResult(
            autoc=float(autoc),
            autoc_se=autoc_se,
            autoc_ci_lower=autoc_ci_lower,
            autoc_ci_upper=autoc_ci_upper,
            qini=qini,
            toc_curve=toc_curve,
            n_obs=n,
            n_bootstrap=self.n_bootstrap,
        )

    def _compute_dr_pseudo_outcomes(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        X: np.ndarray,
    ) -> np.ndarray:
        """Compute doubly-robust pseudo-outcomes for RATE estimation.

        For continuous treatments (which insurance price changes are),
        we use the partial-out residual form:

            Gamma_i = (Y_i - mu_hat_i) / (D_i - e_hat_i) * sign(D_i - e_hat_i)

        This is the approximate DR estimand for the marginal effect of D on Y.

        For a simpler implementation that avoids the division-by-small-number
        problem, we use:

            Gamma_i ≈ residual-on-residual (the DML Frisch-Waugh residual)

        which is the standard DML score and directly represents the causal
        effect via the Neyman orthogonal score.
        """
        from sklearn.ensemble import GradientBoostingRegressor

        # Fit outcome nuisance: E[Y|X]
        mu_model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        mu_model.fit(X, Y)
        mu_hat = mu_model.predict(X)

        # Fit treatment nuisance: E[D|X]
        e_model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        e_model.fit(X, D)
        e_hat = e_model.predict(X)

        # Neyman-orthogonal score (continuous treatment DML pseudo-outcome)
        Y_resid = Y - mu_hat
        D_resid = D - e_hat

        var_D_resid = float(np.var(D_resid, ddof=1))
        if var_D_resid < 1e-10:
            warnings.warn(
                "Treatment residuals have near-zero variance. "
                "DR pseudo-outcomes will be unreliable. "
                "Consider increasing price variation in the data.",
                UserWarning,
                stacklevel=3,
            )

        # DR pseudo-outcome: (Y - mu) / Var(D - e) * (D - e)
        # This is the Frisch-Waugh / partial-out causal score
        gamma = Y_resid * D_resid / (var_D_resid + 1e-12)
        return gamma

    def _compute_toc_curve(
        self,
        gamma: np.ndarray,
        S: np.ndarray,
    ) -> pl.DataFrame:
        """Compute the TOC curve at n_toc_points quantile thresholds."""
        n = len(gamma)
        q_vals = np.linspace(0.1, 1.0, self.n_toc_points)

        # Sort by CATE proxy (descending: high CATE = treated first)
        order = np.argsort(-S)
        gamma_sorted = gamma[order]

        rows = []
        for q in q_vals:
            cutoff = max(1, int(np.round(q * n)))
            toc_val = float(np.mean(gamma_sorted[:cutoff]))
            rows.append({"q": float(q), "toc": toc_val})

        return pl.DataFrame(rows)

    def _compute_autoc(
        self,
        gamma: np.ndarray,
        S: np.ndarray,
    ) -> float:
        """Compute AUTOC = integral of (1/q) * TOC(q) dq.

        Yadlowsky et al. (2025) Eq. 2: AUTOC = E[Gamma_i * w_i] where w_i
        is the rank-weight 1/F(S_i). We approximate via sorted pseudo-outcomes:
        AUTOC = sum_i Gamma_{sigma(i)} * (1 / rank_i) / n
        """
        n = len(gamma)
        order = np.argsort(-S)  # descending by CATE proxy
        gamma_sorted = gamma[order]

        # Rank weights: 1/q where q = rank/n
        ranks = np.arange(1, n + 1)
        weights = 1.0 / (ranks / n)

        # Normalise weights to sum to 1
        weights = weights / weights.sum()
        autoc = float(np.dot(gamma_sorted, weights))
        return autoc
