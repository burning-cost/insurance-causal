"""
Formal HTE inference via BLP, GATES, and CLAN.

Implements the Chernozhukov et al. (2022/2025) framework for testing and
characterising treatment effect heterogeneity from causal forest CATE estimates.

The three-part procedure:
1. BLP (Best Linear Predictor): Test whether the CATE proxy S(x) = tau_hat(x)
   explains any variation in individual treatment effects. beta_2 > 0 confirms
   heterogeneity exists. Estimated via regression on pooled data-splits.

2. GATES (Group Average Treatment Effects): Partition observations into K
   quantile groups based on S(x). Estimate average TE per group. Should be
   monotone increasing if heterogeneity is correctly ordered.

3. CLAN (Classification Analysis): Compare covariate means between the most
   and least treated groups. Identifies which risk factors drive heterogeneity.

Why repeated data-splitting:
    Each BLP/GATES regression requires an independent sample for valid
    inference (the CATE proxy S(x) was fit on the same data, creating
    dependence). We repeat 100 splits and aggregate the t-statistics via
    the median to control size. See Chernozhukov et al. (2022) Section 3.

BLP specification:
    Y_i = alpha + beta_1 * (W_i - e_hat_i)
          + beta_2 * (W_i - e_hat_i) * (S_i - S_bar)
          + gamma' * X_i + epsilon_i

    where W = treatment, e_hat = E[W|X] (treatment propensity), S = CATE proxy.
    beta_1 = ATE, beta_2 = HTE magnitude. beta_2 > 0 confirms heterogeneity.

References
----------
Chernozhukov, V., Demirer, M., Duflo, E., & Fernandez-Val, I. (2020).
    "Generic Machine Learning Inference on Heterogeneous Treatment Effects
    in Randomized Experiments, with an Application to Immunization in India."
    NBER Working Paper 24678. (Published 2025, Econometrica)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np
import polars as pl

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

try:
    from scipy import stats as _scipy_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


DataFrameLike = Union[pl.DataFrame, "pd.DataFrame"]

# Number of repeated splits for BLP/GATES aggregation
_N_SPLITS: int = 100
# Number of GATE groups
_DEFAULT_K_GROUPS: int = 5


@dataclass
class BLPResult:
    """Best Linear Predictor regression results.

    Attributes
    ----------
    beta_1:
        ATE estimate (coefficient on W - e_hat).
    beta_2:
        HTE magnitude (coefficient on (W - e_hat) * (S - S_bar)).
        Positive and significant implies heterogeneity. In practice,
        this is the central test: does the CATE proxy explain variation
        in realised treatment effects?
    beta_1_se, beta_2_se:
        Median standard errors across splits.
    beta_1_tstat, beta_2_tstat:
        Median t-statistics (Chernozhukov 2020 Eq. 2.11).
    beta_2_pvalue:
        p-value for H0: beta_2 = 0 (no heterogeneity). From median t-stat.
    heterogeneity_detected:
        True if beta_2 > 0 and p < 0.05.
    n_splits:
        Number of data splits used.
    """

    beta_1: float
    beta_2: float
    beta_1_se: float
    beta_2_se: float
    beta_1_tstat: float
    beta_2_tstat: float
    beta_2_pvalue: float
    heterogeneity_detected: bool
    n_splits: int

    def __repr__(self) -> str:
        star = "***" if self.beta_2_pvalue < 0.001 else (
            "**" if self.beta_2_pvalue < 0.01 else (
                "*" if self.beta_2_pvalue < 0.05 else ""
            )
        )
        return (
            f"BLPResult(beta_1={self.beta_1:.4f}, beta_2={self.beta_2:.4f}{star}, "
            f"p={self.beta_2_pvalue:.4f}, heterogeneity={self.heterogeneity_detected})"
        )


@dataclass
class GATESResult:
    """Group Average Treatment Effects by CATE quantile.

    Attributes
    ----------
    table:
        polars.DataFrame with columns: group, cate_lower_bound,
        cate_upper_bound, gate, gate_se, n.
        Rows are ordered from lowest to highest CATE group.
    gates_increasing:
        Whether the GATE estimates are monotone increasing across groups
        (expected if the CATE proxy correctly orders heterogeneity).
    n_groups:
        Number of GATE groups.
    """

    table: pl.DataFrame
    gates_increasing: bool
    n_groups: int

    def __repr__(self) -> str:
        return (
            f"GATESResult(n_groups={self.n_groups}, "
            f"gates_increasing={self.gates_increasing})"
        )


@dataclass
class CLANResult:
    """Classification Analysis: covariate comparison between extreme groups.

    Attributes
    ----------
    table:
        polars.DataFrame with columns: feature, mean_top, mean_bottom,
        diff, t_stat, p_value. Compares most and least treated groups.
    top_group:
        Group index (1-indexed) with highest estimated CATE.
    bottom_group:
        Group index (1-indexed) with lowest estimated CATE.
    """

    table: pl.DataFrame
    top_group: int
    bottom_group: int


@dataclass
class HeterogeneousInferenceResult:
    """Full heterogeneity inference result.

    Contains BLP, GATES, and CLAN outputs. Use summary() for a printable
    report or individual attributes for downstream analysis.
    """

    blp: BLPResult
    gates: GATESResult
    clan: CLANResult
    n_obs: int
    n_splits: int

    def summary(self) -> str:
        """Return a human-readable inference report."""
        lines = [
            "Heterogeneous Treatment Effect Inference",
            "=" * 50,
            f"N observations: {self.n_obs:,}",
            f"N data splits:  {self.n_splits}",
            "",
            "--- BLP (Best Linear Predictor) ---",
            f"  ATE (beta_1):    {self.blp.beta_1:.4f}  (SE={self.blp.beta_1_se:.4f})",
            f"  HTE (beta_2):    {self.blp.beta_2:.4f}  (SE={self.blp.beta_2_se:.4f})",
            f"  H0: beta_2=0 -> p={self.blp.beta_2_pvalue:.4f}  "
            f"{'[HETEROGENEITY DETECTED]' if self.blp.heterogeneity_detected else '[no sig. heterogeneity]'}",
            "",
            "--- GATES (Group Average Treatment Effects) ---",
        ]

        gate_tbl = self.gates.table
        for row in gate_tbl.iter_rows(named=True):
            grp = row["group"]
            gate_val = row["gate"]
            se_val = row.get("gate_se", float("nan"))
            n_val = row["n"]
            lines.append(f"  Group {grp}: GATE={gate_val:.4f}  (SE={se_val:.4f})  n={n_val:,}")

        lines.append(
            f"  Monotone increasing: {self.gates.gates_increasing}"
        )
        lines.append("")
        lines.append("--- CLAN (Classification Analysis) ---")
        if len(self.clan.table) > 0:
            top_features = (
                self.clan.table
                .sort("p_value")
                .head(5)
            )
            lines.append("  Top 5 differentiating features (most vs least treated):")
            for row in top_features.iter_rows(named=True):
                sig = "*" if row["p_value"] < 0.05 else ""
                lines.append(
                    f"    {row['feature']}: diff={row['diff']:.4f}  "
                    f"p={row['p_value']:.4f}{sig}"
                )
        return "\n".join(lines)

    def plot_gates(self) -> None:
        """Plot GATE estimates with confidence intervals across groups."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not installed. Cannot produce plot.", stacklevel=2)
            return

        tbl = self.gates.table
        groups = tbl["group"].to_list()
        gates = tbl["gate"].to_list()
        gate_se = tbl["gate_se"].to_list() if "gate_se" in tbl.columns else [0.0] * len(groups)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(groups, gates, yerr=[1.96 * se for se in gate_se],
               capsize=4, color="steelblue", alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("CATE Quantile Group (1 = lowest, K = highest)")
        ax.set_ylabel("Group Average Treatment Effect")
        ax.set_title("GATES: Group Average Treatment Effects by CATE Quantile")
        plt.tight_layout()
        plt.show()

    def plot_clan(self) -> None:
        """Plot CLAN feature differences between top and bottom GATE groups."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not installed. Cannot produce plot.", stacklevel=2)
            return

        tbl = self.clan.table.sort("diff")
        features = tbl["feature"].to_list()
        diffs = tbl["diff"].to_list()
        pvals = tbl["p_value"].to_list()

        colors = ["steelblue" if p < 0.05 else "lightgray" for p in pvals]

        fig, ax = plt.subplots(figsize=(8, max(4, len(features) * 0.35)))
        ax.barh(features, diffs, color=colors)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Mean difference (top group - bottom group)")
        ax.set_title(
            f"CLAN: Feature differences between GATE group {self.clan.top_group} "
            f"and group {self.clan.bottom_group}"
        )
        plt.tight_layout()
        plt.show()


class HeterogeneousInference:
    """Formal BLP/GATES/CLAN inference for causal forest estimates.

    Implements the Chernozhukov et al. (2020/2025) testing procedure for
    detecting and characterising heterogeneous treatment effects. Uses repeated
    data-splitting to control for the dependence between the CATE proxy and
    the regression sample.

    Parameters
    ----------
    n_splits:
        Number of repeated data splits for BLP and GATES estimation.
        100 is standard (Chernozhukov et al. 2020). More splits give
        more stable estimates at the cost of computation time.
    k_groups:
        Number of GATE groups. 5 (quintiles) is the default; use 4 for
        smaller datasets (n < 5,000) to avoid small group warnings.
    alpha:
        Significance level for heterogeneity tests.
    random_state:
        Random seed for split reproducibility.

    Examples
    --------
    >>> from insurance_causal.causal_forest.data import make_hte_renewal_data
    >>> from insurance_causal.causal_forest.estimator import HeterogeneousElasticityEstimator
    >>> from insurance_causal.causal_forest.inference import HeterogeneousInference
    >>> df = make_hte_renewal_data(n=5000)
    >>> confounders = ["age", "ncd_years", "vehicle_group", "channel"]
    >>> est = HeterogeneousElasticityEstimator(n_estimators=100, catboost_iterations=100)
    >>> est.fit(df, outcome="renewed", treatment="log_price_change",
    ...         confounders=confounders)
    >>> cates = est.cate(df)
    >>> inf = HeterogeneousInference(n_splits=20, k_groups=5)
    >>> result = inf.run(df, estimator=est, cate_proxy=cates)
    >>> print(result.summary())
    """

    def __init__(
        self,
        n_splits: int = _N_SPLITS,
        k_groups: int = _DEFAULT_K_GROUPS,
        alpha: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.n_splits = n_splits
        self.k_groups = k_groups
        self.alpha = alpha
        self.random_state = random_state

    def run(
        self,
        df: DataFrameLike,
        estimator: object,
        cate_proxy: np.ndarray,
        confounders: Optional[Sequence[str]] = None,
    ) -> HeterogeneousInferenceResult:
        """Run BLP, GATES, and CLAN inference.

        Parameters
        ----------
        df:
            Dataset used to fit the estimator (or a held-out sample).
        estimator:
            A fitted HeterogeneousElasticityEstimator (or compatible object
            with _Y_train, _D_train, _X_train attributes).
        cate_proxy:
            Per-row CATE estimates from the estimator, shape (n,). Used as
            the CATE proxy S(x) in the BLP and for grouping in GATES/CLAN.
        confounders:
            Feature names for CLAN comparison. If None, uses estimator's
            _confounders attribute.

        Returns
        -------
        HeterogeneousInferenceResult
        """
        from insurance_causal.causal_forest.estimator import _to_pandas, _extract_arrays

        df_pd = _to_pandas(df)
        Y = df_pd[estimator._outcome_col].values.astype(float)
        D = df_pd[estimator._treatment_col].values.astype(float)
        X = estimator._X_train
        feature_names = estimator._feature_names

        if confounders is None:
            confounders = estimator._confounders

        n = len(Y)
        rng = np.random.default_rng(self.random_state)

        # Run repeated splits for BLP
        blp = self._run_blp_splits(Y, D, X, cate_proxy, rng)

        # GATES: single pass on full data, grouping by CATE quantile
        gates = self._run_gates(Y, D, X, cate_proxy)

        # CLAN: covariate comparison between extreme groups
        clan = self._run_clan(df_pd, cate_proxy, list(confounders), estimator._feature_names)

        return HeterogeneousInferenceResult(
            blp=blp,
            gates=gates,
            clan=clan,
            n_obs=n,
            n_splits=self.n_splits,
        )

    def _run_blp_splits(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        X: np.ndarray,
        S: np.ndarray,
        rng: np.random.Generator,
    ) -> BLPResult:
        """Estimate BLP via repeated data-splitting."""
        from sklearn.linear_model import LinearRegression

        n = len(Y)
        S_bar = float(np.mean(S))

        beta_1_list: list[float] = []
        beta_2_list: list[float] = []
        beta_1_se_list: list[float] = []
        beta_2_se_list: list[float] = []

        for _ in range(self.n_splits):
            # Split: auxiliary half for nuisance (e_hat), main half for BLP regression
            idx = rng.permutation(n)
            split_pt = n // 2
            aux_idx = idx[:split_pt]
            main_idx = idx[split_pt:]

            # Estimate propensity on auxiliary half
            e_hat_main = _fit_propensity(
                D[aux_idx], X[aux_idx], D[main_idx], X[main_idx]
            )

            # BLP regression on main half
            Y_m = Y[main_idx]
            D_m = D[main_idx]
            S_m = S[main_idx]
            W_resid = D_m - e_hat_main  # W - e_hat
            S_centered = S_m - S_bar

            # Regressors: intercept, W_resid, W_resid * (S - S_bar)
            # (we don't include full X for speed; intercept absorbs average)
            Z = np.column_stack([
                W_resid,
                W_resid * S_centered,
            ])

            try:
                b1, b2, b1_se, b2_se = _ols_with_se(Y_m, Z)
                beta_1_list.append(b1)
                beta_2_list.append(b2)
                beta_1_se_list.append(b1_se)
                beta_2_se_list.append(b2_se)
            except Exception:
                continue

        if len(beta_1_list) == 0:
            # All splits failed; return degenerate result
            return BLPResult(
                beta_1=float("nan"), beta_2=float("nan"),
                beta_1_se=float("nan"), beta_2_se=float("nan"),
                beta_1_tstat=float("nan"), beta_2_tstat=float("nan"),
                beta_2_pvalue=float("nan"), heterogeneity_detected=False,
                n_splits=0,
            )

        # Aggregate via median (Chernozhukov 2020 Eq. 2.11)
        beta_1 = float(np.median(beta_1_list))
        beta_2 = float(np.median(beta_2_list))
        beta_1_se = float(np.median(beta_1_se_list))
        beta_2_se = float(np.median(beta_2_se_list))

        # t-statistics
        beta_1_tstat = beta_1 / (beta_1_se + 1e-12)
        beta_2_tstat = beta_2 / (beta_2_se + 1e-12)

        # p-value from normal approximation (valid under repeated splitting)
        if _SCIPY_AVAILABLE:
            beta_2_pvalue = float(2 * (1 - _scipy_stats.norm.cdf(abs(beta_2_tstat))))
        else:
            # Rough approximation: p ≈ 2 * Phi(-|t|)
            beta_2_pvalue = float(2 * np.exp(-0.5 * beta_2_tstat ** 2) / np.sqrt(2 * np.pi))

        heterogeneity_detected = (beta_2 > 0) and (beta_2_pvalue < self.alpha)

        return BLPResult(
            beta_1=beta_1,
            beta_2=beta_2,
            beta_1_se=beta_1_se,
            beta_2_se=beta_2_se,
            beta_1_tstat=beta_1_tstat,
            beta_2_tstat=beta_2_tstat,
            beta_2_pvalue=beta_2_pvalue,
            heterogeneity_detected=heterogeneity_detected,
            n_splits=len(beta_1_list),
        )

    def _run_gates(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        X: np.ndarray,
        S: np.ndarray,
    ) -> GATESResult:
        """Compute GATES by CATE quantile groups."""
        K = self.k_groups
        n = len(Y)

        # Assign quantile groups based on CATE proxy
        quantiles = np.linspace(0, 100, K + 1)
        thresholds = np.percentile(S, quantiles)
        groups = np.digitize(S, thresholds[1:-1], right=True)  # 0-indexed

        # Estimate propensity on full data (for group regression)
        e_hat = _fit_propensity(D, X, D, X)

        rows: list[dict] = []
        for k in range(K):
            mask = groups == k
            n_k = int(np.sum(mask))

            if n_k < 10:
                rows.append({
                    "group": k + 1,
                    "cate_lower": float(thresholds[k]),
                    "cate_upper": float(thresholds[k + 1]),
                    "gate": float("nan"),
                    "gate_se": float("nan"),
                    "n": n_k,
                })
                continue

            if n_k < 500:
                warnings.warn(
                    f"GATE group {k + 1} has only {n_k} policies. "
                    "Estimates may be noisy.",
                    UserWarning,
                    stacklevel=3,
                )

            Y_k = Y[mask]
            D_k = D[mask]
            e_k = e_hat[mask]
            W_resid_k = D_k - e_k

            # Simple GATE estimate: cov(Y, W-e) / var(W-e)
            try:
                gate_val, gate_se = _ols_simple(Y_k, W_resid_k)
            except Exception:
                gate_val, gate_se = float("nan"), float("nan")

            rows.append({
                "group": k + 1,
                "cate_lower": float(thresholds[k]),
                "cate_upper": float(thresholds[k + 1]),
                "gate": gate_val,
                "gate_se": gate_se,
                "n": n_k,
            })

        table = pl.DataFrame(rows).sort("group")
        gate_vals = table.filter(pl.col("gate").is_finite())["gate"].to_list()
        gates_increasing = all(gate_vals[i] <= gate_vals[i + 1] for i in range(len(gate_vals) - 1))

        return GATESResult(
            table=table,
            gates_increasing=gates_increasing,
            n_groups=K,
        )

    def _run_clan(
        self,
        df_pd: "pd.DataFrame",
        S: np.ndarray,
        confounders: list[str],
        feature_names: list[str],
    ) -> CLANResult:
        """Compare covariate means between most and least treated groups."""
        K = self.k_groups
        quantiles = np.linspace(0, 100, K + 1)
        thresholds = np.percentile(S, quantiles)
        groups = np.digitize(S, thresholds[1:-1], right=True)

        top_group = K - 1   # highest CATE group (0-indexed)
        bottom_group = 0    # lowest CATE group

        mask_top = groups == top_group
        mask_bottom = groups == bottom_group

        # Use encoded features for comparison
        import pandas as pd_mod
        subset = df_pd[confounders].copy()
        obj_cols = subset.select_dtypes(include=["object", "category"]).columns.tolist()
        if obj_cols:
            subset = pd_mod.get_dummies(subset, columns=obj_cols, drop_first=True)
        # Cast all columns to float to avoid object/bool dtypes from get_dummies
        # (pandas 2.x get_dummies produces bool columns; numpy.object_ fails in scipy)
        subset = subset.astype(float)
        subset = subset.fillna(subset.mean())

        X_top = subset.values[mask_top].astype(float)
        X_bottom = subset.values[mask_bottom].astype(float)

        rows = []
        for i, fname in enumerate(list(subset.columns)):
            top_vals = X_top[:, i]
            bottom_vals = X_bottom[:, i]

            diff = float(np.mean(top_vals) - np.mean(bottom_vals))
            if _SCIPY_AVAILABLE and len(top_vals) > 1 and len(bottom_vals) > 1:
                t_stat, p_val = _scipy_stats.ttest_ind(top_vals, bottom_vals)
                t_stat = float(t_stat)
                p_val = float(p_val)
            else:
                t_stat = float("nan")
                p_val = float("nan")

            rows.append({
                "feature": fname,
                "mean_top": float(np.mean(top_vals)),
                "mean_bottom": float(np.mean(bottom_vals)),
                "diff": diff,
                "t_stat": t_stat,
                "p_value": p_val,
            })

        table = pl.DataFrame(rows)
        return CLANResult(
            table=table,
            top_group=top_group + 1,   # 1-indexed
            bottom_group=bottom_group + 1,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _fit_propensity(
    D_train: np.ndarray,
    X_train: np.ndarray,
    D_test: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """Fit a treatment nuisance model and return predictions on test data."""
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        m = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        m.fit(X_train, D_train)
        return m.predict(X_test)
    except Exception:
        # Fallback to mean
        return np.full(len(D_test), np.mean(D_train))


def _ols_with_se(
    Y: np.ndarray,
    Z: np.ndarray,
) -> tuple[float, float, float, float]:
    """OLS with HC1 standard errors. Returns (beta_1, beta_2, se_1, se_2).

    Z has shape (n, 2) with columns [W_resid, W_resid * (S - S_bar)].
    """
    n, p = Z.shape
    if n < p + 2:
        raise ValueError("Too few observations for OLS.")

    # Add intercept
    Zi = np.column_stack([np.ones(n), Z])
    try:
        beta, _, _, _ = np.linalg.lstsq(Zi, Y, rcond=None)
    except np.linalg.LinAlgError:
        raise ValueError("OLS singular.")

    resid = Y - Zi @ beta
    # HC1 sandwich variance
    meat = Zi.T @ np.diag(resid ** 2) @ Zi
    bread = np.linalg.pinv(Zi.T @ Zi)
    V = (n / (n - p - 1)) * bread @ meat @ bread
    se = np.sqrt(np.clip(np.diag(V), 0, None))

    return float(beta[1]), float(beta[2]), float(se[1]), float(se[2])


def _ols_simple(Y: np.ndarray, W: np.ndarray) -> tuple[float, float]:
    """Simple OLS of Y on W with intercept. Returns (beta, se)."""
    n = len(Y)
    if n < 3:
        raise ValueError("Too few observations.")

    Zi = np.column_stack([np.ones(n), W])
    beta, _, _, _ = np.linalg.lstsq(Zi, Y, rcond=None)
    resid = Y - Zi @ beta
    var_W = float(np.var(W, ddof=1))
    if var_W < 1e-12:
        return float("nan"), float("nan")

    # HC1 SE for the slope
    meat = float(np.mean((W - np.mean(W)) ** 2 * resid ** 2))
    se = float(np.sqrt(max(meat / (var_W ** 2) / n, 0)))
    return float(beta[1]), se
