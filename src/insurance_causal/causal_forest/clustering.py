"""
Forest-kernel spectral clustering for CATE-based subgroup discovery.

The standard GATES/CLAN workflow characterises heterogeneity across pre-defined
groups (NCD band, channel). CausalClusteringAnalyzer turns the problem around:
let the forest itself decide which customers are alike, then ask what treatment
effect each group has.

The kernel is the core idea. A causal forest induces a similarity measure: two
observations are "close" if the forest routes them to the same leaf across many
trees. This leaf-proximity kernel captures causal heterogeneity structure in a
way that raw covariate distance cannot — it reflects *what matters for the
outcome*, not just what varies in X.

Spectral clustering on this kernel finds segments that are internally coherent
in the forest's learned feature space. AIPW pseudo-outcomes give doubly-robust
ATE estimates for each segment, with bootstrap confidence intervals.

Practical limitations:
- The n×n kernel matrix is O(n²) in memory. For n > 5000 we subsample and
  assign the remaining observations via nearest-neighbour. Above 10,000
  observations a warning is emitted.
- Honest causal forests split the sample into a building half and an estimation
  half. Only the estimation half contributes to the kernel (the building half
  has no out-of-bag leaf assignments). This means for large forests the
  effective n for kernel computation is approximately n/2. We handle this by
  working with the full prediction array and accepting mild approximation.

Design decision on kernel extraction:
    EconML's CausalForestDML does not expose leaf indices directly. We access
    the underlying GRFForest via ``estimator._estimator.forest_`` (or the
    fitted estimator's internal `._model_final_fit` attribute), but the API is
    private and version-dependent. The reliable approach is to use
    ``apply(X)`` on the underlying sklearn ExtraTreesRegressor components,
    but CausalForestDML does not expose this either. Our solution: use the
    effect predictions at training points as a proxy for leaf assignments —
    two observations share a "soft leaf" if their predicted effects are close.
    For the RBF and linear kernels we use standard sklearn similarity directly.
    For the forest kernel, we iterate through the estimator's estimators_
    attribute if available, falling back to an RBF on CATE estimates.

References
----------
Wager & Athey (2018). "Estimation and Inference of Heterogeneous Treatment
    Effects using Random Forests." JASA 113(523): 1228-1242.
Athey et al. (2019). "Generalized Random Forests." AoS 47(2): 1148-1178.
Farrell et al. (2021). "Deep Neural Networks for Estimation and Inference."
    Econometrica 89(1): 181-213.  (AIPW pseudo-outcomes)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence, Union

import numpy as np
import polars as pl

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

try:
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import pairwise_kernels, silhouette_score
    from sklearn.neighbors import NearestNeighbors
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    from scipy.linalg import eigh
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


DataFrameLike = Union[pl.DataFrame, "pd.DataFrame"]


@dataclass
class ClusteringResult:
    """Result of CausalClusteringAnalyzer.fit().

    Attributes
    ----------
    n_clusters:
        Number of clusters found.
    labels:
        Integer cluster assignment for each observation, shape (n,).
    summary:
        Polars DataFrame with columns: cluster, n, cate_mean, ate_aipw,
        ate_ci_lower, ate_ci_upper, share.
    profile:
        Polars DataFrame with cluster-level covariate means.
    silhouette_score:
        Silhouette coefficient of the clustering (range [-1, 1]).
    within_cluster_cate_var:
        Mean within-cluster CATE variance (lower = more homogeneous clusters).
    between_cluster_cate_var:
        Between-cluster CATE variance (higher = more separated clusters).
    kernel_type:
        Kernel used ('forest', 'rbf', or 'linear').
    n_obs:
        Number of observations used.
    """

    n_clusters: int
    labels: np.ndarray
    summary: pl.DataFrame
    profile: pl.DataFrame
    silhouette_score: float
    within_cluster_cate_var: float
    between_cluster_cate_var: float
    kernel_type: str
    n_obs: int


class CausalClusteringAnalyzer:
    """Forest-kernel spectral clustering for CATE-based subgroup discovery.

    Identifies customer segments with internally homogeneous treatment effects
    by running spectral clustering on the causal forest's induced kernel. Each
    segment receives an AIPW-based ATE estimate with bootstrap confidence
    intervals, enabling rigorous comparison across segments.

    Parameters
    ----------
    n_clusters:
        Number of clusters. If None, auto-select via eigengap heuristic.
    max_clusters:
        Maximum k to consider when auto-selecting (ignored if n_clusters
        is given).
    min_cluster_size:
        Minimum acceptable cluster size. Clusters smaller than this trigger
        a warning. Does not prevent small clusters from forming.
    kernel_type:
        'forest' uses the causal forest leaf-proximity kernel (best for
        CATE heterogeneity). 'rbf' and 'linear' operate directly on the
        confounder feature matrix and serve as baselines.
    n_bootstrap:
        Bootstrap replicates for cluster ATE confidence intervals.
    random_state:
        Random seed.

    Examples
    --------
    >>> from insurance_causal.causal_forest import (
    ...     HeterogeneousElasticityEstimator, make_hte_renewal_data
    ... )
    >>> from insurance_causal.causal_forest.clustering import CausalClusteringAnalyzer
    >>> df = make_hte_renewal_data(n=5000, seed=42)
    >>> confounders = ["age", "ncd_years", "vehicle_group", "channel"]
    >>> est = HeterogeneousElasticityEstimator(n_estimators=100, catboost_iterations=100)
    >>> est.fit(df, outcome="renewed", treatment="log_price_change",
    ...         confounders=confounders)
    >>> cates = est.cate(df)
    >>> ca = CausalClusteringAnalyzer(n_clusters=4, n_bootstrap=50)
    >>> result = ca.fit(df, estimator=est, cates=cates, confounders=confounders)
    >>> print(result.summary)
    """

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        max_clusters: int = 8,
        min_cluster_size: int = 200,
        kernel_type: Literal["forest", "rbf", "linear"] = "forest",
        n_bootstrap: int = 200,
        random_state: int = 42,
    ) -> None:
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for CausalClusteringAnalyzer. "
                "Install with: pip install scikit-learn"
            )
        if not _SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for CausalClusteringAnalyzer. "
                "Install with: pip install scipy"
            )

        self.n_clusters = n_clusters
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size
        self.kernel_type = kernel_type
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

        self._labels: Optional[np.ndarray] = None
        self._result: Optional[ClusteringResult] = None
        self._is_fitted: bool = False

    def fit(
        self,
        df: DataFrameLike,
        estimator: object,
        cates: np.ndarray,
        confounders: Sequence[str],
    ) -> "CausalClusteringAnalyzer":
        """Fit the clustering model.

        Parameters
        ----------
        df:
            Dataset used for fitting. Must contain confounder columns.
        estimator:
            Fitted HeterogeneousElasticityEstimator.
        cates:
            Per-row CATE estimates from estimator.cate(df), shape (n,).
        confounders:
            List of confounder column names used in the estimator.

        Returns
        -------
        self
        """
        from insurance_causal.causal_forest.estimator import (
            _to_pandas,
            _extract_features,
        )

        confounders = list(confounders)
        df_pd = _to_pandas(df)
        n = len(df_pd)
        cates = np.asarray(cates, dtype=float).flatten()
        assert len(cates) == n, f"cates length {len(cates)} != df rows {n}"

        if n > 10_000:
            warnings.warn(
                f"CausalClusteringAnalyzer received n={n:,} observations. "
                "The forest kernel is O(n²) in memory. Subsampling to 5000 "
                "for kernel computation and assigning the rest via nearest-neighbour.",
                UserWarning,
                stacklevel=2,
            )

        X, _ = _extract_features(df_pd, confounders)

        # ------------------------------------------------------------------ #
        # 1. Compute kernel                                                   #
        # ------------------------------------------------------------------ #
        use_subsample = n > 5000
        rng = np.random.default_rng(self.random_state)

        if use_subsample:
            sub_idx = rng.choice(n, size=5000, replace=False)
            sub_idx_sorted = np.sort(sub_idx)
            X_sub = X[sub_idx_sorted]
            cates_sub = cates[sub_idx_sorted]
        else:
            sub_idx_sorted = np.arange(n)
            X_sub = X
            cates_sub = cates

        K_sub = self._compute_kernel(X_sub, cates_sub, estimator)
        K_sub = _symmetrise_and_clip(K_sub)

        # ------------------------------------------------------------------ #
        # 2. Select k                                                         #
        # ------------------------------------------------------------------ #
        k = self.n_clusters
        if k is None:
            k = _eigengap_k(K_sub, max_k=self.max_clusters, random_state=self.random_state)

        # ------------------------------------------------------------------ #
        # 3. Spectral clustering on subsample                                 #
        # ------------------------------------------------------------------ #
        sc = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            random_state=self.random_state,
            assign_labels="kmeans",
        )
        sub_labels = sc.fit_predict(K_sub)

        # ------------------------------------------------------------------ #
        # 4. Assign full dataset via nearest-neighbour if subsampling         #
        # ------------------------------------------------------------------ #
        if use_subsample:
            labels = _assign_by_nn(X, X_sub, sub_labels, k=5, random_state=self.random_state)
        else:
            labels = sub_labels

        # ------------------------------------------------------------------ #
        # 5. Warn on small clusters                                           #
        # ------------------------------------------------------------------ #
        counts = np.bincount(labels, minlength=k)
        small = [int(i) for i, c in enumerate(counts) if c < self.min_cluster_size]
        if small:
            warnings.warn(
                f"Clusters {small} have fewer than {self.min_cluster_size} observations. "
                "ATE estimates in small clusters are noisy. Consider reducing n_clusters.",
                UserWarning,
                stacklevel=2,
            )

        # ------------------------------------------------------------------ #
        # 6. Cluster-level AIPW ATEs with bootstrap CIs                      #
        # ------------------------------------------------------------------ #
        Y = estimator._Y_train if hasattr(estimator, "_Y_train") and estimator._Y_train is not None else None
        D = estimator._D_train if hasattr(estimator, "_D_train") and estimator._D_train is not None else None

        summary = _compute_cluster_ates(
            cates=cates,
            labels=labels,
            k=k,
            Y=Y,
            D=D,
            n_bootstrap=self.n_bootstrap,
            rng=rng,
        )

        # ------------------------------------------------------------------ #
        # 7. Cluster profiles                                                 #
        # ------------------------------------------------------------------ #
        profile = _compute_cluster_profiles(df_pd, confounders, labels, k)

        # ------------------------------------------------------------------ #
        # 8. Diagnostics                                                      #
        # ------------------------------------------------------------------ #
        # Silhouette on X (feature space) — forest kernel silhouette would
        # require the full n×n matrix which may not exist for large n
        X_for_sil = X_sub if use_subsample else X
        labels_for_sil = labels[sub_idx_sorted] if use_subsample else labels
        try:
            sil = float(silhouette_score(X_for_sil, labels_for_sil))
        except Exception:
            sil = float("nan")

        cluster_means = np.array([cates[labels == c].mean() for c in range(k)])
        within_var = float(np.mean([
            cates[labels == c].var() for c in range(k) if (labels == c).sum() > 1
        ]))
        between_var = float(np.var(cluster_means))

        self._labels = labels
        self._result = ClusteringResult(
            n_clusters=k,
            labels=labels,
            summary=summary,
            profile=profile,
            silhouette_score=sil,
            within_cluster_cate_var=within_var,
            between_cluster_cate_var=between_var,
            kernel_type=self.kernel_type,
            n_obs=n,
        )
        self._is_fitted = True
        return self

    def cluster_labels(self) -> np.ndarray:
        """Return integer cluster assignment for each observation.

        Returns
        -------
        numpy.ndarray of shape (n,) with integer cluster indices.
        """
        self._check_fitted()
        return self._labels

    def summary(self) -> pl.DataFrame:
        """Return cluster-level ATE summary table.

        Returns
        -------
        polars.DataFrame with columns: cluster, n, cate_mean, ate_aipw,
        ate_ci_lower, ate_ci_upper, share.
        """
        self._check_fitted()
        return self._result.summary

    def profile(self, df: DataFrameLike, confounders: Sequence[str]) -> pl.DataFrame:
        """Return cluster-level covariate means.

        Parameters
        ----------
        df:
            Dataset used during fit (or same-schema dataset).
        confounders:
            Confounder columns to profile.

        Returns
        -------
        polars.DataFrame with cluster as first column, one column per
        confounder with cluster-level means.
        """
        self._check_fitted()
        from insurance_causal.causal_forest.estimator import _to_pandas
        df_pd = _to_pandas(df)
        return _compute_cluster_profiles(df_pd, list(confounders), self._labels, self._result.n_clusters)

    def suggest_n_clusters(
        self,
        df: DataFrameLike,
        estimator: object,
        cates: np.ndarray,
        confounders: Sequence[str],
    ) -> int:
        """Suggest k via the eigengap heuristic without fitting the full model.

        Parameters
        ----------
        df:
            Dataset containing confounder columns.
        estimator:
            Fitted HeterogeneousElasticityEstimator.
        cates:
            CATE estimates, shape (n,).
        confounders:
            Confounder column names.

        Returns
        -------
        Suggested number of clusters (int).
        """
        from insurance_causal.causal_forest.estimator import (
            _to_pandas,
            _extract_features,
        )
        df_pd = _to_pandas(df)
        cates = np.asarray(cates, dtype=float).flatten()
        n = len(df_pd)
        X, _ = _extract_features(df_pd, list(confounders))

        rng = np.random.default_rng(self.random_state)
        if n > 5000:
            sub_idx = np.sort(rng.choice(n, size=5000, replace=False))
            X = X[sub_idx]
            cates = cates[sub_idx]

        K = self._compute_kernel(X, cates, estimator)
        K = _symmetrise_and_clip(K)
        return _eigengap_k(K, max_k=self.max_clusters, random_state=self.random_state)

    # ---------------------------------------------------------------------- #
    # Private helpers                                                          #
    # ---------------------------------------------------------------------- #

    def _compute_kernel(
        self,
        X: np.ndarray,
        cates: np.ndarray,
        estimator: object,
    ) -> np.ndarray:
        """Compute the similarity kernel matrix.

        For 'forest': attempt to extract leaf-proximity from the econml
        CausalForestDML, falling back to RBF on CATEs. For 'rbf' and
        'linear': use sklearn pairwise_kernels directly on X.
        """
        if self.kernel_type == "rbf":
            return pairwise_kernels(X, metric="rbf")
        elif self.kernel_type == "linear":
            K = pairwise_kernels(X, metric="linear")
            # Normalise to [0, 1]
            K_min, K_max = K.min(), K.max()
            if K_max > K_min:
                K = (K - K_min) / (K_max - K_min)
            return K
        else:
            # 'forest' kernel
            return self._forest_kernel(X, cates, estimator)

    def _forest_kernel(
        self,
        X: np.ndarray,
        cates: np.ndarray,
        estimator: object,
    ) -> np.ndarray:
        """Extract or approximate the causal forest kernel.

        Strategy (in order of preference):
        1. If the internal estimator exposes estimators_ (a list of decision
           trees), compute exact leaf-co-occurrence: K[i,j] = fraction of
           trees where i and j share a leaf.
        2. Otherwise, use an RBF kernel on the CATE estimates. This is a
           weaker proxy but always available.
        """
        internal = _get_internal_forest(estimator)
        if internal is not None and hasattr(internal, "estimators_"):
            try:
                return _leaf_proximity_kernel(X, internal.estimators_)
            except Exception as e:
                warnings.warn(
                    f"Leaf-proximity kernel extraction failed ({e}). "
                    "Falling back to RBF on CATE estimates.",
                    UserWarning,
                    stacklevel=4,
                )

        # Fallback: RBF on stacked [X, cate] — down-weights X slightly so
        # the CATE dimension drives the kernel
        cate_col = cates.reshape(-1, 1) * 5.0  # amplify CATE signal
        X_aug = np.hstack([X, cate_col])
        return pairwise_kernels(X_aug, metric="rbf")

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "CausalClusteringAnalyzer is not fitted. Call .fit() first."
            )


# --------------------------------------------------------------------------- #
# Module-level helpers                                                         #
# --------------------------------------------------------------------------- #

def _get_internal_forest(estimator: object) -> Optional[object]:
    """Try to get the underlying sklearn forest from a CausalForestDML estimator.

    EconML's internal structure has changed across versions. We try a few
    known attribute paths and return None if none work.
    """
    # econml >= 0.15 stores the fitted final model here
    for attr in ("_model_final_fit", "_model_final", "_ortho_learner_model_final_"):
        obj = getattr(estimator._estimator, attr, None) if hasattr(estimator, "_estimator") else None
        if obj is None:
            obj = getattr(estimator, attr, None)
        if obj is not None:
            # May itself be a wrapper — try to get forest
            for inner in ("model_", "estimator_", "forest_", "_forest"):
                inner_obj = getattr(obj, inner, None)
                if inner_obj is not None and hasattr(inner_obj, "estimators_"):
                    return inner_obj
            if hasattr(obj, "estimators_"):
                return obj

    # Direct path: estimator._estimator is the CausalForest, which may expose
    # the underlying ExtraTreesRegressor
    internal = getattr(estimator, "_estimator", None)
    if internal is not None:
        for attr in ("estimators_", "forest_", "_model_final_fit"):
            obj = getattr(internal, attr, None)
            if obj is not None:
                if hasattr(obj, "estimators_"):
                    return obj

    return None


def _leaf_proximity_kernel(
    X: np.ndarray,
    trees: list,
) -> np.ndarray:
    """Compute leaf-proximity kernel: K[i,j] = fraction of trees i,j share a leaf.

    Parameters
    ----------
    X:
        Feature matrix, shape (n, p).
    trees:
        List of fitted decision tree estimators.

    Returns
    -------
    K: np.ndarray of shape (n, n) with values in [0, 1].
    """
    n = X.shape[0]
    K = np.zeros((n, n), dtype=np.float32)
    n_trees = len(trees)

    for tree in trees:
        # apply returns leaf index for each observation
        leaf_ids = tree.apply(X)  # shape (n,)
        # Two obs share a leaf iff their leaf ids are equal
        # Outer comparison: (n, n) boolean
        shared = (leaf_ids[:, None] == leaf_ids[None, :])  # shape (n, n)
        K += shared.astype(np.float32)

    K /= n_trees
    return K.astype(np.float64)


def _symmetrise_and_clip(K: np.ndarray) -> np.ndarray:
    """Ensure K is symmetric and values are in [0, 1]."""
    K = (K + K.T) / 2.0
    np.fill_diagonal(K, 1.0)
    K = np.clip(K, 0.0, 1.0)
    return K


def _eigengap_k(
    K: np.ndarray,
    max_k: int = 8,
    random_state: int = 42,
) -> int:
    """Select k via the eigengap of the normalised Laplacian.

    The eigengap heuristic: compute the spectrum of the normalised graph
    Laplacian, and find the largest gap between consecutive eigenvalues.
    The number of clusters equals the index of the largest gap.

    Parameters
    ----------
    K:
        Affinity matrix, shape (n, n).
    max_k:
        Maximum k to consider.
    random_state:
        Unused — kept for API consistency.

    Returns
    -------
    Suggested k (int, in range [2, max_k]).
    """
    n = K.shape[0]
    max_k = min(max_k, n - 1)

    # Normalised Laplacian: L = I - D^{-1/2} K D^{-1/2}
    d = K.sum(axis=1)
    # Guard against zero-degree nodes
    d = np.where(d == 0, 1.0, d)
    d_inv_sqrt = 1.0 / np.sqrt(d)
    L_sym = np.eye(n) - (d_inv_sqrt[:, None] * K * d_inv_sqrt[None, :])

    # We only need the first max_k+1 eigenvalues
    # eigh returns eigenvalues in ascending order
    try:
        eigvals = eigh(L_sym, subset_by_index=[0, max_k], eigvals_only=True)
    except Exception:
        # Fallback: full eigendecomposition
        eigvals = np.sort(np.linalg.eigvalsh(L_sym))[:max_k + 1]

    # Gaps between consecutive eigenvalues
    gaps = np.diff(eigvals[:max_k + 1])  # length max_k
    best_k = int(np.argmax(gaps)) + 1  # +1 because gap[i] = eigval[i+1]-eigval[i]
    return max(2, min(best_k, max_k))


def _assign_by_nn(
    X_full: np.ndarray,
    X_sub: np.ndarray,
    sub_labels: np.ndarray,
    k: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """Assign full dataset to clusters via nearest-neighbour majority vote.

    Parameters
    ----------
    X_full:
        Full feature matrix, shape (n, p).
    X_sub:
        Subsample feature matrix, shape (m, p).
    sub_labels:
        Cluster labels for the subsample, shape (m,).
    k:
        Number of neighbours for majority vote.

    Returns
    -------
    labels: np.ndarray of shape (n,) with integer cluster indices.
    """
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=-1)
    nn.fit(X_sub)
    _, indices = nn.kneighbors(X_full)
    # Majority vote among k neighbours
    neighbour_labels = sub_labels[indices]  # shape (n, k)
    n_clusters = sub_labels.max() + 1
    counts = np.apply_along_axis(
        lambda row: np.bincount(row, minlength=n_clusters), axis=1, arr=neighbour_labels
    )
    labels = counts.argmax(axis=1)
    return labels.astype(np.int32)


def _aipw_pseudo_outcome(
    Y: np.ndarray,
    D: np.ndarray,
    cates: np.ndarray,
) -> np.ndarray:
    """Compute AIPW pseudo-outcomes for ATE estimation.

    AIPW (Augmented Inverse Propensity Weighting) pseudo-outcome:
        psi_i = tau_hat_i + (D_i - D_bar) / var(D) * (Y_i - tau_hat_i * D_i)

    This is the influence-function-based doubly-robust estimator. When
    Y and D are not available (no training arrays stored), fall back to
    mean(cates).

    Parameters
    ----------
    Y:
        Outcome array, shape (n,).
    D:
        Treatment array, shape (n,).
    cates:
        CATE estimates, shape (n,).

    Returns
    -------
    pseudo_outcomes: np.ndarray, shape (n,).
    """
    D_bar = D.mean()
    D_var = D.var()
    if D_var < 1e-10:
        return cates.copy()

    # Residualised treatment
    D_res = D - D_bar
    # Simple outcome residual using CATE as the conditional mean model
    Y_res = Y - cates * D
    psi = cates + D_res / D_var * Y_res
    return psi


def _compute_cluster_ates(
    cates: np.ndarray,
    labels: np.ndarray,
    k: int,
    Y: Optional[np.ndarray],
    D: Optional[np.ndarray],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Compute per-cluster ATE with bootstrap CIs.

    Uses AIPW pseudo-outcomes when Y and D are available, otherwise falls
    back to mean CATE with bootstrap variance.

    Returns
    -------
    polars.DataFrame with columns: cluster, n, cate_mean, ate_aipw,
    ate_ci_lower, ate_ci_upper, share.
    """
    n_total = len(cates)
    records = []

    for c in range(k):
        mask = labels == c
        n_c = int(mask.sum())
        cate_c = cates[mask]
        cate_mean = float(cate_c.mean())

        if Y is not None and D is not None and len(Y) == n_total:
            Y_c = Y[mask]
            D_c = D[mask]
            psi_c = _aipw_pseudo_outcome(Y_c, D_c, cate_c)
        else:
            psi_c = cate_c

        ate_aipw = float(psi_c.mean())

        # Bootstrap CI on AIPW pseudo-outcomes
        boot_means = np.array([
            rng.choice(psi_c, size=n_c, replace=True).mean()
            for _ in range(n_bootstrap)
        ])
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))

        records.append({
            "cluster": c,
            "n": n_c,
            "cate_mean": cate_mean,
            "ate_aipw": ate_aipw,
            "ate_ci_lower": ci_lower,
            "ate_ci_upper": ci_upper,
            "share": n_c / n_total,
        })

    return pl.DataFrame(records).with_columns([
        pl.col("cluster").cast(pl.Int32),
        pl.col("n").cast(pl.Int32),
        pl.col("cate_mean").cast(pl.Float64),
        pl.col("ate_aipw").cast(pl.Float64),
        pl.col("ate_ci_lower").cast(pl.Float64),
        pl.col("ate_ci_upper").cast(pl.Float64),
        pl.col("share").cast(pl.Float64),
    ])


def _compute_cluster_profiles(
    df_pd: "pd.DataFrame",
    confounders: list[str],
    labels: np.ndarray,
    k: int,
) -> pl.DataFrame:
    """Compute per-cluster covariate means for numeric confounders.

    Categorical confounders are excluded (profiling them requires mode, not mean).
    String/object columns are skipped and listed in a warning.

    Returns
    -------
    polars.DataFrame with columns: cluster, n, <confounder_1>, ..., <confounder_p>
    """
    import pandas as pd

    numeric_cols = [
        c for c in confounders
        if c in df_pd.columns and pd.api.types.is_numeric_dtype(df_pd[c])
    ]
    categorical_cols = [
        c for c in confounders
        if c in df_pd.columns and not pd.api.types.is_numeric_dtype(df_pd[c])
    ]

    if categorical_cols:
        warnings.warn(
            f"Categorical columns {categorical_cols} are excluded from profile() "
            "covariate means. Numeric columns only.",
            UserWarning,
            stacklevel=3,
        )

    records = []
    for c in range(k):
        mask = labels == c
        row: dict = {"cluster": int(c), "n": int(mask.sum())}
        for col in numeric_cols:
            row[col] = float(df_pd.loc[mask, col].mean())
        records.append(row)

    if not records:
        return pl.DataFrame({"cluster": pl.Series([], dtype=pl.Int32)})

    return pl.DataFrame(records).with_columns(
        pl.col("cluster").cast(pl.Int32),
        pl.col("n").cast(pl.Int32),
    )
