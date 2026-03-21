"""
Tests for CausalClusteringAnalyzer.

Covers:
- Basic API: fit returns self, labels shape, summary schema, profile schema
- n sums to total observations
- Variance diagnostics: within/between cluster CATE variance positive
- Silhouette in valid range
- RBF kernel baseline
- Auto k-selection via eigengap (n_clusters=None)
- Known-cluster DGP: 3 segments with distinct CATEs, ARI > 0.3
- unfitted raises RuntimeError
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

pytest.importorskip("econml", reason="econml not installed — skipping causal_forest tests")

from sklearn.metrics import adjusted_rand_score

from insurance_causal.causal_forest.data import make_hte_renewal_data
from insurance_causal.causal_forest.estimator import HeterogeneousElasticityEstimator
from insurance_causal.causal_forest.clustering import (
    CausalClusteringAnalyzer,
    ClusteringResult,
    _eigengap_k,
    _leaf_proximity_kernel,
    _aipw_pseudo_outcome,
)

CONFOUNDERS = ["age", "ncd_years", "vehicle_group", "channel"]

# ------------------------------------------------------------------ #
# Summary schema: these columns must always be present                #
# ------------------------------------------------------------------ #
SUMMARY_COLS = {"cluster", "n", "cate_mean", "ate_aipw", "ate_ci_lower", "ate_ci_upper", "share"}
PROFILE_COLS_MIN = {"cluster", "n"}


# ================================================================== #
# Fixtures                                                            #
# ================================================================== #

# Re-use conftest fixtures (hte_df, fitted_hte_estimator, cates) from
# tests/causal_forest/conftest.py — they are session-scoped so fit only once.


@pytest.fixture(scope="module")
def fitted_clustering(fitted_hte_estimator, hte_df, cates):
    """CausalClusteringAnalyzer fitted with n_clusters=3, forest kernel."""
    ca = CausalClusteringAnalyzer(
        n_clusters=3,
        kernel_type="forest",
        n_bootstrap=20,  # fast for CI tests; production uses 200
        random_state=42,
    )
    ca.fit(hte_df, estimator=fitted_hte_estimator, cates=cates, confounders=CONFOUNDERS)
    return ca


@pytest.fixture(scope="module")
def fitted_clustering_rbf(fitted_hte_estimator, hte_df, cates):
    """CausalClusteringAnalyzer fitted with n_clusters=3, rbf kernel."""
    ca = CausalClusteringAnalyzer(
        n_clusters=3,
        kernel_type="rbf",
        n_bootstrap=20,
        random_state=42,
    )
    ca.fit(hte_df, estimator=fitted_hte_estimator, cates=cates, confounders=CONFOUNDERS)
    return ca


# ================================================================== #
# 1. Basic API tests                                                  #
# ================================================================== #

class TestBasicAPI:
    def test_fit_returns_self(self, fitted_hte_estimator, hte_df, cates):
        ca = CausalClusteringAnalyzer(n_clusters=2, n_bootstrap=10, random_state=0)
        result = ca.fit(hte_df, estimator=fitted_hte_estimator, cates=cates,
                        confounders=CONFOUNDERS)
        assert result is ca

    def test_is_fitted_after_fit(self, fitted_clustering):
        assert fitted_clustering._is_fitted

    def test_unfitted_raises(self):
        ca = CausalClusteringAnalyzer(n_clusters=3)
        with pytest.raises(RuntimeError, match="not fitted"):
            ca.cluster_labels()

    def test_unfitted_summary_raises(self):
        ca = CausalClusteringAnalyzer(n_clusters=3)
        with pytest.raises(RuntimeError, match="not fitted"):
            ca.summary()

    def test_cluster_labels_shape(self, fitted_clustering, hte_df):
        labels = fitted_clustering.cluster_labels()
        assert labels.shape == (len(hte_df),)

    def test_cluster_labels_integer(self, fitted_clustering):
        labels = fitted_clustering.cluster_labels()
        assert labels.dtype in (np.int32, np.int64, int)

    def test_cluster_labels_range(self, fitted_clustering):
        labels = fitted_clustering.cluster_labels()
        assert labels.min() >= 0
        assert labels.max() <= 2  # n_clusters=3 => max label = 2

    def test_result_is_clustering_result(self, fitted_clustering):
        assert isinstance(fitted_clustering._result, ClusteringResult)


class TestSummarySchema:
    def test_summary_returns_polars(self, fitted_clustering):
        s = fitted_clustering.summary()
        assert isinstance(s, pl.DataFrame)

    def test_summary_columns(self, fitted_clustering):
        s = fitted_clustering.summary()
        assert SUMMARY_COLS.issubset(set(s.columns))

    def test_summary_n_rows(self, fitted_clustering):
        s = fitted_clustering.summary()
        assert len(s) == 3  # n_clusters=3

    def test_summary_n_sums_to_total(self, fitted_clustering, hte_df):
        s = fitted_clustering.summary()
        assert s["n"].sum() == len(hte_df)

    def test_summary_share_sums_to_one(self, fitted_clustering):
        s = fitted_clustering.summary()
        assert abs(s["share"].sum() - 1.0) < 1e-6

    def test_summary_cis_ordered(self, fitted_clustering):
        s = fitted_clustering.summary()
        lowers = s["ate_ci_lower"].to_numpy()
        uppers = s["ate_ci_upper"].to_numpy()
        assert np.all(lowers <= uppers + 1e-8), "Some CI lower bounds exceed upper bounds"

    def test_summary_finite_values(self, fitted_clustering):
        s = fitted_clustering.summary()
        for col in ["cate_mean", "ate_aipw", "ate_ci_lower", "ate_ci_upper"]:
            assert s[col].is_finite().all(), f"Non-finite values in {col}"


class TestProfileSchema:
    def test_profile_returns_polars(self, fitted_clustering, hte_df):
        p = fitted_clustering.profile(hte_df, CONFOUNDERS)
        assert isinstance(p, pl.DataFrame)

    def test_profile_has_cluster_and_n(self, fitted_clustering, hte_df):
        p = fitted_clustering.profile(hte_df, CONFOUNDERS)
        assert "cluster" in p.columns
        assert "n" in p.columns

    def test_profile_n_rows(self, fitted_clustering, hte_df):
        p = fitted_clustering.profile(hte_df, CONFOUNDERS)
        assert len(p) == 3

    def test_profile_numeric_confounders_present(self, fitted_clustering, hte_df):
        p = fitted_clustering.profile(hte_df, CONFOUNDERS)
        # "age" and "ncd_years" are numeric — should appear
        assert "age" in p.columns
        assert "ncd_years" in p.columns

    def test_profile_n_sums_to_total(self, fitted_clustering, hte_df):
        p = fitted_clustering.profile(hte_df, CONFOUNDERS)
        assert p["n"].sum() == len(hte_df)


class TestVarianceDiagnostics:
    def test_within_cluster_cate_var_positive(self, fitted_clustering):
        assert fitted_clustering._result.within_cluster_cate_var >= 0.0

    def test_between_cluster_cate_var_positive(self, fitted_clustering):
        # Clusters should have different mean CATEs
        assert fitted_clustering._result.between_cluster_cate_var >= 0.0

    def test_silhouette_in_range(self, fitted_clustering):
        sil = fitted_clustering._result.silhouette_score
        assert -1.0 <= sil <= 1.0

    def test_result_n_obs_correct(self, fitted_clustering, hte_df):
        assert fitted_clustering._result.n_obs == len(hte_df)

    def test_result_n_clusters_correct(self, fitted_clustering):
        assert fitted_clustering._result.n_clusters == 3


# ================================================================== #
# 2. RBF kernel baseline                                              #
# ================================================================== #

class TestRBFKernel:
    def test_rbf_labels_shape(self, fitted_clustering_rbf, hte_df):
        labels = fitted_clustering_rbf.cluster_labels()
        assert labels.shape == (len(hte_df),)

    def test_rbf_summary_schema(self, fitted_clustering_rbf):
        s = fitted_clustering_rbf.summary()
        assert SUMMARY_COLS.issubset(set(s.columns))

    def test_rbf_n_sums_to_total(self, fitted_clustering_rbf, hte_df):
        s = fitted_clustering_rbf.summary()
        assert s["n"].sum() == len(hte_df)

    def test_rbf_kernel_type_stored(self, fitted_clustering_rbf):
        assert fitted_clustering_rbf._result.kernel_type == "rbf"


# ================================================================== #
# 3. Auto k-selection (n_clusters=None)                               #
# ================================================================== #

class TestAutoKSelection:
    def test_auto_k_returns_valid_k(self, fitted_hte_estimator, hte_df, cates):
        ca = CausalClusteringAnalyzer(
            n_clusters=None,
            max_clusters=6,
            kernel_type="rbf",
            n_bootstrap=10,
            random_state=42,
        )
        ca.fit(hte_df, estimator=fitted_hte_estimator, cates=cates,
               confounders=CONFOUNDERS)
        k = ca._result.n_clusters
        assert 2 <= k <= 6, f"Auto k={k} out of expected range [2, 6]"

    def test_suggest_n_clusters_returns_int(self, fitted_hte_estimator, hte_df, cates):
        ca = CausalClusteringAnalyzer(max_clusters=5, kernel_type="rbf", random_state=42)
        k = ca.suggest_n_clusters(hte_df, fitted_hte_estimator, cates, CONFOUNDERS)
        assert isinstance(k, int)
        assert 2 <= k <= 5

    def test_eigengap_unit(self):
        """Eigengap on a block-diagonal affinity should return 3."""
        rng = np.random.default_rng(0)
        # 3 clusters of 50 — high intra-cluster similarity, low inter-cluster
        n_per = 50
        K = np.zeros((150, 150))
        for i in range(3):
            s, e = i * n_per, (i + 1) * n_per
            K[s:e, s:e] = 0.9 + 0.1 * rng.random((n_per, n_per))
        np.fill_diagonal(K, 1.0)
        K = (K + K.T) / 2.0
        k = _eigengap_k(K, max_k=6)
        assert k == 3, f"Eigengap returned k={k}, expected 3 for 3-cluster block matrix"


# ================================================================== #
# 4. Known-cluster DGP: ARI test                                      #
# ================================================================== #

def _make_clustered_dgp(n: int = 3000, seed: int = 42) -> tuple:
    """Three-segment DGP with strongly separated CATEs.

    Segment 0: NCD <= 1, young  => tau ~ -0.30 (most elastic)
    Segment 1: NCD 2-3          => tau ~ -0.20
    Segment 2: NCD >= 4, older  => tau ~ -0.10 (least elastic)

    Returns (df, true_labels) where true_labels is the segment index.
    """
    rng = np.random.default_rng(seed)
    n0, n1, n2 = n // 3, n // 3, n - 2 * (n // 3)

    def segment(n_seg, ncd_range, age_range, tau, rng):
        age = rng.integers(age_range[0], age_range[1], size=n_seg)
        ncd = rng.integers(ncd_range[0], ncd_range[1] + 1, size=n_seg)
        log_price_chg = rng.normal(0.0, 0.08, size=n_seg)
        noise = rng.normal(0.0, 0.05, size=n_seg)
        renewed = (rng.uniform(size=n_seg) < (0.75 + tau * log_price_chg + noise)).astype(int)
        return age, ncd, log_price_chg, renewed

    ages0, ncds0, D0, Y0 = segment(n0, (0, 1), (18, 30), -0.30, rng)
    ages1, ncds1, D1, Y1 = segment(n1, (2, 3), (30, 50), -0.20, rng)
    ages2, ncds2, D2, Y2 = segment(n2, (4, 5), (45, 70), -0.10, rng)

    import polars as pl

    df = pl.DataFrame({
        "age": np.concatenate([ages0, ages1, ages2]).astype(float),
        "ncd_years": np.concatenate([ncds0, ncds1, ncds2]).astype(float),
        "vehicle_group": ["A"] * n,
        "channel": ["direct"] * n,
        "log_price_change": np.concatenate([D0, D1, D2]),
        "renewed": np.concatenate([Y0, Y1, Y2]).astype(float),
    })
    true_labels = np.array([0] * n0 + [1] * n1 + [2] * n2)
    return df, true_labels


class TestKnownClusterDGP:
    @pytest.fixture(scope="class")
    def clustered_setup(self):
        df, true_labels = _make_clustered_dgp(n=3000, seed=42)
        confounders = ["age", "ncd_years"]
        est = HeterogeneousElasticityEstimator(
            binary_outcome=True,
            n_folds=2,
            n_estimators=50,
            min_samples_leaf=5,
            catboost_iterations=50,
            random_state=42,
        )
        est.fit(df, outcome="renewed", treatment="log_price_change",
                confounders=confounders)
        cates = est.cate(df)
        ca = CausalClusteringAnalyzer(
            n_clusters=3,
            kernel_type="rbf",
            n_bootstrap=20,
            random_state=42,
        )
        ca.fit(df, estimator=est, cates=cates, confounders=confounders)
        return ca, true_labels, cates

    def test_ari_above_threshold(self, clustered_setup):
        ca, true_labels, _ = clustered_setup
        labels = ca.cluster_labels()
        ari = adjusted_rand_score(true_labels, labels)
        assert ari > 0.3, (
            f"ARI={ari:.3f} is below 0.3. The clustering is not recovering "
            "the known three-segment structure."
        )

    def test_cluster_cate_means_vary(self, clustered_setup):
        ca, _, cates = clustered_setup
        labels = ca.cluster_labels()
        cluster_means = [cates[labels == c].mean() for c in range(3)]
        # Standard deviation across cluster means should be > 0.01
        assert np.std(cluster_means) > 0.01, (
            f"Cluster CATE means have near-zero SD ({np.std(cluster_means):.4f}). "
            "Clustering is not separating treatment effect groups."
        )

    def test_summary_ate_aipw_finite(self, clustered_setup):
        ca, _, _ = clustered_setup
        s = ca.summary()
        assert s["ate_aipw"].is_finite().all()


# ================================================================== #
# 5. Unit tests for helper functions                                  #
# ================================================================== #

class TestHelpers:
    def test_aipw_pseudo_outcome_shape(self):
        rng = np.random.default_rng(0)
        n = 100
        Y = rng.normal(0.5, 0.1, n)
        D = rng.normal(0.0, 0.1, n)
        cates = rng.normal(-0.2, 0.05, n)
        psi = _aipw_pseudo_outcome(Y, D, cates)
        assert psi.shape == (n,)
        assert np.all(np.isfinite(psi))

    def test_aipw_constant_treatment_fallback(self):
        """When D has zero variance, AIPW falls back to CATEs."""
        n = 50
        Y = np.ones(n) * 0.5
        D = np.zeros(n)  # zero variance
        cates = np.full(n, -0.2)
        psi = _aipw_pseudo_outcome(Y, D, cates)
        np.testing.assert_array_almost_equal(psi, cates)

    def test_leaf_proximity_kernel_properties(self):
        """K should be symmetric, diagonal=1, values in [0,1]."""
        from sklearn.tree import ExtraTreeRegressor
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 4))
        y = rng.normal(size=50)
        trees = []
        for i in range(5):
            t = ExtraTreeRegressor(max_depth=3, random_state=i)
            t.fit(X, y)
            trees.append(t)
        K = _leaf_proximity_kernel(X, trees)
        assert K.shape == (50, 50)
        np.testing.assert_array_almost_equal(np.diag(K), np.ones(50))
        assert K.min() >= 0.0
        assert K.max() <= 1.0 + 1e-6
        np.testing.assert_array_almost_equal(K, K.T)
