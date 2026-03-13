"""Tests for _inference.py."""
import numpy as np
import pytest
from insurance_causal.autodml._inference import (
    eif_inference,
    score_bootstrap,
    run_inference,
)


class TestEifInference:
    def test_returns_four_values(self):
        psi = np.random.randn(1000)
        result = eif_inference(psi)
        assert len(result) == 4

    def test_estimate_is_mean(self):
        psi = np.array([1.0, 2.0, 3.0, 4.0])
        est, se, _, _ = eif_inference(psi)
        assert abs(est - 2.5) < 1e-10

    def test_ci_contains_estimate(self):
        rng = np.random.RandomState(0)
        psi = rng.randn(1000) + 1.0
        est, se, ci_lo, ci_hi = eif_inference(psi, level=0.95)
        assert ci_lo < est < ci_hi

    def test_ci_width_decreases_with_n(self):
        rng = np.random.RandomState(0)
        psi_small = rng.randn(100)
        psi_large = rng.randn(10000)
        _, se_small, _, _ = eif_inference(psi_small)
        _, se_large, _, _ = eif_inference(psi_large)
        assert se_large < se_small

    def test_with_sample_weight(self):
        psi = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.array([1.0, 1.0, 1.0, 1.0])
        est, se, ci_lo, ci_hi = eif_inference(psi, sample_weight=w)
        assert np.isfinite(est)

    def test_cluster_robust_se(self):
        rng = np.random.RandomState(0)
        psi = rng.randn(100)
        clusters = np.repeat(np.arange(10), 10)
        est, se, ci_lo, ci_hi = eif_inference(psi, cluster_ids=clusters)
        # Cluster SE should be finite
        assert np.isfinite(se)
        assert se > 0

    def test_level_affects_width(self):
        rng = np.random.RandomState(0)
        psi = rng.randn(1000)
        _, _, lo_90, hi_90 = eif_inference(psi, level=0.90)
        _, _, lo_99, hi_99 = eif_inference(psi, level=0.99)
        width_90 = hi_90 - lo_90
        width_99 = hi_99 - lo_99
        assert width_99 > width_90


class TestScoreBootstrap:
    def test_returns_four_values(self):
        psi = np.random.randn(200)
        result = score_bootstrap(psi, n_bootstrap=50, random_state=0)
        assert len(result) == 4

    def test_estimate_is_mean(self):
        psi = np.array([1.0, 2.0, 3.0, 4.0])
        est, se, _, _ = score_bootstrap(psi, n_bootstrap=100, random_state=0)
        assert abs(est - 2.5) < 1e-10

    def test_ci_ordering(self):
        psi = np.random.randn(500)
        _, _, ci_lo, ci_hi = score_bootstrap(psi, n_bootstrap=200, random_state=0)
        assert ci_lo < ci_hi

    def test_reproducible_with_seed(self):
        psi = np.random.randn(200)
        r1 = score_bootstrap(psi, n_bootstrap=100, random_state=42)
        r2 = score_bootstrap(psi, n_bootstrap=100, random_state=42)
        assert r1 == r2

    def test_different_seeds_differ(self):
        psi = np.random.randn(200)
        r1 = score_bootstrap(psi, n_bootstrap=100, random_state=0)
        r2 = score_bootstrap(psi, n_bootstrap=100, random_state=1)
        assert r1[1] != r2[1]  # SEs will differ


class TestRunInference:
    def test_eif_dispatch(self):
        psi = np.random.randn(500)
        result_eif = run_inference(psi, inference="eif")
        result_direct = eif_inference(psi)
        assert abs(result_eif[0] - result_direct[0]) < 1e-10

    def test_bootstrap_dispatch(self):
        psi = np.random.randn(500)
        result = run_inference(psi, inference="bootstrap", n_bootstrap=50, random_state=0)
        assert len(result) == 4

    def test_invalid_inference_raises(self):
        psi = np.random.randn(100)
        with pytest.raises(ValueError, match="Unknown inference method"):
            run_inference(psi, inference="invalid")

    def test_cluster_ids_passed_through(self):
        psi = np.random.randn(100)
        clusters = np.repeat(np.arange(10), 10)
        result = run_inference(psi, inference="eif", cluster_ids=clusters)
        assert len(result) == 4
        assert np.isfinite(result[1])
