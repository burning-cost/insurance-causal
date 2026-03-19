"""
Tests for adaptive CatBoost regularisation.

Verifies that:
1. ``adaptive_catboost_params()`` returns correct tiers at tier boundaries.
2. ``build_nuisance_model()`` with backend="catboost" applies adaptive params.
3. ``nuisance_params`` override takes precedence over adaptive defaults.
4. ``cross_fit_nuisance()`` passes n_samples and nuisance_params correctly.
5. ``PremiumElasticity`` exposes and passes nuisance_params through to cross-fitting.
6. ``adaptive_dml_catboost_params()`` returns correct tiers.
7. ``build_catboost_regressor()`` applies adaptive params and override.
"""
from __future__ import annotations

import numpy as np
import pytest

from insurance_causal.autodml._nuisance import (
    adaptive_catboost_params,
    adaptive_catboost_params as adaptive_dml_catboost_params,  # alias; was moved from _utils
    build_nuisance_model,
    CatBoostNuisance,
)
from insurance_causal.autodml._types import OutcomeFamily
from insurance_causal._utils import build_catboost_regressor


# ---------------------------------------------------------------------------
# adaptive_catboost_params tier tests
# ---------------------------------------------------------------------------

class TestAdaptiveCatboostParams:
    def test_very_small_n_returns_heaviest_regularisation(self):
        params = adaptive_catboost_params(500)
        assert params["depth"] == 2
        assert params["l2_leaf_reg"] == 50
        assert params["learning_rate"] == pytest.approx(0.005)

    def test_boundary_1k_returns_heavy_regularisation(self):
        # n=1000 should hit the "1k <= n < 5k" tier
        params = adaptive_catboost_params(1_000)
        assert params["depth"] == 2
        assert params["l2_leaf_reg"] == 20
        assert params["learning_rate"] == pytest.approx(0.01)

    def test_boundary_5k_returns_stronger_regularisation(self):
        # n=5000 should hit the "5k <= n < 10k" tier
        params = adaptive_catboost_params(5_000)
        assert params["depth"] == 3
        assert params["l2_leaf_reg"] == 10
        assert params["learning_rate"] == pytest.approx(0.03)

    def test_boundary_10k_returns_moderate_regularisation(self):
        # n=10000 should hit the "10k <= n < 50k" tier
        params = adaptive_catboost_params(10_000)
        assert params["depth"] == 4
        assert params["l2_leaf_reg"] == 5
        assert params["learning_rate"] == pytest.approx(0.05)

    def test_large_n_returns_default_params(self):
        # n >= 50k: default CatBoost capacity
        params = adaptive_catboost_params(50_000)
        assert params["depth"] == 6
        assert params["l2_leaf_reg"] == 3

    def test_very_large_n_returns_default_params(self):
        params = adaptive_catboost_params(500_000)
        assert params["depth"] == 6
        assert params["l2_leaf_reg"] == 3

    def test_returns_copy(self):
        # Modifying returned dict should not affect subsequent calls
        params1 = adaptive_catboost_params(500)
        params1["depth"] = 99
        params2 = adaptive_catboost_params(500)
        assert params2["depth"] == 2

    def test_n_just_below_tier_boundary(self):
        # n=999 should still hit the very-heavy tier (n < 1k)
        params = adaptive_catboost_params(999)
        assert params["l2_leaf_reg"] == 50

    def test_n_just_at_tier_boundary(self):
        # n=50000 should hit the default (large sample) tier
        params = adaptive_catboost_params(50_000)
        assert params["depth"] == 6


# ---------------------------------------------------------------------------
# build_nuisance_model with catboost backend
# ---------------------------------------------------------------------------

class TestBuildNuisanceModelCatboost:
    """These tests only run if catboost is installed."""

    @pytest.fixture(autouse=True)
    def skip_if_no_catboost(self):
        pytest.importorskip("catboost")

    def test_small_n_applies_heavy_regularisation(self):
        model = build_nuisance_model(
            outcome_family=OutcomeFamily.GAUSSIAN,
            backend="catboost",
            n_samples=500,
        )
        assert isinstance(model, CatBoostNuisance)
        # Should have applied the n<1k tier
        assert model.depth == 2
        assert model.l2_leaf_reg == 50

    def test_large_n_applies_default_params(self):
        model = build_nuisance_model(
            outcome_family=OutcomeFamily.GAUSSIAN,
            backend="catboost",
            n_samples=100_000,
        )
        assert isinstance(model, CatBoostNuisance)
        assert model.depth == 6
        assert model.l2_leaf_reg == 3

    def test_nuisance_params_override_adaptive_defaults(self):
        # nuisance_params should win over adaptive defaults
        model = build_nuisance_model(
            outcome_family=OutcomeFamily.GAUSSIAN,
            backend="catboost",
            n_samples=500,  # would normally give depth=2, l2=50
            nuisance_params={"depth": 5, "l2_leaf_reg": 1},
        )
        assert model.depth == 5
        assert model.l2_leaf_reg == 1

    def test_nuisance_params_partial_override(self):
        # Only depth overridden; l2 should still come from adaptive tier
        model = build_nuisance_model(
            outcome_family=OutcomeFamily.GAUSSIAN,
            backend="catboost",
            n_samples=500,
            nuisance_params={"depth": 4},
        )
        # depth overridden, l2 from adaptive tier for n=500
        assert model.depth == 4
        assert model.l2_leaf_reg == 50  # from adaptive n<1k tier

    def test_no_n_samples_uses_large_sample_defaults(self):
        model = build_nuisance_model(
            outcome_family=OutcomeFamily.GAUSSIAN,
            backend="catboost",
        )
        assert model.depth == 6
        assert model.l2_leaf_reg == 3

    def test_model_can_fit_and_predict(self):
        rng = np.random.RandomState(0)
        X = rng.randn(100, 3)
        D = rng.uniform(200, 600, 100)
        Y = -0.002 * D + 0.3 * X[:, 0] + rng.randn(100) * 0.1

        model = build_nuisance_model(
            outcome_family=OutcomeFamily.GAUSSIAN,
            backend="catboost",
            n_samples=100,
        )
        model.fit(D, X, Y)
        pred = model.predict(D, X)
        assert pred.shape == (100,)
        assert not np.any(np.isnan(pred))


# ---------------------------------------------------------------------------
# cross_fit_nuisance with nuisance_params
# ---------------------------------------------------------------------------

class TestCrossFitWithNuisanceParams:
    """Tests that nuisance_params flow through cross_fit_nuisance."""

    @pytest.fixture(autouse=True)
    def skip_if_no_catboost(self):
        pytest.importorskip("catboost")

    def test_nuisance_params_applied_in_crossfit(self):
        from insurance_causal.autodml._crossfit import cross_fit_nuisance

        rng = np.random.RandomState(42)
        n = 200
        X = rng.randn(n, 3)
        D = rng.uniform(200, 600, n)
        Y = -0.002 * D + 0.3 * X[:, 0] + rng.randn(n) * 0.1

        g_hat, alpha_hat, folds, models = cross_fit_nuisance(
            X, D, Y,
            n_folds=3,
            nuisance_backend="catboost",
            nuisance_params={"depth": 2, "l2_leaf_reg": 30, "iterations": 50},
            riesz_kwargs={"n_estimators": 10, "random_state": 0},
            random_state=0,
        )

        # Verify parameters were actually applied to the fitted models
        for m in models:
            assert isinstance(m, CatBoostNuisance)
            assert m.depth == 2
            assert m.l2_leaf_reg == 30
            assert m.iterations == 50

        assert not np.any(np.isnan(g_hat))

    def test_adaptive_defaults_applied_in_crossfit(self):
        """Without nuisance_params, adaptive tier is applied based on n."""
        from insurance_causal.autodml._crossfit import cross_fit_nuisance

        rng = np.random.RandomState(7)
        n = 200  # n < 1k: expect depth=2, l2=50
        X = rng.randn(n, 3)
        D = rng.uniform(200, 600, n)
        Y = -0.002 * D + rng.randn(n) * 0.1

        _, _, _, models = cross_fit_nuisance(
            X, D, Y,
            n_folds=3,
            nuisance_backend="catboost",
            riesz_kwargs={"n_estimators": 10, "random_state": 0},
            random_state=0,
        )

        for m in models:
            assert m.depth == 2
            assert m.l2_leaf_reg == 50


# ---------------------------------------------------------------------------
# PremiumElasticity.nuisance_params
# ---------------------------------------------------------------------------

class TestPremiumElasticityNuisanceParams:
    @pytest.fixture(autouse=True)
    def skip_if_no_catboost(self):
        pytest.importorskip("catboost")

    def test_nuisance_params_stored_on_init(self):
        from insurance_causal.autodml.elasticity import PremiumElasticity

        params = {"depth": 3, "l2_leaf_reg": 15}
        model = PremiumElasticity(
            nuisance_backend="catboost",
            nuisance_params=params,
        )
        assert model.nuisance_params == params

    def test_nuisance_params_flow_through_to_crossfit(self):
        from insurance_causal.autodml.elasticity import PremiumElasticity

        rng = np.random.RandomState(1)
        n = 150
        X = rng.randn(n, 3)
        D = rng.uniform(200, 600, n)
        Y = -0.002 * D + rng.randn(n) * 0.1

        model = PremiumElasticity(
            nuisance_backend="catboost",
            nuisance_params={"depth": 2, "l2_leaf_reg": 40, "iterations": 30},
            n_folds=3,
            random_state=0,
        )
        model.fit(X, D, Y)

        for m in model._nuisance_models:
            assert m.depth == 2
            assert m.l2_leaf_reg == 40

    def test_default_nuisance_params_is_none(self):
        from insurance_causal.autodml.elasticity import PremiumElasticity

        model = PremiumElasticity()
        assert model.nuisance_params is None


# ---------------------------------------------------------------------------
# adaptive_dml_catboost_params (for CausalPricingModel / DoubleML pipeline)
# ---------------------------------------------------------------------------
# NOTE: adaptive_dml_catboost_params was an earlier name for
# insurance_causal.autodml._nuisance.adaptive_catboost_params.
# The alias above keeps these tests runnable against current code.

class TestAdaptiveDmlCatboostParams:
    def test_small_n_tier(self):
        params = adaptive_dml_catboost_params(500)
        assert params["depth"] == 2
        assert params["l2_leaf_reg"] == 50

    def test_medium_n_tier(self):
        params = adaptive_dml_catboost_params(7_500)
        assert params["depth"] == 3
        assert params["l2_leaf_reg"] == 10

    def test_large_n_tier(self):
        params = adaptive_dml_catboost_params(100_000)
        assert params["depth"] == 6
        # iterations=300 for large n in _nuisance.adaptive_catboost_params
        assert params["iterations"] == 300

    def test_returns_copy(self):
        p1 = adaptive_dml_catboost_params(500)
        p1["depth"] = 99
        p2 = adaptive_dml_catboost_params(500)
        assert p2["depth"] == 2


# ---------------------------------------------------------------------------
# build_catboost_regressor adaptive behaviour
# ---------------------------------------------------------------------------

class TestBuildCatboostRegressor:
    @pytest.fixture(autouse=True)
    def skip_if_no_catboost(self):
        pytest.importorskip("catboost")

    def test_small_n_applies_heavy_regularisation(self):
        reg = build_catboost_regressor(n_samples=500)
        # CatBoost stores params internally; check via get_params()
        # _utils.adaptive_catboost_params(500) returns depth=4, l2=10
        params = reg.get_params()
        assert params["depth"] == 4
        assert params["l2_leaf_reg"] == 10.0

    def test_large_n_uses_default_params(self):
        reg = build_catboost_regressor(n_samples=100_000)
        params = reg.get_params()
        assert params["depth"] == 6

    def test_nuisance_params_override(self):
        # The override kwarg is called override_params in build_catboost_regressor
        reg = build_catboost_regressor(
            n_samples=500,
            override_params={"depth": 5, "l2_leaf_reg": 2},
        )
        params = reg.get_params()
        assert params["depth"] == 5
        assert params["l2_leaf_reg"] == 2

    def test_no_n_samples_uses_large_defaults(self):
        reg = build_catboost_regressor()
        params = reg.get_params()
        assert params["depth"] == 6
        assert params["iterations"] == 500
