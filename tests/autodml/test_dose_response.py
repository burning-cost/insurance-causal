"""Tests for DoseResponseCurve."""
import numpy as np
import pytest
from insurance_causal.autodml.dose_response import DoseResponseCurve
from insurance_causal.autodml.dgp import SyntheticContinuousDGP
from insurance_causal.autodml._types import DoseResponseResult


def make_data(n=400, seed=0):
    dgp = SyntheticContinuousDGP(n=n, n_features=4, outcome_family="gaussian", random_state=seed)
    X, D, Y, _ = dgp.generate()
    return X, D, Y


class TestDoseResponseCurveBasic:
    def test_fit_returns_self(self):
        X, D, Y = make_data()
        model = DoseResponseCurve(n_folds=2, random_state=0)
        result = model.fit(X, D, Y)
        assert result is model

    def test_is_fitted_after_fit(self):
        X, D, Y = make_data()
        model = DoseResponseCurve(n_folds=2, random_state=0)
        model.fit(X, D, Y)
        assert model._is_fitted

    def test_predict_raises_before_fit(self):
        model = DoseResponseCurve()
        with pytest.raises(RuntimeError):
            model.predict([300.0, 400.0])

    def test_predict_returns_result(self):
        X, D, Y = make_data()
        model = DoseResponseCurve(n_folds=2, random_state=0)
        model.fit(X, D, Y)
        d_grid = np.linspace(250, 550, 10)
        result = model.predict(d_grid)
        assert isinstance(result, DoseResponseResult)

    def test_result_shapes_match_grid(self):
        X, D, Y = make_data()
        model = DoseResponseCurve(n_folds=2, random_state=0)
        model.fit(X, D, Y)
        d_grid = np.linspace(250, 550, 15)
        result = model.predict(d_grid)
        assert len(result.ate) == 15
        assert len(result.se) == 15
        assert len(result.ci_low) == 15
        assert len(result.ci_high) == 15

    def test_ci_ordering(self):
        X, D, Y = make_data()
        model = DoseResponseCurve(n_folds=2, random_state=0)
        model.fit(X, D, Y)
        d_grid = np.linspace(280, 520, 8)
        result = model.predict(d_grid)
        valid = ~np.isnan(result.ate)
        assert np.all(result.ci_low[valid] < result.ate[valid])
        assert np.all(result.ate[valid] < result.ci_high[valid])

    def test_bandwidth_set_silverman(self):
        X, D, Y = make_data()
        model = DoseResponseCurve(n_folds=2, bandwidth="silverman", random_state=0)
        model.fit(X, D, Y)
        assert model._bw > 0

    def test_bandwidth_fixed_float(self):
        X, D, Y = make_data()
        model = DoseResponseCurve(n_folds=2, bandwidth=20.0, random_state=0)
        model.fit(X, D, Y)
        assert model._bw == 20.0

    def test_bandwidth_stored_in_result(self):
        X, D, Y = make_data()
        model = DoseResponseCurve(n_folds=2, bandwidth=25.0, random_state=0)
        model.fit(X, D, Y)
        result = model.predict([350.0])
        assert result.bandwidth == 25.0


class TestDoseResponseKernels:
    def test_gaussian_kernel(self):
        model = DoseResponseCurve(kernel="gaussian")
        u = np.array([0.0, 1.0, 2.0])
        k = model._kernel_fn(u)
        assert k[0] > k[1] > k[2]

    def test_epanechnikov_kernel_zero_outside(self):
        model = DoseResponseCurve(kernel="epanechnikov")
        u = np.array([0.5, 1.5])
        k = model._kernel_fn(u)
        assert k[0] > 0
        assert k[1] == 0.0

    def test_uniform_kernel(self):
        model = DoseResponseCurve(kernel="uniform")
        u = np.array([0.5, 1.5])
        k = model._kernel_fn(u)
        assert k[0] == 0.5
        assert k[1] == 0.0

    def test_invalid_kernel_raises(self):
        model = DoseResponseCurve(kernel="invalid")
        with pytest.raises(ValueError):
            model._kernel_fn(np.array([0.0]))


class TestDoseResponseSilverman:
    def test_silverman_positive(self):
        D = np.random.randn(1000) * 50 + 350
        bw = DoseResponseCurve._silverman_bw(D)
        assert bw > 0

    def test_silverman_scales_with_std(self):
        D_narrow = np.random.randn(500) * 10 + 350
        D_wide = np.random.randn(500) * 100 + 350
        bw_narrow = DoseResponseCurve._silverman_bw(D_narrow)
        bw_wide = DoseResponseCurve._silverman_bw(D_wide)
        assert bw_wide > bw_narrow


class TestDoseResponsePlot:
    def test_plot_raises_without_matplotlib(self):
        """If matplotlib is not importable, plot() should raise ImportError."""
        X, D, Y = make_data(n=200)
        model = DoseResponseCurve(n_folds=2, random_state=0)
        model.fit(X, D, Y)
        try:
            import matplotlib
            matplotlib.use("Agg")
            ax = model.plot()
            import matplotlib.pyplot as plt
            plt.close("all")
            assert ax is not None
        except ImportError:
            with pytest.raises(ImportError):
                model.plot()

    def test_plot_raises_before_fit(self):
        model = DoseResponseCurve()
        with pytest.raises(RuntimeError):
            model.plot()


class TestDoseResponseOutsideSupport:
    def test_warns_for_extreme_d(self):
        X, D, Y = make_data(n=400)
        model = DoseResponseCurve(n_folds=2, bandwidth=1.0, random_state=0)
        model.fit(X, D, Y)
        # Predict at a point far outside observed support with tiny bandwidth
        with pytest.warns(RuntimeWarning, match="Kernel weight sum"):
            model.predict([10000.0])
