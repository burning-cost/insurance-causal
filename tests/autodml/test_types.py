"""Tests for _types module."""
import numpy as np
import pytest
from insurance_causal.autodml._types import (
    EstimationResult,
    SegmentResult,
    DoseResponseResult,
    OutcomeFamily,
)


class TestOutcomeFamily:
    def test_values(self):
        assert OutcomeFamily.GAUSSIAN.value == "gaussian"
        assert OutcomeFamily.POISSON.value == "poisson"
        assert OutcomeFamily.GAMMA.value == "gamma"
        assert OutcomeFamily.TWEEDIE.value == "tweedie"

    def test_from_string(self):
        assert OutcomeFamily("gaussian") == OutcomeFamily.GAUSSIAN
        assert OutcomeFamily("poisson") == OutcomeFamily.POISSON

    def test_invalid_string(self):
        with pytest.raises(ValueError):
            OutcomeFamily("invalid")


class TestEstimationResult:
    def setup_method(self):
        self.result = EstimationResult(
            estimate=0.05,
            se=0.01,
            ci_low=0.03,
            ci_high=0.07,
            ci_level=0.95,
            n_obs=1000,
            n_folds=5,
            psi=np.array([0.04, 0.05, 0.06]),
        )

    def test_basic_attributes(self):
        assert self.result.estimate == 0.05
        assert self.result.se == 0.01
        assert self.result.ci_low == 0.03
        assert self.result.ci_high == 0.07
        assert self.result.ci_level == 0.95
        assert self.result.n_obs == 1000

    def test_pvalue_is_float(self):
        p = self.result.pvalue
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0

    def test_pvalue_significant(self):
        r = EstimationResult(
            estimate=10.0, se=0.1, ci_low=9.8, ci_high=10.2, psi=np.array([])
        )
        assert r.pvalue < 0.05

    def test_pvalue_zero_se(self):
        r = EstimationResult(
            estimate=1.0, se=0.0, ci_low=0.0, ci_high=2.0, psi=np.array([])
        )
        assert np.isnan(r.pvalue)

    def test_summary_returns_string(self):
        s = self.result.summary()
        assert isinstance(s, str)
        assert "estimate" in s
        assert "se" in s

    def test_repr(self):
        r = repr(self.result)
        assert "EstimationResult" in r
        assert "0.0500" in r

    def test_default_notes(self):
        assert self.result.notes == ""

    def test_default_psi(self):
        r = EstimationResult(
            estimate=0.1, se=0.01, ci_low=0.08, ci_high=0.12
        )
        assert isinstance(r.psi, np.ndarray)
        assert len(r.psi) == 0


class TestSegmentResult:
    def test_repr(self):
        result = EstimationResult(
            estimate=0.05, se=0.01, ci_low=0.03, ci_high=0.07, psi=np.array([])
        )
        sr = SegmentResult(segment_name="age=25-34", result=result, n_obs=200)
        r = repr(sr)
        assert "age=25-34" in r
        assert "200" in r


class TestDoseResponseResult:
    def setup_method(self):
        d = np.linspace(300, 500, 20)
        self.dr = DoseResponseResult(
            d_grid=d,
            ate=0.1 * np.ones(20),
            se=0.01 * np.ones(20),
            ci_low=0.08 * np.ones(20),
            ci_high=0.12 * np.ones(20),
            bandwidth=15.0,
            n_obs=5000,
        )

    def test_shapes_consistent(self):
        assert len(self.dr.d_grid) == len(self.dr.ate) == len(self.dr.se)

    def test_repr(self):
        r = repr(self.dr)
        assert "DoseResponseResult" in r
        assert "20" in r

    def test_bandwidth_set(self):
        assert self.dr.bandwidth == 15.0
