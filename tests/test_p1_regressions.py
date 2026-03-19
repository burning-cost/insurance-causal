"""
Regression tests for P1 important bugs fixed in insurance-causal.

P1-3: cate_by_segment() with small segments overfits — threshold extended from
      n < 500 to n < 1000 to cap CatBoost iterations and issue a warning.

DRLearner error message: previously said "binary treatment" but DRLearner
      operates on binary *outcomes* (renewal/lapse), not treatments.
"""
import warnings

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# P1-3: Small segment iteration cap
# ---------------------------------------------------------------------------

class TestSmallSegmentIterationCap:
    """
    The original threshold was n < 500.  Benchmarks showed that segments of
    500-1000 observations also overfit severely with the default CatBoost
    capacity (300+ iterations, depth 6).  The threshold has been extended to
    n < 1000.

    We test this by inspecting the source, not by fitting (fitting requires
    Databricks due to the Pi crash constraint).
    """

    def test_threshold_is_1000_not_500(self):
        """The small-segment guard must check n < 1000, not n < 500."""
        import inspect
        from insurance_causal._model import CausalPricingModel
        src = inspect.getsource(CausalPricingModel.cate_by_segment)
        assert "n_seg < 1000" in src, (
            "Small segment threshold should be 1000, not 500. "
            "Segments of 500-999 observations also overfit with default CatBoost capacity."
        )
        # The old threshold should not still be the primary guard
        # (it's acceptable for it to appear in comments; the live code must say 1000)
        lines = [l.strip() for l in src.splitlines()]
        guard_lines = [l for l in lines if "if n_seg < " in l and "min_segment_size" not in l]
        assert len(guard_lines) >= 1
        for guard in guard_lines:
            assert "1000" in guard, (
                f"Guard line still references old threshold: {guard!r}. "
                "Should be n_seg < 1000."
            )

    def test_iteration_cap_capped_at_100_for_small_segments(self):
        """
        For segments < 1000 obs, max_iter = max(50, min(100, n_seg // 5)).
        This ensures iterations never exceed 100 for small segments.
        """
        import inspect
        from insurance_causal._model import CausalPricingModel
        src = inspect.getsource(CausalPricingModel.cate_by_segment)
        assert "min(100" in src, (
            "Iteration cap for small segments should use min(100, ...) "
            "to prevent overfitting with up to 999 observations."
        )


# ---------------------------------------------------------------------------
# DRLearner error message fix
# ---------------------------------------------------------------------------

class TestDRLearnerErrorMessage:
    """
    The original error message said 'binary treatment' but DRLearner is for
    binary *outcomes* (lapse/retention indicator).  The message now correctly
    says 'binary outcome'.
    """

    def test_error_message_says_binary_outcome_not_treatment(self):
        """Error should say 'binary outcome', not 'binary treatment'."""
        import inspect
        from insurance_causal.elasticity.fit import RenewalElasticityEstimator

        src = inspect.getsource(RenewalElasticityEstimator._build_estimator)
        # Find the DRLearner error message block
        assert "binary outcome" in src.lower(), (
            "DRLearner error message should say 'binary outcome' — "
            "DRLearner is for binary outcomes (renewal/lapse), not binary treatments."
        )
        # The old wrong phrasing should be gone
        assert "binary treatment (binary_outcome" not in src, (
            "Old error message 'binary treatment (binary_outcome=True)' was "
            "confusing — treatment and outcome are different concepts. "
            "The message now correctly refers to 'binary outcome'."
        )

    def test_error_is_raised_for_continuous_outcome(self):
        """
        Building a dr_learner with binary_outcome=False should raise ValueError
        with a message that mentions 'binary outcome'.
        """
        try:
            from econml.dr import DRLearner  # noqa: F401
        except ImportError:
            pytest.skip("econml not available")

        from insurance_causal.elasticity.fit import RenewalElasticityEstimator

        model = RenewalElasticityEstimator(cate_model="dr_learner", binary_outcome=False)
        # Provide dummy models so _build_estimator can reach the DRLearner branch
        with pytest.raises(ValueError) as exc_info:
            model._build_estimator(model_y=None, model_t=None)
        msg = str(exc_info.value).lower()
        assert "binary outcome" in msg, (
            f"Error message should mention 'binary outcome', got: {exc_info.value!r}"
        )
