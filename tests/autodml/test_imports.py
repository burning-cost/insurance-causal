"""Tests for public API imports and __init__.py."""
import pytest


class TestImports:
    def test_main_estimators_importable(self):
        from insurance_causal.autodml import (
            PremiumElasticity,
            DoseResponseCurve,
            PolicyShiftEffect,
            SelectionCorrectedElasticity,
        )
        assert PremiumElasticity is not None
        assert DoseResponseCurve is not None
        assert PolicyShiftEffect is not None
        assert SelectionCorrectedElasticity is not None

    def test_types_importable(self):
        from insurance_causal.autodml import (
            EstimationResult,
            SegmentResult,
            DoseResponseResult,
            OutcomeFamily,
        )
        assert EstimationResult is not None
        assert SegmentResult is not None
        assert DoseResponseResult is not None
        assert OutcomeFamily is not None

    def test_riesz_importable(self):
        from insurance_causal.autodml import ForestRiesz, LinearRiesz
        assert ForestRiesz is not None
        assert LinearRiesz is not None

    def test_dgp_importable(self):
        from insurance_causal.autodml import SyntheticContinuousDGP
        assert SyntheticContinuousDGP is not None

    def test_report_importable(self):
        from insurance_causal.autodml import ElasticityReport
        assert ElasticityReport is not None

    def test_version(self):
        import insurance_causal.autodml as autodml
        assert hasattr(autodml, "__version__")
        assert isinstance(autodml.__version__, str)
        # Should be semantic version
        parts = autodml.__version__.split(".")
        assert len(parts) == 3

    def test_all_in_dunder_all(self):
        import insurance_causal.autodml as autodml
        for name in autodml.__all__:
            assert hasattr(autodml, name), f"{name} in __all__ but not importable"
