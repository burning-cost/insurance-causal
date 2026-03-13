"""Tests for ElasticityReport."""
import json
import numpy as np
import pytest
import tempfile
import os

from insurance_causal.autodml.report import ElasticityReport
from insurance_causal.autodml._types import EstimationResult, SegmentResult


class MockEstimator:
    """Minimal estimator stub for testing ElasticityReport."""

    def __init__(self, estimate=0.05):
        self.result_ = EstimationResult(
            estimate=estimate,
            se=0.01,
            ci_low=estimate - 0.02,
            ci_high=estimate + 0.02,
            ci_level=0.95,
            n_obs=5000,
            n_folds=5,
            psi=np.random.randn(5000),
        )


class TestElasticityReportBasic:
    def test_raises_without_result(self):
        class NoResult:
            result_ = None

        with pytest.raises(ValueError, match="estimate()"):
            ElasticityReport(NoResult())

    def test_to_html_returns_string(self):
        est = MockEstimator()
        report = ElasticityReport(est)
        html = report.to_html()
        assert isinstance(html, str)
        assert "<html" in html

    def test_to_html_contains_estimate(self):
        est = MockEstimator(estimate=0.1234)
        report = ElasticityReport(est)
        html = report.to_html()
        assert "0.1234" in html

    def test_to_html_contains_fca_section(self):
        est = MockEstimator()
        report = ElasticityReport(est)
        html = report.to_html()
        assert "FCA" in html

    def test_to_html_writes_file(self):
        est = MockEstimator()
        report = ElasticityReport(est)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            report.to_html(path=path)
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert "<html" in content
        finally:
            os.unlink(path)

    def test_to_json_returns_dict(self):
        est = MockEstimator()
        report = ElasticityReport(est)
        data = report.to_json()
        assert isinstance(data, dict)

    def test_to_json_keys(self):
        est = MockEstimator()
        report = ElasticityReport(est)
        data = report.to_json()
        assert "overall_ame" in data
        assert "segments" in data
        assert "version" in data

    def test_to_json_estimate_correct(self):
        est = MockEstimator(estimate=0.0789)
        report = ElasticityReport(est)
        data = report.to_json()
        assert abs(data["overall_ame"]["estimate"] - 0.0789) < 1e-6

    def test_to_json_writes_file(self):
        est = MockEstimator()
        report = ElasticityReport(est)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            report.to_json(path=path)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert "overall_ame" in data
        finally:
            os.unlink(path)


class TestElasticityReportWithSegments:
    def make_segment_results(self):
        results = []
        for name in ["young", "middle", "old"]:
            r = EstimationResult(
                estimate=0.05,
                se=0.01,
                ci_low=0.03,
                ci_high=0.07,
                psi=np.random.randn(100),
            )
            results.append(SegmentResult(segment_name=name, result=r, n_obs=100))
        return results

    def test_segment_table_in_html(self):
        est = MockEstimator()
        segs = self.make_segment_results()
        report = ElasticityReport(est, segment_results=segs)
        html = report.to_html()
        assert "young" in html
        assert "middle" in html

    def test_segment_data_in_json(self):
        est = MockEstimator()
        segs = self.make_segment_results()
        report = ElasticityReport(est, segment_results=segs)
        data = report.to_json()
        assert len(data["segments"]) == 3
        names = [s["segment"] for s in data["segments"]]
        assert "young" in names

    def test_no_segment_html_placeholder(self):
        est = MockEstimator()
        report = ElasticityReport(est)
        html = report.to_html()
        assert "No segment results provided" in html


class TestElasticityReportSensitivity:
    def test_sensitivity_in_html(self):
        est = MockEstimator()
        sens = {
            1.0: {"lower": 0.04, "upper": 0.06, "gamma": 1.0},
            2.0: {"lower": 0.02, "upper": 0.08, "gamma": 2.0},
        }
        report = ElasticityReport(est, sensitivity_bounds=sens)
        html = report.to_html()
        assert "Gamma" in html or "1.0" in html

    def test_no_sensitivity_html_placeholder(self):
        est = MockEstimator()
        report = ElasticityReport(est)
        html = report.to_html()
        assert "No sensitivity analysis provided" in html

    def test_sensitivity_in_json(self):
        est = MockEstimator()
        sens = {1.0: {"lower": 0.03, "upper": 0.07, "gamma": 1.0}}
        report = ElasticityReport(est, sensitivity_bounds=sens)
        data = report.to_json()
        assert "sensitivity_bounds" in data
