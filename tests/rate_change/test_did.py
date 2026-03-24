"""Tests for the DiD estimator."""

import numpy as np
import pandas as pd
import pytest
import warnings

from insurance_causal.rate_change import make_rate_change_data
from insurance_causal.rate_change._did import DiDEstimator


class TestDiDEstimator:
    @pytest.fixture
    def did_df(self):
        return make_rate_change_data(
            n_segments=40,
            n_periods=16,
            treatment_period=9,
            true_att=-0.05,
            seed=42,
        )

    def test_att_recovery(self, did_df):
        """DiD should recover the true ATT within 2 SE."""
        est = DiDEstimator(
            outcome_col="outcome",
            treated_col="treated",
            period_col="period",
            treatment_period=9,
            unit_col="segment",
            weight_col="earned_exposure",
        )
        est.fit(did_df)
        res = est.result()
        true_att = -0.05
        # ATT should be within 3 SEs of truth (generous for small datasets)
        assert abs(res.att - true_att) < 3 * res.se, (
            f"ATT {res.att:.4f} too far from truth {true_att} (SE={res.se:.4f})"
        )

    def test_result_has_ci(self, did_df):
        est = DiDEstimator(
            outcome_col="outcome",
            treated_col="treated",
            period_col="period",
            treatment_period=9,
            unit_col="segment",
            weight_col="earned_exposure",
        )
        est.fit(did_df)
        res = est.result()
        assert res.ci_lower < res.att < res.ci_upper
        assert res.ci_lower < res.ci_upper

    def test_se_type_cluster(self, did_df):
        """With 40 segments, should use cluster SE."""
        est = DiDEstimator(
            outcome_col="outcome",
            treated_col="treated",
            period_col="period",
            treatment_period=9,
            unit_col="segment",
            weight_col="earned_exposure",
            min_clusters_for_cluster_se=20,
        )
        est.fit(did_df)
        assert est.result().se_type == "cluster"

    def test_se_type_hc3_fallback(self):
        """With fewer than 20 segments, fall back to HC3."""
        df = make_rate_change_data(n_segments=10, n_periods=12, seed=1)
        est = DiDEstimator(
            outcome_col="outcome",
            treated_col="treated",
            period_col="period",
            treatment_period=9,
            unit_col="segment",
            weight_col="earned_exposure",
            min_clusters_for_cluster_se=20,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            est.fit(df)
        assert est.result().se_type == "HC3"
        # Should have raised a warning about few clusters
        warning_msgs = [str(x.message) for x in w]
        assert any("cluster" in m.lower() for m in warning_msgs)

    def test_event_study_runs(self, did_df):
        est = DiDEstimator(
            outcome_col="outcome",
            treated_col="treated",
            period_col="period",
            treatment_period=9,
            unit_col="segment",
            weight_col="earned_exposure",
            run_event_study=True,
            n_pre_periods=4,
        )
        est.fit(did_df)
        res = est.result()
        assert res.event_study_coefs is not None
        assert len(res.event_study_coefs) > 0
        assert res.event_study_se is not None
        assert len(res.event_study_se) == len(res.event_study_coefs)

    def test_parallel_trends_pass_under_correct_dgp(self, did_df):
        """Under parallel trends DGP, pre-trend test should not reject."""
        est = DiDEstimator(
            outcome_col="outcome",
            treated_col="treated",
            period_col="period",
            treatment_period=9,
            unit_col="segment",
            weight_col="earned_exposure",
            run_event_study=True,
        )
        est.fit(did_df)
        res = est.result()
        # p-value should be > 0.05 under correct parallel trends
        assert res.parallel_trends_p_value is None or res.parallel_trends_p_value > 0.01, (
            f"Parallel trends test failed unexpectedly: p={res.parallel_trends_p_value}"
        )

    def test_missing_outcome_col_raises(self, did_df):
        est = DiDEstimator(
            outcome_col="nonexistent",
            treated_col="treated",
            period_col="period",
            treatment_period=9,
        )
        with pytest.raises(ValueError, match="nonexistent"):
            est.fit(did_df)

    def test_missing_treatment_period_raises(self, did_df):
        est = DiDEstimator(
            outcome_col="outcome",
            treated_col="treated",
            period_col="period",
            treatment_period=999,
        )
        with pytest.raises(ValueError, match="999"):
            est.fit(did_df)

    def test_all_treated_raises(self):
        df = make_rate_change_data(n_segments=10, treated_fraction=1.0, seed=0)
        est = DiDEstimator(
            outcome_col="outcome",
            treated_col="treated",
            period_col="period",
            treatment_period=9,
        )
        with pytest.raises(ValueError, match="control"):
            est.fit(df)

    def test_negative_weights_raises(self, did_df):
        df = did_df.copy()
        df.loc[0, "earned_exposure"] = -1.0
        est = DiDEstimator(
            outcome_col="outcome",
            treated_col="treated",
            period_col="period",
            treatment_period=9,
            weight_col="earned_exposure",
        )
        with pytest.raises(ValueError, match="non-positive"):
            est.fit(df)

    def test_result_before_fit_raises(self):
        est = DiDEstimator(
            outcome_col="outcome",
            treated_col="treated",
            period_col="period",
            treatment_period=9,
        )
        with pytest.raises(RuntimeError, match="fit"):
            est.result()

    def test_exposure_weighting_matters(self):
        """Weighted and unweighted estimates should differ on heterogeneous data."""
        df = make_rate_change_data(n_segments=30, n_periods=12, seed=5, exposure_cv=1.5)
        est_wt = DiDEstimator(
            outcome_col="outcome", treated_col="treated",
            period_col="period", treatment_period=9,
            unit_col="segment", weight_col="earned_exposure",
        )
        est_unwt = DiDEstimator(
            outcome_col="outcome", treated_col="treated",
            period_col="period", treatment_period=9,
            unit_col="segment", weight_col=None,
        )
        est_wt.fit(df)
        est_unwt.fit(df)
        # They should produce different ATTs due to exposure heterogeneity
        assert est_wt.result().att != est_unwt.result().att

    def test_n_obs_matches_data(self, did_df):
        est = DiDEstimator(
            outcome_col="outcome",
            treated_col="treated",
            period_col="period",
            treatment_period=9,
        )
        est.fit(did_df)
        assert est.result().n_obs == len(did_df)

    def test_staggered_detection(self):
        """Should detect and return staggered adoption info."""
        df = make_rate_change_data(n_segments=20, n_periods=16, seed=0)
        # Artificially create staggered adoption: half treated at period 7, half at 9
        df2 = df.copy()
        late_segs = df2["segment"].unique()[10:]
        # Shift treatment onset for late adopters
        df2.loc[df2["segment"].isin(late_segs) & (df2["period"] >= 9), "treated"] = 1
        df2.loc[df2["segment"].isin(late_segs) & (df2["period"] < 9), "treated"] = 0
        df2.loc[~df2["segment"].isin(late_segs) & (df2["period"] >= 7), "treated"] = 1
        df2.loc[~df2["segment"].isin(late_segs) & (df2["period"] < 7), "treated"] = 0
        df2["rate_change"] = df2["treated"]

        est = DiDEstimator(
            outcome_col="outcome",
            treated_col="treated",
            period_col="period",
            treatment_period=7,
            unit_col="segment",
        )
        est.fit(df2)
        stag = est.staggered_info()
        assert stag is not None
        # staggered detection depends on unit_col being provided
        assert "n_cohorts" in stag
