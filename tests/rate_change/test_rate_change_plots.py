"""Smoke tests for rate_change plot functions.

These tests verify that each plotting function can be called with valid inputs
and returns an Axes without raising. We also check one error/edge case per function.

matplotlib Agg backend is forced so tests run headless without a display.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from insurance_causal.rate_change._plots import (
    plot_event_study,
    plot_pre_post_did,
    plot_pre_post_its,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_event_study_df(n_pre=3, n_post=4):
    """Minimal event_study_df with required columns."""
    times = list(range(-n_pre, 0)) + list(range(0, n_post))
    att = np.linspace(-0.01, -0.05, len(times))
    se = np.full(len(times), 0.01)
    return pd.DataFrame(
        {
            "event_time": times,
            "att_e": att,
            "se_e": se,
            "ci_lower_e": att - 1.96 * se,
            "ci_upper_e": att + 1.96 * se,
        }
    )


def _make_did_df():
    """Minimal panel with period, outcome, treated columns."""
    periods = list(range(1, 11)) * 2
    treated = [1] * 10 + [0] * 10
    rng = np.random.default_rng(42)
    outcome = rng.uniform(0.05, 0.15, 20)
    exposure = rng.uniform(50, 200, 20)
    return pd.DataFrame(
        {
            "period": periods,
            "loss_ratio": outcome,
            "treated": treated,
            "exposure": exposure,
        }
    )


def _make_its_df(n=10):
    """Minimal time series with _time_ column required by plot_pre_post_its."""
    rng = np.random.default_rng(7)
    periods = list(range(1, n + 1))
    time_ = list(range(1, n + 1))
    outcome = 0.10 + 0.002 * np.array(time_) + rng.normal(0, 0.005, n)
    return pd.DataFrame(
        {
            "period": periods,
            "_time_": time_,
            "loss_ratio": outcome,
        }
    )


# ---------------------------------------------------------------------------
# plot_event_study
# ---------------------------------------------------------------------------

class TestPlotEventStudy:
    def test_returns_axes(self):
        df = _make_event_study_df()
        ax = plot_event_study(df)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_uses_provided_ax(self):
        fig, ax_in = plt.subplots()
        df = _make_event_study_df()
        ax_out = plot_event_study(df, ax=ax_in)
        assert ax_out is ax_in
        plt.close("all")

    def test_custom_title(self):
        df = _make_event_study_df()
        ax = plot_event_study(df, title="My Title")
        assert ax.get_title() == "My Title"
        plt.close("all")

    def test_pre_only(self):
        """Only pre-treatment periods — post block should be skipped gracefully."""
        df = _make_event_study_df(n_pre=4, n_post=0)
        ax = plot_event_study(df)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_post_only(self):
        """Only post-treatment periods — pre block should be skipped gracefully."""
        df = _make_event_study_df(n_pre=0, n_post=5)
        ax = plot_event_study(df)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_missing_required_column_raises(self):
        df = _make_event_study_df()
        df = df.drop(columns=["att_e"])
        with pytest.raises(KeyError):
            plot_event_study(df)
        plt.close("all")

    def test_empty_df_does_not_raise(self):
        """Empty frame — nothing to plot but should return axes without crashing."""
        df = pd.DataFrame(
            columns=["event_time", "att_e", "se_e", "ci_lower_e", "ci_upper_e"]
        )
        ax = plot_event_study(df)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")


# ---------------------------------------------------------------------------
# plot_pre_post_did
# ---------------------------------------------------------------------------

class TestPlotPrePostDid:
    def test_returns_axes(self):
        df = _make_did_df()
        ax = plot_pre_post_did(
            df,
            period_col="period",
            outcome_col="loss_ratio",
            treated_col="treated",
            change_period=7,
        )
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_with_exposure_col(self):
        df = _make_did_df()
        ax = plot_pre_post_did(
            df,
            period_col="period",
            outcome_col="loss_ratio",
            treated_col="treated",
            change_period=7,
            exposure_col="exposure",
        )
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_uses_provided_ax(self):
        fig, ax_in = plt.subplots()
        df = _make_did_df()
        ax_out = plot_pre_post_did(
            df,
            period_col="period",
            outcome_col="loss_ratio",
            treated_col="treated",
            change_period=7,
            ax=ax_in,
        )
        assert ax_out is ax_in
        plt.close("all")

    def test_custom_title(self):
        df = _make_did_df()
        ax = plot_pre_post_did(
            df,
            period_col="period",
            outcome_col="loss_ratio",
            treated_col="treated",
            change_period=7,
            title="Custom",
        )
        assert ax.get_title() == "Custom"
        plt.close("all")

    def test_treated_only_no_crash(self):
        """No control group — the control plot branch is skipped gracefully."""
        df = _make_did_df()
        df_treated = df[df["treated"] == 1].copy()
        ax = plot_pre_post_did(
            df_treated,
            period_col="period",
            outcome_col="loss_ratio",
            treated_col="treated",
            change_period=7,
        )
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_missing_outcome_column_raises(self):
        df = _make_did_df().drop(columns=["loss_ratio"])
        with pytest.raises((KeyError, Exception)):
            plot_pre_post_did(
                df,
                period_col="period",
                outcome_col="loss_ratio",
                treated_col="treated",
                change_period=7,
            )
        plt.close("all")


# ---------------------------------------------------------------------------
# plot_pre_post_its
# ---------------------------------------------------------------------------

class TestPlotPrePostIts:
    def test_returns_axes(self):
        df = _make_its_df()
        ax = plot_pre_post_its(
            df,
            period_col="period",
            outcome_col="loss_ratio",
            change_period=6,
            t_change=6,
            pre_trend=0.002,
            intercept=0.10,
        )
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_uses_provided_ax(self):
        fig, ax_in = plt.subplots()
        df = _make_its_df()
        ax_out = plot_pre_post_its(
            df,
            period_col="period",
            outcome_col="loss_ratio",
            change_period=6,
            t_change=6,
            pre_trend=0.002,
            intercept=0.10,
            ax=ax_in,
        )
        assert ax_out is ax_in
        plt.close("all")

    def test_custom_title(self):
        df = _make_its_df()
        ax = plot_pre_post_its(
            df,
            period_col="period",
            outcome_col="loss_ratio",
            change_period=6,
            t_change=6,
            pre_trend=0.002,
            intercept=0.10,
            title="ITS Test",
        )
        assert ax.get_title() == "ITS Test"
        plt.close("all")

    def test_missing_time_column_raises(self):
        """_time_ column is required for counterfactual calculation."""
        df = _make_its_df().drop(columns=["_time_"])
        with pytest.raises(KeyError):
            plot_pre_post_its(
                df,
                period_col="period",
                outcome_col="loss_ratio",
                change_period=6,
                t_change=6,
                pre_trend=0.002,
                intercept=0.10,
            )
        plt.close("all")

    def test_zero_pre_trend(self):
        """Flat pre-trend (zero slope) should be a valid input."""
        df = _make_its_df()
        ax = plot_pre_post_its(
            df,
            period_col="period",
            outcome_col="loss_ratio",
            change_period=6,
            t_change=6,
            pre_trend=0.0,
            intercept=0.12,
        )
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")
