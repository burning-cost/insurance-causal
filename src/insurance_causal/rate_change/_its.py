"""
Interrupted Time Series estimator for insurance rate change evaluation.

Segmented regression with HAC (Newey-West) standard errors. Quarter dummies
for seasonal adjustment.

Key parameterisation (Ewusie et al. 2021, IJE 50(3):1011):
    Y_t = beta_0 + beta_1*t + beta_2*D_t + beta_3*(t-T)*D_t + sum(gamma_q*Q_q) + epsilon_t

NOT:
    Y_t = beta_0 + beta_1*t + beta_2*D_t + beta_3*(t-T) + ...

The interaction (t-T)*D_t ensures the slope_change term is zero in the pre-period.

Internal module — not part of the public API.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
except ImportError as exc:
    raise ImportError(
        "statsmodels>=0.14 is required for ITS estimation. "
        "Install with: pip install 'insurance-causal[rate_change]'"
    ) from exc

from ._result import ITSResult
from ._shocks import warn_if_near_shock


class ITSEstimator:
    """
    Interrupted Time Series segmented regression estimator.

    Parameters
    ----------
    outcome_col : str
    period_col : str
    change_period : int
        Integer-encoded change period (internally 1-indexed).
    exposure_col : str | None
    alpha : float
        Significance level for CIs.
    min_pre_periods : int
        Minimum number of pre-treatment periods. Default 4.
        Raises ValueError if fewer. 8+ recommended.
    add_seasonality : bool
        If True, include quarter dummies. Default True.
    change_period_raw : str | int | None
        Original (non-encoded) change period for shock proximity check.
    """

    def __init__(
        self,
        outcome_col: str,
        period_col: str,
        change_period: int,
        exposure_col: str | None = None,
        alpha: float = 0.05,
        min_pre_periods: int = 4,
        add_seasonality: bool = True,
        change_period_raw=None,
    ) -> None:
        self.outcome_col = outcome_col
        self.period_col = period_col
        self.change_period = change_period
        self.exposure_col = exposure_col
        self.alpha = alpha
        self.min_pre_periods = min_pre_periods
        self.add_seasonality = add_seasonality
        self.change_period_raw = change_period_raw

        self._fitted_model = None
        self._df_ts = None
        self._warnings: list[str] = []
        self._shocks: list[str] = []

    def fit(self, df: pd.DataFrame) -> "ITSEstimator":
        """
        Fit the ITS segmented regression.

        Parameters
        ----------
        df : pd.DataFrame
            Aggregate time series data with columns:
            period_col (int), outcome_col, exposure_col (optional),
            and optionally 'quarter' for seasonal dummies.
        """
        df_ts = df.copy().sort_values(self.period_col).reset_index(drop=True)

        # Count pre-treatment periods
        n_pre = int((df_ts[self.period_col] < self.change_period).sum())
        n_post = int((df_ts[self.period_col] >= self.change_period).sum())
        n_total = len(df_ts)

        if n_pre < self.min_pre_periods:
            raise ValueError(
                f"Insufficient pre-treatment periods: {n_pre} found, "
                f"minimum {self.min_pre_periods} required. "
                "Increase the number of pre-treatment periods or lower min_pre_periods."
            )
        if n_pre < 8:
            msg = (
                f"Only {n_pre} pre-treatment periods available. "
                "8+ are recommended for reliable ITS estimation of the pre-trend."
            )
            warnings.warn(msg, UserWarning, stacklevel=3)
            self._warnings.append(msg)

        # Check for shock proximity using the raw (original) change period
        check_period = self.change_period_raw if self.change_period_raw is not None else self.change_period
        shock_list = warn_if_near_shock(check_period)
        self._shocks = shock_list
        for shock in shock_list:
            msg = (
                f"Intervention period '{check_period}' is near a known UK insurance shock: "
                f"{shock}. This may confound the rate change estimate."
            )
            self._warnings.append(msg)

        # Build regression variables
        # t: 1, 2, ..., T_total (time counter)
        df_ts["_time_"] = np.arange(1, n_total + 1)

        # D_t: 1 if period >= change_period
        df_ts["_post_"] = (df_ts[self.period_col] >= self.change_period).astype(int)

        # (t - T) * D_t — interaction term, zero in pre-period
        # T here is the index in our _time_ counter corresponding to change_period
        # Find what _time_ value corresponds to change_period
        t_change = int(df_ts.loc[df_ts[self.period_col] == self.change_period, "_time_"].iloc[0])
        df_ts["_time_since_"] = np.maximum(0, df_ts["_time_"] - t_change) * df_ts["_post_"]

        # Seasonality
        has_quarter = "quarter" in df_ts.columns
        if self.add_seasonality and has_quarter:
            df_ts["_q_"] = df_ts["quarter"].astype(str)
            seasonal_term = " + C(_q_)"
            seasonal_applied = True
        else:
            seasonal_term = ""
            seasonal_applied = False

        formula = (
            f"{self.outcome_col} ~ _time_ + _post_ + _time_since_{seasonal_term}"
        )

        # Weights
        if self.exposure_col is not None:
            weights = df_ts[self.exposure_col].clip(lower=1e-6)
        else:
            weights = pd.Series(np.ones(len(df_ts)), index=df_ts.index)
            if not self._warnings:
                pass  # Already warned upstream by evaluator

        # HAC SE: Newey-West with maxlags = max(1, int(sqrt(T)))
        maxlags = max(1, int(np.sqrt(n_total)))

        model = smf.wls(formula, data=df_ts, weights=weights).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": maxlags},
        )

        self._fitted_model = model
        self._df_ts = df_ts
        self._n_pre = n_pre
        self._n_post = n_post
        self._n_total = n_total
        self._seasonal_applied = seasonal_applied
        self._t_change = t_change

        return self

    def results(self) -> ITSResult:
        """Extract ITSResult from the fitted model."""
        if self._fitted_model is None:
            raise RuntimeError("Call fit() before results().")

        from scipy import stats as scipy_stats

        model = self._fitted_model
        z = scipy_stats.norm.ppf(1 - self.alpha / 2)

        # Level shift (beta_2)
        level_shift = float(model.params["_post_"])
        level_shift_se = float(model.bse["_post_"])
        level_shift_ci_lower = level_shift - z * level_shift_se
        level_shift_ci_upper = level_shift + z * level_shift_se
        level_shift_pvalue = float(model.pvalues["_post_"])

        # Slope change (beta_3)
        slope_change = float(model.params["_time_since_"])
        slope_change_se = float(model.bse["_time_since_"])
        slope_change_pvalue = float(model.pvalues["_time_since_"])

        # Pre-trend (beta_1)
        pre_trend = float(model.params["_time_"])
        pre_trend_se = float(model.bse["_time_"])

        # Effect at various horizons post-intervention
        # Total effect at k periods post: beta_2 + beta_3 * k
        effect_at_periods = {}
        for k in [1, 4, 8]:
            if k <= self._n_post:
                effect_at_periods[k] = level_shift + slope_change * k

        return ITSResult(
            level_shift=level_shift,
            level_shift_se=level_shift_se,
            level_shift_ci_lower=level_shift_ci_lower,
            level_shift_ci_upper=level_shift_ci_upper,
            level_shift_pvalue=level_shift_pvalue,
            slope_change=slope_change,
            slope_change_se=slope_change_se,
            slope_change_pvalue=slope_change_pvalue,
            pre_trend=pre_trend,
            pre_trend_se=pre_trend_se,
            effect_at_periods=effect_at_periods,
            seasonal_adjustment=self._seasonal_applied,
            shocks_near_intervention=self._shocks,
        )
