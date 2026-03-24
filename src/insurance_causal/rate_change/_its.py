"""Interrupted Time Series estimator for rate change evaluation.

Implements segmented regression with Newey-West HAC standard errors and
optional seasonal quarter dummies. Used when no control group is available.
"""

from __future__ import annotations

from typing import Any, Optional
import warnings
import numpy as np
import pandas as pd

from ._result import ITSResult

# Minimum pre-treatment periods required for credible ITS
MIN_PRE_PERIODS_CREDIBLE = 8
MIN_PRE_PERIODS_HARD = 4


class ITSEstimator:
    """Interrupted Time Series via segmented regression.

    The model is:

        Y_t = beta_0 + beta_1*t + beta_2*D_t + beta_3*(t-T)*D_t
              + sum(gamma_q * Q_q) + epsilon_t

    where D_t = 1{t >= T} and (t-T)*D_t is the correct post-intervention
    time counter (zero in pre-period, active only post-intervention).

    Standard errors use Newey-West HAC to handle autocorrelation in the
    time series residuals.

    Parameters
    ----------
    outcome_col : str
        Column name for the outcome variable.
    period_col : str
        Column identifying the time period (must be sortable).
    treatment_period : any
        The period in which the rate change took effect.
    weight_col : str or None
        Column for exposure weights. Used as WLS weights if provided.
    add_seasonality : bool, default True
        Whether to include quarter-of-year dummy variables. Requires
        ``season_col`` to be present, or a 'quarter' column derivable
        from the period.
    season_col : str or None
        Column identifying the quarter/season. If None and add_seasonality
        is True, the estimator will attempt to derive quarter from the
        period column.
    """

    def __init__(
        self,
        outcome_col: str,
        period_col: str,
        treatment_period: Any,
        weight_col: Optional[str] = None,
        add_seasonality: bool = True,
        season_col: Optional[str] = None,
    ) -> None:
        self.outcome_col = outcome_col
        self.period_col = period_col
        self.treatment_period = treatment_period
        self.weight_col = weight_col
        self.add_seasonality = add_seasonality
        self.season_col = season_col

        self._result: Optional[ITSResult] = None
        self._model_result = None
        self._df_fitted: Optional[pd.DataFrame] = None

    def fit(self, df: pd.DataFrame) -> "ITSEstimator":
        """Estimate the ITS segmented regression.

        Parameters
        ----------
        df : pd.DataFrame
            Time series data. One row per time period.

        Returns
        -------
        ITSEstimator
            Self for method chaining.

        Raises
        ------
        ImportError
            If statsmodels is not installed.
        ValueError
            If the data does not meet minimum requirements.
        """
        try:
            import statsmodels.formula.api as smf
        except ImportError as exc:
            raise ImportError(
                "statsmodels>=0.14 is required for ITS estimation. "
                "Install with: pip install statsmodels"
            ) from exc

        df = df.copy()
        self._validate(df)

        sorted_periods = sorted(df[self.period_col].unique())
        t_idx = sorted_periods.index(self.treatment_period)

        n_pre = t_idx  # periods strictly before treatment
        n_post = len(sorted_periods) - t_idx  # treatment period + after

        if n_pre < MIN_PRE_PERIODS_HARD:
            raise ValueError(
                f"Only {n_pre} pre-intervention periods available. "
                f"ITS requires at least {MIN_PRE_PERIODS_HARD} pre-periods to "
                f"estimate the pre-trend. Cannot proceed."
            )

        if n_pre < MIN_PRE_PERIODS_CREDIBLE:
            warnings.warn(
                f"Only {n_pre} pre-intervention periods (fewer than "
                f"{MIN_PRE_PERIODS_CREDIBLE} recommended). ITS estimate of the "
                f"pre-trend may be unreliable. Interpret results with caution.",
                stacklevel=3,
            )

        # Build time counter t (1-indexed for interpretable intercept)
        period_to_t = {p: i + 1 for i, p in enumerate(sorted_periods)}
        df["_t"] = df[self.period_col].map(period_to_t)

        T = period_to_t[self.treatment_period]

        # Post-intervention indicator D_t
        df["_post"] = (df["_t"] >= T).astype(float)

        # Correct ITS slope change: (t - T) * D_t
        # This is ZERO for all pre-period observations
        df["_time_since"] = (df["_t"] - T) * df["_post"]

        # Seasonality
        has_seasonality = False
        if self.add_seasonality:
            df, has_seasonality = self._add_seasonality(df)

        # Build formula
        formula = f"{self.outcome_col} ~ _t + _post + _time_since"

        if has_seasonality:
            # Drop one quarter to avoid perfect collinearity (statsmodels C() handles this)
            formula += " + C(_quarter)"

        weights_arr = df[self.weight_col].values if self.weight_col else None

        if weights_arr is not None:
            mod = smf.wls(formula, data=df, weights=weights_arr)
        else:
            mod = smf.ols(formula, data=df)

        hac_lags = max(1, int(np.sqrt(len(df))))
        res = mod.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

        self._model_result = res
        self._df_fitted = df

        # Extract key parameters
        beta_1 = float(res.params["_t"])           # pre-trend
        beta_2 = float(res.params["_post"])         # level shift
        beta_3 = float(res.params["_time_since"])   # slope change

        ci = res.conf_int()

        self._result = ITSResult(
            level_shift=beta_2,
            level_shift_se=float(res.bse["_post"]),
            level_shift_ci_lower=float(ci.loc["_post"].iloc[0]),
            level_shift_ci_upper=float(ci.loc["_post"].iloc[1]),
            level_shift_p_value=float(res.pvalues["_post"]),
            slope_change=beta_3,
            slope_change_se=float(res.bse["_time_since"]),
            slope_change_ci_lower=float(ci.loc["_time_since"].iloc[0]),
            slope_change_ci_upper=float(ci.loc["_time_since"].iloc[1]),
            slope_change_p_value=float(res.pvalues["_time_since"]),
            pre_trend=beta_1,
            n_pre=n_pre,
            n_post=n_post,
            r_squared=float(res.rsquared),
            hac_lags=hac_lags,
            has_seasonality=has_seasonality,
        )

        return self

    def result(self) -> ITSResult:
        """Return the ITS result.

        Returns
        -------
        ITSResult

        Raises
        ------
        RuntimeError
            If fit() has not been called.
        """
        if self._result is None:
            raise RuntimeError("Call fit() before accessing results.")
        return self._result

    def counterfactual(self) -> Optional[pd.Series]:
        """Return the counterfactual (pre-trend extrapolated) series.

        Returns
        -------
        pd.Series or None
            Counterfactual predicted values indexed by period, or None
            if fit() has not been called.
        """
        if self._df_fitted is None or self._model_result is None:
            return None

        df = self._df_fitted.copy()
        # Counterfactual: set post=0 and time_since=0 everywhere
        df_cf = df.copy()
        df_cf["_post"] = 0.0
        df_cf["_time_since"] = 0.0

        try:
            cf_pred = self._model_result.predict(df_cf)
            return pd.Series(cf_pred.values, index=df[self.period_col].values)
        except Exception:
            return None

    def _add_seasonality(self, df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        """Add quarter dummy column to df. Returns (df, success_flag)."""
        # Use provided season_col if available
        if self.season_col and self.season_col in df.columns:
            df["_quarter"] = df[self.season_col].astype(str)
            return df, True

        # Try to derive quarter from period column
        period_vals = df[self.period_col]
        quarters = self._extract_quarters(period_vals)

        if quarters is not None:
            df["_quarter"] = quarters
            return df, True
        else:
            warnings.warn(
                "Could not determine quarter from period column. "
                "Seasonal dummies will not be included. "
                "Provide season_col to override.",
                stacklevel=4,
            )
            return df, False

    def _extract_quarters(self, periods: pd.Series) -> Optional[pd.Series]:
        """Attempt to extract quarter information from period values."""
        import re

        sample = str(periods.iloc[0])

        # "YYYY-QN" format
        if re.match(r"^\d{4}-Q[1-4]$", sample):
            return periods.astype(str).str.extract(r"Q([1-4])")[0].astype(int)

        # pandas Period with quarterly frequency
        try:
            pds = pd.PeriodIndex(periods)
            if hasattr(pds, "quarter"):
                return pd.Series(pds.quarter, index=periods.index)
        except Exception:
            pass

        # "YYYY-MM" or "YYYY-MM-DD"
        if re.match(r"^\d{4}-\d{2}", sample):
            try:
                dts = pd.to_datetime(periods)
                return dts.dt.quarter
            except Exception:
                pass

        # Integer year — no quarter info
        return None

    def _validate(self, df: pd.DataFrame) -> None:
        """Validate input data for ITS."""
        required = [self.outcome_col, self.period_col]
        if self.weight_col:
            required.append(self.weight_col)
        if self.season_col:
            required.append(self.season_col)

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")

        if self.treatment_period not in df[self.period_col].values:
            raise ValueError(
                f"treatment_period {self.treatment_period!r} not found in "
                f"column {self.period_col!r}. "
                f"Available periods: {sorted(df[self.period_col].unique()).tolist()}"
            )

        if self.weight_col and (df[self.weight_col] <= 0).any():
            raise ValueError(
                f"Weight column {self.weight_col!r} contains non-positive values. "
                f"Exposure weights must be strictly positive."
            )

        n_periods = df[self.period_col].nunique()
        if n_periods < 6:
            raise ValueError(
                f"ITS requires at least 6 time periods; found {n_periods}."
            )
