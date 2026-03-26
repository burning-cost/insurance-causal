"""
RateChangeEvaluator — main public class for post-hoc rate change evaluation.

Estimates the causal effect of a historical insurance rate change using
Difference-in-Differences (when a control group exists) or Interrupted
Time Series (when the entire book was treated).

References
----------
Angrist & Pischke (2009). Mostly Harmless Econometrics, Ch. 5.
Rambachan & Roth (2023). ReStud 90(5):2555 (parallel trends robustness).
Goodman-Bacon (2021). JoE 225(2):254 (staggered adoption bias).
Kontopantelis et al. (2015). BMJ 350:h2750 (ITS best practice).
Ewusie et al. (2021). IJE 50(3):1011 (ITS parameterisation).
Cameron & Miller (2015). JHR 50(2):317-372 (cluster SE with few clusters).
KB entries: 3548, 3549, 3550.
"""

from __future__ import annotations

import re
import warnings
from typing import Literal

import numpy as np
import pandas as pd

from ._did import DiDEstimator
from ._its import ITSEstimator
from ._result import DiDResult, ITSResult, RateChangeResult
from ._diagnostics import ParallelTrendsResult, StaggeredAdoptionChecker


def _parse_period(value) -> int | None:
    """
    Try to parse a quarter string like "2023Q1" or "2023-Q1" into an
    integer representation (year * 4 + quarter_number).
    Returns None if the value is not a quarter string.
    """
    if value is None:
        return None
    s = str(value).strip().upper().replace("-", "")
    match = re.match(r"^(\d{4})Q([1-4])$", s)
    if match:
        year, q = int(match.group(1)), int(match.group(2))
        return year * 4 + q  # unique integer per quarter
    return None


def _encode_periods(series: pd.Series) -> tuple[pd.Series, dict]:
    """
    Convert a period column (any orderable type) to a dense integer index
    starting at 1.

    Returns (encoded_series, period_map) where period_map maps
    original values to integers.
    """
    # Try quarter string parsing first
    parsed_ints = series.map(lambda v: _parse_period(v) if isinstance(v, str) else None)
    if parsed_ints.notna().all():
        sorted_unique = sorted(parsed_ints.unique())
        mapping = {v: i + 1 for i, v in enumerate(sorted_unique)}
        encoded = parsed_ints.map(mapping)
        # Build reverse: original value -> int
        orig_to_int = {}
        for orig, parsed in zip(series, parsed_ints):
            orig_to_int[orig] = mapping[parsed]
        return encoded, orig_to_int

    # Fallback: sort unique values and rank
    sorted_unique = sorted(series.unique())
    orig_to_int = {v: i + 1 for i, v in enumerate(sorted_unique)}
    encoded = series.map(orig_to_int)
    return encoded, orig_to_int


def _encode_change_period(change_period, orig_to_int: dict):
    """Map the change_period (original value) to its encoded integer."""
    if change_period in orig_to_int:
        return orig_to_int[change_period]
    # May be a quarter string: try to find in the mapping by parsed value
    parsed = _parse_period(change_period)
    if parsed is not None:
        for orig, enc in orig_to_int.items():
            orig_parsed = _parse_period(orig) if isinstance(orig, str) else None
            if orig_parsed == parsed:
                return enc
    # Try to cast as integer directly (e.g. change_period=7 when dict keys are 7)
    try:
        return int(change_period)
    except (ValueError, TypeError):
        pass
    raise ValueError(
        f"change_period {change_period!r} could not be found in the period column. "
        "Ensure change_period uses the same format as the period column values. "
        f"Available periods: {sorted(orig_to_int.keys())[:5]}..."
    )


class RateChangeEvaluator:
    """
    Post-hoc causal evaluation of an insurance rate change.

    Estimates the causal effect of a historical rate change on a portfolio
    outcome (conversion rate, claim frequency, loss ratio) using
    Difference-in-Differences (when a control group exists) or Interrupted
    Time Series (when the entire book was treated).

    Parameters
    ----------
    method : {"auto", "did", "its"}
        Estimation method.
        "auto" (default): use DiD if a control group is present (treated == 0
        observations exist), otherwise fall back to ITS.
        "did": force DiD. Raises ValueError if no control group found.
        "its": force ITS. Aggregates to period-level if policy-level data supplied.
    outcome_col : str
        Column name of the outcome variable.
    period_col : str
        Column name identifying the time period.
    treated_col : str | None
        Column name of the binary treatment indicator (0 = control, 1 = treated).
        Required for DiD. Not used for ITS.
    change_period : int | str | None
        The period in which the rate change was implemented.
    exposure_col : str | None
        Column name of earned exposure (policy years) or policy count.
        Strongly recommended. If None, equal weighting is used with a warning.
    unit_col : str | None
        Column name of the unit identifier (segment, territory, channel).
        Required for DiD.
    cluster_col : str | None
        Column to cluster SE on. Defaults to unit_col.
    alpha : float
        Significance level for CIs. Default 0.05 (95% CI).
    min_pre_periods : int
        Minimum number of pre-treatment periods required. Default 4.
    random_state : int
        Seed for any randomised operations. Default 42.

    Examples
    --------
    >>> from insurance_causal.rate_change import RateChangeEvaluator, make_rate_change_data
    >>> df = make_rate_change_data(n_policies=10_000, true_att=-0.03)
    >>> result = RateChangeEvaluator(
    ...     method="auto",
    ...     outcome_col="loss_ratio",
    ...     period_col="period",
    ...     treated_col="treated",
    ...     change_period=7,
    ...     exposure_col="exposure",
    ...     unit_col="segment_id",
    ... ).fit(df).summary()
    >>> print(result)

    References
    ----------
    Angrist & Pischke (2009). Mostly Harmless Econometrics, Ch. 5 (DiD).
    Rambachan & Roth (2023). ReStud 90(5):2555 (parallel trends).
    Goodman-Bacon (2021). JoE 225(2):254 (staggered adoption).
    Kontopantelis et al. (2015). BMJ 350:h2750 (ITS best practice).
    Ewusie et al. (2021). IJE 50(3):1011 (ITS parameterisation).
    KB entries: 3548, 3549, 3550.

    Notes
    -----
    For staggered adoption (multiple treatment cohorts), use
    insurance_causal_policy.StaggeredEstimator (Callaway-Sant'Anna 2021).
    For HonestDiD parallel-trends robustness (Rambachan & Roth 2023), no
    confirmed stable Python port exists as of 2026. See R package HonestDiD.
    """

    def __init__(
        self,
        method: Literal["auto", "did", "its"] = "auto",
        outcome_col: str = "outcome",
        period_col: str = "period",
        treated_col: str | None = "treated",
        change_period: "int | str | None" = None,
        exposure_col: str | None = None,
        unit_col: str | None = None,
        cluster_col: str | None = None,
        alpha: float = 0.05,
        min_pre_periods: int = 4,
        random_state: int = 42,
    ) -> None:
        if method not in ("auto", "did", "its"):
            raise ValueError(f"method must be 'auto', 'did', or 'its'. Got: {method!r}")
        if change_period is None:
            raise ValueError("change_period is required.")

        self.method = method
        self.outcome_col = outcome_col
        self.period_col = period_col
        self.treated_col = treated_col
        self.change_period = change_period
        self.exposure_col = exposure_col
        self.unit_col = unit_col
        self.cluster_col = cluster_col or unit_col
        self.alpha = alpha
        self.min_pre_periods = min_pre_periods
        self.random_state = random_state

        self._result: RateChangeResult | None = None
        self._fitted = False
        self._period_map_: dict = {}
        self._change_period_int: int | None = None
        self._method_used: str | None = None
        self._estimator = None
        self._df_agg: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "RateChangeEvaluator":
        """
        Fit the rate change evaluator.

        Parameters
        ----------
        df : pd.DataFrame
            Policy-level or segment-period aggregated data.

        Returns
        -------
        self : RateChangeEvaluator
        """
        all_warnings: list[str] = []

        # Validate inputs
        self._validate_df(df)

        # Encode periods
        df_work = df.copy()
        encoded_periods, orig_to_int = _encode_periods(df_work[self.period_col])
        df_work["_period_enc_"] = encoded_periods
        self._period_map_ = orig_to_int

        # Encode change_period
        change_period_int = _encode_change_period(self.change_period, orig_to_int)
        self._change_period_int = change_period_int

        # Warn if no exposure column
        if self.exposure_col is None:
            msg = (
                "No exposure_col provided. All observations will be equally weighted. "
                "Exposure weighting is strongly recommended for insurance data "
                "(unweighted loss ratios for segments with 10 vs 10,000 policies "
                "carry the same statistical weight)."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
            all_warnings.append(msg)

        # Determine method
        method = self._resolve_method(df_work)
        self._method_used = method

        if method == "did":
            result = self._fit_did(df_work, change_period_int, all_warnings)
        else:
            result = self._fit_its(df_work, change_period_int, all_warnings)

        self._result = result
        self._fitted = True
        return self

    def summary(self) -> RateChangeResult:
        """
        Return the estimated causal effect with confidence interval.

        Returns
        -------
        RateChangeResult
        """
        if not self._fitted:
            raise RuntimeError(
                "RateChangeEvaluator has not been fitted. Call fit(df) first."
            )
        return self._result

    def parallel_trends_test(self) -> ParallelTrendsResult:
        """
        Return pre-treatment event study coefficients and joint F-test.

        Only available after fitting with method="did".

        Returns
        -------
        ParallelTrendsResult
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        if self._method_used != "did":
            raise RuntimeError("parallel_trends_test() is only available for DiD.")

        did: DiDEstimator = self._estimator
        did_result: DiDResult = self._result.method_detail

        return ParallelTrendsResult(
            event_study_df=did_result.event_study_df,
            joint_pt_fstat=did_result.joint_pt_fstat,
            joint_pt_pvalue=did_result.joint_pt_pvalue,
            passes=did_result.joint_pt_pvalue > 0.05 if not np.isnan(did_result.joint_pt_pvalue) else False,
            n_pre_periods=self._result.n_periods_pre,
        )

    def plot_event_study(
        self,
        ax=None,
        title: str | None = None,
    ):
        """
        Plot event study coefficients with 95% CI.

        Pre-treatment periods in grey; post-treatment in blue.
        """
        if not self._fitted or self._method_used != "did":
            raise RuntimeError("plot_event_study() requires a fitted DiD model.")

        from ._plots import plot_event_study

        did_result: DiDResult = self._result.method_detail
        return plot_event_study(
            did_result.event_study_df,
            ax=ax,
            title=title,
            alpha=self.alpha,
        )

    def plot_pre_post(
        self,
        ax=None,
        title: str | None = None,
    ):
        """
        Plot observed outcome over time.

        For DiD: treated and control observed outcomes with intervention line.
        For ITS: observed vs counterfactual trend.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        if self._method_used == "did":
            from ._plots import plot_pre_post_did
            return plot_pre_post_did(
                df_agg=self._df_agg,
                period_col="_period_enc_",
                outcome_col=self.outcome_col,
                treated_col=self.treated_col,
                change_period=self._change_period_int,
                exposure_col=self.exposure_col,
                ax=ax,
                title=title,
            )
        else:
            from ._plots import plot_pre_post_its
            its: ITSEstimator = self._estimator
            model = its._fitted_model
            return plot_pre_post_its(
                df_ts=its._df_ts,
                period_col="_period_enc_",
                outcome_col=self.outcome_col,
                change_period=self._change_period_int,
                t_change=its._t_change,
                pre_trend=float(model.params["_time_"]),
                intercept=float(model.params["Intercept"]),
                ax=ax,
                title=title,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_df(self, df: pd.DataFrame) -> None:
        required = [self.outcome_col, self.period_col]
        if self.method in ("did", "auto") and self.treated_col:
            # For auto we check whether treated_col exists before requiring it
            if self.method == "did" and self.treated_col not in df.columns:
                raise ValueError(
                    f"treated_col '{self.treated_col}' not found in DataFrame."
                )
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame.")

    def _resolve_method(self, df: pd.DataFrame) -> str:
        """Determine whether to use DiD or ITS."""
        if self.method == "did":
            # Validate that a control group exists
            if self.treated_col not in df.columns:
                raise ValueError(
                    f"treated_col '{self.treated_col}' not found. "
                    "DiD requires a control group column."
                )
            if (df[self.treated_col] == 0).sum() == 0:
                raise ValueError(
                    "No control group found: all observations have treated==1. "
                    "DiD requires at least some control units (treated==0). "
                    "Use method='its' for whole-book evaluations."
                )
            return "did"
        elif self.method == "its":
            return "its"
        else:  # auto
            has_control = (
                self.treated_col is not None
                and self.treated_col in df.columns
                and (df[self.treated_col] == 0).sum() > 0
            )
            return "did" if has_control else "its"

    def _aggregate_to_segment_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate policy-level data to segment-period level.

        Uses exposure-weighted mean for the outcome column.
        """
        period_col = "_period_enc_"
        group_cols = [self.unit_col, period_col, self.treated_col]

        def _agg(g: pd.DataFrame) -> pd.Series:
            if self.exposure_col and self.exposure_col in g.columns:
                w = g[self.exposure_col].clip(lower=0)
                total_w = w.sum()
                outcome_mean = (
                    np.average(g[self.outcome_col], weights=w) if total_w > 0
                    else g[self.outcome_col].mean()
                )
                total_exposure = total_w
            else:
                outcome_mean = g[self.outcome_col].mean()
                total_exposure = float(len(g))

            return pd.Series({
                self.outcome_col: outcome_mean,
                self.exposure_col if self.exposure_col else "_exposure_": total_exposure,
                "n_policies": len(g),
            })

        df_agg = (
            df.groupby(group_cols, observed=True)
            .apply(_agg)
            .reset_index()
        )

        # If exposure_col was None, we added a dummy column; clean up
        if self.exposure_col is None and "_exposure_" in df_agg.columns:
            df_agg = df_agg.drop(columns=["_exposure_"])

        return df_agg

    def _aggregate_to_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate to period level (for ITS when data is policy/segment-level)."""
        period_col = "_period_enc_"

        if self.exposure_col and self.exposure_col in df.columns:
            w = df[self.exposure_col].clip(lower=0)

            def _agg_period(g: pd.DataFrame) -> pd.Series:
                weights = g[self.exposure_col].clip(lower=0)
                total_w = weights.sum()
                outcome_mean = (
                    np.average(g[self.outcome_col], weights=weights) if total_w > 0
                    else g[self.outcome_col].mean()
                )
                return pd.Series({
                    self.outcome_col: outcome_mean,
                    self.exposure_col: total_w,
                })

            df_agg = df.groupby(period_col).apply(
                _agg_period, include_groups=False
            ).reset_index()
        else:
            df_agg = (
                df.groupby(period_col)[self.outcome_col]
                .mean()
                .reset_index()
            )

        # Carry quarter column if present
        if "quarter" in df.columns:
            period_to_quarter = df.groupby(period_col)["quarter"].first()
            df_agg = df_agg.merge(
                period_to_quarter.reset_index(), on=period_col, how="left"
            )

        return df_agg

    def _check_is_policy_level(self, df: pd.DataFrame) -> bool:
        """Return True if df has multiple rows per unit-period combination."""
        if self.unit_col is None or "_period_enc_" not in df.columns:
            return False
        counts = df.groupby([self.unit_col, "_period_enc_"]).size()
        return (counts > 1).any()

    def _fit_did(
        self,
        df_work: pd.DataFrame,
        change_period_int: int,
        all_warnings: list[str],
    ) -> RateChangeResult:
        """Fit the DiD estimator and build RateChangeResult."""
        if self.unit_col is None:
            raise ValueError(
                "unit_col is required for DiD. Provide the segment/territory identifier."
            )

        # Validate control group
        if (df_work[self.treated_col] == 0).sum() == 0:
            raise ValueError(
                "No control group found: all observations have treated==1. "
                "DiD requires at least some control units (treated==0). "
                "Use method='its' for whole-book evaluations."
            )

        # Add unit_id encoding for staggered adoption checker
        df_work["_unit_id_enc_"] = df_work[self.unit_col]

        # Aggregate to segment-period if policy-level
        is_policy_level = self._check_is_policy_level(df_work)
        if is_policy_level:
            df_agg = self._aggregate_to_segment_period(df_work)
        else:
            df_agg = df_work.copy()
            # Ensure exposure col is correct
            if self.exposure_col and self.exposure_col not in df_agg.columns:
                df_agg[self.exposure_col] = 1.0

        # Carry _unit_id_enc_ into df_agg if it got lost in aggregation
        if "_unit_id_enc_" not in df_agg.columns:
            unit_to_enc = df_work[[self.unit_col, "_unit_id_enc_"]].drop_duplicates()
            df_agg = df_agg.merge(unit_to_enc, on=self.unit_col, how="left")

        self._df_agg = df_agg

        # Check regression-to-mean: warn if treated pre-treatment LR >> control
        if self.treated_col in df_agg.columns:
            pre_mask = df_agg["_period_enc_"] < change_period_int
            treated_pre = df_agg[pre_mask & (df_agg[self.treated_col] == 1)]
            control_pre = df_agg[pre_mask & (df_agg[self.treated_col] == 0)]

            if len(treated_pre) > 0 and len(control_pre) > 0:
                w_t = treated_pre[self.exposure_col].values if self.exposure_col else None
                w_c = control_pre[self.exposure_col].values if self.exposure_col else None

                pre_mean_treated = float(
                    np.average(treated_pre[self.outcome_col], weights=w_t)
                    if w_t is not None else treated_pre[self.outcome_col].mean()
                )
                pre_mean_control = float(
                    np.average(control_pre[self.outcome_col], weights=w_c)
                    if w_c is not None else control_pre[self.outcome_col].mean()
                )

                if pre_mean_control > 0 and abs(pre_mean_treated - pre_mean_control) / abs(pre_mean_control) > 0.5:
                    msg = (
                        f"Pre-treatment mean for treated group ({pre_mean_treated:.3f}) "
                        f"differs by > 50% from control group ({pre_mean_control:.3f}). "
                        "If treated segments were selected for rate change due to high loss "
                        "ratios, regression to the mean may bias the DiD estimate."
                    )
                    warnings.warn(msg, UserWarning, stacklevel=2)
                    all_warnings.append(msg)
            else:
                pre_mean_treated = float("nan")
                pre_mean_control = None
        else:
            pre_mean_treated = float("nan")
            pre_mean_control = None

        # Compute pre_mean_treated and pre_mean_control properly
        pre_mask = df_agg["_period_enc_"] < change_period_int
        treated_pre = df_agg[pre_mask & (df_agg[self.treated_col] == 1)]
        control_pre = df_agg[pre_mask & (df_agg[self.treated_col] == 0)]

        def _wmean(grp, col, w_col):
            if w_col and w_col in grp.columns:
                w = grp[w_col].clip(lower=0)
                if w.sum() > 0:
                    return float(np.average(grp[col], weights=w))
            return float(grp[col].mean()) if len(grp) > 0 else float("nan")

        pre_mean_treated = _wmean(treated_pre, self.outcome_col, self.exposure_col)
        pre_mean_control = (
            _wmean(control_pre, self.outcome_col, self.exposure_col)
            if len(control_pre) > 0 else None
        )

        # Fit DiD
        did = DiDEstimator(
            outcome_col=self.outcome_col,
            period_col="_period_enc_",
            treated_col=self.treated_col,
            unit_col=self.unit_col,
            change_period=change_period_int,
            exposure_col=self.exposure_col,
            cluster_col=self.cluster_col,
            alpha=self.alpha,
        )
        did.fit(df_agg)
        self._estimator = did

        all_warnings.extend(did._warnings)
        did_result = did.results()

        n_periods_pre = int((df_agg["_period_enc_"] < change_period_int).any())
        n_periods_pre = int(df_agg.loc[
            df_agg[self.treated_col] == 1, "_period_enc_"
        ].lt(change_period_int).sum() // max(1, did_result.n_units_treated))

        # Use unique period counts
        all_periods = df_agg["_period_enc_"].unique()
        n_periods_pre = int(sum(1 for p in all_periods if p < change_period_int))
        n_periods_post = int(sum(1 for p in all_periods if p >= change_period_int))

        n_treated = int(df_agg[df_agg[self.treated_col] == 1]["_period_enc_"].count())
        n_control = int(df_agg[df_agg[self.treated_col] == 0]["_period_enc_"].count())

        # ATT %
        att_pct = None
        if pre_mean_treated and pre_mean_treated != 0:
            att_pct = float(did_result.att / pre_mean_treated * 100)

        # PT pvalue
        pt_pvalue = did_result.joint_pt_pvalue if not np.isnan(did_result.joint_pt_pvalue) else None

        return RateChangeResult(
            method="did",
            outcome_col=self.outcome_col,
            att=did_result.att,
            att_pct=att_pct,
            se=did_result.se,
            ci_lower=did_result.ci_lower,
            ci_upper=did_result.ci_upper,
            p_value=did_result.p_value,
            n_treated=n_treated,
            n_control=n_control,
            n_periods_pre=n_periods_pre,
            n_periods_post=n_periods_post,
            pre_mean_treated=pre_mean_treated,
            pre_mean_control=pre_mean_control,
            parallel_trends_pvalue=pt_pvalue,
            staggered_adoption_detected=did._staggered_detected,
            cluster_se_used=did._cluster_se_used,
            n_clusters=did._n_clusters,
            warnings=all_warnings,
            method_detail=did_result,
        )

    def _fit_its(
        self,
        df_work: pd.DataFrame,
        change_period_int: int,
        all_warnings: list[str],
    ) -> RateChangeResult:
        """Fit the ITS estimator and build RateChangeResult."""
        # Aggregate to period level if necessary
        has_unit = self.unit_col and self.unit_col in df_work.columns
        is_multi_row = False
        if has_unit:
            counts = df_work.groupby("_period_enc_").size()
            is_multi_row = (counts > 1).any()

        if is_multi_row or has_unit:
            df_ts = self._aggregate_to_period(df_work)
        else:
            # df_work already has _period_enc_ added by fit(); copy directly.
            # Do NOT rename period_col to _period_enc_ here — that would create
            # a duplicate column since _period_enc_ already exists in df_work.
            df_ts = df_work.copy()

        # Ensure _period_enc_ exists (guard for edge cases)
        if "_period_enc_" not in df_ts.columns:
            if self.period_col in df_ts.columns:
                df_ts["_period_enc_"] = df_ts[self.period_col]

        # Quarter column
        if "quarter" not in df_ts.columns:
            # Derive quarter from period index
            sorted_periods = sorted(df_ts["_period_enc_"].unique())
            period_to_quarter = {p: ((i % 4) + 1) for i, p in enumerate(sorted_periods)}
            df_ts["quarter"] = df_ts["_period_enc_"].map(period_to_quarter)

        # Fit ITS
        its = ITSEstimator(
            outcome_col=self.outcome_col,
            period_col="_period_enc_",
            change_period=change_period_int,
            exposure_col=self.exposure_col,
            alpha=self.alpha,
            min_pre_periods=self.min_pre_periods,
            add_seasonality=True,
            change_period_raw=self.change_period,
        )
        its.fit(df_ts)
        self._estimator = its

        all_warnings.extend(its._warnings)
        its_result = its.results()

        # Period counts
        all_periods = df_ts["_period_enc_"].unique()
        n_periods_pre = int(sum(1 for p in all_periods if p < change_period_int))
        n_periods_post = int(sum(1 for p in all_periods if p >= change_period_int))

        # Pre-treatment mean
        pre_mask = df_ts["_period_enc_"] < change_period_int
        pre_df = df_ts[pre_mask]
        if self.exposure_col and self.exposure_col in pre_df.columns:
            w = pre_df[self.exposure_col].clip(lower=0)
            pre_mean_treated = float(np.average(pre_df[self.outcome_col], weights=w))
        else:
            pre_mean_treated = float(pre_df[self.outcome_col].mean())

        # ATT for ITS is the level shift (primary estimate)
        att = its_result.level_shift
        se = its_result.level_shift_se
        ci_lower = its_result.level_shift_ci_lower
        ci_upper = its_result.level_shift_ci_upper
        p_value = its_result.level_shift_pvalue

        att_pct = None
        if pre_mean_treated and pre_mean_treated != 0:
            att_pct = float(att / pre_mean_treated * 100)

        n_total = len(df_ts)

        return RateChangeResult(
            method="its",
            outcome_col=self.outcome_col,
            att=att,
            att_pct=att_pct,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            n_treated=n_total,
            n_control=0,
            n_periods_pre=n_periods_pre,
            n_periods_post=n_periods_post,
            pre_mean_treated=pre_mean_treated,
            pre_mean_control=None,
            parallel_trends_pvalue=None,
            staggered_adoption_detected=False,
            cluster_se_used=False,
            n_clusters=None,
            warnings=all_warnings,
            method_detail=its_result,
        )
