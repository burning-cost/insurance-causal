"""Difference-in-Differences estimator for rate change evaluation.

Implements exposure-weighted TWFE via statsmodels WLS with cluster-robust
standard errors. Falls back to HC3 when cluster count < 20.
"""

from __future__ import annotations

from typing import Any, Optional
import warnings
import numpy as np
import pandas as pd

from ._result import DiDResult
from ._diagnostics import check_parallel_trends, check_staggered_adoption


class DiDEstimator:
    """Two-way fixed effects DiD with exposure weighting.

    Parameters
    ----------
    outcome_col : str
        Column name for the outcome variable.
    treated_col : str
        Column indicating treatment group membership (0/1, time-invariant).
    period_col : str
        Column indicating time period.
    treatment_period : any
        The period in which treatment begins.
    unit_col : str or None
        Column identifying cross-sectional units. Required for clustering.
    weight_col : str or None
        Column for exposure weights (WLS). If None, OLS is used.
    min_clusters_for_cluster_se : int, default 20
        Minimum number of clusters to use cluster-robust SE. Below this
        threshold, HC3 is used with a warning.
    run_event_study : bool, default True
        Whether to run the event study for parallel trends diagnostics.
    n_pre_periods : int, default 4
        Number of pre-treatment periods for event study.
    """

    def __init__(
        self,
        outcome_col: str,
        treated_col: str,
        period_col: str,
        treatment_period: Any,
        unit_col: Optional[str] = None,
        weight_col: Optional[str] = None,
        min_clusters_for_cluster_se: int = 20,
        run_event_study: bool = True,
        n_pre_periods: int = 4,
    ) -> None:
        self.outcome_col = outcome_col
        self.treated_col = treated_col
        self.period_col = period_col
        self.treatment_period = treatment_period
        self.unit_col = unit_col
        self.weight_col = weight_col
        self.min_clusters_for_cluster_se = min_clusters_for_cluster_se
        self.run_event_study = run_event_study
        self.n_pre_periods = n_pre_periods

        self._result: Optional[DiDResult] = None
        self._staggered_info: Optional[dict] = None
        self._model_result = None

    def fit(self, df: pd.DataFrame) -> "DiDEstimator":
        """Estimate the DiD ATT.

        Parameters
        ----------
        df : pd.DataFrame
            Panel data with columns as specified in __init__.

        Returns
        -------
        DiDEstimator
            Self for method chaining.
        """
        try:
            import statsmodels.formula.api as smf
        except ImportError as exc:
            raise ImportError(
                "statsmodels>=0.14 is required for DiD estimation. "
                "Install with: pip install statsmodels"
            ) from exc

        df = df.copy()

        # Validate inputs
        self._validate(df)

        # Check for staggered adoption
        self._staggered_info = check_staggered_adoption(
            df, self.treated_col, self.period_col, self.unit_col
        )

        # Build the treatment indicator D_it = treated_i AND period_t >= T
        sorted_periods = sorted(df[self.period_col].unique())
        post_periods = set(p for p in sorted_periods if p >= self.treatment_period)
        df["_post"] = df[self.period_col].isin(post_periods).astype(int)
        df["_D"] = (df[self.treated_col] * df["_post"]).astype(float)

        # Build formula with unit and period FE
        # Use C() for categorical FE dummies
        formula_parts = [self.outcome_col, "~ _D"]

        if self.unit_col and df[self.unit_col].nunique() > 1:
            formula_parts.append(f"+ C({self.unit_col})")

        if df[self.period_col].nunique() > 1:
            formula_parts.append(f"+ C({self.period_col})")

        formula = " ".join(formula_parts)

        weights_arr = df[self.weight_col].values if self.weight_col else None

        # Determine SE type
        n_units = df[self.unit_col].nunique() if self.unit_col else 1
        use_cluster = (
            self.unit_col is not None
            and n_units >= self.min_clusters_for_cluster_se
        )

        if weights_arr is not None:
            mod = smf.wls(formula, data=df, weights=weights_arr)
        else:
            mod = smf.ols(formula, data=df)

        if use_cluster:
            res = mod.fit(
                cov_type="cluster",
                cov_kwds={"groups": df[self.unit_col]},
            )
            se_type = "cluster"
        else:
            res = mod.fit(cov_type="HC3")
            se_type = "HC3"
            if self.unit_col and n_units < self.min_clusters_for_cluster_se:
                warnings.warn(
                    f"Only {n_units} clusters available (< {self.min_clusters_for_cluster_se}). "
                    f"Using HC3 heteroskedasticity-robust SE instead of cluster SE. "
                    f"Estimates are valid but SE may be less reliable with few clusters.",
                    stacklevel=3,
                )

        self._model_result = res

        att = float(res.params["_D"])
        se = float(res.bse["_D"])
        ci = res.conf_int().loc["_D"]
        ci_lower = float(ci.iloc[0])
        ci_upper = float(ci.iloc[1])
        p_value = float(res.pvalues["_D"])

        n_treated = int(df[self.treated_col].sum()) if df[self.treated_col].nunique() <= 2 else int(
            (df[self.treated_col] == 1).sum()
        )

        # Run parallel trends event study
        pt_result = None
        event_coefs = None
        event_ses = None
        event_periods = None

        if self.run_event_study:
            try:
                pt_result = check_parallel_trends(
                    df=df,
                    outcome_col=self.outcome_col,
                    treated_col=self.treated_col,
                    period_col=self.period_col,
                    treatment_period=self.treatment_period,
                    unit_col=self.unit_col,
                    weight_col=self.weight_col,
                    n_pre_periods=self.n_pre_periods,
                )
                event_coefs = pt_result["coefs"]
                event_ses = pt_result["ses"]
                event_periods = pt_result["periods"]
            except Exception as exc:
                warnings.warn(f"Event study failed: {exc}", stacklevel=3)

        self._result = DiDResult(
            att=att,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            n_obs=int(res.nobs),
            n_units=int(df[self.unit_col].nunique()) if self.unit_col else 1,
            n_treated=n_treated,
            n_periods=int(df[self.period_col].nunique()),
            se_type=se_type,
            parallel_trends_f_stat=pt_result["f_stat"] if pt_result else None,
            parallel_trends_p_value=pt_result["p_value"] if pt_result else None,
            event_study_coefs=event_coefs,
            event_study_se=event_ses,
            event_study_periods=event_periods,
        )

        return self

    def result(self) -> DiDResult:
        """Return the DiD result.

        Returns
        -------
        DiDResult

        Raises
        ------
        RuntimeError
            If fit() has not been called.
        """
        if self._result is None:
            raise RuntimeError("Call fit() before accessing results.")
        return self._result

    def staggered_info(self) -> Optional[dict]:
        """Return staggered adoption detection results."""
        return self._staggered_info

    def _validate(self, df: pd.DataFrame) -> None:
        """Validate input data."""
        required = [self.outcome_col, self.treated_col, self.period_col]
        if self.unit_col:
            required.append(self.unit_col)
        if self.weight_col:
            required.append(self.weight_col)

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")

        if df[self.treated_col].nunique() < 2:
            raise ValueError(
                f"Column {self.treated_col!r} must have both treated (1) and "
                f"control (0) units for DiD. Found only: "
                f"{df[self.treated_col].unique().tolist()}"
            )

        if self.treatment_period not in df[self.period_col].values:
            raise ValueError(
                f"treatment_period {self.treatment_period!r} not found in "
                f"column {self.period_col!r}."
            )

        if self.weight_col and (df[self.weight_col] <= 0).any():
            raise ValueError(
                f"Weight column {self.weight_col!r} contains non-positive values. "
                f"Exposure weights must be strictly positive."
            )
