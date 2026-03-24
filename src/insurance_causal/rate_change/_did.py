"""
Difference-in-Differences estimator for insurance rate change evaluation.

Implements exposure-weighted TWFE via statsmodels WLS. Cluster-robust SE
are used when G >= 20 clusters; HC3 otherwise (Cameron & Miller 2015).

The event study (for parallel trends testing) is fit as a separate model
with individual event-time dummies replacing the single _D_ indicator.
This is standard practice: the main model gives the ATT; the event study
model gives the pre-treatment coefficients for the parallel trends test.

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
        "statsmodels>=0.14 is required for DiD estimation. "
        "Install with: pip install 'insurance-causal[rate_change]'"
    ) from exc

from ._diagnostics import StaggeredAdoptionChecker, _run_joint_ftest_pre
from ._result import DiDResult


class DiDEstimator:
    """
    Two-way fixed effects DiD estimator.

    Fits two models:
    1. Main ATT model: Y ~ _D_ + C(unit) + C(period)   [WLS]
    2. Event study model: Y ~ evt_dummies + C(unit) + C(period)   [WLS, same weights/SE]

    The event study model is fit lazily when results() is called. Both models
    use the same exposure weights and cluster/HC3 SE specification.

    Parameters
    ----------
    outcome_col : str
    period_col : str
    treated_col : str
    unit_col : str
    change_period : int
        Integer-encoded change period.
    exposure_col : str | None
    cluster_col : str | None
        Column to cluster SE on. Defaults to unit_col.
    alpha : float
        Significance level for CIs.
    """

    def __init__(
        self,
        outcome_col: str,
        period_col: str,
        treated_col: str,
        unit_col: str,
        change_period: int,
        exposure_col: str | None = None,
        cluster_col: str | None = None,
        alpha: float = 0.05,
    ) -> None:
        self.outcome_col = outcome_col
        self.period_col = period_col
        self.treated_col = treated_col
        self.unit_col = unit_col
        self.change_period = change_period
        self.exposure_col = exposure_col
        self.cluster_col = cluster_col or unit_col
        self.alpha = alpha

        self._fitted_model = None
        self._evt_model = None
        self._df_reg = None
        self._cluster_se_used: bool = False
        self._n_clusters: int | None = None
        self._warnings: list[str] = []
        self._staggered_detected: bool = False
        self._staggered_cohorts: list[int] = []
        self._cov_type: str = "cluster"
        self._cov_kwds: dict = {}
        self._weights: pd.Series | None = None
        self._event_times_ex_ref: list[int] = []

    def fit(self, df: pd.DataFrame) -> "DiDEstimator":
        """
        Fit the DiD model on segment-period aggregated data.

        The data must already be aggregated to unit-period level before calling
        this method (handled by RateChangeEvaluator).
        """
        df_reg = df.copy()

        # Treatment indicator D_it
        df_reg["_D_"] = (
            (df_reg[self.treated_col] == 1) &
            (df_reg[self.period_col] >= self.change_period)
        ).astype(int)

        # Check staggered adoption
        checker = StaggeredAdoptionChecker()
        is_staggered, cohorts = checker.check(
            df_reg, self.treated_col, self.period_col, self.change_period
        )
        self._staggered_detected = is_staggered
        self._staggered_cohorts = cohorts
        if is_staggered:
            msg = checker.warn_if_staggered(is_staggered, cohorts)
            if msg:
                self._warnings.append(msg)

        # Build event study dummies (for the event study model)
        df_reg["_event_time_"] = df_reg[self.period_col] - self.change_period

        treated_event_times = sorted(
            df_reg.loc[df_reg[self.treated_col] == 1, "_event_time_"].unique().tolist()
        )
        # Exclude e=-1 as reference period
        event_times_ex_ref = [e for e in treated_event_times if e != -1]
        self._event_times_ex_ref = event_times_ex_ref

        # Create event study dummy columns in df_reg
        for e in event_times_ex_ref:
            col = f"_evt_{abs(e)}_" if e < 0 else f"_evt_pos_{e}_"
            df_reg[col] = (
                (df_reg["_event_time_"] == e) & (df_reg[self.treated_col] == 1)
            ).astype(int)

        # Cluster SE logic
        n_clusters = df_reg[self.cluster_col].nunique()
        self._n_clusters = n_clusters

        if n_clusters < 20:
            cov_type = "HC3"
            cov_kwds: dict = {}
            self._cluster_se_used = False
            msg = (
                f"Only {n_clusters} clusters in '{self.cluster_col}'. "
                "Clustered SE are unreliable with G < 20 (Cameron & Miller 2015). "
                "Using HC3 heteroskedasticity-robust SE. CIs may be anti-conservative."
            )
            warnings.warn(msg, UserWarning, stacklevel=3)
            self._warnings.append(msg)
        else:
            cov_type = "cluster"
            cov_kwds = {"groups": df_reg[self.cluster_col]}
            self._cluster_se_used = True

        self._cov_type = cov_type
        self._cov_kwds = cov_kwds

        # Exposure weights
        if self.exposure_col is not None:
            weights = df_reg[self.exposure_col].clip(lower=1e-6)
        else:
            weights = pd.Series(np.ones(len(df_reg)), index=df_reg.index)

        self._weights = weights

        # Main ATT formula (TWFE with unit and time FE via C() dummies)
        formula = (
            f"{self.outcome_col} ~ _D_ + C({self.unit_col}) + C({self.period_col})"
        )

        model = smf.wls(formula, data=df_reg, weights=weights).fit(
            cov_type=cov_type,
            cov_kwds=cov_kwds,
        )

        self._fitted_model = model
        self._df_reg = df_reg

        return self

    def results(self) -> DiDResult:
        """Extract DiDResult from the fitted model. Fits event study model lazily."""
        if self._fitted_model is None:
            raise RuntimeError("Call fit() before results().")

        from scipy import stats

        model = self._fitted_model
        z = stats.norm.ppf(1 - self.alpha / 2)

        # ATT coefficient
        att_param = "_D_"
        att = float(model.params[att_param])
        se = float(model.bse[att_param])
        ci_lower = att - z * se
        ci_upper = att + z * se
        p_value = float(model.pvalues[att_param])

        # Unit counts
        df_reg = self._df_reg
        n_treated = int(
            df_reg[df_reg[self.treated_col] == 1][self.unit_col].nunique()
        )
        n_control = int(
            df_reg[df_reg[self.treated_col] == 0][self.unit_col].nunique()
        )
        n_periods = int(df_reg[self.period_col].nunique())

        # Fit event study model and run parallel trends test
        self._fit_event_study_model()
        event_study_df, fstat, pvalue = self._run_event_study()

        return DiDResult(
            att=att,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            formula=str(self._fitted_model.model.formula),
            n_units_treated=n_treated,
            n_units_control=n_control,
            n_periods=n_periods,
            event_study_df=event_study_df,
            joint_pt_fstat=fstat,
            joint_pt_pvalue=pvalue,
        )

    def _fit_event_study_model(self) -> None:
        """
        Fit a separate event study model for the parallel trends diagnostic.

        The event study model replaces _D_ with individual event-time dummies.
        Uses the same WLS weights and SE specification as the main model.
        This is the standard approach: ATT from the main model, pre-treatment
        coefficients from the event study model.
        """
        if self._df_reg is None:
            return

        df_reg = self._df_reg
        event_times_ex_ref = self._event_times_ex_ref

        if not event_times_ex_ref:
            self._evt_model = None
            return

        # Only include event time dummies that exist in df_reg
        evt_cols = []
        for e in event_times_ex_ref:
            col = f"_evt_{abs(e)}_" if e < 0 else f"_evt_pos_{e}_"
            if col in df_reg.columns:
                evt_cols.append(col)

        if not evt_cols:
            self._evt_model = None
            return

        evt_terms = " + ".join(evt_cols)
        formula = (
            f"{self.outcome_col} ~ {evt_terms} + "
            f"C({self.unit_col}) + C({self.period_col})"
        )

        try:
            evt_model = smf.wls(
                formula, data=df_reg, weights=self._weights
            ).fit(
                cov_type=self._cov_type,
                cov_kwds=self._cov_kwds,
            )
            self._evt_model = evt_model
        except Exception:
            # If event study model fails (e.g. perfect collinearity), gracefully skip
            self._evt_model = None

    def _run_event_study(self) -> tuple[pd.DataFrame, float, float]:
        """
        Extract event study coefficients from the event study model and run
        a joint F-test on pre-treatment period dummies.

        Uses self._evt_model (fitted by _fit_event_study_model).
        """
        from scipy import stats

        z = stats.norm.ppf(1 - self.alpha / 2)
        event_times_ex_ref = self._event_times_ex_ref

        rows = []
        # Reference period (e = -1) is always zero by construction
        rows.append({
            "event_time": -1,
            "att_e": 0.0,
            "se_e": 0.0,
            "ci_lower_e": 0.0,
            "ci_upper_e": 0.0,
        })

        if self._evt_model is None:
            event_study_df = pd.DataFrame(rows)
            return event_study_df, float("nan"), float("nan")

        model = self._evt_model
        pre_param_names = []

        for e in sorted(event_times_ex_ref):
            col_base = f"_evt_{abs(e)}_" if e < 0 else f"_evt_pos_{e}_"

            # Find the actual param name in the model
            param_name = None
            for pname in model.params.index:
                if col_base in pname:
                    param_name = pname
                    break

            if param_name is None:
                continue

            coef = float(model.params[param_name])
            se_val = float(model.bse[param_name])

            rows.append({
                "event_time": e,
                "att_e": coef,
                "se_e": se_val,
                "ci_lower_e": coef - z * se_val,
                "ci_upper_e": coef + z * se_val,
            })

            if e < 0:
                pre_param_names.append(param_name)

        event_study_df = pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)

        fstat, pvalue = _run_joint_ftest_pre(model, pre_param_names)

        return event_study_df, fstat, pvalue
