"""
Diagnostic tools for DiD validity.

ParallelTrendsDiagnostic : event study + joint F-test on pre-treatment periods.
StaggeredAdoptionChecker : detects whether treated units have different
                           first-treatment periods.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ParallelTrendsResult:
    """
    Result of the parallel trends pre-test.

    Attributes
    ----------
    event_study_df : pd.DataFrame
        Columns: event_time, att_e, se_e, ci_lower_e, ci_upper_e.
    joint_pt_fstat : float
        F-statistic for H0: all pre-treatment event study coefficients are zero.
    joint_pt_pvalue : float
        P-value for the joint F-test.
    passes : bool
        True if joint_pt_pvalue > 0.05.
    n_pre_periods : int
        Number of pre-treatment event-time bins tested.
    """

    event_study_df: pd.DataFrame
    joint_pt_fstat: float
    joint_pt_pvalue: float
    passes: bool
    n_pre_periods: int


def run_event_study(
    df_reg: pd.DataFrame,
    fitted_model,
    event_times_ex_ref: list[int],
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, float, float]:
    """
    Extract event study coefficients and run a joint F-test on pre-treatment
    coefficients.

    Parameters
    ----------
    df_reg : pd.DataFrame
        Regression data (needed to identify column names).
    fitted_model : statsmodels fitted WLS result
        The fitted model containing event study dummy coefficients.
    event_times_ex_ref : list[int]
        Sorted list of event times (excluding the reference period e=-1).
    alpha : float
        Significance level for CIs.

    Returns
    -------
    (event_study_df, fstat, pvalue) : tuple
        event_study_df has columns:
            event_time, att_e, se_e, ci_lower_e, ci_upper_e
        fstat, pvalue from the joint test on pre-treatment dummies.
    """
    from scipy import stats

    z = stats.norm.ppf(1 - alpha / 2)
    rows = []

    # Reference period e=-1 is always included as zeros
    rows.append({
        "event_time": -1,
        "att_e": 0.0,
        "se_e": 0.0,
        "ci_lower_e": 0.0,
        "ci_upper_e": 0.0,
    })

    pre_param_names = []

    for e in sorted(event_times_ex_ref):
        if e < 0:
            col_name = f"_evt_{abs(e)}_"
        else:
            col_name = f"_evt_pos_{e}_"

        # statsmodels may encode the dummy with [T.1] suffix if categorical
        param_name = None
        for pname in fitted_model.params.index:
            if col_name in pname or pname == col_name:
                param_name = pname
                break

        if param_name is None:
            continue

        coef = fitted_model.params[param_name]
        se = fitted_model.bse[param_name]
        rows.append({
            "event_time": e,
            "att_e": coef,
            "se_e": se,
            "ci_lower_e": coef - z * se,
            "ci_upper_e": coef + z * se,
        })

        if e < 0:
            pre_param_names.append(param_name)

    event_study_df = pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)

    # Joint F-test on pre-treatment coefficients
    fstat = float("nan")
    pvalue = float("nan")

    if len(pre_param_names) >= 1:
        try:
            joint_test = fitted_model.wald_test(
                " = ".join([f"({p})" for p in pre_param_names]) + " = 0"
                if len(pre_param_names) == 1
                else [(f"({p})", 0) for p in pre_param_names],
                use_f=True,
            )
            # Robustly extract fstat and pvalue from whatever wald_test returns
            try:
                fstat = float(joint_test.fvalue)
                pvalue = float(joint_test.pvalue)
            except Exception:
                fstat = float("nan")
                pvalue = float("nan")
        except Exception:
            # If Wald test fails (e.g. singular matrix), fall back to nan
            pass

    return event_study_df, fstat, pvalue


def _run_joint_ftest_pre(fitted_model, pre_param_names: list[str]) -> tuple[float, float]:
    """
    Run a joint Wald test that all pre_param_names coefficients are zero.

    Returns
    -------
    (fstat, pvalue) : tuple[float, float]
    """
    if not pre_param_names:
        return float("nan"), float("nan")

    try:
        # Build restriction matrix: R * beta = 0, one row per parameter
        all_params = list(fitted_model.params.index)
        n_params = len(all_params)
        n_rest = len(pre_param_names)

        R = np.zeros((n_rest, n_params))
        for i, pname in enumerate(pre_param_names):
            j = all_params.index(pname)
            R[i, j] = 1.0

        result = fitted_model.f_test(R)
        fstat = float(result.fvalue)
        pvalue = float(result.pvalue)
        return fstat, pvalue
    except Exception:
        return float("nan"), float("nan")


class StaggeredAdoptionChecker:
    """
    Checks whether treated units have different first-treatment periods.

    If max(first_period) - min(first_period) > 0 among treated units,
    staggered adoption is present. Standard TWFE is biased under
    heterogeneous treatment effects with staggered adoption
    (Goodman-Bacon 2021).
    """

    def check(
        self,
        df: pd.DataFrame,
        treated_col: str,
        period_col: str,
        change_period: int,
    ) -> tuple[bool, list[int]]:
        """
        Parameters
        ----------
        df : pd.DataFrame
            Panel data with at least treated_col and period_col.
        treated_col : str
            Column name of treatment indicator.
        period_col : str
            Column name of period (integer-encoded).
        change_period : int
            Nominal change period (used to identify first treatment for
            each unit via period_col).

        Returns
        -------
        (is_staggered, cohorts) : tuple
            is_staggered: True if multiple first-treatment periods detected.
            cohorts: sorted list of distinct first-treatment periods.
        """
        treated_df = df[df[treated_col] == 1]
        if treated_df.empty:
            return False, []

        # For each treated unit (or group of rows), find first period >= change_period
        # where the unit actually appears. If unit_col is not available, we look at
        # the treated column directly.
        # We assume df has a unit identifier; use treated_col groupby to find first period.
        # If the data has a unit_col, the caller should pass that. Here we work with
        # the data as-is: find first period in the post-treatment portion per unit.

        # Find first treatment period per unit if a unit_id column exists
        if "_unit_id_enc_" in df.columns:
            unit_col = "_unit_id_enc_"
        elif "unit_id" in df.columns:
            unit_col = "unit_id"
        elif "segment_id" in df.columns:
            unit_col = "segment_id"
        else:
            # No unit column — can't detect staggered adoption reliably
            return False, []

        # Find first period where treated==1 for each unit.
        # Do NOT filter by >= change_period — the whole point is to detect
        # units that were treated at different times.
        first_treatment = (
            treated_df
            .groupby(unit_col)[period_col]
            .min()
        )

        if first_treatment.empty:
            return False, []

        cohorts = sorted(first_treatment.unique().tolist())
        is_staggered = len(cohorts) > 1

        return is_staggered, cohorts

    def warn_if_staggered(
        self,
        is_staggered: bool,
        cohorts: list[int],
    ) -> str | None:
        """
        Emit a UserWarning if staggered adoption is detected.

        Returns
        -------
        str | None
            Warning message if staggered, else None.
        """
        if not is_staggered:
            return None

        msg = (
            f"Multiple treatment cohorts detected (cohorts: {cohorts}). "
            "Standard TWFE DiD is biased under heterogeneous treatment effects with staggered "
            "adoption (Goodman-Bacon 2021). Use insurance_causal_policy.StaggeredEstimator "
            "(Callaway-Sant'Anna 2021) for valid staggered DiD. "
            "RateChangeEvaluator will proceed with standard TWFE but results should be "
            "treated as approximate."
        )
        warnings.warn(msg, UserWarning, stacklevel=4)
        return msg
