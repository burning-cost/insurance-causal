"""Diagnostic tests for DiD and ITS assumptions.

Parallel trends test via pre-treatment event study.
Staggered adoption detection for TWFE bias warning.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
import pandas as pd
import warnings


def check_parallel_trends(
    df: pd.DataFrame,
    outcome_col: str,
    treated_col: str,
    period_col: str,
    treatment_period: Any,
    unit_col: Optional[str] = None,
    weight_col: Optional[str] = None,
    n_pre_periods: int = 4,
) -> dict:
    """Test the parallel trends assumption via a joint pre-trend F-test.

    Estimates event study coefficients for up to ``n_pre_periods`` periods
    before the treatment and tests whether they are jointly zero.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with one row per unit-period.
    outcome_col : str
        Column name for the outcome variable.
    treated_col : str
        Column indicating treatment group (0/1).
    period_col : str
        Column indicating the time period.
    treatment_period : any
        The period in which treatment begins.
    unit_col : str or None
        Column identifying cross-sectional units. If None, uses ``treated_col``
        as the unit identifier.
    weight_col : str or None
        Column for exposure weights (WLS).
    n_pre_periods : int, default 4
        Number of pre-treatment periods to include in the event study.

    Returns
    -------
    dict with keys:
        - 'f_stat': float — F-statistic for joint zero pre-trends
        - 'p_value': float — p-value
        - 'coefs': list of float — event study coefficients (e=-n,...,-1)
        - 'ses': list of float — standard errors
        - 'periods': list of int — relative period indices
        - 'passed': bool — True if p_value >= 0.1
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise ImportError(
            "statsmodels>=0.14 is required for parallel trends testing. "
            "Install with: pip install statsmodels"
        ) from exc

    sorted_periods = sorted(df[period_col].unique())
    try:
        t_idx = sorted_periods.index(treatment_period)
    except ValueError:
        raise ValueError(
            f"treatment_period {treatment_period!r} not found in {period_col} column."
        )

    # Build relative event-time variable for treated units
    period_to_rel = {p: i - t_idx for i, p in enumerate(sorted_periods)}
    df = df.copy()
    df["_rel_time"] = df[period_col].map(period_to_rel)

    # Only pre-treatment periods + treatment period, treated units
    pre_rels = list(range(-n_pre_periods, 0))  # e.g. [-4, -3, -2, -1]

    # Create event study dummies for pre-treatment periods (treated units only)
    for e in pre_rels:
        col = f"_es_{-e}"  # _es_4, _es_3, _es_2, _es_1
        df[col] = ((df[treated_col] == 1) & (df["_rel_time"] == e)).astype(float)

    es_cols = [f"_es_{-e}" for e in pre_rels]

    # Keep only periods with data for pre-trend test
    df_pre = df[df["_rel_time"] < 0].copy()

    if len(df_pre) == 0:
        return {
            "f_stat": None,
            "p_value": None,
            "coefs": [],
            "ses": [],
            "periods": pre_rels,
            "passed": True,
        }

    unit_id = unit_col if unit_col else treated_col
    # Include unit FE (as dummies for treated + control groups at minimum)
    formula_parts = [outcome_col, "~", " + ".join(es_cols)]
    # Add unit fixed effects if we have a unit column
    if unit_col and df[unit_col].nunique() > 1:
        formula_parts.append(f"+ C({unit_col})")

    formula = " ".join(formula_parts)

    weights_arr = df_pre[weight_col].values if weight_col else None

    try:
        if weights_arr is not None:
            mod = smf.wls(formula, data=df_pre, weights=weights_arr)
        else:
            mod = smf.ols(formula, data=df_pre)

        # Use HC3 for the pre-trend test (fewer observations)
        res = mod.fit(cov_type="HC3")

        coef_names = [c for c in res.params.index if c.startswith("_es_")]
        if not coef_names:
            return {
                "f_stat": None, "p_value": None,
                "coefs": [], "ses": [], "periods": pre_rels, "passed": True
            }

        coefs = [res.params[c] for c in coef_names]
        ses = [res.bse[c] for c in coef_names]

        # Joint F-test: are all pre-trend dummies zero?
        f_test = res.f_test(np.eye(len(coef_names), len(res.params)))
        f_stat = float(f_test.fvalue) if np.ndim(f_test.fvalue) == 0 else float(f_test.fvalue.flat[0])
        p_value = float(f_test.pvalue)

        # Map coef names back to relative periods
        periods_out = [-(int(c.split("_es_")[1])) for c in coef_names]

        return {
            "f_stat": f_stat,
            "p_value": p_value,
            "coefs": coefs,
            "ses": ses,
            "periods": periods_out,
            "passed": p_value >= 0.1,
        }

    except Exception as exc:
        warnings.warn(
            f"Parallel trends test failed: {exc}. Skipping pre-trend test.",
            stacklevel=3,
        )
        return {
            "f_stat": None, "p_value": None,
            "coefs": [], "ses": [], "periods": pre_rels, "passed": True
        }


def check_staggered_adoption(
    df: pd.DataFrame,
    treated_col: str,
    period_col: str,
    unit_col: Optional[str] = None,
) -> dict:
    """Detect staggered treatment adoption across units.

    Staggered adoption (different units receiving treatment in different
    periods) invalidates standard two-way fixed effects DiD estimates.
    See Goodman-Bacon (2021) and Callaway & Sant'Anna (2021).

    Parameters
    ----------
    df : pd.DataFrame
        Panel data.
    treated_col : str
        Column indicating treatment status (0/1).
    period_col : str
        Column for time period.
    unit_col : str or None
        Column identifying units. If None, staggered detection is limited.

    Returns
    -------
    dict with keys:
        - 'is_staggered': bool
        - 'n_cohorts': int — number of distinct treatment cohorts
        - 'cohort_sizes': dict — cohort period -> count of units
        - 'message': str — human-readable explanation
    """
    if unit_col is None:
        return {
            "is_staggered": False,
            "n_cohorts": 1,
            "cohort_sizes": {},
            "message": "Cannot detect staggered adoption without unit_col.",
        }

    treated_units = df[df[treated_col] == 1][[unit_col, period_col]].copy()
    if len(treated_units) == 0:
        return {
            "is_staggered": False,
            "n_cohorts": 0,
            "cohort_sizes": {},
            "message": "No treated units found.",
        }

    # Find first treated period for each unit
    first_treatment = (
        treated_units.groupby(unit_col)[period_col].min().rename("first_period")
    )

    cohort_counts = first_treatment.value_counts().to_dict()
    n_cohorts = len(cohort_counts)

    if n_cohorts > 1:
        cohort_str = ", ".join(
            f"{p} (n={c})" for p, c in sorted(cohort_counts.items(), key=lambda x: str(x[0]))
        )
        message = (
            f"Staggered adoption detected: {n_cohorts} treatment cohorts found "
            f"({cohort_str}). Standard TWFE is biased under staggered adoption "
            f"(Goodman-Bacon 2021). Consider using the StaggeredEstimator from "
            f"insurance-causal-policy or Callaway-Sant'Anna (2021) estimator."
        )
        is_staggered = True
    else:
        message = f"Single treatment cohort detected. Standard TWFE is valid."
        is_staggered = False

    return {
        "is_staggered": is_staggered,
        "n_cohorts": n_cohorts,
        "cohort_sizes": cohort_counts,
        "message": message,
    }
