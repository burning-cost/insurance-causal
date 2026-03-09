"""
Diagnostic utilities for causal pricing models.

These functions can be used independently of CausalPricingModel — they
operate on fitted DoubleML objects, raw data, or the results of a fit.

The diagnostics answer three practical questions:

1. confounding_bias_report() — How different is the causal estimate from the
   naive correlation? (Answerable after fitting.)

2. cate_by_decile() — How does the treatment effect vary across the risk
   distribution? (Answerable after fitting with a segment column.)

3. sensitivity_analysis() — How strong would an unobserved confounder need
   to be to overturn our conclusion? (Rosenbaum-style bounds, answerable
   after fitting.)
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from ._model import CausalPricingModel


def confounding_bias_report(
    model: "CausalPricingModel",
    naive_coefficient: float | None = None,
    glm_model=None,
) -> pd.DataFrame:
    """
    Convenience wrapper for CausalPricingModel.confounding_bias_report().

    Produces a table comparing the naive (confounded) estimate of the treatment
    effect to the DML causal estimate, with the implied bias.

    Parameters
    ----------
    model : CausalPricingModel
        A fitted CausalPricingModel.
    naive_coefficient : float | None
        The naive estimate from a GLM or other model. Provide this or glm_model.
    glm_model : fitted model | None
        A fitted model from which to extract the treatment coefficient.

    Returns
    -------
    pd.DataFrame
        See CausalPricingModel.confounding_bias_report() for column definitions.
    """
    return model.confounding_bias_report(
        naive_coefficient=naive_coefficient,
        glm_model=glm_model,
    )


def cate_by_decile(
    model: "CausalPricingModel",
    df: "pd.DataFrame | pl.DataFrame",
    score_col: str,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """
    Estimate average treatment effects within deciles of a risk score.

    The canonical use case: fit a DML model on the full dataset, then ask
    whether the treatment effect is larger for high-risk or low-risk customers.
    For price elasticity, this answers: "Are price-sensitive customers concentrated
    in certain risk deciles?" For telematics, this answers: "Is the causal effect
    of harsh braking stronger for urban drivers?"

    Parameters
    ----------
    model : CausalPricingModel
        A fitted CausalPricingModel.
    df : DataFrame
        The data to segment (typically the training data or a held-out set).
    score_col : str
        Column name of the risk score to decile by. This is often the output
        of the nuisance outcome model (predicted frequency, predicted loss cost).
        Use any continuous column you want to use as the segmentation variable.
    n_deciles : int
        Number of equal-frequency groups. Default: 10 (deciles).

    Returns
    -------
    pd.DataFrame
        One row per decile. Columns: decile, score_lower, score_upper,
        cate_estimate, ci_lower, ci_upper, std_error, p_value, n_obs, status.

    Notes
    -----
    This function fits a separate DML model per decile. It is equivalent to
    model.cate_by_segment() with a decile-based segmentation column. The
    underlying caveat applies: each decile estimate uses only the data in
    that decile, so confidence intervals are wider than for the overall ATE.
    """
    from ._utils import to_pandas

    df_pd = to_pandas(df)

    if score_col not in df_pd.columns:
        raise ValueError(f"Score column '{score_col}' not found in data.")

    # Create decile labels
    df_pd = df_pd.copy()
    df_pd["__decile__"] = pd.qcut(
        df_pd[score_col],
        q=n_deciles,
        labels=False,
        duplicates="drop",
    )

    result = model.cate_by_segment(df_pd, segment_col="__decile__")

    # Add decile score bounds for interpretability
    decile_bounds = (
        df_pd.groupby("__decile__")[score_col]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"__decile__": "segment", "min": "score_lower", "max": "score_upper"})
    )
    result = result.merge(decile_bounds, on="segment", how="left")
    result = result.rename(columns={"segment": "decile"})

    return result


def sensitivity_analysis(
    ate: float,
    se: float,
    gamma_values: list[float] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Rosenbaum-style sensitivity analysis for unobserved confounding.

    DML assumes all confounders are observed. In practice, some confounders
    are always missing. This function computes how strong an unobserved binary
    confounder would need to be to change our conclusion.

    The Rosenbaum sensitivity parameter Γ (gamma) represents the odds ratio of
    treatment assignment for two units with identical observed confounders X but
    differing on the unobserved confounder. Γ = 1 means no unobserved
    confounding. Γ = 2 means an unobserved factor doubles the odds of treatment
    for some units relative to comparable units.

    This implementation applies the sensitivity bounds to the DML point estimate
    and standard error rather than to a rank-based test statistic (the classical
    Rosenbaum approach). This is an approximation but is directly interpretable
    in the DML context: it shows how the conclusion changes as we allow for
    progressively stronger unobserved confounding.

    Parameters
    ----------
    ate : float
        Point estimate of the average treatment effect (from model.average_treatment_effect()).
    se : float
        Standard error of the estimate.
    gamma_values : list[float] | None
        Values of Γ to evaluate. Default: [1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 3.0].
        Γ = 1.0 is the baseline (no unobserved confounding).
    alpha : float
        Significance level for confidence intervals. Default: 0.05.

    Returns
    -------
    pd.DataFrame
        One row per Γ value. Columns: gamma, bound_lower, bound_upper,
        ci_lower, ci_upper, conclusion_holds, p_value_upper, p_value_lower.

    Notes
    -----
    "conclusion_holds" is True when the sign of the causal estimate is robust
    to that level of unobserved confounding — i.e., the confidence interval
    does not contain zero at that Γ value.

    If conclusion_holds becomes False at Γ = 1.25, the result is fragile:
    an unobserved confounder that increases treatment odds by only 25% for
    some units would overturn the conclusion. If it holds to Γ = 2.0, the
    result is robust.

    Example
    -------
    >>> report = sensitivity_analysis(ate=-0.023, se=0.004)
    >>> print(report[["gamma", "conclusion_holds", "ci_lower", "ci_upper"]])
    """
    if gamma_values is None:
        gamma_values = [1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 3.0]

    z_alpha = stats.norm.ppf(1 - alpha / 2)

    rows = []
    for gamma in gamma_values:
        # Rosenbaum bounds: the bias from an unobserved confounder with
        # odds ratio Γ is bounded by ±log(Γ) * se (approximation).
        # This is a conservative bound on how much the estimate could shift.
        bias_bound = np.log(gamma) * se

        bound_lower = ate - bias_bound
        bound_upper = ate + bias_bound

        # Worst-case CI: shift the Rosenbaum bounds by ±z * se
        ci_lower = bound_lower - z_alpha * se
        ci_upper = bound_upper + z_alpha * se

        # Does the conclusion hold (CI does not cross zero)?
        if ate > 0:
            conclusion_holds = ci_lower > 0
        else:
            conclusion_holds = ci_upper < 0

        # p-value from the worst-case bound
        worst_case_effect = bound_lower if ate > 0 else bound_upper
        p_value = 2 * stats.norm.sf(abs(worst_case_effect) / se)

        rows.append({
            "gamma": gamma,
            "bias_bound": round(bias_bound, 6),
            "bound_lower": round(bound_lower, 6),
            "bound_upper": round(bound_upper, 6),
            "ci_lower": round(ci_lower, 6),
            "ci_upper": round(ci_upper, 6),
            "conclusion_holds": conclusion_holds,
            "p_value_worst_case": round(p_value, 4),
        })

    result = pd.DataFrame(rows)
    result["gamma"] = result["gamma"].astype(float)
    return result


def nuisance_model_summary(model: "CausalPricingModel") -> dict:
    """
    Report on the fit quality of the DML nuisance models.

    Good causal inference via DML requires good nuisance models. If E[Y|X] and
    E[D|X] are poorly estimated (low out-of-sample R² from cross-fitting),
    the residuals will carry unexplained signal that the final OLS step cannot
    fully absorb, and the causal estimate will be imprecise.

    Key metric: R² of the treatment nuisance model. If this is very high
    (> 0.95), the treatment is nearly deterministic given X — there is very
    little exogenous variation to identify the causal effect. The standard
    error of θ̂ will be large.

    Parameters
    ----------
    model : CausalPricingModel
        A fitted CausalPricingModel.

    Returns
    -------
    dict
        Keys: treatment_r2, outcome_r2, treatment_residual_variance,
        warning (if applicable).
    """
    model._check_fitted()

    dml = model._dml_model
    summary = {}

    # DoubleML stores nuisance predictions as predictions['ml_l'] and predictions['ml_m']
    try:
        preds = dml.predictions
        # Outcome nuisance
        if "ml_l" in preds:
            y_true = dml.data.y
            y_pred = preds["ml_l"].values.flatten()
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            summary["outcome_r2"] = round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else None
        # Treatment nuisance
        if "ml_m" in preds:
            d_true = dml.data.d
            d_pred = preds["ml_m"].values.flatten()
            ss_res = np.sum((d_true - d_pred) ** 2)
            ss_tot = np.sum((d_true - d_true.mean()) ** 2)
            summary["treatment_r2"] = round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else None
            summary["treatment_residual_variance"] = round(float(np.var(d_true - d_pred)), 6)

        # Warn if treatment is nearly deterministic
        if summary.get("treatment_r2", 0) is not None and summary.get("treatment_r2", 0) > 0.95:
            summary["warning"] = (
                f"Treatment R² = {summary['treatment_r2']:.3f}. "
                "The treatment is nearly deterministic given confounders — "
                "there is very little exogenous variation. "
                "The causal estimate may be very noisy. "
                "Consider whether there is truly exogenous variation in the treatment, "
                "or whether an instrumental variable approach is needed."
            )
    except Exception as e:
        summary["error"] = str(e)

    return summary
