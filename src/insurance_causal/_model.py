"""
CausalPricingModel — the main user-facing class.

This is a thin, opinionated wrapper over DoubleML's DoubleMLPLR (partially
linear regression) estimator. The goal is to hide the econometrics plumbing
and expose an interface that a pricing actuary can use without reading
Chernozhukov et al. (2018).

The key steps under the hood:

1. Prepare data: convert polars to pandas, validate, apply outcome/treatment
   transformations.

2. Build nuisance models: two CatBoost models — one for E[Y|X] (outcome
   nuisance) and one for E[D|X] (treatment nuisance / propensity).

3. Fit DoubleMLPLR with cross-fitting (K-fold, default 5). Cross-fitting
   ensures the nuisance estimation errors are asymptotically independent of
   the score, giving valid inference on θ.

4. Extract the coefficient (ATE), standard error, and confidence interval
   from the DoubleML object.

5. Optionally compute CATE by segment by splitting the data and returning
   per-segment ATE estimates. Full nonparametric CATE via causal forest is
   not yet implemented (it requires EconML; planned for v0.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .treatments import AnyTreatment, BinaryTreatment, PriceChangeTreatment
from ._utils import (
    to_pandas,
    build_catboost_regressor,
    build_catboost_classifier,
    make_doubleml_data,
    poisson_outcome_transform,
    gamma_outcome_transform,
    check_overlap,
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AverageTreatmentEffect:
    """
    Result of a DML average treatment effect estimation.

    Attributes
    ----------
    estimate : float
        Point estimate of the causal treatment effect (θ in the DML literature).
    std_error : float
        Standard error of the estimate, from the asymptotic normal distribution.
    ci_lower : float
        Lower bound of the 95% confidence interval.
    ci_upper : float
        Upper bound of the 95% confidence interval.
    p_value : float
        Two-sided p-value for H₀: θ = 0.
    n_obs : int
        Number of observations used in fitting.
    treatment_col : str
        Name of the treatment column.
    outcome_col : str
        Name of the outcome column.
    """

    estimate: float
    std_error: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_obs: int
    treatment_col: str
    outcome_col: str

    def __str__(self) -> str:
        return (
            f"Average Treatment Effect\n"
            f"  Treatment: {self.treatment_col}\n"
            f"  Outcome:   {self.outcome_col}\n"
            f"  Estimate:  {self.estimate:.4f}\n"
            f"  Std Error: {self.std_error:.4f}\n"
            f"  95% CI:    ({self.ci_lower:.4f}, {self.ci_upper:.4f})\n"
            f"  p-value:   {self.p_value:.4f}\n"
            f"  N:         {self.n_obs:,}"
        )


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------


class CausalPricingModel:
    """
    Causal treatment effect estimation for insurance pricing data.

    Uses Double Machine Learning (Chernozhukov et al., 2018) to estimate the
    causal effect of a treatment on an insurance outcome, controlling for
    observed confounders. The estimation uses CatBoost as the nuisance model
    for both the outcome and treatment equations.

    Parameters
    ----------
    outcome : str
        Column name of the outcome variable. For claim frequency: claim count.
        For renewal: binary renewal indicator. For severity: claim amount.
    outcome_type : {"continuous", "binary", "poisson", "gamma"}
        Distribution family of the outcome. Determines how the outcome is
        transformed before DML fitting:
        - "continuous" : used as-is. Appropriate for log loss cost or
          any approximately symmetric continuous outcome.
        - "binary" : used as-is (0/1). The DML coefficient is the ATE on
          the probability scale.
        - "poisson" : divided by exposure (if provided) to give frequency.
          Appropriate for claim count outcomes.
        - "gamma" : log-transformed. Appropriate for claim severity.
    treatment : PriceChangeTreatment | BinaryTreatment | ContinuousTreatment
        Treatment specification. See treatments.py for details.
    confounders : list[str]
        Column names of the confounders X. These are the variables that
        causally affect both the treatment and the outcome — the variables
        we need to control for to identify the causal effect of treatment.
        Include all standard rating factors: age, vehicle, postcode, NCB,
        prior claims, etc.
    exposure_col : str | None
        Column name of earned years / policy duration. Required if
        outcome_type="poisson". Ignored for other outcome types.
    nuisance_model : {"catboost"}
        Nuisance model to use for E[Y|X] and E[D|X]. Currently only CatBoost
        is supported. v0.2 will add arbitrary sklearn estimators.
    cv_folds : int
        Number of cross-fitting folds. Default: 5. More folds give more stable
        estimates at the cost of fitting time. 5 is the standard choice.
    random_state : int
        Random seed for reproducibility.

    Notes
    -----
    The DML assumption is that all relevant confounders are in the confounders
    list — the "conditional ignorability" or "no unobserved confounders"
    assumption. DML cannot adjust for confounders you did not include. If
    important confounders are missing (e.g. actual annual mileage when you
    only have stated mileage), the estimate will be biased. Use
    sensitivity_analysis() to bound the potential bias from unobserved
    confounders.

    References
    ----------
    Chernozhukov, V. et al. (2018). "Double/Debiased Machine Learning for
    Treatment and Structural Parameters." The Econometrics Journal, 21(1).
    """

    def __init__(
        self,
        outcome: str,
        outcome_type: Literal["continuous", "binary", "poisson", "gamma"],
        treatment: AnyTreatment,
        confounders: list[str],
        exposure_col: str | None = None,
        nuisance_model: Literal["catboost"] = "catboost",
        cv_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        self.outcome = outcome
        self.outcome_type = outcome_type
        self.treatment = treatment
        self.confounders = confounders
        self.exposure_col = exposure_col
        self.nuisance_model = nuisance_model
        self.cv_folds = cv_folds
        self.random_state = random_state

        # Set after fitting
        self._dml_model = None
        self._fitted = False
        self._n_obs: int = 0
        self._treatment_overlap: dict = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, df: "pd.DataFrame | pl.DataFrame") -> "CausalPricingModel":
        """
        Fit the DML model on observational data.

        Parameters
        ----------
        df : polars.DataFrame or pandas.DataFrame
            Policy-level data. Must contain the outcome column, the treatment
            column, and all confounder columns.

        Returns
        -------
        self : CausalPricingModel
            Returns self so you can chain: model.fit(df).average_treatment_effect()
        """
        import doubleml as dml

        df_pd = to_pandas(df)
        self._n_obs = len(df_pd)

        # --- Validate inputs -----------------------------------------------
        self._validate_columns(df_pd)
        treatment_series = df_pd[self.treatment.column]
        self.treatment.validate(treatment_series)

        # --- Transform treatment -------------------------------------------
        treatment_transformed = self.treatment.transform(treatment_series)
        self._treatment_overlap = check_overlap(treatment_transformed.values)

        # --- Transform outcome ----------------------------------------------
        exposure = None
        if self.exposure_col is not None:
            exposure = df_pd[self.exposure_col].values

        outcome_transformed = self._transform_outcome(
            df_pd[self.outcome].values, exposure
        )

        # --- Build working DataFrame for DoubleML --------------------------
        # Use internal column names to avoid clashes with user's column names
        _outcome_col = "__y__"
        _treatment_col = "__d__"

        df_dml = df_pd[self.confounders].copy()
        df_dml[_treatment_col] = treatment_transformed.values
        df_dml[_outcome_col] = outcome_transformed

        dml_data = dml.DoubleMLData(
            df_dml,
            y_col=_outcome_col,
            d_cols=_treatment_col,
            x_cols=self.confounders,
        )

        # --- Build nuisance models -----------------------------------------
        ml_y, ml_d = self._build_nuisance_models()

        # --- Fit DoubleMLPLR -----------------------------------------------
        dml_plr = dml.DoubleMLPLR(
            obj_dml_data=dml_data,
            ml_l=ml_y,   # nuisance model for outcome E[Y|X]
            ml_m=ml_d,   # nuisance model for treatment E[D|X]
            n_folds=self.cv_folds,
            score="partialling out",
        )
        dml_plr.fit()

        self._dml_model = dml_plr
        self._fitted = True

        return self

    def _transform_outcome(
        self,
        y: np.ndarray,
        exposure: np.ndarray | None,
    ) -> np.ndarray:
        """Apply outcome-type-specific transformation."""
        if self.outcome_type == "poisson":
            return poisson_outcome_transform(y, exposure)
        elif self.outcome_type == "gamma":
            return gamma_outcome_transform(y, exposure)
        else:
            # "continuous" and "binary" — use as-is
            return y.astype(float)

    def _validate_columns(self, df_pd: pd.DataFrame) -> None:
        """Check all required columns are present."""
        required = set(self.confounders) | {self.outcome, self.treatment.column}
        if self.exposure_col:
            required.add(self.exposure_col)
        missing = required - set(df_pd.columns)
        if missing:
            raise ValueError(
                f"Missing columns in input data: {sorted(missing)}. "
                f"Available columns: {sorted(df_pd.columns)}."
            )

    def _build_nuisance_models(self) -> tuple:
        """
        Build CatBoost nuisance models for outcome and treatment.

        For binary treatments (propensity model), we use CatBoostClassifier.
        For continuous treatments and all outcomes, we use CatBoostRegressor.
        """
        ml_y = build_catboost_regressor(self.random_state)
        if isinstance(self.treatment, BinaryTreatment):
            ml_d = build_catboost_classifier(self.random_state)
        else:
            ml_d = build_catboost_regressor(self.random_state)
        return ml_y, ml_d

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "Model has not been fitted. Call .fit(df) first."
            )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def average_treatment_effect(self) -> AverageTreatmentEffect:
        """
        Return the estimated average treatment effect with confidence interval.

        The ATE (θ) is the causal effect of a unit increase in the (transformed)
        treatment on the outcome, averaged over the population. For a
        PriceChangeTreatment with scale="log", this is the elasticity: a 1-unit
        increase in log(1 + price_change) corresponds to a θ change in the outcome.

        Returns
        -------
        AverageTreatmentEffect
            Dataclass with estimate, std_error, ci_lower, ci_upper, p_value.

        Notes
        -----
        The 95% CI uses the asymptotic normal approximation: estimate ± 1.96 * SE.
        This is valid when sample sizes are large (n > 500) and the nuisance
        functions are estimated consistently. For small samples, consider
        bootstrap CIs (available on the underlying DoubleML object via
        model._dml_model.bootstrap()).
        """
        self._check_fitted()

        coef = float(self._dml_model.coef[0])
        se = float(self._dml_model.se[0])
        t_stat = float(self._dml_model.t_stat[0])
        p_val = float(self._dml_model.pval[0])

        # DoubleML returns 95% CI directly
        ci = self._dml_model.confint(level=0.95)
        ci_lower = float(ci.iloc[0, 0])
        ci_upper = float(ci.iloc[0, 1])

        return AverageTreatmentEffect(
            estimate=coef,
            std_error=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_val,
            n_obs=self._n_obs,
            treatment_col=self.treatment.column,
            outcome_col=self.outcome,
        )

    def cate_by_segment(
        self,
        df: "pd.DataFrame | pl.DataFrame",
        segment_col: str,
        min_segment_size: int = 200,
    ) -> pd.DataFrame:
        """
        Estimate average treatment effects separately within each segment.

        This is an approximation of CATE (Conditional Average Treatment Effect)
        by subgroup. Each segment gets its own DML estimate, using the same
        nuisance model architecture and CV fold count as the main model.

        This is computationally expensive: it fits a new DML model for each
        segment. For large datasets with many segments, restrict to the most
        commercially important segments.

        Full nonparametric CATE via causal forest is planned for v0.2 and will
        be substantially more efficient.

        Parameters
        ----------
        df : DataFrame
            The same data used for fitting (or held-out data).
        segment_col : str
            Column name of the segmentation variable. Typically a categorical
            such as age_band, vehicle_group, or ncb_years.
        min_segment_size : int
            Minimum number of observations for a segment to be estimated.
            Segments smaller than this are marked as insufficient_data.

        Returns
        -------
        pd.DataFrame
            One row per segment. Columns: segment, cate_estimate, ci_lower,
            ci_upper, std_error, p_value, n_obs.
        """
        self._check_fitted()
        df_pd = to_pandas(df)

        if segment_col not in df_pd.columns:
            raise ValueError(
                f"Segment column '{segment_col}' not found in data."
            )

        results = []
        for segment_val in sorted(df_pd[segment_col].unique()):
            mask = df_pd[segment_col] == segment_val
            n_seg = mask.sum()

            row: dict = {
                "segment": segment_val,
                "n_obs": n_seg,
                "cate_estimate": None,
                "ci_lower": None,
                "ci_upper": None,
                "std_error": None,
                "p_value": None,
                "status": "ok",
            }

            if n_seg < min_segment_size:
                row["status"] = "insufficient_data"
                results.append(row)
                continue

            df_seg = df_pd[mask].copy()
            seg_model = CausalPricingModel(
                outcome=self.outcome,
                outcome_type=self.outcome_type,
                treatment=self.treatment,
                confounders=self.confounders,
                exposure_col=self.exposure_col,
                nuisance_model=self.nuisance_model,
                cv_folds=min(self.cv_folds, max(2, n_seg // 100)),
                random_state=self.random_state,
            )
            try:
                seg_model.fit(df_seg)
                ate = seg_model.average_treatment_effect()
                row["cate_estimate"] = ate.estimate
                row["ci_lower"] = ate.ci_lower
                row["ci_upper"] = ate.ci_upper
                row["std_error"] = ate.std_error
                row["p_value"] = ate.p_value
            except Exception as e:
                row["status"] = f"error: {e}"

            results.append(row)

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def confounding_bias_report(
        self,
        naive_coefficient: float | None = None,
        glm_model=None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Compare the DML causal estimate to a naive (confounded) estimate.

        This is the key diagnostic for pricing actuaries: how much does
        confounding inflate or deflate the naive estimate of the treatment
        effect? The naive estimate is either supplied directly or extracted
        from a fitted GLM/GBM.

        Parameters
        ----------
        naive_coefficient : float | None
            The naive estimate of the treatment effect from a standard model
            (e.g. the coefficient on price_change from a GLM). If None,
            glm_model must be provided.
        glm_model : fitted model | None
            A fitted sklearn-compatible model with .coef_ or a glum/statsmodels
            model with .params. The treatment coefficient is extracted
            automatically if possible. Currently supports:
            - sklearn linear models (LogisticRegression, LinearRegression)
            - glum / statsmodels GeneralisedLinearModel (via .params)
            If extraction fails, pass naive_coefficient directly.
        alpha : float
            Significance level for the confidence interval. Default: 0.05 (95% CI).

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame with columns:
            treatment, outcome, naive_estimate, causal_estimate, bias,
            bias_pct, ci_lower, ci_upper, p_value, interpretation.

        Notes
        -----
        "Confounding bias" is defined as naive_estimate - causal_estimate.
        A positive bias means the naive model overstates the true treatment
        effect (e.g. price sensitivity appears higher than it really is).
        A negative bias means the naive model understates the effect.

        The interpretation column gives a plain-English summary of what the
        bias implies for pricing decisions.
        """
        self._check_fitted()

        ate = self.average_treatment_effect()
        causal = ate.estimate

        # --- Extract naive coefficient --------------------------------------
        naive = None
        if naive_coefficient is not None:
            naive = float(naive_coefficient)
        elif glm_model is not None:
            naive = self._extract_naive_from_model(glm_model)
        else:
            raise ValueError(
                "Provide either naive_coefficient or glm_model."
            )

        # --- Compute bias --------------------------------------------------
        bias = naive - causal
        bias_pct = (bias / abs(causal) * 100) if causal != 0 else float("nan")

        # --- Interpretation ------------------------------------------------
        interpretation = self._interpret_bias(naive, causal, bias, bias_pct)

        return pd.DataFrame([{
            "treatment": self.treatment.column,
            "outcome": self.outcome,
            "naive_estimate": round(naive, 6),
            "causal_estimate": round(causal, 6),
            "bias": round(bias, 6),
            "bias_pct": round(bias_pct, 1),
            "causal_ci_lower": round(ate.ci_lower, 6),
            "causal_ci_upper": round(ate.ci_upper, 6),
            "causal_p_value": round(ate.p_value, 4),
            "n_obs": ate.n_obs,
            "interpretation": interpretation,
        }])

    def _extract_naive_from_model(self, glm_model) -> float:
        """
        Extract the treatment coefficient from a fitted model object.

        Tries multiple attribute patterns used by sklearn, glum, and statsmodels.
        """
        treatment_col = self.treatment.column

        # statsmodels / glum: .params is a Series indexed by feature name
        if hasattr(glm_model, "params"):
            params = glm_model.params
            if hasattr(params, "__getitem__") and treatment_col in params:
                return float(params[treatment_col])
            raise ValueError(
                f"Could not find '{treatment_col}' in model.params. "
                f"Available: {list(params.index) if hasattr(params, 'index') else 'unknown'}."
            )

        # sklearn linear models: .coef_ is an array, need .feature_names_in_
        if hasattr(glm_model, "coef_"):
            if hasattr(glm_model, "feature_names_in_"):
                feature_names = list(glm_model.feature_names_in_)
                if treatment_col in feature_names:
                    idx = feature_names.index(treatment_col)
                    coef = glm_model.coef_
                    if coef.ndim > 1:
                        coef = coef[0]
                    return float(coef[idx])
            raise ValueError(
                f"sklearn model has .coef_ but no .feature_names_in_. "
                f"Fit the model on a DataFrame (not a numpy array) to preserve "
                f"feature names, or pass naive_coefficient directly."
            )

        raise ValueError(
            "Cannot extract coefficient from model. "
            "Pass the naive_coefficient parameter directly."
        )

    @staticmethod
    def _interpret_bias(
        naive: float,
        causal: float,
        bias: float,
        bias_pct: float,
    ) -> str:
        """Generate a plain-English interpretation of confounding bias."""
        if abs(bias_pct) < 10:
            return (
                f"Confounding bias is small ({bias_pct:.1f}%). "
                "The naive estimate is close to the causal estimate."
            )
        direction = "overstates" if bias > 0 else "understates"
        magnitude = "substantially" if abs(bias_pct) > 50 else "moderately"
        sign_agreement = "both" if (naive * causal > 0) else "opposite signs —"
        if naive * causal <= 0:
            return (
                f"The naive estimate and causal estimate have opposite signs. "
                f"Confounding completely reverses the apparent direction of the effect. "
                f"Naive: {naive:.4f}, Causal: {causal:.4f}. "
                "Do not rely on the naive estimate for pricing decisions."
            )
        return (
            f"The naive estimate {direction} the true causal effect by "
            f"{magnitude} ({bias_pct:.1f}%). "
            f"Pricing decisions based on the naive estimate are {magnitude} biased."
        )

    # ------------------------------------------------------------------
    # Model access
    # ------------------------------------------------------------------

    @property
    def dml_model(self):
        """The underlying DoubleMLPLR object, for advanced use."""
        self._check_fitted()
        return self._dml_model

    def treatment_overlap_stats(self) -> dict:
        """
        Summary statistics on the (transformed) treatment distribution.

        Low treatment variance — which occurs when price is nearly deterministic
        given X — makes the DML estimate noisy. If SD is very small, the
        model is identifying off very little exogenous variation.
        """
        self._check_fitted()
        return self._treatment_overlap

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return (
            f"CausalPricingModel("
            f"outcome='{self.outcome}', "
            f"treatment='{self.treatment.column}', "
            f"confounders={self.confounders}, "
            f"status={status})"
        )
