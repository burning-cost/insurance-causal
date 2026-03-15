"""
DML-based renewal price elasticity estimation.

This module wraps EconML's CausalForestDML and DRLearner estimators with
insurance-specific defaults: CatBoost nuisance models (which handle categorical
features natively), binary outcome support, and log-log specification.

The estimator follows sklearn's fit/predict API pattern. After fitting, you can
retrieve the average treatment effect (ATE), per-row CATE, and group average
treatment effects (GATEs) by any categorical variable.

Why CausalForestDML by default:
    It is fully non-parametric, requires no pre-specified feature interactions,
    and provides valid pointwise confidence intervals via honest splitting.
    For a portfolio-level average, it is overkill — PLR with CatBoost is faster.
    But for the elasticity surface (the actual deliverable for pricing teams),
    CausalForestDML is the right tool.

Why CatBoost for nuisance models:
    UK motor and home insurance data contains many categorical features (region,
    vehicle group, occupation, payment method). CatBoost handles these natively
    without manual encoding. It also provides calibrated probability outputs for
    the binary outcome nuisance model, which matters when computing Pearson-style
    residuals.

Log-log specification:
    We set treatment = log(offer_price / last_year_price). With binary outcome Y
    and log treatment D, the DML coefficient theta is the semi-elasticity:
    a 1-unit change in D (i.e., a 100% price increase) changes renewal probability
    by theta percentage points. For typical price changes of 5–20%, interpret as
    approximately: a 10% price increase changes renewal probability by theta/10
    percentage points. This is the standard interpretation used in the UK pricing
    literature.

econml version notes:
    LinearDML does not accept ``discrete_outcome`` in all versions. We do not
    pass it for LinearDML to maintain compatibility. CausalForestDML requires
    ``n_estimators`` to be divisible by ``n_folds * 2`` (honest splitting creates
    two sub-samples per fold). The ``ate_interval()`` method requires X to be
    passed explicitly in econml>=0.15.

References
----------
Chernozhukov et al. (2018), Econometrics Journal 21(1).
Athey & Wager (2019), Annals of Statistics 47(2).
Kennedy (2023), Electronic Journal of Statistics 17(2).
KB entries 593, 594, 600.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union
import warnings

import numpy as np
import polars as pl

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
DataFrameLike = Union[pl.DataFrame, "pd.DataFrame"]


class RenewalElasticityEstimator:
    """Double Machine Learning estimator for insurance renewal price elasticity.

    Wraps EconML's CausalForestDML (default) or DRLearner with CatBoost
    nuisance models and insurance-specific defaults.

    Parameters
    ----------
    cate_model:
        CATE estimator backend. ``"causal_forest"`` (default) uses
        :class:`econml.dml.CausalForestDML`. ``"dr_learner"`` uses
        :class:`econml.dr.DRLearner`. ``"linear_dml"`` uses
        :class:`econml.dml.LinearDML` (fastest; assumes constant treatment
        effect).
    outcome_model:
        Nuisance model for E[Y|X]. ``"catboost"`` (default) uses
        CatBoostClassifier for binary outcomes and CatBoostRegressor otherwise.
        Pass any sklearn-compatible estimator to override.
    treatment_model:
        Nuisance model for E[D|X]. ``"catboost"`` (default) uses
        CatBoostRegressor. Pass any sklearn-compatible estimator to override.
    binary_outcome:
        Whether the outcome is binary (0/1 renewal indicator). When True,
        CatBoostClassifier is used as the outcome nuisance model and
        ``discrete_outcome=True`` is passed to CausalForestDML.
    n_folds:
        Number of cross-fitting folds. 5 is the default (Chernozhukov 2018
        recommends at least 3). For CausalForestDML, ``n_estimators`` must
        be divisible by ``n_folds * 2``.
    n_estimators:
        Number of trees in the CausalForestDML. Larger values give more
        accurate CATE estimates at higher compute cost. Must be divisible
        by ``n_folds * 2`` for CausalForestDML.
    catboost_iterations:
        Training iterations for CatBoost nuisance models. 500 is a good
        default for datasets of 10k–500k rows.
    random_state:
        Random seed.

    Examples
    --------
    >>> from insurance_causal.elasticity.data import make_renewal_data
    >>> from insurance_causal.elasticity.fit import RenewalElasticityEstimator
    >>> df = make_renewal_data(n=5000)
    >>> confounders = ["age", "ncd_years", "vehicle_group", "region", "channel"]
    >>> est = RenewalElasticityEstimator(n_estimators=100, catboost_iterations=100)
    >>> est.fit(df, outcome="renewed", treatment="log_price_change",
    ...         confounders=confounders)
    >>> ate, lb, ub = est.ate()
    >>> print(f"ATE: {ate:.3f}  95% CI: [{lb:.3f}, {ub:.3f}]")
    """

    def __init__(
        self,
        cate_model: str = "causal_forest",
        outcome_model: Union[str, object] = "catboost",
        treatment_model: Union[str, object] = "catboost",
        binary_outcome: bool = True,
        n_folds: int = 5,
        n_estimators: int = 200,
        catboost_iterations: int = 500,
        random_state: int = 42,
    ) -> None:
        self.cate_model = cate_model
        self.outcome_model = outcome_model
        self.treatment_model = treatment_model
        self.binary_outcome = binary_outcome
        self.n_folds = n_folds
        self.n_estimators = n_estimators
        self.catboost_iterations = catboost_iterations
        self.random_state = random_state

        self._estimator: Optional[object] = None
        self._feature_names: list[str] = []
        self._outcome_col: str = ""
        self._treatment_col: str = ""
        self._confounders: list[str] = []
        self._X_train: Optional[np.ndarray] = None  # stored for ate_interval()
        self._is_fitted: bool = False

    def fit(
        self,
        df: DataFrameLike,
        outcome: str = "renewed",
        treatment: str = "log_price_change",
        confounders: Optional[Sequence[str]] = None,
    ) -> "RenewalElasticityEstimator":
        """Fit the DML elasticity estimator.

        Parameters
        ----------
        df:
            Renewal dataset. Accepts polars or pandas DataFrames.
        outcome:
            Column name of the binary renewal indicator (Y).
        treatment:
            Column name of the treatment variable, typically
            log(offer_price / last_year_price) (D).
        confounders:
            Column names to control for (X). These are the observable risk
            factors that jointly determine price and renewal probability.
            If None, raises ValueError.

        Returns
        -------
        self
        """
        if confounders is None:
            raise ValueError(
                "confounders must be provided. Pass the list of risk factor "
                "column names that determine both price and renewal probability "
                "(e.g. age, ncd_years, region, vehicle_group, channel)."
            )

        self._outcome_col = outcome
        self._treatment_col = treatment
        self._confounders = list(confounders)

        df_pd = _to_pandas(df)
        Y, D, X, feature_names = _extract_arrays(df_pd, outcome, treatment, self._confounders)
        self._feature_names = feature_names
        self._X_train = X  # store for use in ate_interval()

        model_y = self._build_outcome_model()
        model_t = self._build_treatment_model()
        estimator = self._build_estimator(model_y, model_t)

        estimator.fit(Y, D, X=X)
        self._estimator = estimator
        self._is_fitted = True
        return self

    def ate(self) -> tuple[float, float, float]:
        """Return the average treatment effect with 95% confidence interval.

        The ATE is the portfolio-average semi-elasticity: the expected change in
        renewal probability for a 1-unit increase in log price change.

        For typical log price changes in [0.0, 0.20], a 10% increase
        (log_price_change = 0.095) changes renewal probability by approximately
        ATE * 0.095 percentage points.

        Returns
        -------
        tuple of (ate, lower_bound, upper_bound)
            ATE point estimate and 95% confidence interval bounds.
        """
        self._check_fitted()
        X = self._X_train
        # Use effect(X).mean() as point estimate — works for all estimator types
        # (CausalForestDML.ate_() only works for discrete treatments; LinearDML.ate_
        # does not exist in all versions). ate_interval() provides the CI.
        try:
            result = self._estimator.ate_interval(X=X, alpha=0.05)
        except TypeError:
            # Older econml versions may not require X
            result = self._estimator.ate_interval(alpha=0.05)
        ate_point = float(np.mean(self._estimator.effect(X)))
        lb = float(result[0])
        ub = float(result[1])
        return ate_point, lb, ub

    def cate(self, df: DataFrameLike) -> np.ndarray:
        """Return per-row CATE estimates (individual-level elasticity).

        Each value is the estimated semi-elasticity for that customer: the
        expected change in renewal probability per unit change in log price.
        Customers with high price sensitivity (large negative values) should
        receive smaller price increases or targeted discounts.

        Parameters
        ----------
        df:
            Dataset to predict on. Must contain the confounder columns used
            during fitting.

        Returns
        -------
        numpy.ndarray of shape (n,)
        """
        self._check_fitted()
        df_pd = _to_pandas(df)
        _, _, X, _ = _extract_arrays(df_pd, self._outcome_col, self._treatment_col, self._confounders)
        return self._estimator.effect(X).flatten()

    def cate_interval(self, df: DataFrameLike, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
        """Return per-row CATE confidence intervals.

        Parameters
        ----------
        df:
            Dataset to predict on.
        alpha:
            Significance level. Default 0.05 gives 95% intervals.

        Returns
        -------
        tuple of (lower_bounds, upper_bounds), each of shape (n,)
        """
        self._check_fitted()
        df_pd = _to_pandas(df)
        _, _, X, _ = _extract_arrays(df_pd, self._outcome_col, self._treatment_col, self._confounders)

        if not hasattr(self._estimator, "effect_interval"):
            warnings.warn(
                "This estimator does not support per-row confidence intervals. "
                "Returning ATE interval for all rows.",
                stacklevel=2,
            )
            ate, lb, ub = self.ate()
            n = len(df_pd)
            return np.full(n, lb), np.full(n, ub)

        lb, ub = self._estimator.effect_interval(X, alpha=alpha)
        return lb.flatten(), ub.flatten()

    def gate(
        self,
        df: DataFrameLike,
        by: str,
    ) -> pl.DataFrame:
        """Return Group Average Treatment Effects (GATEs) by a categorical variable.

        GATEs answer: "what is the average elasticity for NCD band 5 customers?"
        This is the primary output for pricing actuaries building segment-level
        discount strategies.

        Parameters
        ----------
        df:
            Dataset. Must contain ``by`` and the confounder columns.
        by:
            Column name to group by (e.g. ``"ncd_years"``, ``"age_band"``,
            ``"channel"``).

        Returns
        -------
        polars.DataFrame with columns: ``by``, ``elasticity``,
        ``ci_lower``, ``ci_upper``, ``n``
        """
        self._check_fitted()
        df_pl = _to_polars(df)
        cate_vals = self.cate(df)
        lb_vals, ub_vals = self.cate_interval(df)

        df_with_cate = df_pl.with_columns([
            pl.Series("_cate", cate_vals),
            pl.Series("_ci_lower", lb_vals),
            pl.Series("_ci_upper", ub_vals),
        ])

        result = (
            df_with_cate
            .group_by(by)
            .agg([
                pl.col("_cate").mean().alias("elasticity"),
                pl.col("_ci_lower").mean().alias("ci_lower"),
                pl.col("_ci_upper").mean().alias("ci_upper"),
                pl.len().alias("n"),
            ])
            .sort(by)
        )
        return result

    # ---------------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------------

    def _build_outcome_model(self) -> object:
        """Instantiate the outcome nuisance model."""
        if isinstance(self.outcome_model, str) and self.outcome_model == "catboost":
            try:
                if self.binary_outcome:
                    from catboost import CatBoostClassifier
                    return CatBoostClassifier(
                        iterations=self.catboost_iterations,
                        verbose=0,
                        random_seed=self.random_state,
                        eval_metric="Logloss",
                    )
                else:
                    from catboost import CatBoostRegressor
                    return CatBoostRegressor(
                        iterations=self.catboost_iterations,
                        verbose=0,
                        random_seed=self.random_state,
                    )
            except ImportError:
                warnings.warn(
                    "CatBoost is not installed. Falling back to GradientBoostingClassifier/Regressor. "
                    "Install catboost for better performance with categorical features.",
                    stacklevel=3,
                )
                from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
                if self.binary_outcome:
                    return GradientBoostingClassifier(
                        n_estimators=100, random_state=self.random_state
                    )
                else:
                    return GradientBoostingRegressor(
                        n_estimators=100, random_state=self.random_state
                    )
        return self.outcome_model

    def _build_treatment_model(self) -> object:
        """Instantiate the treatment nuisance model."""
        if isinstance(self.treatment_model, str) and self.treatment_model == "catboost":
            try:
                from catboost import CatBoostRegressor
                return CatBoostRegressor(
                    iterations=self.catboost_iterations,
                    verbose=0,
                    random_seed=self.random_state,
                )
            except ImportError:
                warnings.warn(
                    "CatBoost is not installed. Falling back to GradientBoostingRegressor.",
                    stacklevel=3,
                )
                from sklearn.ensemble import GradientBoostingRegressor
                return GradientBoostingRegressor(
                    n_estimators=100, random_state=self.random_state
                )
        return self.treatment_model

    def _build_estimator(self, model_y: object, model_t: object) -> object:
        """Instantiate the CATE estimator."""
        try:
            from econml.dml import CausalForestDML, LinearDML
            from econml.dr import DRLearner
        except ImportError as e:
            raise ImportError(
                "EconML is required for fitting the elasticity estimator. "
                "Install it with: pip install econml"
            ) from e

        if self.cate_model == "causal_forest":
            # n_estimators must be divisible by n_folds * 2 (honest splitting)
            n_est = self.n_estimators
            divisor = self.n_folds * 2
            if n_est % divisor != 0:
                n_est = ((n_est // divisor) + 1) * divisor
                warnings.warn(
                    f"n_estimators={self.n_estimators} is not divisible by n_folds*2={divisor}. "
                    f"Rounding up to {n_est}.",
                    UserWarning,
                    stacklevel=3,
                )
            return CausalForestDML(
                model_y=model_y,
                model_t=model_t,
                discrete_outcome=self.binary_outcome,
                n_estimators=n_est,
                cv=self.n_folds,
                random_state=self.random_state,
            )
        elif self.cate_model == "linear_dml":
            # LinearDML does not support discrete_outcome in all econml versions
            return LinearDML(
                model_y=model_y,
                model_t=model_t,
                cv=self.n_folds,
                random_state=self.random_state,
            )
        elif self.cate_model == "dr_learner":
            if not self.binary_outcome:
                raise ValueError(
                    "DRLearner requires a binary treatment (binary_outcome=True). "
                    "DRLearner estimates the ATE for binary treatments, not continuous ones. "
                    "For continuous treatment use 'causal_forest' or 'linear_dml' instead."
                )
            return DRLearner(
                model_regression=model_y,
                model_propensity=model_t,
                cv=self.n_folds,
                random_state=self.random_state,
            )
        else:
            raise ValueError(
                f"Unknown cate_model '{self.cate_model}'. "
                "Choose from: 'causal_forest', 'linear_dml', 'dr_learner'."
            )

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Estimator is not fitted. Call .fit() first."
            )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _to_pandas(df: DataFrameLike) -> "pd.DataFrame":
    """Convert polars DataFrame to pandas, pass pandas through."""
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    return df


def _to_polars(df: DataFrameLike) -> pl.DataFrame:
    """Convert pandas DataFrame to polars, pass polars through."""
    if isinstance(df, pl.DataFrame):
        return df
    return pl.from_pandas(df)


def _extract_arrays(
    df_pd: "pd.DataFrame",
    outcome: str,
    treatment: str,
    confounders: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Extract Y, D, X arrays from a pandas DataFrame.

    Categorical/string columns in confounders are one-hot encoded.

    Returns
    -------
    Y, D, X, feature_names
    """
    import pandas as pd

    Y = df_pd[outcome].values.astype(float)
    D = df_pd[treatment].values.astype(float)

    subset = df_pd[confounders].copy()
    obj_cols = subset.select_dtypes(include=["object", "category"]).columns.tolist()
    if obj_cols:
        subset = pd.get_dummies(subset, columns=obj_cols, drop_first=True)

    # Fill any NaNs with column mean (defensive)
    subset = subset.fillna(subset.mean())

    X = subset.values.astype(float)
    feature_names = list(subset.columns)

    return Y, D, X, feature_names
