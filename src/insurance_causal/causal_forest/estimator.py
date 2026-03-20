"""
Heterogeneous treatment effect estimator for insurance pricing.

HeterogeneousElasticityEstimator wraps EconML's CausalForestDML with
insurance-specific defaults:
- CatBoost nuisance models (handles UK insurance categoricals natively)
- min_samples_leaf=20 (larger than default; sparse rating segments need
  more data per leaf to avoid overfit in small groups)
- honest=True (Athey & Imbens 2016: sample splitting for valid inference)
- Exposure weighting support for Poisson/rate claim outcomes
- Accepts polars or pandas DataFrames

The key difference from RenewalElasticityEstimator in the elasticity
subpackage: this estimator focuses on HTE inference (BLP, GATES, CLAN,
RATE), not on pricing optimisation. The methods here are the building
blocks for the HeterogeneousInference and TargetingEvaluator classes.

Design decision on n_estimators:
    CausalForestDML requires n_estimators % (n_folds * 2) == 0 due to
    honest splitting — the training sample is split in two, and each
    fold-sub-half pair gets its own tree subsample. We auto-round up
    rather than raising an error, emitting a warning. The round-up is
    at most (n_folds * 2 - 1) trees, which is negligible for n_estimators
    in the hundreds.

References
----------
Athey, Tibshirani & Wager (2019). "Generalized Random Forests."
    Annals of Statistics 47(2): 1148-1178.
Chernozhukov et al. (2018). "Double/Debiased Machine Learning."
    Econometrics Journal 21(1): C1-C68.
Wager & Athey (2018). "Estimation and Inference of Heterogeneous Treatment
    Effects using Random Forests." JASA 113(523): 1228-1242.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence, Union

import numpy as np
import polars as pl

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


DataFrameLike = Union[pl.DataFrame, "pd.DataFrame"]


class HeterogeneousElasticityEstimator:
    """Causal forest estimator for heterogeneous price elasticity.

    Wraps CausalForestDML with insurance-specific defaults and exposes
    methods for producing CATE estimates, ATE with confidence intervals,
    and group average treatment effects (GATEs) by any categorical variable.

    This class is the foundation for :class:`~insurance_causal.causal_forest.inference.HeterogeneousInference`
    and :class:`~insurance_causal.causal_forest.targeting.TargetingEvaluator`. You can also use it
    standalone if you only need CATE estimates without the full inference machinery.

    Parameters
    ----------
    binary_outcome:
        Whether the outcome is binary (0/1 renewal indicator). When True,
        CatBoostClassifier is used as the outcome nuisance model.
    n_folds:
        Number of cross-fitting folds. 5 is the standard (Chernozhukov 2018).
        CausalForestDML requires n_estimators divisible by n_folds * 2.
    n_estimators:
        Trees in the causal forest. Will be rounded up to the nearest
        multiple of n_folds * 2 if needed.
    min_samples_leaf:
        Minimum observations per leaf. Default 20 (larger than econml's
        default of 5). Insurance rating segments can be sparse; leaves with
        < 20 observations produce unreliable CATE estimates.
    catboost_iterations:
        Training iterations for CatBoost nuisance models.
    random_state:
        Random seed.
    exposure_col:
        If provided, exposure-weight the fit (for Poisson/rate outcomes).
        The outcome column will be divided by this exposure column, and
        exposure values used as sample weights.

    Examples
    --------
    >>> from insurance_causal.causal_forest.data import make_hte_renewal_data
    >>> from insurance_causal.causal_forest.estimator import HeterogeneousElasticityEstimator
    >>> df = make_hte_renewal_data(n=5000, seed=42)
    >>> confounders = ["age", "ncd_years", "vehicle_group", "channel"]
    >>> est = HeterogeneousElasticityEstimator(n_estimators=100, catboost_iterations=100)
    >>> est.fit(df, outcome="renewed", treatment="log_price_change",
    ...         confounders=confounders)
    >>> ate, lb, ub = est.ate()
    >>> cates = est.cate(df)
    >>> gates = est.gate(df, by="ncd_years")
    """

    def __init__(
        self,
        binary_outcome: bool = True,
        n_folds: int = 5,
        n_estimators: int = 200,
        min_samples_leaf: int = 20,
        catboost_iterations: int = 500,
        random_state: int = 42,
        exposure_col: Optional[str] = None,
    ) -> None:
        self.binary_outcome = binary_outcome
        self.n_folds = n_folds
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.catboost_iterations = catboost_iterations
        self.random_state = random_state
        self.exposure_col = exposure_col

        self._estimator: Optional[object] = None
        self._feature_names: list[str] = []
        self._outcome_col: str = ""
        self._treatment_col: str = ""
        self._confounders: list[str] = []
        self._X_train: Optional[np.ndarray] = None
        self._Y_train: Optional[np.ndarray] = None
        self._D_train: Optional[np.ndarray] = None
        self._e_hat: Optional[np.ndarray] = None  # propensity/treatment residuals
        self._is_fitted: bool = False

    def fit(
        self,
        df: DataFrameLike,
        outcome: str = "renewed",
        treatment: str = "log_price_change",
        confounders: Optional[Sequence[str]] = None,
    ) -> "HeterogeneousElasticityEstimator":
        """Fit the causal forest estimator.

        Parameters
        ----------
        df:
            Renewal dataset. Accepts polars or pandas DataFrames.
        outcome:
            Column name for the outcome variable Y.
        treatment:
            Column name for the treatment variable D (typically
            log_price_change).
        confounders:
            Observable risk factors that jointly determine price and
            outcome. Must be provided.

        Returns
        -------
        self
        """
        if confounders is None:
            raise ValueError(
                "confounders must be provided. Pass the list of risk factor "
                "column names that confound both treatment and outcome."
            )

        self._outcome_col = outcome
        self._treatment_col = treatment
        self._confounders = list(confounders)

        df_pd = _to_pandas(df)
        Y, D, X, feature_names, sample_weight = _extract_arrays(
            df_pd, outcome, treatment, self._confounders,
            exposure_col=self.exposure_col,
        )
        self._feature_names = feature_names
        self._X_train = X
        self._Y_train = Y
        self._D_train = D

        model_y = self._build_outcome_model()
        model_t = self._build_treatment_model()
        estimator = self._build_estimator(model_y, model_t)

        fit_kwargs: dict = {"X": X}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        estimator.fit(Y, D, **fit_kwargs)
        self._estimator = estimator
        self._is_fitted = True

        # Cache treatment residuals (used by inference and diagnostics)
        try:
            self._e_hat = estimator.nuisance_scores_t
        except AttributeError:
            self._e_hat = None

        return self

    def cate(self, df: DataFrameLike) -> np.ndarray:
        """Return per-row CATE estimates.

        Parameters
        ----------
        df:
            Dataset to predict on. Must contain the confounder columns
            used during fitting.

        Returns
        -------
        numpy.ndarray of shape (n,)
            Individual-level estimated treatment effects (semi-elasticities).
        """
        self._check_fitted()
        df_pd = _to_pandas(df)
        _, _, X, _, _ = _extract_arrays(
            df_pd, self._outcome_col, self._treatment_col, self._confounders,
            exposure_col=None,
        )
        return self._estimator.effect(X).flatten()

    def cate_interval(
        self,
        df: DataFrameLike,
        alpha: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return per-row CATE confidence intervals.

        Parameters
        ----------
        df:
            Dataset to predict on.
        alpha:
            Significance level. 0.05 gives 95% intervals.

        Returns
        -------
        tuple of (lower_bounds, upper_bounds), each shape (n,)
        """
        self._check_fitted()
        df_pd = _to_pandas(df)
        _, _, X, _, _ = _extract_arrays(
            df_pd, self._outcome_col, self._treatment_col, self._confounders,
            exposure_col=None,
        )
        lb, ub = self._estimator.effect_interval(X, alpha=alpha)
        return lb.flatten(), ub.flatten()

    def ate(self) -> tuple[float, float, float]:
        """Return the average treatment effect with 95% confidence interval.

        Returns
        -------
        tuple of (ate, lower_bound, upper_bound)
        """
        self._check_fitted()
        X = self._X_train
        try:
            result = self._estimator.ate_interval(X=X, alpha=0.05)
        except TypeError:
            result = self._estimator.ate_interval(alpha=0.05)
        ate_point = float(np.mean(self._estimator.effect(X)))
        lb = float(result[0])
        ub = float(result[1])
        return ate_point, lb, ub

    def gate(
        self,
        df: DataFrameLike,
        by: str,
    ) -> pl.DataFrame:
        """Return Group Average Treatment Effects (GATEs) by a categorical variable.

        Parameters
        ----------
        df:
            Dataset. Must contain ``by`` and the confounder columns.
        by:
            Column name to group by.

        Returns
        -------
        polars.DataFrame with columns: ``by``, ``cate``, ``ci_lower``,
        ``ci_upper``, ``n``, with a warning if any group has < 500 observations.
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
                pl.col("_cate").mean().alias("cate"),
                pl.col("_ci_lower").mean().alias("ci_lower"),
                pl.col("_ci_upper").mean().alias("ci_upper"),
                pl.len().alias("n"),
            ])
            .sort(by)
        )

        # Warn on small groups
        small_groups = result.filter(pl.col("n") < 500)
        if len(small_groups) > 0:
            group_sizes = small_groups.select([by, "n"]).to_dict(as_series=False)
            warnings.warn(
                f"GATE groups with < 500 policies: {group_sizes}. "
                "CATE estimates in small segments are noisy and confidence "
                "intervals will be wide. Consider merging small groups.",
                UserWarning,
                stacklevel=2,
            )

        return result

    # ---------------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------------

    def _build_outcome_model(self) -> object:
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
                "CatBoost not installed. Falling back to GradientBoosting.",
                stacklevel=3,
            )
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
            if self.binary_outcome:
                return GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)
            else:
                return GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)

    def _build_treatment_model(self) -> object:
        try:
            from catboost import CatBoostRegressor
            return CatBoostRegressor(
                iterations=self.catboost_iterations,
                verbose=0,
                random_seed=self.random_state,
            )
        except ImportError:
            warnings.warn(
                "CatBoost not installed. Falling back to GradientBoostingRegressor.",
                stacklevel=3,
            )
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)

    def _build_estimator(self, model_y: object, model_t: object) -> object:
        try:
            from econml.dml import CausalForestDML
        except ImportError as e:
            raise ImportError(
                "EconML is required. Install with: pip install econml"
            ) from e

        # n_estimators must be divisible by n_folds * 2 (honest splitting)
        n_est = self.n_estimators
        divisor = self.n_folds * 2
        if n_est % divisor != 0:
            n_est = ((n_est // divisor) + 1) * divisor
            warnings.warn(
                f"n_estimators={self.n_estimators} is not divisible by "
                f"n_folds*2={divisor}. Rounding up to {n_est}.",
                UserWarning,
                stacklevel=3,
            )

        return CausalForestDML(
            model_y=model_y,
            model_t=model_t,
            discrete_outcome=self.binary_outcome,
            n_estimators=n_est,
            min_samples_leaf=self.min_samples_leaf,
            honest=True,
            cv=self.n_folds,
            random_state=self.random_state,
        )

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Estimator is not fitted. Call .fit() first."
            )


# ---------------------------------------------------------------------------
# Module-level helpers (shared across causal_forest subpackage)
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
    exposure_col: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], Optional[np.ndarray]]:
    """Extract Y, D, X, feature_names, sample_weight from a pandas DataFrame.

    Categorical/string columns are one-hot encoded. If exposure_col is
    provided, Y is divided by exposure and the exposure array is returned
    as sample_weight.

    Returns
    -------
    Y, D, X, feature_names, sample_weight
    """
    import pandas as pd_mod

    Y = df_pd[outcome].values.astype(float)
    D = df_pd[treatment].values.astype(float)

    # Handle exposure weighting
    sample_weight: Optional[np.ndarray] = None
    if exposure_col is not None:
        exposure = df_pd[exposure_col].values.astype(float)
        if np.any(exposure <= 0):
            raise ValueError("All exposure values must be strictly positive.")
        Y = Y / exposure
        sample_weight = exposure

    # Encode features
    subset = df_pd[confounders].copy()
    obj_cols = subset.select_dtypes(include=["object", "category"]).columns.tolist()
    if obj_cols:
        subset = pd_mod.get_dummies(subset, columns=obj_cols, drop_first=True)

    # Fill NaNs defensively
    subset = subset.fillna(subset.mean())

    X = subset.values.astype(float)
    feature_names = list(subset.columns)

    return Y, D, X, feature_names, sample_weight
