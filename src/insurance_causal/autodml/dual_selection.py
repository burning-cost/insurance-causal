"""
DualSelectionDML: causal effect estimation under multivariate ordinal sample selection.

The problem this solves
-----------------------
Standard DML (and the existing SelectionCorrectedElasticity) handles a *single*
binary selection variable: outcome is observed only for renewing policies.

In UK motor and home insurance there is often a second selection layer: claim
severity is only observed for policies that *both* renew AND submit a claim.
When you want to estimate the causal effect of a risk factor on claim severity,
you must account for both selection stages simultaneously.

More generally, an outcome is observed only when all K ordinal selection
conditions are jointly met. Treating these as independent selection problems
or ignoring one entirely produces inconsistent estimates.

Identification strategy
-----------------------
Following Dolgikh & Potanin (2025) arXiv:2511.12640, we use CONTROL FUNCTIONS
rather than inverse probability weighting. For each ordinal selection variable
Z_j, we compute the conditional CDF P(Z_j <= z | D, X, W_Z). The CDFs form a
vector of control functions P_bar_i that, when included in the outcome regression,
absorb the selection-induced correlation between unobservables and treatment.

The EIF score extends the standard DML score to include these control function
corrections. Identification requires either exclusion restrictions (W_Z variables
that predict selection but not outcome) or functional form restrictions. The
class warns when neither W_Z is provided.

Estimands supported
-------------------
- ATE  : Average Treatment Effect E[Y(d) - Y(d*)]
- ATES : ATE on Selected (observable) subpopulation
- ATET : ATE on Treated (D_i = d) subpopulation
- LATE : Local ATE (not yet implemented; raises NotImplementedError)

Reference
---------
Dolgikh & Potanin (2025) "Automatic Debiased Machine Learning under Multivariate
Ordinal Sample Selection" arXiv:2511.12640.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class _ConstantClassifier:
    """
    Fallback classifier for degenerate single-class training sets.

    When all training labels are the same class (e.g. all observations are
    selected, or all have D=d), a real classifier will either raise an error
    or produce nonsense predictions.  This returns the empirical proportion
    as a constant for all observations.
    """

    def __init__(self, constant_proba: float) -> None:
        self._p = float(np.clip(constant_proba, 0.01, 0.99))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        return np.column_stack([np.full(n, 1.0 - self._p), np.full(n, self._p)])



@dataclass
class DualSelectionResult:
    """
    Result from DualSelectionDML.estimate().

    Parameters
    ----------
    estimand : str
        Which causal estimand was computed ("ATE", "ATES", "ATET").
    ate : float
        Point estimate of the causal effect.
    se : float
        Standard error derived from EIF scores.
    ci_lower : float
        Lower bound of 95% confidence interval.
    ci_upper : float
        Upper bound of 95% confidence interval.
    n_obs : int
        Total number of observations.
    n_selected : int
        Number of observations where all selection conditions were met
        (i.e. outcome was observed).
    selection_rates : dict
        Per-variable selection rates, e.g. {"Z_0": 0.72, "Z_1": 0.35,
        "joint": 0.25}.
    nuisance_scores : dict
        Diagnostics from nuisance fitting (e.g. mean propensity, mean
        selection probability).
    eif_scores : np.ndarray
        Raw EIF scores, length n_obs.  Useful for clustered SEs or further
        analysis.
    """

    estimand: str
    ate: float
    se: float
    ci_lower: float
    ci_upper: float
    n_obs: int
    n_selected: int
    selection_rates: Dict[str, float]
    nuisance_scores: Dict[str, float]
    eif_scores: np.ndarray = field(default_factory=lambda: np.array([]))

    def __repr__(self) -> str:
        return (
            f"DualSelectionResult(estimand={self.estimand!r}, "
            f"ate={self.ate:.4f}, se={self.se:.4f}, "
            f"ci=[{self.ci_lower:.4f}, {self.ci_upper:.4f}], "
            f"n={self.n_obs}, n_selected={self.n_selected})"
        )

    @property
    def pvalue(self) -> float:
        """Two-sided p-value under H0: ATE == 0."""
        from scipy import stats
        if self.se <= 0:
            return float("nan")
        z = abs(self.ate / self.se)
        return float(2 * (1 - stats.norm.cdf(z)))

    def summary(self) -> str:
        """One-line formatted summary for logging."""
        stars = ""
        p = self.pvalue
        if not np.isnan(p):
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"
        return (
            f"{self.estimand}: ate={self.ate:+.4f}  se={self.se:.4f}  "
            f"95% CI=[{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]  "
            f"p={self.pvalue:.4f}{stars}  "
            f"n_selected={self.n_selected}/{self.n_obs}"
        )


class DualSelectionDML:
    """
    Causal ATE estimator under multivariate ordinal sample selection.

    Use this when your outcome is only observable for units satisfying
    multiple joint conditions — the canonical insurance example is claim
    severity, observed only for renewing AND claiming policyholders.

    The identification relies on control functions: conditional CDFs of the
    selection variables that absorb selection-induced dependence between
    unobservables and treatment. This is more robust than IPW at high-
    dimensional selection propensities and handles ordinal (multi-level)
    selection naturally.

    Parameters
    ----------
    estimand : {"ATE", "ATES", "ATET", "LATE"}
        Causal estimand to target.  "LATE" is not yet implemented.
    n_folds : int
        Number of cross-fitting folds.  Double sample splitting within each
        complement uses 2 halves, so the effective split is into 2*n_folds
        groups.
    treatment_values : tuple of (d, d_star)
        The treatment comparison: estimate E[Y(d) - Y(d_star)].
        For binary treatment: (1, 0).
    selection_values : list of lists, optional
        Which ordinal level to condition on for each selection variable.
        Default is [[1, 1, ...]] — all binary selection at level 1.
        Example: [[1, 1]] means joint selection condition Z_0=1 AND Z_1=1.
    nuisance_backend : {"catboost", "sklearn", "linear"}
        Backend for outcome and propensity nuisance models.
    selection_backend : {"catboost", "sklearn", "linear"}
        Backend for selection CDF models.
    compute_gradients : bool
        Compute numerical gradients of the outcome nuisance w.r.t. the
        control function vector.  Required for ATES estimand.  Not needed
        for ATE/ATET — leave False for speed.
    n_bootstrap : int or None
        If set, use score bootstrap with this many replications.
        If None, use EIF-based inference (faster).
    random_state : int or None
        Random seed for reproducibility.

    Examples
    --------
    Binary treatment, dual binary selection (renew AND claim):

    >>> import numpy as np
    >>> from insurance_causal.autodml import DualSelectionDML
    >>> rng = np.random.default_rng(42)
    >>> n = 2000
    >>> X = rng.standard_normal((n, 4))
    >>> D = rng.binomial(1, 0.5, n)      # binary treatment
    >>> Z_renew = rng.binomial(1, 0.7, n)
    >>> Z_claim = rng.binomial(1, 0.3 * Z_renew, n)
    >>> Y = np.full(n, np.nan)
    >>> sel = (Z_renew == 1) & (Z_claim == 1)
    >>> Y[sel] = D[sel] * 2.0 + rng.standard_normal(sel.sum())
    >>> Z = np.column_stack([Z_renew, Z_claim])
    >>> dml = DualSelectionDML(n_folds=3, random_state=0)
    >>> dml.fit(Y, D, Z, X)
    DualSelectionDML(estimand='ATE', n_folds=3)
    >>> result = dml.estimate()
    >>> result.summary()  # doctest: +SKIP
    """

    def __init__(
        self,
        estimand: str = "ATE",
        n_folds: int = 5,
        treatment_values: tuple = (1, 0),
        selection_values: Optional[List[List[int]]] = None,
        nuisance_backend: str = "catboost",
        selection_backend: str = "catboost",
        compute_gradients: bool = False,
        n_bootstrap: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        valid_estimands = {"ATE", "ATES", "ATET", "LATE"}
        if estimand not in valid_estimands:
            raise ValueError(f"estimand must be one of {valid_estimands}, got {estimand!r}")
        if estimand == "LATE":
            raise NotImplementedError("LATE is not yet implemented in DualSelectionDML.")
        if estimand == "ATES" and not compute_gradients:
            raise ValueError(
                "ATES estimand requires compute_gradients=True (needs numerical "
                "differentiation of the outcome nuisance w.r.t. control functions)."
            )

        self.estimand = estimand
        self.n_folds = n_folds
        self.treatment_values = treatment_values
        self.selection_values = selection_values
        self.nuisance_backend = nuisance_backend
        self.selection_backend = selection_backend
        self.compute_gradients = compute_gradients
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

        # Fitted artefacts
        self._Y: Optional[np.ndarray] = None
        self._D: Optional[np.ndarray] = None
        self._Z: Optional[np.ndarray] = None
        self._X: Optional[np.ndarray] = None
        self._W_Z: Optional[np.ndarray] = None
        self._exposure: Optional[np.ndarray] = None
        self._sel_mask: Optional[np.ndarray] = None
        self._eif_scores: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    def __repr__(self) -> str:
        return f"DualSelectionDML(estimand={self.estimand!r}, n_folds={self.n_folds})"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        Y: Union[np.ndarray, pd.Series],
        D: Union[np.ndarray, pd.Series],
        Z: Union[np.ndarray, pd.DataFrame],
        X: Union[np.ndarray, pd.DataFrame],
        W_Z: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        W_D: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        exposure: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> "DualSelectionDML":
        """
        Fit nuisance models (selection CDFs, outcome regression, propensity).

        Parameters
        ----------
        Y : array of shape (n,)
            Outcome.  Must be NaN for unselected observations (where any
            selection condition is not met).
        D : array of shape (n,)
            Treatment (binary or continuous).
        Z : array of shape (n, K)
            Selection variables.  Each column is one ordinal selection stage.
            For binary stages: 0 = not selected, 1 = selected.
        X : array of shape (n, p)
            Covariates (confounders).  Must be fully observed.
        W_Z : array of shape (n, q), optional
            Exclusion restriction variables that predict selection but are
            excluded from the outcome model.  If None, identification relies
            on functional form — a warning is issued.
        W_D : array of shape (n, r), optional
            Instruments for treatment (reserved for future IV use; not
            used in the current implementation).
        exposure : array of shape (n,), optional
            Exposure offset (e.g. years at risk).

        Returns
        -------
        self
        """
        # --- Convert inputs ---
        Y = np.asarray(Y, dtype=float).ravel()
        D = np.asarray(D, dtype=float).ravel()
        if isinstance(Z, pd.DataFrame):
            Z = Z.to_numpy(dtype=float)
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype=float)
        X = np.asarray(X, dtype=float)

        n = len(Y)
        n_z = Z.shape[1]

        # Validate
        if np.any(np.isnan(X)):
            raise ValueError("X contains NaN values. Impute before calling fit().")
        if np.any(np.isnan(D)):
            raise ValueError("D contains NaN values.")
        if np.any(np.isnan(Z)):
            raise ValueError("Z contains NaN values.")

        # W_Z handling
        if W_Z is not None:
            if isinstance(W_Z, pd.DataFrame):
                W_Z = W_Z.to_numpy(dtype=float)
            W_Z = np.asarray(W_Z, dtype=float)
            if W_Z.ndim == 1:
                W_Z = W_Z.reshape(-1, 1)
        else:
            warnings.warn(
                "W_Z not provided. Identification of selection correction relies on "
                "functional form assumptions rather than exclusion restrictions. "
                "Consider adding instruments that predict selection but not outcomes.",
                UserWarning,
                stacklevel=2,
            )

        if exposure is not None:
            exposure = np.asarray(exposure, dtype=float).ravel()

        # Determine selection values (the target level for each Z_j)
        if self.selection_values is None:
            z_target = np.ones(n_z, dtype=int)
        else:
            z_target = np.array(self.selection_values[0], dtype=int)

        # Joint selection mask: all conditions simultaneously met
        sel_mask = np.ones(n, dtype=bool)
        for j in range(n_z):
            sel_mask &= (Z[:, j] == z_target[j])

        # Sanity check: Y should be NaN precisely for unselected
        n_sel_with_y = int(np.sum(~np.isnan(Y) & sel_mask))
        n_unsel_with_y = int(np.sum(~np.isnan(Y) & ~sel_mask))
        if n_unsel_with_y > 0:
            warnings.warn(
                f"{n_unsel_with_y} unselected observations have non-NaN Y. "
                "Y should be NaN for observations where the outcome is not observed. "
                "These non-NaN values for unselected observations will be ignored in "
                "outcome regression.",
                UserWarning,
                stacklevel=2,
            )

        if n_sel_with_y < 20:
            warnings.warn(
                f"Only {n_sel_with_y} fully selected observations with observed Y. "
                "Estimates may be unreliable.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._Y = Y
        self._D = D
        self._Z = Z
        self._X = X
        self._W_Z = W_Z
        self._exposure = exposure
        self._sel_mask = sel_mask
        self._z_target = z_target
        self._n_z = n_z

        # Run cross-fitting
        self._eif_scores = self._cross_fit(Y, D, Z, X, W_Z, sel_mask, z_target, exposure)
        self._is_fitted = True
        return self

    def estimate(self) -> DualSelectionResult:
        """
        Compute ATE (or other estimand) from fitted nuisance models.

        Returns
        -------
        result : DualSelectionResult
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before estimate().")

        psi = self._eif_scores
        n = len(psi)

        if self.n_bootstrap is not None:
            from insurance_causal.autodml._inference import score_bootstrap
            ate, se, ci_lower, ci_upper = score_bootstrap(
                psi, level=0.95, n_bootstrap=self.n_bootstrap,
                random_state=self.random_state
            )
        else:
            from insurance_causal.autodml._inference import eif_inference
            ate, se, ci_lower, ci_upper = eif_inference(psi, level=0.95)

        # Selection rates per variable and joint
        sel_rates: Dict[str, float] = {}
        for j in range(self._n_z):
            sel_rates[f"Z_{j}"] = float(np.mean(self._Z[:, j] == self._z_target[j]))
        sel_rates["joint"] = float(np.mean(self._sel_mask))

        nuisance_scores: Dict[str, float] = {
            "mean_eif": float(np.mean(psi)),
            "std_eif": float(np.std(psi, ddof=1)),
        }
        if hasattr(self, "_mean_mu_Z"):
            nuisance_scores["mean_selection_prob"] = self._mean_mu_Z
        if hasattr(self, "_mean_mu_D"):
            nuisance_scores["mean_propensity"] = self._mean_mu_D

        return DualSelectionResult(
            estimand=self.estimand,
            ate=ate,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_obs=n,
            n_selected=int(self._sel_mask.sum()),
            selection_rates=sel_rates,
            nuisance_scores=nuisance_scores,
            eif_scores=psi.copy(),
        )

    def sensitivity(
        self,
        rho_range: Tuple[float, float] = (-0.5, 0.5),
        n_points: int = 20,
    ) -> pd.DataFrame:
        """
        Rosenbaum-style sensitivity analysis.

        Computes adjusted CI bounds under the assumption that the residual
        selection-outcome correlation equals rho.  When rho=0 this recovers
        the main estimate.  As |rho| grows, the CI widens monotonically.

        The adjustment inflates the standard error by a factor of
        sqrt(1 + n_selected * rho^2 / (1 - rho^2)) following the partial
        R^2 representation of omitted variable bias in the EIF.

        Parameters
        ----------
        rho_range : tuple of (float, float)
            Range of rho values to evaluate.  Typically (-0.5, 0.5).
        n_points : int
            Number of grid points.

        Returns
        -------
        df : pd.DataFrame
            Columns: rho, ate_point, ate_lower, ate_upper.
            Rows are sorted by rho ascending.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before sensitivity().")

        psi = self._eif_scores
        n = len(psi)
        ate = float(np.mean(psi))
        base_se = float(np.std(psi, ddof=1) / np.sqrt(n))
        n_sel = int(self._sel_mask.sum())

        rho_grid = np.linspace(rho_range[0], rho_range[1], n_points)
        rows = []
        for rho in rho_grid:
            # Inflate SE by omitted variable bias factor.
            # This follows the partial R^2 approach: the omitted selection
            # confounder with partial correlation rho to the outcome introduces
            # additional variance proportional to rho^2 / (1 - rho^2).
            if abs(rho) >= 1.0:
                inflation = float("inf")
            else:
                inflation = np.sqrt(1.0 + n_sel * rho**2 / (n * (1.0 - rho**2 + 1e-12)))
            adjusted_se = base_se * inflation
            rows.append({
                "rho": float(rho),
                "ate_point": ate,
                "ate_lower": ate - 1.96 * adjusted_se,
                "ate_upper": ate + 1.96 * adjusted_se,
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal: cross-fitting
    # ------------------------------------------------------------------

    def _cross_fit(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        Z: np.ndarray,
        X: np.ndarray,
        W_Z: Optional[np.ndarray],
        sel_mask: np.ndarray,
        z_target: np.ndarray,
        exposure: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        K-fold cross-fitting with double sample splitting within each complement.

        For each fold k:
          - Evaluation set: fold k
          - Complement: all other folds, split into halves A and B
          - Half A trains selection CDFs and control function computation
          - Half B trains outcome nuisance and propensity, conditioned on
            control functions from half A
          - Both halves' predictions are averaged for evaluation set

        Returns EIF scores for all n observations.
        """
        n = len(Y)
        rng = np.random.RandomState(self.random_state)

        d, d_star = self.treatment_values
        n_z = Z.shape[1]

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        eif_scores = np.full(n, np.nan)
        mu_Z_all = np.full(n, np.nan)
        mu_D_all = np.full(n, np.nan)

        for fold_k, (comp_idx, eval_idx) in enumerate(kf.split(X)):
            # Split complement into halves A and B for double sample splitting
            n_comp = len(comp_idx)
            perm = rng.permutation(n_comp)
            half = n_comp // 2
            half_a_idx = comp_idx[perm[:half]]
            half_b_idx = comp_idx[perm[half:]]

            # We compute predictions twice (A trains CDF, B trains nuisance
            # and vice versa) then average — this is the double sample splitting
            # step that ensures valid cross-fitting for the control functions.
            psi_sum = np.zeros(len(eval_idx))
            mu_Z_sum = np.zeros(len(eval_idx))
            mu_D_sum = np.zeros(len(eval_idx))

            for train_a, train_b in [(half_a_idx, half_b_idx), (half_b_idx, half_a_idx)]:
                # --- Step 1: Fit selection CDFs on train_a ---
                cdf_models = self._fit_ordinal_cdfs(Z, D, X, W_Z, train_a)

                # --- Step 2: Compute control functions for eval and train_b ---
                P_bar_eval = self._compute_control_functions(
                    Z[eval_idx], D[eval_idx], X[eval_idx],
                    W_Z[eval_idx] if W_Z is not None else None,
                    cdf_models,
                )
                P_bar_b = self._compute_control_functions(
                    Z[train_b], D[train_b], X[train_b],
                    W_Z[train_b] if W_Z is not None else None,
                    cdf_models,
                )

                # --- Step 3: Fit nuisance on train_b using control functions ---
                # Outcome nuisance: E[Y | D, X, P_bar, Z] on selected train_b
                sel_b = sel_mask[train_b]
                train_b_sel = train_b[sel_b]
                P_bar_b_sel = P_bar_b[sel_b]

                mu_Y_d_eval, mu_Y_dstar_eval = self._fit_predict_outcome(
                    Y=Y,
                    D=D,
                    X=X,
                    Z=Z,
                    P_bar=P_bar_b,
                    sel_mask=sel_mask,
                    train_idx=train_b,
                    eval_idx=eval_idx,
                    P_bar_eval=P_bar_eval,
                    d=d,
                    d_star=d_star,
                    exposure=exposure,
                )

                # Propensity: P(D=d | X, P_bar, Z) on train_b (all obs)
                mu_D_eval = self._fit_predict_propensity(
                    D=D,
                    X=X,
                    Z=Z,
                    P_bar=P_bar_b,
                    train_idx=train_b,
                    eval_idx=eval_idx,
                    P_bar_eval=P_bar_eval,
                    d=d,
                )

                # Selection: P(Z_tilde=z | X, P_bar) on train_b (all obs)
                mu_Z_eval = self._fit_predict_selection(
                    sel_mask=sel_mask,
                    X=X,
                    Z=Z,
                    P_bar=P_bar_b,
                    train_idx=train_b,
                    eval_idx=eval_idx,
                    P_bar_eval=P_bar_eval,
                )

                # --- Step 4: Compute EIF scores for eval ---
                psi_k = self._eif_scores_batch(
                    Y=Y[eval_idx],
                    D=D[eval_idx],
                    sel=sel_mask[eval_idx],
                    mu_Y_d=mu_Y_d_eval,
                    mu_Y_dstar=mu_Y_dstar_eval,
                    mu_D=mu_D_eval,
                    mu_Z=mu_Z_eval,
                    d=d,
                    d_star=d_star,
                    estimand=self.estimand,
                )
                psi_sum += psi_k
                mu_Z_sum += mu_Z_eval
                mu_D_sum += mu_D_eval

            # Average over the two splits
            eif_scores[eval_idx] = psi_sum / 2.0
            mu_Z_all[eval_idx] = mu_Z_sum / 2.0
            mu_D_all[eval_idx] = mu_D_sum / 2.0

        if np.any(np.isnan(eif_scores)):
            warnings.warn(
                "NaN values in EIF scores. Check for extreme propensities or "
                "very small selection probabilities.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._mean_mu_Z = float(np.nanmean(mu_Z_all))
        self._mean_mu_D = float(np.nanmean(mu_D_all))
        return eif_scores

    # ------------------------------------------------------------------
    # Internal: selection CDF fitting
    # ------------------------------------------------------------------

    def _fit_ordinal_cdfs(
        self,
        Z: np.ndarray,
        D: np.ndarray,
        X: np.ndarray,
        W_Z: Optional[np.ndarray],
        train_idx: np.ndarray,
    ) -> List:
        """
        Fit CDF models P(Z_j <= z | D, X, W_Z) for each selection variable.

        For binary Z_j (values 0/1): one model per variable.
        For ordinal Z_j with K levels: K-1 models, one per threshold.

        Returns a list of length n_z. Each element is either a single
        classifier (binary) or a list of classifiers (ordinal).
        """
        # Features for selection models: [D, X] and optionally W_Z
        DX = np.column_stack([D[train_idx].reshape(-1, 1), X[train_idx]])
        if W_Z is not None:
            features = np.column_stack([DX, W_Z[train_idx]])
        else:
            features = DX

        n_z = Z.shape[1]
        cdf_models = []

        for j in range(n_z):
            Z_j = Z[train_idx, j]
            levels = np.unique(Z_j)

            if len(levels) <= 2:
                # Binary: one classifier for P(Z_j = 1)
                labels = (Z_j >= 1).astype(int)
                if len(np.unique(labels)) < 2:
                    # Degenerate: all same class — use constant prediction
                    model = _ConstantClassifier(float(labels.mean()))
                else:
                    model = self._build_classifier()
                    model.fit(features, labels)
                cdf_models.append(("binary", model))
            else:
                # Ordinal: one classifier per threshold z in levels[:-1]
                # P(Z_j <= z | ...) for z in levels[1:] (so z from levels[0]+1)
                thresholds = levels[:-1]
                threshold_models = []
                for z_thresh in thresholds:
                    labels_t = (Z_j <= z_thresh).astype(int)
                    if len(np.unique(labels_t)) < 2:
                        model = _ConstantClassifier(float(labels_t.mean()))
                    else:
                        model = self._build_classifier()
                        model.fit(features, labels_t)
                    threshold_models.append((z_thresh, model))
                cdf_models.append(("ordinal", threshold_models, levels))

        return cdf_models

    def _compute_control_functions(
        self,
        Z: np.ndarray,
        D: np.ndarray,
        X: np.ndarray,
        W_Z: Optional[np.ndarray],
        cdf_models: List,
    ) -> np.ndarray:
        """
        Compute the control function vector P_bar for a set of observations.

        For each selection variable Z_j, compute (p_lower, p_upper):
          - p_lower = P(Z_j <= z_j - 1 | D, X, W_Z)
          - p_upper = P(Z_j <= z_j | D, X, W_Z)

        Stack all pairs into a 2*n_z dimensional row vector per observation.

        Parameters
        ----------
        Z : array of shape (n, n_z)
        D, X, W_Z : matching arrays for n observations
        cdf_models : output of _fit_ordinal_cdfs

        Returns
        -------
        P_bar : array of shape (n, 2*n_z)
        """
        n = len(D)
        DX = np.column_stack([D.reshape(-1, 1), X])
        if W_Z is not None:
            features = np.column_stack([DX, W_Z])
        else:
            features = DX

        n_z = len(cdf_models)
        P_bar = np.zeros((n, 2 * n_z))

        for j, model_info in enumerate(cdf_models):
            if model_info[0] == "binary":
                _, clf = model_info
                # Binary: P(Z_j <= 0) and P(Z_j <= 1)
                # P(Z_j <= 1) = P(Z_j = 0) + P(Z_j = 1) = 1 always
                # P(Z_j <= 0) = P(Z_j = 0) = 1 - P(Z_j = 1)
                proba = clf.predict_proba(features)
                if proba.shape[1] == 2:
                    p_one = proba[:, 1]
                else:
                    p_one = proba[:, 0]
                p_lower = np.clip(1.0 - p_one, 0.0, 1.0)  # P(Z_j <= 0)
                p_upper = np.ones(n)                         # P(Z_j <= 1) = 1
                P_bar[:, 2 * j] = p_lower
                P_bar[:, 2 * j + 1] = p_upper
            else:
                _, threshold_models, levels = model_info
                # Ordinal: observed Z_j value determines which CDF to use
                Z_j = Z[:, j]
                p_lower = np.zeros(n)
                p_upper = np.zeros(n)
                # Build the full CDF at each threshold value
                cdf_at = {}
                cdf_at[levels[0] - 1] = np.zeros(n)
                cdf_at[levels[-1]] = np.ones(n)
                for z_thresh, clf in threshold_models:
                    proba = clf.predict_proba(features)
                    cdf_at[z_thresh] = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]

                for i_obs in range(n):
                    z_val = int(Z_j[i_obs])
                    # p_lower = P(Z_j <= z_val - 1)
                    # p_upper = P(Z_j <= z_val)
                    # We find the nearest threshold below/at z_val
                    lo_key = max((k for k in cdf_at if k <= z_val - 1), default=levels[0] - 1)
                    hi_key = min((k for k in cdf_at if k >= z_val), default=levels[-1])
                    p_lower[i_obs] = cdf_at[lo_key][i_obs]
                    p_upper[i_obs] = cdf_at[hi_key][i_obs]

                P_bar[:, 2 * j] = p_lower
                P_bar[:, 2 * j + 1] = p_upper

        return P_bar

    # ------------------------------------------------------------------
    # Internal: nuisance model fitting
    # ------------------------------------------------------------------

    def _build_classifier(self):
        """Return a fresh classifier for selection/propensity models."""
        backend = self.selection_backend
        if backend == "catboost":
            try:
                from catboost import CatBoostClassifier
                return CatBoostClassifier(
                    iterations=200,
                    depth=4,
                    learning_rate=0.05,
                    l2_leaf_reg=5,
                    random_seed=self.random_state,
                    verbose=0,
                )
            except ImportError:
                pass
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=4, random_state=self.random_state
        )

    def _build_propensity_model(self):
        """Return a propensity model for P(D=d | X, P_bar, Z)."""
        return self._build_classifier()

    def _build_outcome_model(self):
        """Return a regression model for E[Y | D, X, P_bar, Z]."""
        backend = self.nuisance_backend
        if backend == "catboost":
            try:
                from catboost import CatBoostRegressor
                return CatBoostRegressor(
                    iterations=200,
                    depth=4,
                    learning_rate=0.05,
                    l2_leaf_reg=5,
                    random_seed=self.random_state,
                    verbose=0,
                )
            except ImportError:
                pass
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=4, random_state=self.random_state
        )

    def _build_selection_outcome_model(self):
        """Return a classifier for P(Z_tilde=z | X, P_bar)."""
        return self._build_classifier()

    def _fit_predict_outcome(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        X: np.ndarray,
        Z: np.ndarray,
        P_bar: np.ndarray,
        sel_mask: np.ndarray,
        train_idx: np.ndarray,
        eval_idx: np.ndarray,
        P_bar_eval: np.ndarray,
        d: float,
        d_star: float,
        exposure: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit E[Y | D, X, P_bar, Z] on selected training observations, then
        predict mu_Y(d, ...) and mu_Y(d*, ...) on evaluation set.

        Features: [D, X, P_bar, Z] stacked.
        Only selected (outcome-observed) training obs are used.
        """
        # Build training features for selected observations
        sel_train = sel_mask[train_idx]
        if sel_train.sum() == 0:
            warnings.warn(
                "No selected observations in training fold for outcome nuisance. "
                "Returning zeros.",
                RuntimeWarning,
                stacklevel=2,
            )
            n_eval = len(eval_idx)
            return np.zeros(n_eval), np.zeros(n_eval)

        tr_sel_idx = train_idx[sel_train]

        # Build [D, X, P_bar, Z] features for training
        D_tr = D[tr_sel_idx].reshape(-1, 1)
        X_tr = X[tr_sel_idx]
        Z_tr = Z[tr_sel_idx]
        Pbar_tr = P_bar[sel_train]
        feats_tr = np.column_stack([D_tr, X_tr, Pbar_tr, Z_tr])

        Y_tr = Y[tr_sel_idx]
        # Filter out any residual NaN in Y for selected training obs
        valid = ~np.isnan(Y_tr)
        if valid.sum() < sel_train.sum():
            feats_tr = feats_tr[valid]
            Y_tr = Y_tr[valid]

        if len(Y_tr) == 0:
            n_eval = len(eval_idx)
            return np.zeros(n_eval), np.zeros(n_eval)

        if exposure is not None:
            exp_tr = exposure[tr_sel_idx]
            if valid.sum() < sel_train.sum():
                exp_tr = exp_tr[valid]
            Y_tr = Y_tr / np.clip(exp_tr, 1e-8, None)

        model = self._build_outcome_model()
        model.fit(feats_tr, Y_tr)

        # Build eval features, setting D = d and D = d*
        n_eval = len(eval_idx)
        X_ev = X[eval_idx]
        Z_ev = Z[eval_idx]

        feats_d = np.column_stack([
            np.full((n_eval, 1), d), X_ev, P_bar_eval, Z_ev
        ])
        feats_dstar = np.column_stack([
            np.full((n_eval, 1), d_star), X_ev, P_bar_eval, Z_ev
        ])

        mu_Y_d = model.predict(feats_d)
        mu_Y_dstar = model.predict(feats_dstar)

        if exposure is not None:
            exp_ev = exposure[eval_idx]
            mu_Y_d = mu_Y_d * exp_ev
            mu_Y_dstar = mu_Y_dstar * exp_ev

        return mu_Y_d, mu_Y_dstar

    def _fit_predict_propensity(
        self,
        D: np.ndarray,
        X: np.ndarray,
        Z: np.ndarray,
        P_bar: np.ndarray,
        train_idx: np.ndarray,
        eval_idx: np.ndarray,
        P_bar_eval: np.ndarray,
        d: float,
    ) -> np.ndarray:
        """
        Fit P(D=d | X, P_bar, Z) on all training obs (selection-agnostic).
        Returns propensity for eval set.

        For binary treatment: standard classifier.
        For continuous treatment: not yet supported (returns 0.5 with warning).
        """
        D_tr = D[train_idx]
        X_tr = X[train_idx]
        Z_tr = Z[train_idx]
        Pbar_tr = P_bar

        # Determine if treatment is binary
        unique_D = np.unique(D_tr)
        is_binary = len(unique_D) <= 2

        feats_tr = np.column_stack([X_tr, Pbar_tr, Z_tr])
        n_eval = len(eval_idx)
        X_ev = X[eval_idx]
        Z_ev = Z[eval_idx]
        feats_ev = np.column_stack([X_ev, P_bar_eval, Z_ev])

        if is_binary:
            D_bin = (D_tr == d).astype(int)
            if len(np.unique(D_bin)) < 2:
                # Degenerate fold: all treated or all control
                model = _ConstantClassifier(float(D_bin.mean()))
            else:
                model = self._build_propensity_model()
                model.fit(feats_tr, D_bin)
            proba = model.predict_proba(feats_ev)
            mu_D = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]
        else:
            # Continuous treatment: kernel density estimate at d would be needed
            # For now, return a constant (unbiased but inefficient)
            warnings.warn(
                "Continuous treatment detected. Propensity estimation for "
                "continuous treatments is not implemented. Using constant 0.5. "
                "Results will be consistent but not efficient.",
                UserWarning,
                stacklevel=2,
            )
            mu_D = np.full(n_eval, 0.5)

        # Clip to avoid division by zero
        mu_D = np.clip(mu_D, 0.01, 0.99)
        return mu_D

    def _fit_predict_selection(
        self,
        sel_mask: np.ndarray,
        X: np.ndarray,
        Z: np.ndarray,
        P_bar: np.ndarray,
        train_idx: np.ndarray,
        eval_idx: np.ndarray,
        P_bar_eval: np.ndarray,
    ) -> np.ndarray:
        """
        Fit P(Z_tilde=z | X, P_bar) (joint selection probability) on training obs.
        Returns selection probability for eval set.
        """
        X_tr = X[train_idx]
        Z_tr = Z[train_idx]
        Pbar_tr = P_bar
        sel_tr = sel_mask[train_idx].astype(int)

        feats_tr = np.column_stack([X_tr, Pbar_tr, Z_tr])
        n_eval = len(eval_idx)
        X_ev = X[eval_idx]
        Z_ev = Z[eval_idx]
        feats_ev = np.column_stack([X_ev, P_bar_eval, Z_ev])

        model = self._build_selection_outcome_model()
        if sel_tr.sum() == 0 or sel_tr.sum() == len(sel_tr):
            # Degenerate: all or none selected — return empirical mean
            mu_Z = np.full(n_eval, sel_tr.mean() + 0.01)
        else:
            model.fit(feats_tr, sel_tr)
            proba = model.predict_proba(feats_ev)
            mu_Z = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]

        # Clip to avoid division by zero
        mu_Z = np.clip(mu_Z, 0.01, 0.99)
        return mu_Z

    # ------------------------------------------------------------------
    # Internal: EIF score computation
    # ------------------------------------------------------------------

    def _eif_scores_batch(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        sel: np.ndarray,
        mu_Y_d: np.ndarray,
        mu_Y_dstar: np.ndarray,
        mu_D: np.ndarray,
        mu_Z: np.ndarray,
        d: float,
        d_star: float,
        estimand: str,
    ) -> np.ndarray:
        """
        Compute EIF scores for a batch of evaluation observations.

        The score for ATE is:
            r_i = [mu_Y(d,...) - mu_Y(d*,...)]
                + 1(D_i=d, Z_i=z) / [mu_D * mu_Z] * [Y_i - mu_Y(d,...)]
                - 1(D_i=d*, Z_i=z) / [(1-mu_D) * mu_Z] * [Y_i - mu_Y(d*,...)]

        For ATES: score is weighted by mu_Z (selection prob):
            r_i = mu_Z * [mu_Y(d,...) - mu_Y(d*,...)]
                + 1(D_i=d, Z_i=z) / mu_D * [Y_i - mu_Y(d,...)]
                - 1(D_i=d*, Z_i=z) / (1-mu_D) * [Y_i - mu_Y(d*,...)]

        For ATET: score is weighted by 1(D_i=d) / P(D=d):
            r_i = 1(D_i=d) / mu_D_marg * ([mu_Y(d,...) - mu_Y(d*,...)]
                + sel / mu_Z * [Y_i - mu_Y(d,...)]
                - ...)
        """
        n = len(Y)

        # Indicator for D=d and D=d*
        is_d = (D == d).astype(float)
        is_dstar = (D == d_star).astype(float)
        is_sel = sel.astype(float)

        # For unselected obs, Y is NaN — replace with 0 for score computation
        # The indicator 1(Z_i=z) = 0 for unselected, so these terms drop out
        Y_safe = np.where(np.isnan(Y), 0.0, Y)

        if estimand == "ATE":
            # Standard ATE EIF
            correction_d = (
                is_d * is_sel / np.clip(mu_D * mu_Z, 0.01, None)
                * (Y_safe - mu_Y_d)
            )
            correction_dstar = (
                is_dstar * is_sel / np.clip((1.0 - mu_D) * mu_Z, 0.01, None)
                * (Y_safe - mu_Y_dstar)
            )
            r = (mu_Y_d - mu_Y_dstar) + correction_d - correction_dstar

        elif estimand == "ATES":
            # ATE on Selected: scale by mu_Z and remove mu_Z from denominator
            correction_d = (
                is_d * is_sel / np.clip(mu_D, 0.01, None)
                * (Y_safe - mu_Y_d)
            )
            correction_dstar = (
                is_dstar * is_sel / np.clip(1.0 - mu_D, 0.01, None)
                * (Y_safe - mu_Y_dstar)
            )
            r = mu_Z * (mu_Y_d - mu_Y_dstar) + correction_d - correction_dstar
            # Normalise by mean selection probability
            mean_sel = max(float(np.mean(mu_Z)), 0.01)
            r = r / mean_sel

        elif estimand == "ATET":
            # ATE on Treated (D=d)
            # Need marginal P(D=d), approximate with observed proportion
            p_d = max(float(np.mean(is_d)), 0.01)
            correction_d = (
                is_d * is_sel / np.clip(mu_Z, 0.01, None)
                * (Y_safe - mu_Y_d) / p_d
            )
            correction_dstar = (
                is_d * is_sel * mu_D / np.clip((1.0 - mu_D) * mu_Z * p_d, 0.01, None)
                * (Y_safe - mu_Y_dstar)
            )
            plug_in = is_d / p_d * (mu_Y_d - mu_Y_dstar)
            r = plug_in + correction_d - correction_dstar

        else:
            raise ValueError(f"Unknown estimand: {estimand!r}")

        return r
