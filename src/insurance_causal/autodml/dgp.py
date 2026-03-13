"""
SyntheticContinuousDGP: data-generating process with known causal truth.

Used for validating estimators against ground truth and for the Databricks
notebook demonstrations.

The DGP models a stylised UK motor insurance renewal portfolio:
- X: policyholder characteristics (age, NCB, vehicle age, postcode risk)
- D: premium set by a technical pricing model plus random noise
- Y: outcome (claim indicator / pure premium)
- S: renewal indicator (binary, depends on D and X)

The confounding structure: higher-risk policyholders (older vehicles, lower
NCB) tend to be quoted higher premiums, so D and Y are correlated via X.

Known ground truth
------------------
The true AME is E[dE[Y|D,X]/dD] which we compute analytically from the DGP.

For the default DGP:
    log E[Y | D, X] = beta_D * D + f(X)
    AME = beta_D * E[E[Y | D, X]]   (multiplicative model)

For a linear model:
    E[Y | D, X] = beta_D * D + f(X)
    AME = beta_D

The DGP can simulate selection bias by making the renewal probability a
decreasing function of D.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SyntheticContinuousDGP:
    """
    Synthetic DGP for a UK motor insurance renewal portfolio.

    Parameters
    ----------
    n : int
        Number of observations.
    n_features : int
        Number of covariates (at least 4 are meaningful; extras are noise).
    outcome_family : {"gaussian", "poisson", "gamma"}
        Distribution of the outcome.  "poisson" simulates claim counts;
        "gamma" simulates severity; "gaussian" simulates log-pure-premium.
    beta_D : float
        True causal effect of premium on outcome (per unit D).
        AME = beta_D for linear, beta_D * E[Y] for log-linear.
    confounding_strength : float
        How strongly X affects both D and Y.  0 = no confounding;
        1 = strong confounding.  Higher values make naive OLS more biased.
    selection_strength : float
        How strongly D affects the renewal probability S.
        0 = no selection (all observe Y); positive values = higher premiums
        cause more lapses.
    sigma_D : float
        Standard deviation of treatment noise (pricing random component).
    base_premium : float
        Mean premium in pounds.
    exposure_rate : float
        Mean exposure (years at risk per policy).  Used for Poisson models.
    random_state : int or None
        Random seed.

    Attributes
    ----------
    true_ame_ : float
        Analytical true AME, set during generate().
    true_dose_response_ : callable
        Function d -> E[Y(d)] (population average outcome at premium d).
    """

    n: int = 5000
    n_features: int = 8
    outcome_family: str = "gaussian"
    beta_D: float = -0.002
    confounding_strength: float = 0.5
    selection_strength: float = 1.5
    sigma_D: float = 30.0
    base_premium: float = 350.0
    exposure_rate: float = 1.0
    random_state: Optional[int] = 42

    true_ame_: float = field(default=0.0, init=False)
    true_dose_response_: Optional[object] = field(default=None, init=False)

    def generate(
        self,
        include_selection: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Generate synthetic data.

        Parameters
        ----------
        include_selection : bool
            If True, returns a renewal indicator S and sets Y to NaN for
            non-renewers (simulating the renewal portfolio selection problem).

        Returns
        -------
        X : array of shape (n, n_features)
        D : array of shape (n,)
        Y : array of shape (n,)
        S : array of shape (n,) or None
            Renewal indicator (1 = observed). Only returned if
            include_selection=True; otherwise None.
        """
        rng = np.random.RandomState(self.random_state)
        n = self.n

        # --- Covariates ---
        # X[:,0]: normalised age (0=young, 1=old)
        age = rng.beta(2, 3, size=n)
        # X[:,1]: NCB (0=no NCB, 1=full NCB). Negatively correlated with risk.
        ncb = rng.beta(3, 2, size=n)
        # X[:,2]: vehicle age (0=new, 1=old). Positively correlated with risk.
        veh_age = rng.beta(2, 2, size=n)
        # X[:,3]: postcode risk score (0=low risk, 1=high risk).
        postcode_risk = rng.beta(1.5, 2, size=n)
        # Remaining features: noise
        noise_features = rng.randn(n, max(0, self.n_features - 4))

        X_list = [age, ncb, veh_age, postcode_risk]
        if noise_features.shape[1] > 0:
            X_list.extend([noise_features[:, k] for k in range(noise_features.shape[1])])
        X = np.column_stack(X_list)

        # --- Latent risk score (true log-rate) ---
        # Higher NCB = lower risk; higher vehicle age and postcode risk = higher risk
        log_risk = (
            0.5 * age
            - 1.0 * ncb
            + 0.8 * veh_age
            + 1.2 * postcode_risk
        )

        # --- Treatment: premium ---
        # Technical premium = base * exp(log_risk) * confounding + noise
        technical_premium = self.base_premium * np.exp(
            self.confounding_strength * log_risk
        )
        D = technical_premium + rng.randn(n) * self.sigma_D
        D = np.clip(D, 50.0, 2000.0)  # Realistic premium range

        # --- Outcome ---
        # True structural equation: mu = exp(gamma * log_risk + beta_D * D)
        gamma = 1.0  # Risk coefficient
        mu_log = gamma * log_risk + self.beta_D * D

        if self.outcome_family == "poisson":
            exposure = rng.exponential(self.exposure_rate, size=n)
            exposure = np.clip(exposure, 0.1, 3.0)
            rate = np.exp(mu_log)
            Y = rng.poisson(rate * exposure).astype(float)
        elif self.outcome_family == "gamma":
            mean_sev = np.exp(mu_log + 5.0)  # Severity: mean ~£150-300
            shape_param = 2.0
            scale_param = mean_sev / shape_param
            Y = rng.gamma(shape_param, scale_param)
            exposure = None
        elif self.outcome_family == "gaussian":
            Y = mu_log + rng.randn(n) * 0.3
            exposure = None
        else:
            raise ValueError(f"Unknown outcome_family: {self.outcome_family!r}")

        # --- True AME ---
        # For the log-linear model: E[Y|D,X] = exp(mu_log)
        # dE[Y|D,X]/dD = beta_D * exp(mu_log)
        # AME = E[beta_D * exp(mu_log)] = beta_D * E[exp(mu_log)]
        if self.outcome_family in ("poisson", "gamma"):
            self.true_ame_ = float(self.beta_D * np.mean(np.exp(mu_log)))
        else:
            # Gaussian: linear in D so AME = beta_D
            self.true_ame_ = float(self.beta_D)

        # True dose-response function
        _log_risk_mean = float(np.mean(log_risk))
        _beta_D = self.beta_D
        _family = self.outcome_family

        def true_dose_response(d, _lrm=_log_risk_mean, _bd=_beta_D, _fam=_family):
            """E[Y(d)] = E_X[exp(log_risk + beta_D * d)]"""
            if _fam in ("poisson", "gamma"):
                return np.exp(_lrm + _bd * d)
            else:
                return _lrm + _bd * d

        self.true_dose_response_ = true_dose_response

        # --- Selection ---
        if include_selection:
            # Renewal probability: logistic(-selection_strength * D_std)
            # When selection_strength=0: all policies renew (S=1 everywhere).
            D_std = (D - D.mean()) / (D.std() + 1e-8)
            if self.selection_strength == 0.0:
                S = np.ones(n, dtype=float)
            else:
                logit_s = -self.selection_strength * D_std
                pi_s = 1.0 / (1.0 + np.exp(-logit_s))
                S = rng.binomial(1, pi_s).astype(float)
            # Mask outcomes for non-renewers
            Y_obs = Y.copy()
            Y_obs[S == 0] = np.nan
            if self.outcome_family == "poisson":
                return X, D, Y_obs, S, exposure
            return X, D, Y_obs, S
        else:
            if self.outcome_family == "poisson":
                return X, D, Y, None, exposure
            return X, D, Y, None

    def as_dataframe(
        self,
        include_selection: bool = False,
    ) -> pd.DataFrame:
        """
        Generate data and return as a pandas DataFrame.

        Parameters
        ----------
        include_selection : bool
            Whether to include the selection indicator column.

        Returns
        -------
        df : pd.DataFrame
            Columns: X0..X{p-1}, premium, outcome, [selection], [exposure].
        """
        result = self.generate(include_selection=include_selection)

        if self.outcome_family == "poisson":
            X, D, Y, S, exposure = result if include_selection else (*result[:3], None, result[4])
        else:
            X, D, Y, S = result

        df = pd.DataFrame(
            X,
            columns=[f"X{k}" for k in range(X.shape[1])],
        )
        df.rename(
            columns={
                "X0": "age_norm",
                "X1": "ncb_norm",
                "X2": "veh_age_norm",
                "X3": "postcode_risk",
            },
            inplace=True,
        )
        df["premium"] = D
        df["outcome"] = Y

        if include_selection and S is not None:
            df["renewed"] = S.astype(int)

        if self.outcome_family == "poisson":
            df["exposure"] = exposure

        return df
