"""
Inference procedures for debiased ML estimators.

Two approaches are provided:

EIF-based (default)
    Uses the empirical variance of the efficient influence function scores
    to compute standard errors.  Achieves the semiparametric efficiency
    bound and requires only one pass through the data.
    SE = std(psi) / sqrt(n)

Bootstrap
    Resamples the EIF scores (not the raw data) to compute CIs.  This is
    the 'score bootstrap' — much faster than full refitting and valid under
    the same regularity conditions as the EIF approach.

For clustered standard errors (e.g. multiple policies from the same
household), pass a ``cluster_ids`` array.  This computes the cluster-robust
variance: SE^2 = sum_c (sum_{i in c} psi_i)^2 / n^2.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import stats


def eif_inference(
    psi: np.ndarray,
    level: float = 0.95,
    cluster_ids: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float]:
    """
    EIF-based point estimate and confidence interval.

    Parameters
    ----------
    psi : array of shape (n,)
        Efficient influence function scores.
    level : float
        Confidence level (e.g. 0.95 for 95% CI).
    cluster_ids : array of shape (n,), optional
        Cluster identifiers for cluster-robust standard errors.
    sample_weight : array of shape (n,), optional
        Observation weights.

    Returns
    -------
    estimate : float
    se : float
    ci_low : float
    ci_high : float
    """
    psi = np.asarray(psi, dtype=float).ravel()

    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float)
        w = w / w.mean()
        estimate = float(np.average(psi, weights=w))
    else:
        estimate = float(np.mean(psi))

    n = len(psi)
    alpha = 1.0 - level
    z = stats.norm.ppf(1 - alpha / 2)

    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)
        clusters = np.unique(cluster_ids)
        cluster_scores = np.array(
            [np.sum(psi[cluster_ids == c]) for c in clusters]
        )
        se = float(np.sqrt(np.sum(cluster_scores**2)) / n)
    else:
        se = float(np.std(psi, ddof=1) / np.sqrt(n))

    ci_low = estimate - z * se
    ci_high = estimate + z * se
    return estimate, se, ci_low, ci_high


def score_bootstrap(
    psi: np.ndarray,
    level: float = 0.95,
    n_bootstrap: int = 500,
    random_state: Optional[int] = None,
) -> Tuple[float, float, float, float]:
    """
    Score bootstrap confidence intervals (resampling EIF scores).

    This is faster than full refitting bootstrap and asymptotically valid
    under standard DML regularity conditions.

    Parameters
    ----------
    psi : array of shape (n,)
        EIF scores.
    level : float
        CI coverage level.
    n_bootstrap : int
        Number of bootstrap replications.
    random_state : int or None
        Random seed.

    Returns
    -------
    estimate : float
    se : float
    ci_low : float
    ci_high : float
    """
    psi = np.asarray(psi, dtype=float).ravel()
    rng = np.random.RandomState(random_state)
    n = len(psi)

    estimate = float(np.mean(psi))
    boot_means = np.array(
        [np.mean(rng.choice(psi, size=n, replace=True)) for _ in range(n_bootstrap)]
    )
    se = float(np.std(boot_means, ddof=1))
    alpha = 1.0 - level
    ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return estimate, se, ci_low, ci_high


def run_inference(
    psi: np.ndarray,
    level: float = 0.95,
    inference: str = "eif",
    cluster_ids: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
    n_bootstrap: int = 500,
    random_state: Optional[int] = None,
) -> Tuple[float, float, float, float]:
    """
    Dispatch to the requested inference procedure.

    Parameters
    ----------
    psi : array of shape (n,)
        EIF scores.
    level : float
        Confidence level.
    inference : {"eif", "bootstrap"}
        Inference method.
    cluster_ids : array of shape (n,), optional
        For cluster-robust EIF inference.
    sample_weight : array of shape (n,), optional
        Observation weights.
    n_bootstrap : int
        Bootstrap replications (ignored for "eif").
    random_state : int or None
        Random seed.

    Returns
    -------
    estimate, se, ci_low, ci_high : float
    """
    if inference == "bootstrap":
        return score_bootstrap(
            psi,
            level=level,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
    elif inference == "eif":
        return eif_inference(
            psi,
            level=level,
            cluster_ids=cluster_ids,
            sample_weight=sample_weight,
        )
    else:
        raise ValueError(f"Unknown inference method: {inference!r}. Choose 'eif' or 'bootstrap'.")
