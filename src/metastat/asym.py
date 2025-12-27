"""Asymmetric error handling for meta-analysis."""
from typing import Callable, Optional, Tuple

import numpy as np
import warnings

from .result import MetaResult
from .stat import chi2


def symmetrize_error(
    diff: np.ndarray,
    sigma_n: np.ndarray,
    sigma_p: np.ndarray,
    method: str = 'pdg',
) -> np.ndarray:
    error = np.full(sigma_n.shape, np.nan)

    error[diff < -sigma_n] = sigma_n[diff < -sigma_n]
    error[diff > sigma_p] = sigma_p[diff > sigma_p]
    between = ~((diff < -sigma_n) | (diff > sigma_p))

    if method == 'pdg':
        error_between = (2*sigma_n*sigma_p+diff*(sigma_p-sigma_n))/(sigma_n+sigma_p)
    error[between] = error_between[between]

    return error


def iter_infer(
    values: np.ndarray,
    sigmas_n: np.ndarray,
    sigmas_p: np.ndarray,
    infer_func: Callable[[np.ndarray, np.ndarray, float], MetaResult],
    coverage: float,
    init: Optional[float] = None,
    tol: float = 0.01,
    max_iter: int = 100,
) -> MetaResult:
    """Iteratively compute the PDG-style weighted mean and the effective errors."""
    sigmas = 2 * (sigmas_n * sigmas_p) / (sigmas_n + sigmas_p)
    chisq_prev = -999
    wms = []
    res = None
    for iteration in range(max_iter + 1):
        if iteration == 0 and init is not None:
            wm = init
        elif iteration == max_iter:
            warnings.warn('weighted mean iterations exceeded')
        else:
            res = infer_func(values, sigmas, coverage=coverage)
            wm = res.point_est
        wms.append(wm)
        chisq = chi2(wm, values, sigmas)
        if np.abs(chisq - chisq_prev) < tol:
            break
        chisq_prev = chisq
        diff = wm - values
        sigmas = symmetrize_error(diff, sigmas_n, sigmas_p)
        if iteration == max_iter - 1:
            warnings.warn('weighted mean iterations exceeded')
    # Ensure we have a result object
    if res is None:
        res = infer_func(values, sigmas, coverage=coverage)
    res.extra["iterations"] = iteration
    res.extra["wms"] = wms
    res.extra["chisq"] = chisq
    res.extra["sigmas"] = sigmas
    return res


def bartlett_linear_ll(
    theta: float,
    values: np.ndarray,
    sigma_n: np.ndarray,
    sigma_p: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Compute log-likelihood using Bartlett's linear approximation.
    
    Args:
        theta: Parameter value
        values: Observed values
        sigma_n: Negative errors
        sigma_p: Positive errors
    
    Returns:
        Tuple of (total log-likelihood, per-point log-likelihood)
    """
    sigma = 2 * (sigma_n * sigma_p) / (sigma_n + sigma_p)
    sigma_prime = (sigma_p - sigma_n) / (sigma_n + sigma_p)
    ll_perpoint = -0.5 * ((theta - values) / (sigma + sigma_prime * (theta - values))) ** 2
    ll = np.sum(ll_perpoint)
    return float(ll), ll_perpoint


def pdg_ll(
    theta: float,
    values: np.ndarray,
    sigma_n: np.ndarray,
    sigma_p: np.ndarray,
) -> float:
    """Compute log-likelihood using PDG symmetrization.
    
    Args:
        theta: Parameter value
        values: Observed values
        sigma_n: Negative errors
        sigma_p: Positive errors
    
    Returns:
        Total log-likelihood
    """
    diff = theta - values
    sigma = symmetrize_error(diff, sigma_n, sigma_p, method='pdg')
    ll = -0.5 * np.sum((diff / sigma) ** 2)
    return float(ll)


def dimidiated_ll(
    theta: float,
    values: np.ndarray,
    sigma_n: np.ndarray,
    sigma_p: np.ndarray,
) -> float:
    """Compute log-likelihood using dimidiated (split) errors.
    
    Uses sigma_p when diff > 0, sigma_n when diff < 0.
    
    Args:
        theta: Parameter value
        values: Observed values
        sigma_n: Negative errors
        sigma_p: Positive errors
    
    Returns:
        Total log-likelihood
    """
    diff = theta - values
    # error p when diff > 0, error n when diff < 0
    error = np.where(diff > 0, sigma_p, sigma_n)
    ll = -0.5 * np.sum((diff / error) ** 2)
    return float(ll)


def ll_interval(
    values: np.ndarray,
    sigma_n: np.ndarray,
    sigma_p: np.ndarray,
    ll_func: Callable[[float, np.ndarray, np.ndarray, np.ndarray], float] = pdg_ll,
    n_points: int = 10000,
) -> Tuple[float, float, float]:
    """Find MLE and confidence interval from log-likelihood.
    
    Args:
        values: Observed values
        sigma_n: Negative errors
        sigma_p: Positive errors
        ll_func: Log-likelihood function to use
        n_points: Number of points in the search grid
    
    Returns:
        Tuple of (mle, lower_error, upper_error)
    """
    lb = np.min(values - 2 * sigma_n)
    ub = np.max(values + 2 * sigma_p)
    xspace = np.linspace(lb, ub, n_points)
    lls = np.array([ll_func(x, values, sigma_n, sigma_p) for x in xspace])
    ll_mle = np.max(lls)
    mle = xspace[np.argmax(lls)]
    in_interval = lls - ll_mle > -0.5
    lower = xspace[np.argmax(in_interval)]
    upper = xspace[-np.argmax(in_interval[::-1])]
    return MetaResult(
        point_est=float(mle),
        interval=[float(lower), float(upper)],
        targeted_cov=0.6827,
        actual_cov=0.6827,
    )
