"""Summary statistics for meta-analysis."""
from typing import Iterable

import numpy as np

from .result import MetaResult


def I2(values: np.ndarray, sigmas: np.ndarray) -> float:
    w = 1 / sigmas**2
    ybar = np.sum(w * values) / np.sum(w)
    Q = np.sum(w * (values - ybar) ** 2)
    n = len(values)
    df = n - 1
    I2_val = 1 - df / Q
    return np.max([0.0, I2_val])

def br(values: np.ndarray, sigmas: np.ndarray) -> float:
    w = 1 / sigmas**2
    ybar = np.sum(w * values) / np.sum(w)
    Q = np.sum(w * (values - ybar) ** 2)
    n = len(values)
    df = n - 1
    return np.sqrt(Q / df)

def chi2(wm: float, values: np.ndarray, sigmas: np.ndarray) -> float:
    return np.sum((values - wm) ** 2 / sigmas**2)

def interval_score(
    interval: Iterable[float], truth: float, coverage: float, percent: bool = False
) -> float:
    alpha = 1 - coverage
    l, u = interval
    width = u - l
    below = (truth < l) * (2 / alpha) * (l - truth)
    above = (u < truth) * (2 / alpha) * (truth - u)
    total = width + below + above
    if percent:
        total /= np.abs(truth)
    return float(total)


def em_t_location_scale(
    values: np.ndarray,
    sigmas: np.ndarray,
    nu: float,
    c_min: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> MetaResult:
    """
    EM for theta, scale c in y_i ~ theta + c * sigma_i * t_nu.
    Returns MetaResult with point_est=theta_hat, interval empty (not CI).
    """
    theta = np.median(values)
    mad = np.median(np.abs(values - theta)) / 0.6745
    c = max(c_min, mad / np.median(sigmas))
    for it in range(max_iter):
        r = (values - theta) / (c * sigmas)
        w = (nu + 1) / (nu + r**2)
        theta_new = np.sum(w * values / sigmas**2) / np.sum(w / sigmas**2)
        c_new = np.sqrt(np.mean(w * (values - theta_new) ** 2 / sigmas**2))
        c_new = max(c_min, c_new)
        if abs(theta_new - theta) < tol and abs(c_new - c) < tol:
            return MetaResult(
                point_est=float(theta_new),
                interval=[],
                targeted_cov=np.nan,
                actual_cov=np.nan,
                scale=float(c_new),
                extra={"iterations": it + 1},
            )
        theta, c = theta_new, c_new
    return MetaResult(
        point_est=float(theta),
        interval=[],
        targeted_cov=np.nan,
        actual_cov=np.nan,
        scale=float(c),
        extra={"iterations": max_iter, "converged": 0.0},
    )

