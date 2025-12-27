"""Statistical tests (p-values) for meta-analysis."""
from itertools import product
from typing import Iterable

import numpy as np

try:  # optional dependency
    import Rmath4  # type: ignore
except Exception:  # pragma: no cover - optional
    Rmath4 = None


def errscale_test(values: np.ndarray, sigmas: np.ndarray) -> float:
    """
    Test of the Birge ratio model (Baker and Jackson, 2013).
    Returns a p-value.
    """
    sigmas2 = sigmas**2
    w = 1 / sigmas2
    muhat = np.sum(w * values) / np.sum(w)
    Q = np.sum(w * (values - muhat) ** 2)
    n = len(values)
    bi = np.log(sigmas2)
    bbar = np.mean(bi)
    S = np.sum((bi - bbar) * ((values - muhat) ** 2 / sigmas2) / (Q / (n - 1)))
    Sprime = np.zeros(1000)
    for i in range(len(Sprime)):
        eps = np.random.normal(loc=0, scale=1, size=n)
        M = np.sum(1 / sigmas2)
        epshat = 1 / M * np.sum(eps / sigmas)
        Sprime[i] = (
            (n - 1)
            * np.sum((bi - bbar) * (eps - epshat / sigmas) ** 2)
            / np.sum((eps - epshat / sigmas) ** 2)
        )
    p = np.mean(Sprime <= S)
    return float(p)


def sign_rank_test(values: Iterable[float], h0_median: float = 0) -> float:
    if Rmath4 is None:
        raise ImportError("Rmath4 is required for sign_rank_test")
    values = np.array(values) - h0_median
    zeroes = any(values == 0)
    if zeroes:
        values = values[values != 0]
    n = len(values)
    w = np.add.outer(values, values) / 2
    w = np.sort(w[np.tril_indices(w.shape[0], 0)])
    count = np.sum(w > 0)
    p_upper_tail = Rmath4.psignrank(count - 1, n, 0, 0)
    p_lower_tail = Rmath4.psignrank(count, n, 1, 0)
    return float(min(min(p_upper_tail, p_lower_tail) * 2, 1))


def flip_test(
    values: np.ndarray,
    h0: float = 0,
    tail: str = "both",
    mode: str = "median",
    boot: bool = False,
) -> float:
    if tail not in ["both", "lower", "upper"]:
        raise ValueError("tail must be one of 'both', 'lower', or 'upper'")
    values = np.array(values) - h0
    B = 10000
    n = len(values)
    if 2**n <= B and not boot:
        exact = True
        B = 2**n
    else:
        exact = False
    center_func = np.median if mode == "median" else np.mean
    T = center_func(values)
    if exact:
        signs = np.array(list(product([-1, 1], repeat=n)))
    else:
        signs = np.random.choice([-1, 1], size=(B, n))
    if boot:
        values = np.random.choice(values, size=(B, n), replace=True)
    signs_centers = center_func(signs * values, axis=1)
    lower_count = np.sum(T < signs_centers)
    upper_count = np.sum(T > signs_centers)
    if tail == "both":
        return float((min(lower_count, upper_count) / B) * 2)
    if tail == "lower":
        return float(lower_count / B)
    return float(upper_count / B)

