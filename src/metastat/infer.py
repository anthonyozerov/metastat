"""Interval estimators for meta-analysis."""
from logging import critical
from typing import Optional, Tuple

import numpy as np
from scipy.stats import binom, norm, t

try:  # optional dependency
    import Rmath4  # type: ignore
except Exception:  # pragma: no cover - optional
    Rmath4 = None

from .result import MetaResult
from .test import flip_test


def precision_weighted_mean(
    values: np.ndarray,
    sigmas: np.ndarray,
) -> float:
    w = 1 / sigmas**2
    return np.sum(w * values) / np.sum(w)


def fixed_effect(
    values: np.ndarray,
    sigmas: np.ndarray,
    coverage: Optional[float] = None,
    zalpha: Optional[float] = None,
) -> MetaResult:
    if zalpha is None:
        assert coverage is not None, "coverage is required when zalpha not provided"
        tail_prob = (1 - coverage) / 2
        zalpha = float(np.abs(norm.ppf(tail_prob)))
    w = 1 / sigmas**2
    yhat = np.sum(w * values) / np.sum(w)
    sigma = np.sqrt(1 / np.sum(w))
    interval = [yhat - zalpha * sigma, yhat + zalpha * sigma]
    cov = coverage if coverage is not None else 1 - 2 * norm.sf(zalpha)
    return MetaResult(
        point_est=float(yhat),
        interval=interval,
        targeted_cov=cov,
        actual_cov=cov,
        sigma=float(sigma),
    )


def _random_effects_dl_base(values: np.ndarray, sigmas: np.ndarray):
    w = 1 / sigmas**2
    ybar = np.sum(w * values) / np.sum(w)
    Q = np.sum(w * (values - ybar) ** 2)
    n = len(values)
    df = n - 1
    s1 = np.sum(w)
    s2 = np.sum(w**2)
    tau2 = np.max([0.0, (Q - df) / (s1 - s2 / s1)])
    wstar = 1 / (sigmas**2 + tau2)
    muhat = np.sum(values * wstar) / np.sum(wstar)
    return float(muhat), wstar, float(tau2)


def _validate_coverage(
    coverage: Optional[float],
    crit: Optional[float],
    dist: str = "z",
    df: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Validate and compute coverage/alpha for normal or t-distribution.
    
    Args:
        coverage: Desired coverage probability (e.g., 0.95)
        crit: Critical value (z-score for normal, t-score for t-distribution)
        dist: Distribution type, either "z" (normal) or "t" (t-distribution)
        df: Degrees of freedom (required when dist="t")
    
    Returns:
        Tuple of (coverage, alpha, tail_prob)
    """
    assert dist in ("z", "t"), f"dist must be 'z' or 't', got {dist}"
    if dist == "t":
        assert df is not None, "df is required when dist='t'"
    
    assert (coverage is None) != (crit is None), f"Exactly one of coverage or {dist}alpha must be provided"
    
    if crit is None:
        tail_prob = (1 - coverage) / 2
        if dist == "z":
            crit = float(np.abs(norm.ppf(tail_prob)))
        else:
            crit = float(np.abs(t.ppf(tail_prob, df)))
    else:
        if dist == "z":
            coverage = 1 - 2 * norm.sf(crit)
        else:
            coverage = 1 - 2 * t.sf(crit, df)
        tail_prob = (1 - coverage) / 2
    
    return coverage, crit, tail_prob


def random_effects_dl(
    values: np.ndarray,
    sigmas: np.ndarray,
    coverage: Optional[float] = None,
    zalpha: Optional[float] = None,
) -> MetaResult:
    coverage, zalpha, _ = _validate_coverage(coverage, zalpha, dist="z")
    muhat, wstar, tau2 = _random_effects_dl_base(values, sigmas)
    sigma = np.sqrt(1 / (np.sum(wstar)))
    interval = [muhat - zalpha * sigma, muhat + zalpha * sigma]
    return MetaResult(
        point_est=float(muhat),
        interval=interval,
        targeted_cov=coverage,
        actual_cov=coverage,
        sigma=float(sigma),
        tau=float(np.sqrt(tau2)),
    )


def random_effects_hksj(
    values: np.ndarray,
    sigmas: np.ndarray,
    coverage: Optional[float] = None,
    talpha: Optional[float] = None,
    trunc: str = "none",
) -> MetaResult:
    n = len(values)
    df = n - 1
    coverage, talpha, tail_prob = _validate_coverage(coverage, talpha, dist="t", df=df)
    muhat, wstar, tau2 = _random_effects_dl_base(values, sigmas)
    Qstar = np.sum(wstar * (values - muhat) ** 2)
    c2 = Qstar / df
    if trunc == "simple":
        c2 = np.max([c2, 1.0])
    elif trunc == "talpha":
        zalpha = float(np.abs(norm.ppf(tail_prob)))
        c2 = np.max([c2, (zalpha / talpha) ** 2])
    sigma2 = c2 / np.sum(wstar)
    sigma = np.sqrt(sigma2)
    interval = [muhat - talpha * sigma, muhat + talpha * sigma]
    return MetaResult(
        point_est=float(muhat),
        interval=interval,
        targeted_cov=coverage,
        actual_cov=coverage,
        sigma=float(sigma),
        tau=float(np.sqrt(tau2)),
        extra={"c2": float(c2)},
    )


def random_effects_mle(
    values: np.ndarray,
    sigmas: np.ndarray,
    coverage: Optional[float] = None,
    zalpha: Optional[float] = None,
    truth: Optional[float] = None,
) -> MetaResult:
    coverage, zalpha, _ = _validate_coverage(coverage, zalpha, dist="z")
    sigmas2 = sigmas**2
    tau2 = np.var(values)
    for _ in range(50):
        w = 1 / (sigmas2 + tau2)
        muhat = np.sum(w * values) / np.sum(w) if truth is None else truth
        tau2 = np.sum(w**2 * ((values - muhat) ** 2 - sigmas2)) / np.sum(w**2)
        tau2 = np.max([0.0, tau2])
    sigma = np.sqrt(1 / np.sum(w))
    interval = [muhat - zalpha * sigma, muhat + zalpha * sigma]
    return MetaResult(
        point_est=float(muhat),
        interval=interval,
        targeted_cov=coverage,
        actual_cov=coverage,
        sigma=float(sigma),
        tau=float(np.sqrt(tau2)),
    )


def random_effects_pm(
    values: np.ndarray,
    sigmas: np.ndarray,
    coverage: Optional[float] = None,
    zalpha: Optional[float] = None,
) -> MetaResult:
    coverage, zalpha, _ = _validate_coverage(coverage, zalpha, dist="z")
    n = len(values)
    sigmas2 = sigmas**2
    tau2 = 0.0
    for _ in range(100):
        w = 1 / (sigmas2 + tau2)
        muhat = np.sum(w * values) / np.sum(w)
        Ftau2 = np.sum(w * (values - muhat) ** 2) - (n - 1)
        if tau2 == 0 and Ftau2 < 0:
            break
        if np.isclose(Ftau2, 0):
            break
        delta = Ftau2 / np.sum(w**2 * (values - muhat) ** 2)
        tau2 += delta
    w = 1 / (sigmas2 + tau2)
    thetahat = np.sum(w * values) / np.sum(w)
    sigma = np.sqrt(1 / np.sum(w))
    interval = [thetahat - zalpha * sigma, thetahat + zalpha * sigma]
    return MetaResult(
        point_est=float(thetahat),
        interval=interval,
        targeted_cov=coverage,
        actual_cov=coverage,
        sigma=float(sigma),
        tau=float(np.sqrt(tau2)),
    )


def birge(
    values: np.ndarray,
    sigmas: np.ndarray,
    coverage: Optional[float] = None,
    dist: str = "normal",
    pdg: bool = False,
    codata: bool = False,
    mle: bool = False,
    truth: Optional[float] = None,
) -> MetaResult:
    assert not (codata and pdg), "codata and pdg cannot both be True"
    n = len(values)
    u = np.sqrt(1 / np.sum(1 / sigmas**2))
    wm = truth if truth is not None else u**2 * np.sum(values / sigmas**2)
    if not codata:
        if pdg:
            thresh = 3 * np.sqrt(n) * u
            good = sigmas < thresh
            if np.sum(good) <= 1:
                good = np.ones(n, dtype=bool)
        else:
            good = np.ones(n, dtype=bool)
        chi2 = np.sum((values[good] - wm) ** 2 / sigmas[good] ** 2)
        chat2 = chi2 / (np.sum(good) - 1) if not mle else chi2 / np.sum(good)
        chat = np.sqrt(chat2)
        chat = np.max((1.0, chat))
    else:
        resids_norm = (values - wm) / sigmas
        max_resid = np.max(np.abs(resids_norm))
        chat = max(max_resid / 2, 1.0)
    sigma = chat * u
    tail_prob = (1 - coverage) / 2
    zalpha = float(np.abs(norm.ppf(tail_prob)))
    talpha = float(np.abs(t.ppf(tail_prob, n - 1)))
    if dist == "normal":
        crit = zalpha
    elif dist == "t":
        crit = zalpha if chat == 1 else talpha
    else:
        raise ValueError(f"Invalid distribution: {dist}")
    interval = [wm - crit * sigma, wm + crit * sigma]
    return MetaResult(
        point_est=float(wm),
        interval=interval,
        targeted_cov=coverage,
        actual_cov=coverage,
        sigma=float(sigma),
        chat=float(chat),
    )


def vniim(
    values: np.ndarray,
    sigmas: np.ndarray,
    coverage: Optional[float] = None,
    zalpha: Optional[float] = None,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> MetaResult:
    assert coverage is not None or zalpha is not None
    if zalpha is None:
        tail_prob = (1 - coverage) / 2
        zalpha = float(np.abs(norm.ppf(tail_prob)))
    n = len(sigmas)
    sigmas2 = sigmas**2
    F = n - 1
    diff = 1.0
    base = birge(values, sigmas, coverage=0.95)
    wm = base.point_est
    rb = base.chat if base.chat is not None else 1.0
    ri2 = np.full(sigmas.shape, rb**2)
    resids2 = (wm - values) ** 2 / (sigmas2 * ri2)
    iter_n = 0
    while diff > tol and iter_n < max_iter:
        rhs = np.zeros(sigmas.shape)
        for i in range(n):
            rhs[i] = (resids2[i] / F) * np.sum(ri2 * (ri2 - 1))
        ri2 = (1 + np.sqrt(1 + 4 * rhs)) / 2
        w = 1 / (sigmas2 * ri2)
        wm = np.sum(values * w) / np.sum(w)
        resids2 = (wm - values) ** 2 / (sigmas2 * ri2)
        chi2 = np.sum(resids2)
        diff = np.abs(chi2 - F)
        iter_n += 1
    u = np.sqrt(1 / np.sum(1 / (sigmas2 * ri2)))
    interval = [wm - zalpha * u, wm + zalpha * u]
    cov = coverage if coverage is not None else 1 - 2 * norm.sf(zalpha)
    return MetaResult(
        point_est=float(wm),
        interval=interval,
        targeted_cov=cov,
        actual_cov=cov,
        sigma=float(u),
        extra={"iterations": iter_n},
    )


def binomial(
    values: np.ndarray,
    coverage: float,
    p: float = 0.5,
    cdf: Optional[np.ndarray] = None,
    shrink: Optional[str] = None,
) -> MetaResult:
    assert coverage is not None
    tail_prob = (1 - coverage) / 2
    n = len(values)
    
    if cdf is None:
        cdf = binom.cdf(np.arange(n + 1), n, p)
    assert len(cdf) == n + 1
    if not np.all(values[:-1] <= values[1:]):
        values = np.sort(values)
    idx_l = np.argmax(cdf > tail_prob) - 1
    tail_l = cdf[idx_l]
    idx_u = np.argmax(cdf >= 1 - tail_prob)
    tail_u = 1 - cdf[idx_u]
    assert idx_l <= idx_u, f"idx_l={idx_l} > idx_u={idx_u}"
    nominal_coverage = 1 - tail_l - tail_u
    assert nominal_coverage >= coverage
    bottom = values[idx_l]
    top = values[idx_u]
    interval = [bottom, top]
    if shrink is None or nominal_coverage == coverage:
        return MetaResult(
            point_est=float(np.median(values)),
            interval=interval,
            targeted_cov=coverage,
            actual_cov=float(nominal_coverage),
        )
    z_nominal_l = -norm.ppf(tail_l)
    z_nominal_u = -norm.ppf(tail_u)
    z_target = -norm.ppf(tail_prob)
    bottom2 = values[idx_l + 1]
    top2 = values[idx_u - 1]
    tail_l2 = cdf[idx_l + 1]
    tail_u2 = 1 - cdf[idx_u - 1]
    z_nominal_l2 = -norm.ppf(tail_l2)
    z_nominal_u2 = -norm.ppf(tail_u2)
    if shrink == "scale":
        halfwidth = (top - bottom) / 2
        middle = (bottom + top) / 2
        new_width_b = halfwidth * z_nominal_l / z_target
        new_width_u = halfwidth * z_nominal_u / z_target
        bottom_shrink = middle - new_width_b
        top_shrink = middle + new_width_u
    elif shrink == "center":
        median = values[np.argmin(np.abs(cdf - 0.5))]
        bottom_shrink = median - (median - bottom) * z_target / z_nominal_l
        top_shrink = median + (top - median) * z_target / z_nominal_u
    elif shrink == "cdf-interp":
        xspace = np.linspace(bottom, top, 100, endpoint=True)
        xspace = np.unique(np.concatenate([values[idx_l : idx_u + 1], xspace]))
        xspace = np.sort(xspace, kind="mergesort")
        cdf_interp = np.interp(xspace, values, cdf[:-1])
        bottom_shrink = xspace[np.argmax(cdf_interp > tail_prob) - 1]
        top_shrink = xspace[np.argmax(cdf_interp >= 1 - tail_prob)]
    elif shrink == "prob-linear":
        bottom_shrink = bottom + (bottom2 - bottom) * (tail_prob - tail_l) / (
            tail_l2 - tail_l
        )
        top_shrink = top - (top - top2) * (tail_prob - tail_u) / (tail_u2 - tail_u)
    elif shrink == "z-linear":
        bottom_shrink = bottom + (bottom2 - bottom) * (z_nominal_l - z_target) / (
            z_nominal_l - z_nominal_l2
        )
        top_shrink = top - (top - top2) * (z_nominal_u - z_target) / (
            z_nominal_u - z_nominal_u2
        )
    else:
        raise ValueError(f"Invalid shrink method: {shrink}")
    interval_shrink = [bottom_shrink, top_shrink]
    assert interval_shrink[0] >= interval[0]
    assert interval_shrink[1] <= interval[1]
    return MetaResult(
        point_est=float(np.median(values)),
        interval=interval_shrink,
        targeted_cov=coverage,
        actual_cov=np.nan, # unknown
        extra={"raw_interval": interval},
    )


def sign_rank(
    values: np.ndarray,
    coverage: float,
) -> MetaResult:
    if Rmath4 is None:
        raise ImportError("Rmath4 is required for sign_rank")
    n = len(values)
    alpha = 1 - coverage
    w = np.add.outer(values, values) / 2
    w = np.sort(w[np.tril_indices(w.shape[0], 0)])
    qu = int(Rmath4.qsignrank(alpha / 2, n, 0, 0))
    if qu == 0:
        qu = 1
    achieved_alpha = 2 * Rmath4.psignrank(qu - 1, n, 0, 0)
    if achieved_alpha > alpha:
        qu = qu + 1
        new_achieved_alpha = 2 * Rmath4.psignrank(qu - 1, n, 0, 0)
        achieved_alpha = new_achieved_alpha
    ql = int(n * (n + 1) / 2 - qu)
    lower = w[ql + 1 - 1]
    upper = w[qu - 1]
    interval = [lower, upper]
    return MetaResult(
        point_est=float(np.median(values)),
        interval=interval,
        targeted_cov=coverage,
        actual_cov=float(1 - achieved_alpha),
    )


def flip_interval(
    values: np.ndarray,
    coverage: float = 0.6827,
    mode: str = "median",
    boot: bool = False,
    max_iter: int = 1000,
) -> MetaResult:
    tail_prob = (1 - coverage) / 2
    rng = max(values) - min(values)
    l = min(values) - rng
    r = max(values) + rng
    pl = 1.0
    iter_n = 0
    while (r - l > rng * 1e-6 or pl > tail_prob) and iter_n < max_iter:
        m = (l + r) / 2
        pl = flip_test(values, h0=m, mode=mode, tail="lower", boot=boot)
        if pl < tail_prob:
            l = m
        else:
            r = m
        iter_n += 1
    lower = m
    if iter_n == max_iter:
        return MetaResult(
            point_est=float(np.nan),
            interval=[np.nan, np.nan],
            targeted_cov=coverage,
            actual_cov=float("nan"),
        )
    l = lower
    r = max(values) + rng
    pu = 1.0
    iter_n = 0
    while (r - l > rng * 1e-6 or pu > tail_prob) and iter_n < max_iter:
        m = (l + r) / 2
        pu = flip_test(values, h0=m, mode=mode, tail="upper", boot=boot)
        if pu < tail_prob:
            r = m
        else:
            l = m
        iter_n += 1
    upper = m
    if iter_n == max_iter:
        return MetaResult(
            point_est=float(np.nan),
            interval=[np.nan, np.nan],
            targeted_cov=coverage,
            actual_cov=float("nan"),
        )
    assert 1 - (pl + pu) >= coverage
    return MetaResult(
        point_est=float(np.median(values) if mode == "median" else np.mean(values)),
        interval=[lower, upper],
        targeted_cov=coverage,
        actual_cov=float(1 - (pl + pu)),
    )


def linear_pool(
    values: np.ndarray, sigmas: np.ndarray, coverage: float = 0.6827, gridn: int = 10000
) -> MetaResult:
    grid = np.linspace(
        min(values) - 2 * max(sigmas), max(values) + 2 * max(sigmas), gridn
    )
    cdfs = norm.cdf(grid[:, np.newaxis], loc=values, scale=sigmas)
    probs = np.mean(np.nan_to_num(cdfs, nan=0.0), axis=1)
    tail_prob = (1 - coverage) / 2
    assert np.any(probs > tail_prob)
    assert np.any(probs >= 1 - tail_prob)
    idx_l = np.argmax(probs > tail_prob) - 1
    idx_u = np.argmax(probs >= 1 - tail_prob)
    assert idx_l <= idx_u
    achieved_coverage = 1 - probs[idx_l] - (1 - probs[idx_u])
    return MetaResult(
        point_est=float(np.mean(values)),
        interval=[grid[idx_l], grid[idx_u]],
        targeted_cov=coverage,
        actual_cov=float(achieved_coverage),
    )


def birge_forecast(
    values: np.ndarray,
    sigmas: np.ndarray,
    coverage: float = 0.6827,
    chat: Optional[float] = None,
) -> MetaResult:
    n = len(values)
    sigmas2 = sigmas**2
    u = np.sqrt(1 / np.sum(1 / sigmas2))
    wm = u**2 * np.sum(values / sigmas2)
    if chat is None:
        chi2 = np.sum((values - wm) ** 2 / sigmas2)
        chat2 = chi2 / (n - 1)
        chat = np.sqrt(chat2)
    sigma_m = chat * u
    if chat < 1:
        sigmas_r2 = (1 - chat**2) * sigmas2
        sigma_r2 = np.median(sigmas_r2)
    else:
        sigma_r2 = 0.0
    sigma = np.sqrt(sigma_m**2 + sigma_r2)
    interval = [wm - sigma, wm + sigma]
    return MetaResult(
        point_est=float(wm),
        interval=interval,
        targeted_cov=coverage,
        actual_cov=coverage,
        sigma=float(sigma),
        chat=float(chat),
        extra={"sigma_random": float(np.sqrt(sigma_r2))},
    )


def boot(
    values: np.ndarray,
    sigmas: np.ndarray,
    coverage: float = 0.6827,
    which: str = "normal",
) -> MetaResult:
    n = len(values)
    B = 100
    w = 1 / sigmas**2
    tail_prob = (1 - coverage) / 2
    ybar = np.sum(values * w) / np.sum(w)
    sigmahat = np.sqrt(1 / np.sum(w))
    ybars = np.full(B, np.nan)
    sigmahats = np.full(B, np.nan)
    for b in range(B):
        sample = np.random.choice(n, size=n, replace=True)
        ybars[b] = np.sum(values[sample] * w[sample]) / np.sum(w[sample])
        sigmahats[b] = np.sqrt(1 / np.sum(w[sample]))
    if which == "normal":
        sigmahat = np.std(ybars)
        zalpha = float(np.abs(norm.ppf(tail_prob)))
        interval = [ybar - zalpha * sigmahat, ybar + zalpha * sigmahat]
    elif which == "studentized":
        tstar = (ybars - ybar) / sigmahats
        interval = list(np.quantile(ybar - tstar * sigmahat, [tail_prob, 1 - tail_prob]))
    else:
        raise ValueError("which must be 'normal' or 'studentized'")
    return MetaResult(
        point_est=float(ybar),
        interval=interval,
        targeted_cov=coverage,
        actual_cov=coverage,
        sigma=float(sigmahat),
    )

