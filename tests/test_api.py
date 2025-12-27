import numpy as np
import pytest

import metastat as ms

try:
    import Rmath4  # type: ignore
except Exception:
    Rmath4 = None


def toy_data():
    values = np.array([0, 1, 2])
    sigmas = np.array([0.1, 0.1, 0.1])
    return values, sigmas


def test_basics():
    values, sigmas = toy_data()
    for infer_func in [ms.infer.fixed_effect, ms.infer.random_effects_dl, ms.infer.random_effects_hksj, ms.infer.random_effects_mle, ms.infer.random_effects_pm, ms.infer.birge, ms.infer.vniim]:
        res = infer_func(values, sigmas, coverage=0.95)
        assert res.interval[0] < res.point_est < res.interval[1]
        if res.sigma is not None:
            assert res.sigma > 0
        if res.tau is not None:
            assert res.tau >= 0
        if res.chat is not None:
            assert res.chat >= 1
    for infer_func in [ms.infer.binomial, ms.infer.sign_rank]:
        res = infer_func(values, coverage=0.6827)
        assert res.interval[0] <= res.point_est <= res.interval[1]


def test_birge():
    values, sigmas = toy_data()
    res_br = ms.infer.birge(values, sigmas, coverage=0.95)
    res_fe = ms.infer.fixed_effect(values, sigmas, coverage=0.95)
    br = ms.stat.br(values, sigmas)
    assert np.isclose(res_br.point_est, res_fe.point_est)
    assert np.isclose(br, res_br.chat)


def test_I2():
    values, sigmas = toy_data()
    i2 = ms.stat.I2(values, sigmas)
    assert 0 <= i2 <= 1


@pytest.mark.skipif(Rmath4 is None, reason="Rmath4 not installed")
def test_sign_rank_optional():
    values, sigmas = toy_data()
    res = ms.infer.sign_rank(values, coverage=0.5)
    assert res.actual_cov <= 1
    assert res.interval[0] <= res.interval[1]

