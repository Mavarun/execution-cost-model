import numpy as np
import pytest

from execution_cost.simulator import MarketParams, simulate_market, volume_profile
from execution_cost.volume_model import (
    DayFactorNowcast,
    fit_nowcast,
    fit_profile,
    lag1_forecast,
    log_forecast_errors,
)


def test_fitted_profile_recovers_the_u_shape():
    mp = MarketParams()
    mk = simulate_market(mp, 3000, seed=7)
    u = fit_profile(mk.volumes)
    assert u.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(u, volume_profile(mp.n_intervals, mp.u_shape), rtol=0.03)


def test_day_factor_nowcast_without_noise_is_exact():
    u = volume_profile(10, 1.5)
    adv = 1e6
    vols = adv * u[None, :] * np.array([[0.5], [1.0], [2.0]])
    m = DayFactorNowcast(u, adv, k0=1e-9)
    for j in range(1, 10):
        np.testing.assert_allclose(m.day_factor(vols[:, :j]), [0.5, 1.0, 2.0], rtol=1e-6)
        np.testing.assert_allclose(m.forecast(vols[:, :j], j), vols[:, j], rtol=1e-6)


def test_no_observation_forecast_is_the_prior():
    u = volume_profile(10, 1.5)
    m = DayFactorNowcast(u, 1e6, k0=2.0)
    np.testing.assert_allclose(m.day_factor(np.zeros((4, 0))), 1.0)
    assert m.remaining_volume(np.zeros((4, 0))).shape == (4, 10)


def test_forecast_never_uses_the_current_or_future_interval():
    mp = MarketParams()
    v = simulate_market(mp, 100, seed=1).volumes
    m = DayFactorNowcast(fit_profile(v), mp.adv, 2.0)
    a = log_forecast_errors(v, m.forecast)
    w = v.copy()
    w[:, 12:] *= 5.0
    b = log_forecast_errors(w, m.forecast)
    np.testing.assert_allclose(a[:, :12], b[:, :12])


def test_nowcast_beats_lag1_out_of_sample_on_normal_days():
    mp = MarketParams()
    train = simulate_market(mp, 500, seed=11).volumes
    test = simulate_market(mp, 2000, seed=12).volumes
    m, info = fit_nowcast(train, mp.adv)
    assert info["k0"] in info["k0_mse"]
    e_now = log_forecast_errors(test, m.forecast)[:, 1:]
    e_lag = log_forecast_errors(test, lambda o, j: lag1_forecast(o, j, m.profile, mp.adv))[:, 1:]
    assert np.sqrt(np.mean(e_now**2)) < 0.8 * np.sqrt(np.mean(e_lag**2))
