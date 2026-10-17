import numpy as np

from execution_cost.adaptive import AdaptiveSqrtOpt, first_leg_fraction
from execution_cost.optimal_sqrt import sqrt_schedule
from execution_cost.simulator import MarketParams, execute, simulate_market, volume_profile
from execution_cost.volume_model import DayFactorNowcast

MP = MarketParams()
X = 0.05 * MP.adv
LAM = 2e-6


def _policy(mp=MP, adapt=True, k0=2.0):
    u = volume_profile(mp.n_intervals, mp.u_shape)
    return AdaptiveSqrtOpt(mp, DayFactorNowcast(u, mp.adv, k0), X, LAM, mp.temp_coeff, mp.perm_coeff, adapt=adapt)


def test_first_leg_at_interval_zero_matches_the_static_optimum():
    u = volume_profile(MP.n_intervals, MP.u_shape)
    static = sqrt_schedule(MP, X, LAM)
    f = first_leg_fraction(MP, u, 0, X, LAM, MP.temp_coeff, MP.perm_coeff)
    assert abs(f - static[0] / X) < 1e-6


def test_noise_free_day_replays_the_static_schedule():
    """Mean-variance with known volume is time consistent: re-solving changes nothing."""
    mp = MarketParams(volume_noise=0.0, day_volume_noise=0.0)
    mk = simulate_market(mp, 4, seed=0)
    res = execute(_policy(mp), mk, mp, X)
    static = sqrt_schedule(mp, X, LAM)
    np.testing.assert_allclose(res.trades[0], static, rtol=0.02, atol=2e-3 * X)
    np.testing.assert_allclose(res.executed, X)


def test_trades_faster_on_a_heavy_day_and_slower_on_a_thin_day():
    pol = _policy()
    left = np.full(2, X * 0.8)
    f = pol.fraction(5, left, np.array([0.6, 1.6]))
    assert f[1] > f[0]


def test_always_completes_and_never_negative():
    mk = simulate_market(MP, 300, seed=4)
    res = execute(_policy(), mk, MP, X)
    np.testing.assert_allclose(res.executed, X)
    assert (res.trades >= -1e-9).all()


def test_non_adaptive_variant_ignores_volume():
    mk = simulate_market(MP, 50, seed=4)
    a = execute(_policy(adapt=False), mk, MP, X)
    mk.volumes[:, :] *= 2.0
    b = execute(_policy(adapt=False), mk, MP, X)
    np.testing.assert_allclose(a.trades, b.trades)
