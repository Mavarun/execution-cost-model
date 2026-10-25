import numpy as np
import pytest

from execution_cost.adaptive import AdaptiveSqrtOpt, PriceAdaptiveSqrtOpt
from execution_cost.adaptive_eval import static_sqrt_on_profile
from execution_cost.simulator import MarketParams, execute, simulate_market
from execution_cost.volume_model import fit_nowcast

MP = MarketParams()
X = 0.05 * MP.adv
Y, G, LAM = 0.6, 0.3, 2e-5


@pytest.fixture(scope="module")
def setup():
    nowcast, _ = fit_nowcast(simulate_market(MP, 200, seed=7).volumes, MP.adv)
    base = PriceAdaptiveSqrtOpt(MP, nowcast, X, LAM, Y, G, n_grid=24)
    return nowcast, base


def clone(base, a):
    pol = object.__new__(PriceAdaptiveSqrtOpt)
    pol.__dict__.update(base.__dict__)
    pol.a = a
    return pol


def test_a_zero_is_the_plan_only_replan_on_the_same_grid(setup):
    nowcast, base = setup
    mk = simulate_market(MP, 300, seed=11)
    replan = AdaptiveSqrtOpt(MP, nowcast, X, LAM, Y, G, n_grid=24, adapt=False, grid_span=float(np.exp(2)))
    a, b = execute(clone(base, 0.0), mk, MP, X), execute(replan, mk, MP, X)
    np.testing.assert_allclose(a.trades, b.trades)
    static = static_sqrt_on_profile(MP, nowcast.profile, X, LAM, Y, G)
    np.testing.assert_allclose(a.trades.mean(0), static, rtol=0.03, atol=X * 2e-3)


def test_multiplier_sign_own_impact_removed_and_clipped(setup):
    _, base = setup
    pol = clone(base, 1.0)
    R = np.array([X / 2] * 3)
    own = G * MP.sigma_price * (X / 2) / MP.adv
    s = MP.sigma_price * np.sqrt(13 / MP.n_intervals)
    m = pol.multiplier(13, own + np.array([-s, 0.0, s]), R)
    np.testing.assert_allclose(m, np.exp([1.0, 0.0, -1.0]))
    assert pol.multiplier(13, np.array([-100 * s]), R[:1])[0] == pytest.approx(pol.m_max)
    np.testing.assert_allclose(pol.multiplier(0, np.array([5.0]), R[:1]), 1.0)


def test_aim_trades_faster_in_the_money_and_pim_slower(setup):
    _, base = setup
    R = np.full(3, X / 2)
    s = MP.sigma_price * np.sqrt(10 / MP.n_intervals)
    own = G * MP.sigma_price * (X / 2) / MP.adv
    moves = own + np.array([-s, 0.0, s])  # favourable, flat, adverse for the order
    obs = np.full((3, 10), MP.adv / MP.n_intervals)
    aim = clone(base, 0.5).shares(10, obs, R.copy(), MP, moves)
    pim = clone(base, -0.5).shares(10, obs, R.copy(), MP, moves)
    assert aim[0] > aim[1] > aim[2]
    assert pim[0] < pim[1] < pim[2]
    assert aim[1] == pytest.approx(pim[1])


def test_completes_every_day_and_never_sees_later_prices(setup):
    _, base = setup
    mk = simulate_market(MP, 150, seed=12)
    pol = clone(base, 0.5)
    r1 = execute(pol, mk, MP, X)
    np.testing.assert_allclose(r1.executed, X)
    mk.noise_increments[:, 15:] *= -2.0
    r2 = execute(pol, mk, MP, X)
    np.testing.assert_allclose(r1.trades[:, :16], r2.trades[:, :16])
    assert not np.allclose(r1.trades[:, 16:], r2.trades[:, 16:])


def test_grid_span_below_one_is_rejected(setup):
    nowcast, _ = setup
    with pytest.raises(ValueError):
        AdaptiveSqrtOpt(MP, nowcast, X, LAM, Y, G, n_grid=4, grid_span=0.5)
