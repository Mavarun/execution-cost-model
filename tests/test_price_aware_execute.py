import numpy as np

from execution_cost.schedules import twap_schedule
from execution_cost.simulator import MarketParams, execute, simulate_market


class Recorder:
    uses_prices = True

    def __init__(self, sched):
        self.sched = np.asarray(sched, float)
        self.moves = []

    def shares(self, j, observed, remaining, params, mid_move):
        self.moves.append(np.array(mid_move, copy=True))
        return np.full(remaining.shape, self.sched[j])


class BuyTheDip:
    """Front-loads when the mid has fallen (a price-reactive toy)."""

    uses_prices = True

    def shares(self, j, observed, remaining, params, mid_move):
        if j == params.n_intervals - 1:
            return remaining
        return np.where(mid_move < 0, 0.2, 0.02) * remaining


def test_mid_move_is_the_signed_mid_before_trading_and_static_replay_is_exact():
    mp = MarketParams()
    mk = simulate_market(mp, 200, seed=4)
    X = 0.05 * mp.adv
    tw = twap_schedule(X, mp.n_intervals)
    for side in (1, -1):
        rec = Recorder(tw)
        a, b = execute(tw, mk, mp, X, side=side), execute(rec, mk, mp, X, side=side)
        np.testing.assert_allclose(a.fill_prices, b.fill_prices)
        moves = np.stack(rec.moves, axis=1)
        np.testing.assert_allclose(moves, side * (b.mids - mp.s0))
        np.testing.assert_allclose(moves[:, 0], 0.0)


def test_price_policy_cannot_see_future_prices():
    mp = MarketParams()
    mk = simulate_market(mp, 100, seed=5)
    X = 0.05 * mp.adv
    a = execute(BuyTheDip(), mk, mp, X)
    mk.noise_increments[:, 12:] *= -3.0  # changes mids from interval 13 on
    b = execute(BuyTheDip(), mk, mp, X)
    np.testing.assert_allclose(a.trades[:, :13], b.trades[:, :13])
    assert not np.allclose(a.trades[:, 13:], b.trades[:, 13:])
    np.testing.assert_allclose(b.executed, X)


def test_volume_only_policies_keep_their_four_argument_signature():
    class VolOnly:
        def shares(self, j, observed, remaining, params):
            return remaining if j == params.n_intervals - 1 else 0.04 * remaining

    mp = MarketParams()
    r = execute(VolOnly(), simulate_market(mp, 20, seed=1), mp, 1000.0)
    np.testing.assert_allclose(r.executed, 1000.0)
