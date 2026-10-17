import numpy as np

from execution_cost.schedules import twap_schedule
from execution_cost.simulator import MarketParams, execute, simulate_market


class StaticAsPolicy:
    def __init__(self, sched):
        self.sched = np.asarray(sched, float)
        self.seen = []

    def shares(self, j, observed, remaining, params):
        self.seen.append(observed.shape)
        return np.full(remaining.shape, self.sched[j])


class PeekPolicy:
    """Trades in proportion to the last observed volume."""

    def shares(self, j, observed, remaining, params):
        if j == 0:
            return np.zeros_like(remaining)
        if j == params.n_intervals - 1:
            return remaining
        return 0.05 * observed[:, -1]


def test_policy_reproducing_a_static_schedule_matches_it_exactly():
    mp = MarketParams()
    mk = simulate_market(mp, 300, seed=9)
    X = 0.05 * mp.adv
    tw = twap_schedule(X, mp.n_intervals)
    pol = StaticAsPolicy(tw)
    a = execute(tw, mk, mp, X)
    b = execute(pol, mk, mp, X)
    np.testing.assert_allclose(a.fill_prices, b.fill_prices)
    np.testing.assert_allclose(a.trades, b.trades)
    # observed block grows one interval at a time: never sees interval j
    assert pol.seen == [(300, j) for j in range(mp.n_intervals)]


def test_policy_cannot_see_future_volume():
    mp = MarketParams()
    mk = simulate_market(mp, 50, seed=2)
    X = 0.05 * mp.adv
    a = execute(PeekPolicy(), mk, mp, X)
    mk.volumes[:, 10:] *= 3.0
    b = execute(PeekPolicy(), mk, mp, X)
    np.testing.assert_allclose(a.trades[:, :11], b.trades[:, :11])
    np.testing.assert_allclose(a.executed, X)
