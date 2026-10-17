import numpy as np
import pytest

from execution_cost.simulator import EventParams, MarketParams, simulate_market


def test_no_events_is_unchanged_and_normal_days_identical():
    mp = MarketParams()
    base = simulate_market(mp, 500, seed=3)
    ev = simulate_market(mp, 500, seed=3, events=EventParams(prob=0.3))
    assert base.is_event is None
    normal = ~ev.is_event
    np.testing.assert_array_equal(ev.volumes[normal], base.volumes[normal])
    np.testing.assert_array_equal(ev.noise_increments, base.noise_increments)
    assert (ev.event_start[normal] == -1).all()


def test_event_days_surge_exactly_in_their_window():
    mp = MarketParams()
    e = EventParams(prob=1.0, surge=4.0, length=3, decay=0.5)
    base = simulate_market(mp, 200, seed=5)
    ev = simulate_market(mp, 200, seed=5, events=e)
    ratio = ev.volumes / base.volumes
    for p in range(200):
        s = ev.event_start[p]
        assert e.min_start <= s <= mp.n_intervals - e.length
        np.testing.assert_allclose(ratio[p, s : s + 3], [4.0, 2.5, 1.75])
        mask = np.ones(mp.n_intervals, bool)
        mask[s : s + 3] = False
        np.testing.assert_allclose(ratio[p, mask], 1.0)


def test_event_share_matches_prob():
    ev = simulate_market(MarketParams(), 20000, seed=1, events=EventParams(prob=0.15))
    assert abs(ev.is_event.mean() - 0.15) < 0.01


def test_bad_event_params():
    with pytest.raises(ValueError):
        EventParams(prob=1.5)
    with pytest.raises(ValueError):
        EventParams(surge=0.5)
    with pytest.raises(ValueError):
        simulate_market(MarketParams(n_intervals=4), 10, events=EventParams(length=4, min_start=2))
