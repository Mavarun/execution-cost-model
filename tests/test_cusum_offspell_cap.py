import numpy as np
import pytest

from execution_cost.decay_gate import CusumGateParams, cusum_gate_states


def _alive_then_dead(n_alive=3000, n_dead=3000, mu=0.0004, sd=0.006, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(mu, sd, n_alive), rng.normal(0.0, sd, n_dead)])


def _spells(on, value):
    d = np.diff(np.r_[1 - value, on == value, 1 - value].astype(int))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return ends - starts


def test_default_is_unchanged():
    x = _alive_then_dead(seed=3)
    a = cusum_gate_states(x, 1500, CusumGateParams(shrink=0.75))
    b = cusum_gate_states(x, 1500, CusumGateParams(shrink=0.75, max_off=None))
    assert np.array_equal(a, b)
    assert CusumGateParams().max_off is None and CusumGateParams(max_off=126).label().endswith("c126")


def test_no_off_spell_exceeds_the_cap_and_the_gate_still_turns_off_after_death():
    dead_off_cap, dead_off_base = [], []
    for s in range(1, 6):
        x = _alive_then_dead(seed=s)
        on = cusum_gate_states(x, 1500, CusumGateParams(shrink=0.75, max_off=126))
        off = _spells(on, 0.0)
        assert off.size == 0 or off.max() <= 126
        dead_off_cap.append((on[1500:] == 0).mean())
        dead_off_base.append((cusum_gate_states(x, 1500, CusumGateParams(shrink=0.75))[1500:] == 0).mean())
    # it still spends time off after the alpha dies, but less than the uncapped gate
    assert 0 < np.mean(dead_off_cap) < np.mean(dead_off_base)


def test_cap_never_shortens_an_on_spell_decision_it_only_ends_off_spells():
    x = _alive_then_dead(seed=7)
    base = cusum_gate_states(x, 1500, CusumGateParams(shrink=0.75))
    cap = cusum_gate_states(x, 1500, CusumGateParams(shrink=0.75, max_off=10_000))
    assert np.array_equal(base, cap)  # a cap longer than the sample changes nothing


def test_bad_cap_raises():
    with pytest.raises(ValueError):
        cusum_gate_states(_alive_then_dead(), 1500, CusumGateParams(max_off=0))
