"""CusumGateParams.on_frac: a lower re-entry level than the off alarm (slice 9)."""

import numpy as np
import pytest

from execution_cost.decay_gate import CusumGateParams, cusum_gate_states


def _shadow(seed=0, n=6000, start=1260, mu=3e-4, sd=4e-3, dead=(2000, 2600)):
    rng = np.random.default_rng(seed)
    x = rng.normal(mu, sd, n)
    x[start + dead[0]:start + dead[1]] = rng.normal(0.0, sd, dead[1] - dead[0])
    return x, start


def test_on_frac_one_is_the_slice8_gate():
    x, s = _shadow()
    a = cusum_gate_states(x, s, CusumGateParams(shrink=0.75))
    b = cusum_gate_states(x, s, CusumGateParams(shrink=0.75, on_frac=1.0))
    assert np.array_equal(a, b)


def test_lower_reentry_raises_the_on_share_and_keeps_the_first_alarm():
    for seed in range(5):
        x, s = _shadow(seed)
        full = cusum_gate_states(x, s, CusumGateParams(h=8.0))
        fast = cusum_gate_states(x, s, CusumGateParams(h=8.0, on_frac=0.25))
        assert fast.mean() >= full.mean()
        # off alarms use the same level, so the first switch-off is the same bar
        assert np.argmax(full == 0) == np.argmax(fast == 0)


def test_label_and_validation():
    assert CusumGateParams(shrink=0.75, on_frac=0.25).label() == "CARL2520s0.75r0.25"
    assert CusumGateParams().label() == "CARL2520"
    x, s = _shadow()
    for bad in (0.0, 1.5):
        with pytest.raises(ValueError):
            cusum_gate_states(x, s, CusumGateParams(on_frac=bad))
