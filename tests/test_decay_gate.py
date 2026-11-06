import numpy as np
import pytest

from execution_cost.alpha import AlphaParams, simulate_alpha_path
from execution_cost.decay_gate import GateParams, gate_states, gated_walk_forward, shadow_record
from execution_cost.rebalance import Rule
from execution_cost.walkforward import run_rules, walk_forward
from execution_cost.walkforward_eval import cost_params

RULES = [Rule("PARTIAL", rate=a) for a in (0.3, 0.1)]


@pytest.fixture(scope="module")
def setup():
    path = simulate_alpha_path(AlphaParams(n=1500), seed=3)
    costs = cost_params()
    runs = run_rules(path, RULES, costs)
    wf = walk_forward(path, RULES, costs, min_train=500, test_len=250, runs=runs)
    return path, costs, runs, wf


def test_gate_states_use_only_past_bars():
    shadow = np.r_[np.full(10, 1.0), np.full(10, -1.0)]
    on = gate_states(shadow, 5, GateParams(lookback=5))
    assert on[:6].tolist() == [1.0] * 6  # bars 5..10: trailing window still all positive
    assert on[-1] == 0.0
    # changing the shadow at or after t never changes the state at t
    s2 = shadow.copy()
    s2[12:] = 100.0
    assert gate_states(s2, 5, GateParams(lookback=5))[:8].tolist() == on[:8].tolist()


def test_gate_validation():
    with pytest.raises(ValueError):
        gate_states(np.zeros(10), 3, GateParams(lookback=5))
    with pytest.raises(ValueError):
        gate_states(np.zeros(10), 5, GateParams(lookback=1))


def test_always_on_gate_reproduces_the_walk_forward(setup):
    path, costs, runs, wf = setup
    g = gated_walk_forward(path, wf, runs, costs, GateParams(lookback=50, threshold=-np.inf))
    assert g.share_on == 1.0
    np.testing.assert_allclose(g.net, wf.net)
    np.testing.assert_allclose(g.weights, wf.weights)


def test_always_off_gate_goes_flat_and_pays_nothing(setup):
    path, costs, runs, wf = setup
    g = gated_walk_forward(path, wf, runs, costs, GateParams(lookback=50, threshold=np.inf))
    assert g.share_on == 0.0
    assert np.all(g.weights == 0.0) and np.all(g.cost == 0.0)


def test_shadow_head_is_first_choice_in_sample(setup):
    path, costs, runs, wf = setup
    sh = shadow_record(wf, runs)
    first = next(r for r in runs if r.rule.label == wf.choices[0])
    assert sh.size == path.returns.size
    np.testing.assert_allclose(sh[:500], first.net[:500])
    np.testing.assert_allclose(sh[500:], wf.net)
