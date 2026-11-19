import numpy as np
import pytest

from execution_cost.alpha import AlphaParams, simulate_alpha_path
from execution_cost.decay_gate import (
    CusumGateParams, GateParams, cusum_gate_states, gate_states, gated_walk_forward, gated_walk_forward_states,
    shadow_record, siegmund_arl, siegmund_h,
)
from execution_cost.walkforward import run_rules, walk_forward
from execution_cost.walkforward_eval import candidate_rules, cost_params


def _alive_then_dead(n_alive=3000, n_dead=3000, mu=0.0004, sd=0.006, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(mu, sd, n_alive), rng.normal(0.0, sd, n_dead)])


def test_siegmund_h_hits_the_arl_target_and_grows_as_the_signal_weakens():
    for delta in (0.02, 0.05, 0.2):
        h = siegmund_h(delta, 2520)
        assert siegmund_arl(h, delta) == pytest.approx(2520, rel=1e-6)
    assert siegmund_h(0.02, 2520) > siegmund_h(0.05, 2520) > siegmund_h(0.2, 2520)
    # a textbook h = 5 has an in-control ARL of weeks at a daily alpha Sharpe of ~0.07
    assert siegmund_arl(5.0, 0.0333) < 60


def test_on_more_while_alive_than_after_death_on_average():
    alive, dead = [], []
    for s in range(1, 9):
        on = cusum_gate_states(_alive_then_dead(seed=s), 1500, CusumGateParams())
        alive.append(on[:1500].mean())
        dead.append(on[1500:].mean())
    assert np.mean(alive) > 0.75 and np.mean(dead) < np.mean(alive) - 0.25


def test_is_causal():
    x = _alive_then_dead(seed=2)
    a = cusum_gate_states(x, 1500, CusumGateParams())
    y = x.copy()
    y[2600:] += 1.0  # change the future only
    b = cusum_gate_states(y, 1500, CusumGateParams())
    np.testing.assert_array_equal(a[: 2600 - 1500 + 1], b[: 2600 - 1500 + 1])


def test_higher_h_switches_less():
    x = _alive_then_dead(seed=3)
    flips = [np.abs(np.diff(cusum_gate_states(x, 1500, CusumGateParams(h=h)))).sum() for h in (5.0, 15.0, 40.0)]
    assert flips[0] >= flips[1] >= flips[2]


def test_nonpositive_reference_keeps_gate_on_and_validation():
    x = np.random.default_rng(4).normal(-0.001, 0.01, 3000)
    assert (cusum_gate_states(x, 1500, CusumGateParams()) == 1.0).all()
    with pytest.raises(ValueError):
        cusum_gate_states(x, 100, CusumGateParams(ref_bars=1260))
    with pytest.raises(ValueError):
        cusum_gate_states(x, 1500, CusumGateParams(h=0.0))
    with pytest.raises(ValueError):
        siegmund_h(0.0, 2520)


def test_state_replay_matches_trailing_gate_replay():
    path = simulate_alpha_path(AlphaParams(n=1600), seed=3)
    rules = candidate_rules()["WF-PARTIAL"]
    costs = cost_params()
    runs = run_rules(path, rules, costs)
    wf = walk_forward(path, rules, costs, min_train=600, test_len=250, runs=runs)
    gp = GateParams(lookback=252)
    ref = gated_walk_forward(path, wf, runs, costs, gp)
    on = gate_states(shadow_record(wf, runs), wf.fold_starts[0], gp)
    got = gated_walk_forward_states(path, wf, runs, costs, on)
    np.testing.assert_allclose(got.net, ref.net)
