import numpy as np
import pytest

from execution_cost.alpha import AlphaParams, simulate_alpha_path
from execution_cost.rebalance import Rule, net_returns
from execution_cost.vol_target import TRADING_DAYS, _train_std, solve_leverage, vol_targeted
from execution_cost.walkforward import run_rules, walk_forward
from execution_cost.walkforward_eval import cost_params

RULES = [Rule("PARTIAL", rate=a) for a in (0.3, 0.1)]


@pytest.fixture(scope="module")
def setup():
    path = simulate_alpha_path(AlphaParams(n=1500), seed=5)
    costs = cost_params()
    runs = run_rules(path, RULES, costs)
    wf = walk_forward(path, RULES, costs, min_train=500, test_len=250, runs=runs)
    return path, costs, runs, wf


def test_solved_leverage_hits_training_target(setup):
    path, costs, runs, wf = setup
    target = 0.10 / np.sqrt(TRADING_DAYS)
    k = solve_leverage(path, runs[0], 1, 500, costs, target)
    assert _train_std(path, runs[0].weights, 1, 500, costs, k) == pytest.approx(target, rel=1e-4)


def test_leverage_uses_training_bars_only(setup):
    path, costs, runs, wf = setup
    before = vol_targeted(path, wf, runs, costs).leverage[0]
    scrambled = path.__class__(**{**path.__dict__, "returns": np.r_[path.returns[:500], -path.returns[500:]]})
    runs2 = run_rules(scrambled, RULES, costs)
    wf2 = walk_forward(scrambled, RULES, costs, min_train=500, test_len=250, runs=runs2)
    assert vol_targeted(scrambled, wf2, runs2, costs).leverage[0] == pytest.approx(before)


def test_unit_leverage_bracket_reproduces_costs(setup):
    path, costs, runs, wf = setup
    vt = vol_targeted(path, wf, runs, costs)
    lev = np.repeat(vt.leverage, [250, 250, 250, 250])[: wf.weights.size]
    _, c, _ = net_returns(path.returns[500:], lev * wf.weights, costs, daily_vol=path.daily_vol[500:])
    np.testing.assert_allclose(vt.cost, c)
    assert all(0.05 <= k <= 20.0 for k in vt.leverage)


def test_impossible_targets_clip_to_bracket(setup):
    path, costs, runs, _ = setup
    assert solve_leverage(path, runs[0], 1, 500, costs, 1e-9) == 0.05
    assert solve_leverage(path, runs[0], 1, 500, costs, 10.0) == 20.0
