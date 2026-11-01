"""Walk-forward selection: fold geometry, no look-ahead, carried positions."""

from __future__ import annotations

import numpy as np
import pytest

from execution_cost.alpha import AlphaParams, simulate_alpha_path
from execution_cost.costs import CostParams
from execution_cost.rebalance import Rule, net_returns
from execution_cost.sharpe_stats import per_bar_sharpe
from execution_cost.walkforward import (
    fixed_rule_oos,
    fold_starts,
    run_rules,
    select_on_window,
    walk_forward,
)

RULES = [Rule("FULL"), Rule("PARTIAL", rate=0.2), Rule("BAND", width=0.3), Rule("PARTIAL+BAND", rate=0.5, width=0.2)]


@pytest.fixture(scope="module")
def path():
    return simulate_alpha_path(AlphaParams(n=1500), seed=42)


def test_fold_starts_cover_the_tail_without_overlap():
    assert fold_starts(1000, min_train=400, test_len=250) == [400, 650, 900]
    with pytest.raises(ValueError):
        fold_starts(100, min_train=100, test_len=10)


def test_oos_record_covers_exactly_the_bars_after_min_train(path):
    res = walk_forward(path, RULES, CostParams(), min_train=600, test_len=250)
    assert res.fold_starts == [600, 850, 1100, 1350]
    np.testing.assert_array_equal(res.oos_index, np.arange(600, 1500))
    assert res.net.size == res.gross.size == res.weights.size == 900
    np.testing.assert_allclose(res.net, res.gross - res.cost)
    assert len(res.choices) == 4


def test_selection_uses_only_training_bars(path):
    """Scrambling the returns after a fold start cannot change that fold's choice."""
    params = CostParams()
    base = walk_forward(path, RULES, params, min_train=600, test_len=250)
    rng = np.random.default_rng(0)
    returns = path.returns.copy()
    returns[850:] = rng.permutation(returns[850:]) * 3.0
    tampered = type(path)(
        returns=returns, target=path.target, factor=path.factor, signal=path.signal,
        daily_vol=path.daily_vol, params=path.params, seed=path.seed,
    )
    other = walk_forward(tampered, RULES, params, min_train=600, test_len=250)
    assert other.choices[:2] == base.choices[:2]
    # Positions in those folds depend only on targets and the held position.
    np.testing.assert_array_equal(other.weights[:500], base.weights[:500])


def test_choice_is_the_training_argmax(path):
    params = CostParams()
    runs = run_rules(path, RULES, params)
    res = walk_forward(path, RULES, params, min_train=600, test_len=250, runs=runs)
    for s, label in zip(res.fold_starts, res.choices):
        best = max(runs, key=lambda r: per_bar_sharpe(r.net[1:s]))
        assert best.rule.label == label
    k, scores = select_on_window(runs, 600)
    assert set(scores) == {r.rule.label for r in runs}


def test_single_rule_walk_forward_equals_the_fixed_rule(path):
    params = CostParams()
    rule = Rule("PARTIAL+BAND", rate=0.5, width=0.2)
    runs = run_rules(path, [rule], params)
    wf = walk_forward(path, [rule], params, min_train=600, test_len=250, runs=runs)
    fx = fixed_rule_oos(runs[0], 600)
    # Position state carries across folds, so the OOS weights match a single
    # pass; only the first OOS bar differs (WF enters from flat, the fixed run
    # is already invested).
    np.testing.assert_allclose(wf.weights[300:], fx["weights"][300:], atol=1e-12)
    np.testing.assert_allclose(wf.net[300:], fx["net"][300:], atol=1e-15)


def test_carried_position_is_charged_at_fold_boundaries(path):
    params = CostParams()
    res = walk_forward(path, RULES, params, min_train=600, test_len=250)
    # Recompute the second fold's costs from the held position at its start.
    i0 = 250
    w_prev = res.weights[i0 - 1]
    g, c, n = net_returns(path.returns[850:1100], res.weights[i0:i0 + 250], params,
                          daily_vol=path.daily_vol[850:1100], w0=w_prev)
    np.testing.assert_allclose(res.cost[i0:i0 + 250], c)
