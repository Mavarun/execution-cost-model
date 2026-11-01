"""Rebalancing rules: closed forms, causality, band geometry, cost accounting."""

from __future__ import annotations

import numpy as np
import pytest

from execution_cost.costs import CostParams, total_unit_cost
from execution_cost.portfolio import _ewma, apply_costs_to_returns
from execution_cost.rebalance import (
    Rule,
    net_returns,
    no_trade_band,
    partial_adjust,
    trades_from_positions,
)


@pytest.fixture()
def target():
    rng = np.random.default_rng(0)
    return np.tanh(np.cumsum(rng.normal(0, 0.3, 400)) * 0.2 + rng.normal(0, 0.5, 400))


def test_partial_adjust_equals_slice1_ewma_from_its_first_value(target):
    span = 5
    a = 2.0 / (span + 1.0)
    # slice 1's _ewma starts at x[0]; partial adjustment from w0 = x[0] matches.
    np.testing.assert_allclose(partial_adjust(target, a, w0=target[0]), _ewma(target, span))


def test_partial_adjust_rate_one_is_full_and_recursion_holds(target):
    np.testing.assert_allclose(partial_adjust(target, 1.0, w0=0.7), target)
    w = partial_adjust(target, 0.2, w0=0.3)
    assert w[0] == pytest.approx(0.3 + 0.2 * (target[0] - 0.3))
    np.testing.assert_allclose(w[1:], w[:-1] + 0.2 * (target[1:] - w[:-1]))


def test_partial_adjust_rejects_bad_rate(target):
    with pytest.raises(ValueError):
        partial_adjust(target, 0.0)


def test_band_keeps_position_within_width_and_trades_to_the_edge(target):
    b = 0.3
    w = no_trade_band(target, b)
    assert np.all(np.abs(target - w) <= b + 1e-12)
    dw = trades_from_positions(w)
    traded = np.abs(dw) > 0
    # Every trade lands exactly on an edge of the band.
    np.testing.assert_allclose(np.abs(target[traded] - w[traded]), b)


def test_band_zero_is_full_and_wider_band_trades_less(target):
    np.testing.assert_allclose(no_trade_band(target, 0.0), target)
    turns = [np.abs(trades_from_positions(no_trade_band(target, b))).sum() for b in (0.0, 0.1, 0.3, 0.6)]
    assert all(x > y for x, y in zip(turns, turns[1:]))


def test_band_idle_when_target_never_leaves_it():
    x = np.array([0.05, -0.05, 0.08, 0.0])
    np.testing.assert_array_equal(no_trade_band(x, 0.1, w0=0.0), np.zeros(4))


@pytest.mark.parametrize(
    "rule",
    [Rule("FULL"), Rule("PARTIAL", rate=0.2), Rule("BAND", width=0.2), Rule("PARTIAL+BAND", rate=0.3, width=0.1)],
)
def test_rules_are_causal(rule, target):
    base = rule.positions(target, w0=0.1)
    changed = target.copy()
    changed[250:] = -changed[250:]
    np.testing.assert_array_equal(rule.positions(changed, w0=0.1)[:250], base[:250])


def test_segmented_run_with_carried_position_equals_one_pass(target):
    for rule in (Rule("PARTIAL", rate=0.25), Rule("PARTIAL+BAND", rate=0.25, width=0.15)):
        _check_split(rule, target)


def _check_split(rule, target):
    whole = rule.positions(target)
    first = rule.positions(target[:200])
    second = rule.positions(target[200:], w0=first[-1])
    # Band state is the held position, so a split run is identical.
    np.testing.assert_allclose(np.concatenate([first, second]), whole)


def test_partial_band_nests_band_and_partial(target):
    np.testing.assert_allclose(
        Rule("PARTIAL+BAND", rate=1.0, width=0.2).positions(target, w0=0.1),
        Rule("BAND", width=0.2).positions(target, w0=0.1),
    )
    np.testing.assert_allclose(
        Rule("PARTIAL+BAND", rate=0.3, width=0.0).positions(target, w0=0.1),
        Rule("PARTIAL", rate=0.3).positions(target, w0=0.1),
    )


def test_partial_band_closes_a_fraction_of_the_excess():
    w = no_trade_band(np.array([1.0]), 0.2, w0=0.0, rate=0.5)
    assert w[0] == pytest.approx(0.4)


def test_rule_labels_and_validation():
    assert Rule("BAND", width=0.2).label == "BAND(b=0.2)"
    assert Rule("PARTIAL+BAND", rate=0.1, width=0.3).label == "PARTIAL+BAND(a=0.1,b=0.3)"
    with pytest.raises(ValueError):
        Rule("MAGIC")


def test_net_returns_match_slice1_cost_accounting(target):
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, target.size)
    params = CostParams()
    g1, c1, n1 = net_returns(r, target, params, daily_vol=0.01)
    g0, c0, n0 = apply_costs_to_returns(r, target, params, daily_vol=0.01)
    np.testing.assert_allclose(g1, g0)
    np.testing.assert_allclose(c1, c0)
    np.testing.assert_allclose(n1, n0)


def test_net_returns_charge_the_carried_first_trade_and_nothing_when_idle():
    params = CostParams()
    w = np.array([0.5, 0.5, 0.5])
    r = np.zeros(3)
    _, cost, _ = net_returns(r, w, params, daily_vol=0.01, w0=0.2)
    assert cost[0] == pytest.approx(0.3 * float(total_unit_cost(0.3, params, daily_vol=0.01)))
    assert cost[1] == 0.0 and cost[2] == 0.0
