"""Slice-5 harness: frozen protocol, grids, reproducibility, decision plumbing."""

from __future__ import annotations

import numpy as np
import pytest

from execution_cost.alpha import AlphaParams
from execution_cost.walkforward_eval import (
    PROTOCOL,
    SELECTORS,
    candidate_rules,
    cost_params,
    decide,
    evaluate_seed,
    null_market,
    paired_seed_test,
    pooled_hac,
    table,
)

SMALL = AlphaParams(n=900)


def test_protocol_is_frozen():
    assert PROTOCOL["market"] == AlphaParams().to_dict()
    assert PROTOCOL["min_train"] == 1260 and PROTOCOL["test_len"] == 252
    assert PROTOCOL["eval_seeds"] == tuple(range(500, 520))
    assert PROTOCOL["null_seeds"] == tuple(range(600, 620))
    assert not set(PROTOCOL["eval_seeds"]) & set(PROTOCOL["null_seeds"])
    assert PROTOCOL["dev_seed"] not in PROTOCOL["eval_seeds"] + PROTOCOL["null_seeds"]
    assert PROTOCOL["primary"] == ("WF-PB", "WF-PARTIAL")
    # 5,040 bars -> 15 annual test folds after a 5-year training window
    assert len(range(PROTOCOL["min_train"], PROTOCOL["market"]["n"], PROTOCOL["test_len"])) == 15


def test_grid_sizes_match_the_trial_count():
    groups = candidate_rules()
    assert len(groups["WF-PARTIAL"]) == 8 and len(groups["WF-BAND"]) == 8 and len(groups["WF-PB"]) == 25
    assert len(groups["WF-ALL"]) == PROTOCOL["n_trials"] == 42
    labels = [r.label for r in groups["WF-ALL"]]
    assert len(set(labels)) == len(labels)


def test_cost_scaling_scales_every_component_but_adv():
    c = cost_params(2.0)
    assert (c.spread_bps, c.slippage_bps, c.impact_coeff, c.adv_notional) == (20.0, 10.0, 0.5, 1.0)


def test_null_market_only_changes_ic():
    m = null_market()
    assert m.ic == 0.0 and m.rho == PROTOCOL["market"]["rho"] and m.n == PROTOCOL["market"]["n"]


@pytest.fixture(scope="module")
def two_seeds():
    return [evaluate_seed(s, market=SMALL, min_train=400, test_len=250) for s in (1, 2)]


def test_evaluate_seed_is_reproducible_and_complete(two_seeds):
    again = evaluate_seed(1, market=SMALL, min_train=400, test_len=250)
    r = two_seeds[0]
    assert set(r.oos) == set(SELECTORS) | {"FULL"}
    for name in r.oos:
        np.testing.assert_array_equal(r.oos_net[name], again.oos_net[name])
        assert r.oos_net[name].size == 500
    assert r.choices == again.choices
    assert 0.0 <= r.insample_best["dsr"] <= 1.0


def test_full_has_the_highest_turnover(two_seeds):
    for r in two_seeds:
        assert all(r.oos["FULL"]["turnover"] >= r.oos[n]["turnover"] for n in SELECTORS)


def test_paired_seed_test_and_table(two_seeds):
    t = paired_seed_test(np.array([1.0, 2.0, 3.5]), np.array([0.5, 1.0, 1.0]), alternative="greater")
    assert t["mean"] == pytest.approx(4.0 / 3.0) and t["wins"] == 3 and t["p"] < 0.1
    tab = table(two_seeds)
    assert tab["FULL"]["net_sharpe"] == pytest.approx(np.mean([r.oos["FULL"]["net_sharpe"] for r in two_seeds]))
    h = pooled_hac(two_seeds, "WF-PB", "WF-PARTIAL")
    assert np.isfinite(h["se"]) and h["se"] > 0


def test_decide_reports_every_hypothesis(two_seeds):
    d = decide(two_seeds + two_seeds, two_seeds)
    assert set(d) == {"H1", "H2", "H3", "H4", "recommended_family"}
    assert d["recommended_family"] in {"PARTIAL", "PARTIAL+BAND"}


def test_levering_by_one_reproduces_the_oos_record(two_seeds):
    from execution_cost.walkforward_eval import levered_net

    r = two_seeds[0]
    np.testing.assert_allclose(levered_net(r, "WF-PB", 1.0), r.oos_net["WF-PB"], atol=1e-15)


def test_equal_risk_leverage_hits_the_target_and_pays_superlinear_costs(two_seeds):
    from execution_cost.walkforward_eval import equal_risk_comparison, equal_risk_leverage, levered_net

    r = two_seeds[0]
    target = float(np.std(r.oos_net["WF-PARTIAL"], ddof=1))
    k = equal_risk_leverage(r, "WF-BAND", target)
    assert np.std(levered_net(r, "WF-BAND", k), ddof=1) == pytest.approx(target, rel=1e-4)
    # Doubling positions more than doubles cost: net mean grows by less than 2x gross.
    w = r.oos_weights["WF-PB"]
    gross = (w * r.oos_returns).mean()
    net1, net2 = levered_net(r, "WF-PB", 1.0).mean(), levered_net(r, "WF-PB", 2.0).mean()
    assert (2 * gross - net2) > 2 * (gross - net1)
    cmp = equal_risk_comparison(two_seeds, "WF-PB", "WF-PARTIAL")
    assert cmp["n"] == 2 and cmp["leverage_mean"] > 0
