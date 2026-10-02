"""Implementation-shortfall decomposition identities and special cases."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from execution_cost.schedules import twap_schedule
from execution_cost.shortfall import decompose_shortfall, perold_shortfall
from execution_cost.simulator import POV, MarketParams, execute, simulate_market

MP = MarketParams()


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("sched", ["twap", "pov"])
def test_components_sum_exactly_to_total(side, sched):
    mk = simulate_market(MP, n_paths=300, seed=0)
    X = 60_000
    s = twap_schedule(X, MP.n_intervals) if sched == "twap" else POV(0.04)
    res = execute(s, mk, MP, X, side=side)
    br = decompose_shortfall(res, decision_price=MP.s0 - 0.1 * side, fee_bps=0.5)
    np.testing.assert_allclose(br.components_sum(), br.total, atol=1e-9)
    assert (br.spread >= 0).all() and (br.temporary >= 0).all()


def test_frictionless_flat_market_costs_only_half_spread():
    p = replace(MP, sigma_daily=1e-12, temp_coeff=0.0, perm_coeff=0.0)
    mk = simulate_market(p, n_paths=4, seed=1)
    br = decompose_shortfall(execute(twap_schedule(10_000, 26), mk, p, 10_000), fee_bps=0.0)
    np.testing.assert_allclose(br.total, 0.5 * p.spread_bps, atol=1e-6)
    np.testing.assert_allclose(br.timing, 0.0, atol=1e-6)
    np.testing.assert_allclose(br.permanent, 0.0, atol=1e-12)


def test_matches_model_free_perold_split():
    mk = simulate_market(MP, n_paths=3, seed=2)
    X = 40_000
    res = execute(POV(0.03), mk, MP, X, side=1)
    br = decompose_shortfall(res, decision_price=99.9, fee_bps=0.0)
    for i in range(3):
        raw = perold_shortfall(
            res.trades[i], res.fill_prices[i], target=X, side=1, decision_price=99.9,
            arrival_price=res.arrival, final_price=res.final_mid[i],
        )
        bps = 1e4 / (X * res.arrival)
        assert raw["total"] * bps == pytest.approx(br.total[i])
        assert raw["opportunity"] * bps == pytest.approx(br.opportunity[i])
        assert raw["delay"] * bps == pytest.approx(br.delay[i])
        assert raw["execution"] * bps == pytest.approx(
            br.spread[i] + br.temporary[i] + br.permanent[i] + br.timing[i]
        )


def test_timing_is_mean_zero_noise_and_impact_terms_are_deterministic_costs():
    mk = simulate_market(MP, n_paths=6000, seed=3)
    br = decompose_shortfall(execute(twap_schedule(50_000, 26), mk, MP, 50_000))
    se = br.timing.std(ddof=1) / np.sqrt(len(br.timing))
    assert abs(br.timing.mean()) < 4 * se
    # permanent impact paid by TWAP: perm_coeff*sigma*q*(N-1)/(2N) in bps
    q = 50_000 / MP.adv
    expected_perm = MP.perm_coeff * MP.sigma_daily * q * (25 / 52) * 1e4
    np.testing.assert_allclose(br.permanent, expected_perm, rtol=1e-9)
    assert br.temporary.mean() > br.spread.mean() > 0


def test_unfinished_pov_carries_opportunity_cost_and_delay_sign():
    mk = simulate_market(MP, n_paths=2000, seed=4)
    X = 60_000
    br = decompose_shortfall(execute(POV(0.03), mk, MP, X), decision_price=MP.s0 - 0.05)
    assert br.completion.mean() < 0.8
    assert (br.opportunity != 0).mean() > 0.9
    # buying after the price rose from decision to arrival is a delay cost
    np.testing.assert_allclose(br.delay, X * 0.05 / (X * MP.s0) * 1e4)
    s = br.summary()
    assert s["mean_completion"] == pytest.approx(br.completion.mean())
    assert {"mean_total_bps", "se_total_bps", "mean_opportunity_bps"} <= set(s)
