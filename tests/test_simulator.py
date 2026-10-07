"""Execution simulator and schedule construction."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from execution_cost.costs import square_root_impact_cost
from execution_cost.schedules import (
    ac_params_from_market,
    ac_schedule,
    expected_profile,
    twap_schedule,
    vwap_schedule,
)
from execution_cost.simulator import (
    POV,
    MarketParams,
    execute,
    simulate_market,
    temporary_impact,
    volume_profile,
)

MP = MarketParams()


def test_volume_profile_is_u_shaped_and_normalised():
    prof = volume_profile(26, 1.5)
    assert prof.sum() == pytest.approx(1.0)
    assert prof[0] > prof[13] and prof[-1] > prof[13]
    np.testing.assert_allclose(volume_profile(10, 0.0), 0.1)


def test_realised_volume_averages_adv():
    mk = simulate_market(MP, n_paths=4000, seed=0)
    assert mk.volumes.sum(1).mean() == pytest.approx(MP.adv, rel=0.02)


def test_frictionless_flat_market_fills_at_mid_plus_half_spread():
    p = replace(MP, sigma_daily=1e-12, temp_coeff=0.0, perm_coeff=0.0)
    mk = simulate_market(p, n_paths=5, seed=1)
    res = execute(twap_schedule(10_000, p.n_intervals), mk, p, 10_000, side=1)
    np.testing.assert_allclose(res.fill_prices, p.s0 + p.half_spread, atol=1e-8)
    sell = execute(twap_schedule(10_000, p.n_intervals), mk, p, 10_000, side=-1)
    np.testing.assert_allclose(sell.fill_prices, p.s0 - p.half_spread, atol=1e-8)


def test_temporary_impact_reuses_square_root_cost_model():
    part = np.array([0.0, 0.01, 0.1])
    expected = square_root_impact_cost(part, impact_coeff=MP.temp_coeff, daily_vol=MP.sigma_daily) * MP.s0
    np.testing.assert_allclose(temporary_impact(part, MP), expected)
    lin = replace(MP, temp_exponent=1.0)
    np.testing.assert_allclose(temporary_impact(part, lin), MP.temp_coeff * MP.sigma_daily * part * MP.s0)


def test_permanent_impact_accumulates_linearly():
    p = replace(MP, sigma_daily=MP.sigma_daily, temp_coeff=0.0)
    mk = simulate_market(p, n_paths=3, seed=2)
    X = 50_000
    res = execute(twap_schedule(X, p.n_intervals), mk, p, X, side=1)
    np.testing.assert_allclose(res.final_perm, p.perm_coeff * p.sigma_price * X / p.adv)
    np.testing.assert_allclose(res.final_mid, p.s0 + res.final_noise + res.final_perm)


def test_static_schedules_complete_and_pov_may_not():
    mk = simulate_market(MP, n_paths=500, seed=3)
    X = 50_000
    for sched in (twap_schedule(X, 26), vwap_schedule(X, expected_profile(MP)), ac_schedule(MP, X, 1e-6)):
        res = execute(sched, mk, MP, X)
        np.testing.assert_allclose(res.executed, X)
    slow = execute(POV(0.03), mk, MP, X)  # 3% of ~1M ADV < 50k on most days
    assert slow.completion.mean() < 0.9
    fast = execute(POV(0.2), mk, MP, X)
    np.testing.assert_allclose(fast.executed, X)


def test_common_random_numbers_shared_across_schedules():
    mk = simulate_market(MP, n_paths=50, seed=4)
    a = execute(twap_schedule(20_000, 26), mk, MP, 20_000)
    b = execute(vwap_schedule(20_000, mk.profile), mk, MP, 20_000)
    np.testing.assert_allclose(a.final_noise, b.final_noise)


def test_ac_mapping_matches_sqrt_law_at_twap_rate():
    X = 50_000
    ac = ac_params_from_market(MP, X, lam=0.0)
    # per-share linear temp cost at TWAP rate equals sqrt-law cost at q = X/ADV
    sqrt_cost = MP.temp_coeff * MP.sigma_price * np.sqrt(X / MP.adv)
    assert ac.eta * X == pytest.approx(sqrt_cost)
    np.testing.assert_allclose(ac_schedule(MP, X, 0.0), twap_schedule(X, 26))
    assert ac_schedule(MP, X, 1e-5)[0] > X / 26
    with pytest.raises(ValueError):
        POV(1.5)
    with pytest.raises(ValueError):
        execute(twap_schedule(10, 26), simulate_market(MP, 2, 0), MP, 10, side=0)


def test_per_path_targets_for_metaorders():
    mk = simulate_market(MP, n_paths=4, seed=5)
    Q = np.array([1_000.0, 10_000.0, 50_000.0, 100_000.0])
    res = execute(Q[:, None] / 26 * np.ones((1, 26)), mk, MP, Q)
    np.testing.assert_allclose(res.executed, Q)
    # bigger orders pay more temporary impact per share
    assert np.all(np.diff(res.temp_impact.mean(1)) > 0)
    with pytest.raises(ValueError):
        execute(twap_schedule(10, 26), mk, MP, np.ones(3))


# --- lagged (implementable) POV -------------------------------------------------

def test_lagged_pov_uses_previous_interval_volume_rescaled_by_profile():
    from execution_cost.simulator import POV, MarketParams, execute, simulate_market

    mp = MarketParams()
    mk = simulate_market(mp, 50, seed=3)
    target = 1e9  # never binding: trades are pure POV
    res = execute(POV(0.05, lag=1), mk, mp, target)
    np.testing.assert_allclose(res.trades[:, 0], 0.05 * mp.adv * mk.profile[0])
    expect = 0.05 * mk.volumes[:, :-1] * mk.profile[1:] / mk.profile[:-1]
    np.testing.assert_allclose(res.trades[:, 1:], expect)
    ideal = execute(POV(0.05), mk, mp, target)
    np.testing.assert_allclose(ideal.trades, 0.05 * mk.volumes)


def test_lagged_pov_participation_is_noisy_around_the_rate():
    from execution_cost.simulator import POV, MarketParams, execute, simulate_market

    mp = MarketParams()
    mk = simulate_market(mp, 2000, seed=4)
    part = execute(POV(0.05, lag=1), mk, mp, 1e9).trades / mk.volumes
    ideal = execute(POV(0.05), mk, mp, 1e9).trades / mk.volumes
    np.testing.assert_allclose(ideal, 0.05)
    assert part.std() > 0.01  # interval volume noise 0.3 -> forecast errors
    assert abs(np.median(part[:, 1:]) - 0.05) < 0.01


def test_pov_lag_validation():
    import pytest

    from execution_cost.simulator import POV

    with pytest.raises(ValueError):
        POV(0.05, lag=2)
