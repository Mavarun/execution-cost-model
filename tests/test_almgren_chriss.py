"""Almgren-Chriss trajectory: limits, optimality, frontier, Monte Carlo."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from execution_cost.almgren_chriss import (
    ACParams,
    ac_kappa,
    ac_trade_list,
    ac_trajectory,
    cost_variance,
    efficient_frontier,
    expected_cost,
    optimal_holdings_by_linear_solve,
    simulate_ac_shortfall,
)

BASE = ACParams()


def test_zero_risk_aversion_is_twap():
    p = replace(BASE, lam=0.0)
    x = ac_trajectory(p)
    np.testing.assert_allclose(x, p.X * (1 - np.arange(p.N + 1) / p.N), atol=1e-6)
    np.testing.assert_allclose(ac_trade_list(p), p.X / p.N)


def test_trajectory_boundaries_and_monotone():
    x = ac_trajectory(BASE)
    assert x[0] == pytest.approx(BASE.X)
    assert x[-1] == 0.0
    assert np.all(np.diff(x) <= 1e-9)
    assert ac_trade_list(BASE).sum() == pytest.approx(BASE.X)


def test_closed_form_matches_independent_linear_solve():
    for lam in (0.0, 1e-7, 1e-6, 1e-5):
        p = replace(BASE, lam=lam)
        np.testing.assert_allclose(ac_trajectory(p), optimal_holdings_by_linear_solve(p), rtol=1e-8, atol=1e-6)


def test_closed_form_beats_perturbed_schedules():
    p = BASE
    x = ac_trajectory(p)
    obj = expected_cost(p, x) + p.lam * cost_variance(p, x)
    rng = np.random.default_rng(0)
    for _ in range(200):
        y = x.copy()
        y[1:-1] += rng.normal(0, 500.0, size=p.N - 1)
        y[1:-1] = np.clip(y[1:-1], 0, p.X)
        y[1:-1] = np.minimum.accumulate(y[1:-1])  # keep it a sell-only path
        assert expected_cost(p, y) + p.lam * cost_variance(p, y) >= obj - 1e-6


def test_higher_risk_aversion_front_loads_and_trades_cost_for_risk():
    lams = [0.0, 1e-7, 1e-6, 1e-5]
    fr = efficient_frontier(BASE, lams)
    E = [f["expected_cost"] for f in fr]
    S = [f["std_cost"] for f in fr]
    assert all(b > a for a, b in zip(E, E[1:]))
    assert all(b < a for a, b in zip(S, S[1:]))
    first = [ac_trade_list(replace(BASE, lam=l))[0] for l in lams]
    assert all(b > a for a, b in zip(first, first[1:]))
    assert fr[0]["kappa"] == pytest.approx(0.0, abs=1e-9)


def test_monte_carlo_matches_analytic_mean_and_variance():
    p = BASE
    sims = simulate_ac_shortfall(p, n_paths=40_000, seed=1)
    E, V = expected_cost(p), cost_variance(p)
    se_mean = np.sqrt(V / len(sims))
    assert abs(sims.mean() - E) < 4 * se_mean
    assert sims.var(ddof=1) == pytest.approx(V, rel=0.03)


def test_kappa_large_lambda_stable_and_validation():
    p = replace(BASE, lam=1e-2)
    x = ac_trajectory(p)
    assert np.isfinite(x).all() and x[1] < 0.3 * p.X and x[6] < 1e-3 * p.X
    assert ac_kappa(p) > 0
    with pytest.raises(ValueError):
        ac_kappa(replace(BASE, eta=1e-12, gamma=1.0))
    with pytest.raises(ValueError):
        ac_kappa(replace(BASE, lam=-1.0))
