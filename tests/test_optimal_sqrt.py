"""Profile-aware sqrt-law optimal schedule: closed-form limits, KKT, dominance."""

from __future__ import annotations

import numpy as np
import pytest

from execution_cost.comparison import lam_for_urgency
from execution_cost.optimal_sqrt import (
    SqrtExecutionProblem,
    kkt_residual,
    solve_sqrt_schedule,
    sqrt_schedule,
)
from execution_cost.schedules import ac_params_from_market, ac_schedule, twap_schedule, vwap_schedule
from execution_cost.simulator import MarketParams, volume_profile

MP = MarketParams()
X = 50_000.0
PROF = volume_profile(MP.n_intervals, MP.u_shape)


def _lam(k: float) -> float:
    return lam_for_urgency(ac_params_from_market(MP, X, lam=0.0), k)


def test_risk_neutral_without_permanent_impact_is_exactly_vwap():
    prob = SqrtExecutionProblem.from_market(MP, X, lam=0.0, perm_coeff=0.0)
    sol = solve_sqrt_schedule(prob)
    assert sol.success
    np.testing.assert_allclose(sol.trades, vwap_schedule(X, PROF), rtol=1e-8, atol=1e-6)


def test_flat_profile_risk_neutral_is_twap():
    flat = MarketParams(u_shape=0.0)
    sol = solve_sqrt_schedule(SqrtExecutionProblem.from_market(flat, X, lam=0.0, perm_coeff=0.0))
    np.testing.assert_allclose(sol.trades, twap_schedule(X, flat.n_intervals), rtol=1e-8)


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(7)
    n = rng.uniform(0.5, 1.5, MP.n_intervals)
    n *= X / n.sum()
    prob = SqrtExecutionProblem.from_market(MP, X, lam=_lam(2.0))
    num = np.array([(prob.objective(n + e) - prob.objective(n - e)) / 2.0 for e in np.eye(len(n))])
    np.testing.assert_allclose(prob.gradient(n), num, rtol=1e-6, atol=1e-9 * np.abs(num).max())


def test_permanent_cost_matches_closed_form():
    n = vwap_schedule(X, PROF)
    prob = SqrtExecutionProblem.from_market(MP, X)
    closed = prob.perm_coeff * prob.sigma_price / (2 * prob.adv) * (X**2 - np.sum(n**2))
    assert prob.permanent_cost(n) == pytest.approx(closed, rel=1e-12)


@pytest.mark.parametrize("k", [0.0, 1.0, 3.0])
def test_solution_satisfies_kkt_and_sums_to_target(k):
    sol = solve_sqrt_schedule(SqrtExecutionProblem.from_market(MP, X, lam=_lam(k)))
    assert sol.success and sol.trades.min() >= 0
    assert sol.trades.sum() == pytest.approx(X, rel=1e-12)
    assert sol.kkt_residual < 1e-4


@pytest.mark.parametrize("k", [0.0, 1.0, 3.0])
def test_beats_twap_vwap_and_linearised_ac_on_its_own_model(k):
    lam = _lam(k)
    prob = SqrtExecutionProblem.from_market(MP, X, lam=lam)
    sol = solve_sqrt_schedule(prob)
    for other in (twap_schedule(X, MP.n_intervals), vwap_schedule(X, PROF), ac_schedule(MP, X, lam)):
        assert sol.objective <= prob.objective(other) * (1 + 1e-9)


def test_beats_random_feasible_perturbations():
    prob = SqrtExecutionProblem.from_market(MP, X, lam=_lam(1.0))
    sol = solve_sqrt_schedule(prob)
    rng = np.random.default_rng(0)
    for _ in range(200):
        z = rng.standard_normal(MP.n_intervals)
        z -= z.mean()
        cand = np.maximum(sol.trades + 0.02 * X / MP.n_intervals * z, 0.0)
        cand *= X / cand.sum()
        assert prob.objective(cand) >= sol.objective * (1 - 1e-9)


def test_risk_aversion_front_loads_and_trades_cost_for_variance():
    sols = [solve_sqrt_schedule(SqrtExecutionProblem.from_market(MP, X, lam=_lam(k))) for k in (0, 1, 3, 6)]
    first = [s.trades[0] for s in sols]
    assert all(a < b for a, b in zip(first, first[1:]))
    costs = [s.expected_cost for s in sols]
    stds = [s.std_cost for s in sols]
    assert all(a < b for a, b in zip(costs, costs[1:]))
    assert all(a > b for a, b in zip(stds, stds[1:]))


def test_kkt_residual_flags_a_non_optimal_schedule():
    prob = SqrtExecutionProblem.from_market(MP, X, lam=_lam(1.0))
    assert kkt_residual(prob, twap_schedule(X, MP.n_intervals)) > 0.05


def test_convenience_wrapper_and_input_validation():
    n = sqrt_schedule(MP, X, lam=_lam(1.0), temp_coeff=0.54, perm_coeff=0.2)
    assert n.sum() == pytest.approx(X)
    with pytest.raises(ValueError):
        solve_sqrt_schedule(SqrtExecutionProblem.from_market(MP, 0.0))
