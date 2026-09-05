"""Turnover cost-curve hypothesis tests."""

from __future__ import annotations

from execution_cost.costs import CostParams
from execution_cost.sensitivity import curve_drag_rises_with_turnover, turnover_cost_curve


def test_higher_turnover_increases_cost_and_drags_sharpe():
    params = CostParams(spread_bps=10.0, slippage_bps=5.0, impact_coeff=0.25, adv_notional=1.0)
    curve = turnover_cost_curve([1, 5, 21], n=2500, seed=11, params=params)
    by_hold = {p.hold_bars: p for p in curve}
    assert by_hold[1].mean_turnover > by_hold[5].mean_turnover > by_hold[21].mean_turnover
    assert by_hold[1].mean_daily_cost > by_hold[21].mean_daily_cost
    assert by_hold[1].sharpe_drag > by_hold[21].sharpe_drag
    # Aggressive turnover destroys more edge after costs
    assert by_hold[1].net_sharpe < by_hold[21].net_sharpe


def test_cost_curve_drag_rises_with_turnover():
    curve = turnover_cost_curve([1, 2, 3, 5, 8, 13], n=2000, seed=42, params=CostParams())
    assert len(curve) == 6
    assert all(p.gross_sharpe > p.net_sharpe for p in curve)
    assert curve_drag_rises_with_turnover(curve, atol=1e-9)
    ordered = sorted(curve, key=lambda p: p.mean_turnover)
    # Highest turnover has worse net Sharpe than lowest-turnover point on the grid
    assert ordered[-1].net_sharpe < ordered[0].net_sharpe
