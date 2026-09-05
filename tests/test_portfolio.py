"""Gross vs net Sharpe and cost application tests."""

from __future__ import annotations

import numpy as np
import pytest

from execution_cost.costs import CostParams
from execution_cost.portfolio import apply_costs_to_returns, run_cost_slice


def test_net_sharpe_strictly_below_gross_when_trading():
    res = run_cost_slice(n=3000, seed=7, hold_bars=3, params=CostParams())
    assert res.mean_turnover > 0.0
    assert res.mean_daily_cost > 0.0
    assert res.net_sharpe < res.gross_sharpe


def test_zero_costs_match_gross_when_params_zero():
    params = CostParams(spread_bps=0.0, slippage_bps=0.0, impact_coeff=0.0)
    res = run_cost_slice(n=1500, seed=1, hold_bars=5, params=params)
    assert res.net_sharpe == pytest.approx(res.gross_sharpe, rel=0, abs=1e-12)
    assert res.mean_daily_cost == pytest.approx(0.0)


def test_apply_costs_shapes_and_identity_with_flat_weights():
    r = np.array([0.01, -0.02, 0.015, 0.0])
    w = np.array([0.5, 0.5, 0.5, 0.5])  # only initial trade
    params = CostParams(spread_bps=10.0, slippage_bps=0.0, impact_coeff=0.0, adv_notional=1.0)
    gross, costs, net = apply_costs_to_returns(r, w, params, daily_vol=0.01)
    assert gross.shape == costs.shape == net.shape == r.shape
    np.testing.assert_allclose(gross, w * r)
    # Only t=0 has a trade of 0.5
    assert costs[0] > 0.0
    assert costs[1:].sum() == pytest.approx(0.0)
    np.testing.assert_allclose(net, gross - costs)
