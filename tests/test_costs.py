"""Unit tests for spread / slippage / square-root impact costs."""

from __future__ import annotations

import numpy as np
import pytest

from execution_cost.costs import (
    CostParams,
    half_spread_cost,
    linear_slippage_cost,
    participation_rate,
    square_root_impact_cost,
    total_unit_cost,
)


def test_half_spread_scales_with_bps():
    assert half_spread_cost(10.0) == pytest.approx(5.0 / 10_000.0)
    assert half_spread_cost(0.0) == 0.0


def test_participation_rate_uses_adv():
    p = participation_rate(0.5, adv_notional=1.0)
    assert float(p) == pytest.approx(0.5)
    p2 = participation_rate([1.0, 2.0], adv_notional=2.0)
    np.testing.assert_allclose(p2, [0.5, 1.0])


def test_linear_slippage_zero_at_zero_participation():
    assert float(linear_slippage_cost(0.0, 10.0)) == 0.0
    assert float(linear_slippage_cost(1.0, 10.0)) == pytest.approx(10.0 / 10_000.0)


def test_square_root_impact_concave_in_participation():
    c_low = float(square_root_impact_cost(0.01, impact_coeff=0.1, daily_vol=0.02))
    c_high = float(square_root_impact_cost(0.04, impact_coeff=0.1, daily_vol=0.02))
    # 4x participation => 2x cost under square-root law
    assert c_high == pytest.approx(2.0 * c_low)
    assert c_low > 0.0


def test_total_unit_cost_positive_and_increases_with_size():
    params = CostParams(spread_bps=5.0, slippage_bps=2.0, impact_coeff=0.1, adv_notional=1.0)
    c_small = float(total_unit_cost(0.01, params, daily_vol=0.01))
    c_large = float(total_unit_cost(0.25, params, daily_vol=0.01))
    assert c_small > half_spread_cost(params.spread_bps) - 1e-15
    assert c_large > c_small
