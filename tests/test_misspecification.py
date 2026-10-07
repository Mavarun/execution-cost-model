"""Exponent misspecification: sqrt optimum vs AC when the true delta != 0.5."""

from __future__ import annotations

import pytest

from execution_cost.calibration import calibrate_impact, generate_metaorders, temp_coeff_at_exponent
from execution_cost.optimal_sqrt import SqrtExecutionProblem, sqrt_schedule
from execution_cost.simulator import MarketParams


def test_temp_coeff_at_half_reproduces_calibrated_sqrt_coefficient():
    orders = generate_metaorders(n_orders=4000, seed=3)
    cal = calibrate_impact(orders, seed=3)
    assert temp_coeff_at_exponent(orders, 0.5, cal.perm_coeff, seed=3) == pytest.approx(cal.temp_sqrt_Y, rel=1e-12)


def test_planner_exponent_is_explicit_not_read_from_true_market():
    mp = MarketParams(temp_exponent=0.8)
    X = 0.05 * mp.adv
    a = sqrt_schedule(mp, X, 1e-6, 0.5, 0.25, temp_exponent=0.5)
    b = sqrt_schedule(MarketParams(), X, 1e-6, 0.5, 0.25)
    assert a == pytest.approx(b)
    assert SqrtExecutionProblem.from_market(mp, X, temp_exponent=0.5).temp_exponent == 0.5
