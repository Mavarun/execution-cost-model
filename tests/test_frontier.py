"""Realised frontiers: sqrt optimum vs linearised AC at equal risk, out of sample."""

from __future__ import annotations

import numpy as np
import pytest

from execution_cost.frontier import _objective_usd, match_std_lam, realised_frontiers
from execution_cost.calibration import calibrate_impact
from execution_cost.simulator import MarketParams, simulate_market

MP = MarketParams()


@pytest.fixture(scope="module")
def points():
    return realised_frontiers(q=0.05, urgencies=(0.5, 1.0, 3.0), n_paths=2500, seed=777)


def test_bisection_hits_the_target_std():
    cal = calibrate_impact(seed=0)
    mk = simulate_market(MP, 800, seed=11)
    X = 0.05 * MP.adv
    lam, tot = match_std_lam(80.0, MP, X, mk, cal.temp_sqrt_Y, cal.perm_coeff)
    assert tot.std(ddof=1) == pytest.approx(80.0, rel=1e-6)
    assert lam > 0


def test_sqrt_optimum_is_cheaper_than_ac_at_equal_realised_risk(points):
    matched = [p for p in points if p.risk_matched]
    assert len(matched) >= 2
    for p in matched:
        assert p.sqrt_std_bps == pytest.approx(p.ac_std_bps, rel=1e-6)
        assert p.paired_saving_bps > 3 * p.paired_saving_se_bps
        assert p.sqrt_lam_matched < p.lam  # AC's lam overstates its own urgency


def test_low_urgency_ac_is_dominated_outright(points):
    p = points[0]
    assert p.kappa_T == 0.5 and not p.risk_matched
    assert p.sqrt_std_bps < p.ac_std_bps and p.paired_saving_bps > 0


def test_realised_mean_variance_objective_is_lower_at_the_same_lam(points):
    for p in points:
        assert p.sqrt_objective_same_lam_usd < p.ac_objective_usd
        assert p.sqrt_std_same_lam_bps < p.ac_std_bps


def test_objective_units():
    tot = np.array([10.0, 30.0])  # bps
    X, s0, lam = 1000.0, 100.0, 1e-3
    usd = tot / 1e4 * X * s0
    assert _objective_usd(tot, X, s0, lam) == pytest.approx(usd.mean() + lam * usd.var(ddof=1))
