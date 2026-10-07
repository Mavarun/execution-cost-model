"""Schedule comparison: costs, risk, OOS pre-trade accuracy (common random numbers)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from execution_cost.almgren_chriss import ac_kappa
from execution_cost.calibration import calibrate_impact
from execution_cost.comparison import compare_schedules, lam_for_urgency, pretrade_estimate_bps
from execution_cost.schedules import ac_params_from_market, twap_schedule
from execution_cost.simulator import MarketParams

MP = MarketParams()


@pytest.fixture(scope="module")
def report():
    return compare_schedules(q=0.05, n_paths=3000, seed=321)


def test_lam_for_urgency_inverts_kappa():
    p = ac_params_from_market(MP, 50_000, lam=0.0)
    for k in (0.0, 0.5, 1.0, 3.0, 6.0):
        assert ac_kappa(replace(p, lam=lam_for_urgency(p, k))) * p.T == pytest.approx(k, abs=1e-9)


def test_pretrade_twap_matches_sqrt_law_closed_form():
    X, Y, G = 50_000, 0.55, 0.2
    q = X / MP.adv
    est = pretrade_estimate_bps(twap_schedule(X, 26), MP, X, Y, G)
    closed = (Y * np.sqrt(q) + G * q * 25 / 52) * MP.sigma_daily * 1e4 + 0.5 * MP.spread_bps
    assert est == pytest.approx(closed, rel=1e-9)


def test_vwap_beats_twap_on_cost_under_u_shaped_volume(report):
    r = report.by_name()
    v = r["VWAP"]
    assert v.paired_diff_vs_twap_bps < 0
    assert abs(v.paired_diff_vs_twap_bps) > 3 * v.paired_diff_se_bps
    assert v.summary["mean_temporary_bps"] < r["TWAP"].summary["mean_temporary_bps"]


def test_urgent_almgren_chriss_trades_cost_for_lower_risk(report):
    r = report.by_name()
    twap, ac = r["TWAP"], r["AC(kT=3)"]
    assert ac.first_interval_share > 2 * twap.first_interval_share
    assert ac.paired_diff_vs_twap_bps > 2 * ac.paired_diff_se_bps  # pays more impact
    assert ac.summary["std_total_bps"] < 0.8 * twap.summary["std_total_bps"]  # less timing risk
    assert ac.p95_total_bps < twap.p95_total_bps


def test_pov_adapts_but_can_leave_shares_unfilled(report):
    pov = [s for s in report.schedules if s.name.startswith("POV")][0]
    assert 0.9 < pov.summary["mean_completion"] < 1.0
    assert pov.pretrade_estimate_bps is None


def test_calibrated_pretrade_estimates_track_realised_impact_out_of_sample(report):
    # Calibration and evaluation use disjoint seeds. Compare against the
    # realised deterministic part (spread + temporary + permanent); timing
    # noise has mean zero and is excluded.
    for s in report.schedules:
        if s.pretrade_estimate_bps is None:
            continue
        sm = s.summary
        realised = sm["mean_spread_bps"] + sm["mean_temporary_bps"] + sm["mean_permanent_bps"]
        assert abs(s.pretrade_estimate_bps - realised) < 1.0, s.name


def test_all_schedules_pay_positive_costs_and_cost_grows_with_size():
    small = compare_schedules(q=0.02, n_paths=1000, seed=5).by_name()
    large = compare_schedules(q=0.10, n_paths=1000, seed=5).by_name()
    for name in ("TWAP", "VWAP", "AC(kT=1)", "AC(kT=3)"):
        assert small[name].summary["mean_temporary_bps"] > 0
        lg, sm = large[name].summary, small[name].summary
        assert lg["mean_temporary_bps"] + lg["mean_permanent_bps"] > sm["mean_temporary_bps"] + sm["mean_permanent_bps"]


def test_miscalibrated_impact_biases_pretrade_estimate():
    good = calibrate_impact(seed=0)
    bad = replace(good, temp_sqrt_Y=2.0 * good.temp_sqrt_Y)
    rb = compare_schedules(q=0.05, n_paths=800, seed=9, calibration=bad).by_name()["TWAP"]
    sm = rb.summary
    realised = sm["mean_spread_bps"] + sm["mean_temporary_bps"] + sm["mean_permanent_bps"]
    assert rb.pretrade_estimate_bps > realised + 15.0


# --- slice 3: profile-aware sqrt optimum and implementable POV -------------------

def test_sqrt_optimum_at_the_same_lam_cuts_risk_at_no_significant_extra_cost(report):
    r = report.by_name()
    ac, sq = r["AC(kT=1)"], r["SQRT-OPT(kT=1)"]
    assert sq.summary["std_total_bps"] < 0.9 * ac.summary["std_total_bps"]
    gap = sq.paired_diff_vs_twap_bps - ac.paired_diff_vs_twap_bps
    assert abs(gap) < 3 * (sq.paired_diff_se_bps + ac.paired_diff_se_bps)
    # it front-loads into the heavy opening bins
    assert sq.first_interval_share > ac.first_interval_share


def test_sqrt_optimum_urgent_end_buys_much_lower_risk_for_more_impact(report):
    r = report.by_name()
    ac, sq = r["AC(kT=3)"], r["SQRT-OPT(kT=3)"]
    assert sq.summary["std_total_bps"] < 0.7 * ac.summary["std_total_bps"]
    assert sq.summary["mean_temporary_bps"] > ac.summary["mean_temporary_bps"]
    assert sq.p95_total_bps < ac.p95_total_bps


def test_lagged_pov_pays_for_its_volume_forecast_error(report):
    pov = [s for s in report.schedules if s.name.startswith("POV(")][0]
    lag = [s for s in report.schedules if s.name.startswith("POV-lag1")][0]
    assert lag.summary["mean_temporary_bps"] > pov.summary["mean_temporary_bps"]
    assert lag.paired_diff_vs_twap_bps > pov.paired_diff_vs_twap_bps + 2 * lag.paired_diff_se_bps
    assert 0.9 < lag.summary["mean_completion"] < 1.0


def test_new_schedules_can_be_switched_off():
    names = [s.name for s in compare_schedules(q=0.05, n_paths=200, seed=1, sqrt_optimal=False, lagged_pov=False).schedules]
    assert not any(n.startswith(("SQRT-OPT", "POV-lag1")) for n in names)
