"""TWAP / VWAP / POV / Almgren-Chriss comparison with costs, out of sample.

Protocol
--------
1. **Calibrate** the sqrt impact law and permanent coefficient on a *train*
   set of synthetic metaorders (:func:`execution_cost.calibration.calibrate_impact`).
2. **Build schedules** for a buy of ``q`` x ADV using only pre-trade
   information: TWAP; VWAP on the expected volume profile; POV at
   ``pov_rate`` (idealised, and the implementable lag-1 version);
   Almgren-Chriss with the *calibrated* coefficients mapped to linear
   impact, at two urgencies ``kappa * T``; and the profile-aware sqrt-law
   optimum (:mod:`execution_cost.optimal_sqrt`) with the same calibrated
   coefficients at the *same* risk aversion ``lam`` as each AC urgency.
3. **Evaluate** every schedule on the same fresh simulated days (common
   random numbers, a seed disjoint from calibration) under the *true*
   simulator parameters, and decompose implementation shortfall (spread,
   temporary, permanent, timing, opportunity) in bps of arrival notional.
4. **Pre-trade vs realised**: a calibrated pre-trade cost estimate for each
   schedule is compared with its realised mean, as an out-of-sample check
   of the cost model.

Paired differences vs TWAP use the same price/volume paths, so their
standard errors are far smaller than those of the raw means (timing noise
cancels).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from execution_cost.almgren_chriss import ACParams, ac_trade_list
from execution_cost.calibration import CalibrationResult, calibrate_impact, generate_metaorders
from execution_cost.optimal_sqrt import sqrt_schedule
from execution_cost.schedules import ac_params_from_market, twap_schedule, vwap_schedule
from execution_cost.shortfall import decompose_shortfall
from execution_cost.simulator import POV, MarketParams, execute, simulate_market, volume_profile


def lam_for_urgency(p: ACParams, kappa_T: float) -> float:
    """Risk aversion giving urgency ``kappa * T`` (inverts the discrete AC relation)."""
    if kappa_T < 0:
        raise ValueError("kappa_T must be >= 0")
    kappa = kappa_T / p.T
    kt2 = 2.0 / p.tau**2 * (np.cosh(kappa * p.tau) - 1.0)
    return float(kt2 * p.eta_tilde / p.sigma**2)


def pretrade_estimate_bps(
    schedule: np.ndarray,
    market: MarketParams,
    target: float,
    temp_coeff: float,
    perm_coeff: float,
) -> float:
    """Expected IS (bps) of a static schedule under a calibrated sqrt model.

    Uses expected interval volume ``ADV u_j``; ``temp_coeff`` is a
    metaorder-level coefficient (it already absorbs volume noise when
    calibrated on realised metaorders). Timing cost has mean zero.
    """
    n = np.asarray(schedule, dtype=float)
    u = volume_profile(market.n_intervals, market.u_shape)
    q = target / market.adv
    part = n / (market.adv * u)
    # temp_coeff is calibrated against total q for TWAP; express per interval
    # relative to the TWAP participation profile so TWAP reproduces Y sqrt(q).
    twap_part = (target / market.n_intervals) / (market.adv * u)
    prof_twap = np.mean(np.sqrt(twap_part))
    temp = temp_coeff * np.sqrt(q) * (n @ np.sqrt(part)) / (target * prof_twap)
    cum_before = np.concatenate([[0.0], np.cumsum(n)[:-1]])
    perm = perm_coeff * (n @ cum_before) / (target * market.adv)
    spread = 0.5 * market.spread_bps / 1e4 / market.sigma_daily
    return float((temp + perm + spread) * market.sigma_daily * 1e4)


@dataclass
class ScheduleResult:
    name: str
    summary: dict
    paired_diff_vs_twap_bps: float
    paired_diff_se_bps: float
    p95_total_bps: float
    pretrade_estimate_bps: float | None
    first_interval_share: float | None

    def to_dict(self) -> dict:
        d = dict(self.summary)
        d.update(
            name=self.name,
            paired_diff_vs_twap_bps=self.paired_diff_vs_twap_bps,
            paired_diff_se_bps=self.paired_diff_se_bps,
            p95_total_bps=self.p95_total_bps,
            pretrade_estimate_bps=self.pretrade_estimate_bps,
            first_interval_share=self.first_interval_share,
        )
        return d


@dataclass
class ComparisonReport:
    q: float
    n_paths: int
    calibration: dict
    schedules: list[ScheduleResult]

    def by_name(self) -> dict[str, ScheduleResult]:
        return {s.name: s for s in self.schedules}

    def to_dict(self) -> dict:
        return {
            "q": self.q,
            "n_paths": self.n_paths,
            "calibration": self.calibration,
            "schedules": [s.to_dict() for s in self.schedules],
        }


def compare_schedules(
    market: MarketParams | None = None,
    q: float = 0.05,
    n_paths: int = 4000,
    seed: int = 123,
    urgencies: tuple[float, ...] = (1.0, 3.0),
    pov_rate: float | None = None,
    calibration: CalibrationResult | None = None,
    calibration_seed: int = 0,
    fee_bps: float = 0.0,
    sqrt_optimal: bool = True,
    lagged_pov: bool = True,
) -> ComparisonReport:
    """Run the protocol in the module docstring for a buy of ``q`` x ADV."""
    mp = market or MarketParams()
    if calibration is None:
        orders = generate_metaorders(seed=calibration_seed, market=mp)
        calibration = calibrate_impact(orders, seed=calibration_seed, market=mp)
    # temporary coefficient net of the (separately estimated) permanent part
    Y, G = calibration.temp_sqrt_Y, max(calibration.perm_coeff, 0.0)
    X = q * mp.adv
    N = mp.n_intervals
    pov_rate = pov_rate if pov_rate is not None else min(1.25 * q, 0.5)

    statics: dict[str, np.ndarray] = {
        "TWAP": twap_schedule(X, N),
        "VWAP": vwap_schedule(X, volume_profile(N, mp.u_shape)),
    }
    base_ac = ac_params_from_market(mp, X, lam=0.0, temp_coeff=Y, perm_coeff=G)
    for k in urgencies:
        lam = lam_for_urgency(base_ac, k)
        statics[f"AC(kT={k:g})"] = ac_trade_list(replace(base_ac, lam=lam))
    if sqrt_optimal:
        for k in urgencies:
            lam = lam_for_urgency(base_ac, k)
            statics[f"SQRT-OPT(kT={k:g})"] = sqrt_schedule(mp, X, lam, temp_coeff=Y, perm_coeff=G, temp_exponent=0.5)

    mk = simulate_market(mp, n_paths, seed=seed)
    results: dict[str, tuple] = {}
    for name, sched in statics.items():
        br = decompose_shortfall(execute(sched, mk, mp, X), fee_bps=fee_bps)
        results[name] = (br, pretrade_estimate_bps(sched, mp, X, Y, G), sched[0] / X)
    br_pov = decompose_shortfall(execute(POV(pov_rate), mk, mp, X), fee_bps=fee_bps)
    results[f"POV({pov_rate:.3f})"] = (br_pov, None, None)
    if lagged_pov:
        br_lag = decompose_shortfall(execute(POV(pov_rate, lag=1), mk, mp, X), fee_bps=fee_bps)
        results[f"POV-lag1({pov_rate:.3f})"] = (br_lag, None, None)

    twap_total = results["TWAP"][0].total
    out = []
    for name, (br, pre, first) in results.items():
        diff = br.total - twap_total
        out.append(
            ScheduleResult(
                name=name,
                summary=br.summary(),
                paired_diff_vs_twap_bps=float(diff.mean()),
                paired_diff_se_bps=float(diff.std(ddof=1) / np.sqrt(len(diff))),
                p95_total_bps=float(np.percentile(br.total, 95)),
                pretrade_estimate_bps=pre,
                first_interval_share=first,
            )
        )
    return ComparisonReport(q=q, n_paths=n_paths, calibration=calibration.to_dict(), schedules=out)
