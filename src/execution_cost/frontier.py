"""Realised efficient frontiers: linearised AC vs the profile-aware sqrt optimum.

The same ``lam`` puts the two schedules at different points of the
cost-risk trade-off (the linearised AC model over-charges fast trading, so it
front-loads less). Comparing them at equal ``lam`` therefore mixes "better
schedule" with "different risk appetite". This module compares them two
fair ways, out of sample and net of every cost:

1. **Equal realised risk.** For each AC urgency, bisect the sqrt optimum's
   ``lam`` until its *realised* IS standard deviation on the evaluation days
   matches AC's, then report the paired mean-IS difference (same price and
   volume paths) with its standard error.
2. **Realised mean-variance objective at equal ``lam``.** ``E[IS] + lam
   Var[IS]`` in dollars, estimated on the evaluation days; that is the
   quantity both schedules claim to minimise.

Both schedules use coefficients calibrated on a disjoint seed. Evaluation
uses the true simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from execution_cost.almgren_chriss import ac_trade_list
from execution_cost.calibration import CalibrationResult, calibrate_impact, generate_metaorders
from execution_cost.comparison import lam_for_urgency
from execution_cost.optimal_sqrt import sqrt_schedule
from execution_cost.schedules import ac_params_from_market
from execution_cost.shortfall import decompose_shortfall
from execution_cost.simulator import MarketParams, execute, simulate_market


@dataclass
class FrontierPoint:
    kappa_T: float
    lam: float
    ac_mean_bps: float
    ac_std_bps: float
    sqrt_lam_matched: float
    sqrt_mean_bps: float
    sqrt_std_bps: float
    paired_saving_bps: float  # AC - SQRT at equal realised std (> 0: sqrt cheaper)
    paired_saving_se_bps: float
    ac_objective_usd: float  # realised E + lam Var at the AC lam
    sqrt_objective_same_lam_usd: float
    sqrt_std_same_lam_bps: float
    # False when even the risk-neutral sqrt optimum is less risky than AC:
    # the comparison is then at *lower* risk for the sqrt schedule.
    risk_matched: bool = True

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _evaluate(sched, mk, mp, X, fee_bps):
    return decompose_shortfall(execute(sched, mk, mp, X), fee_bps=fee_bps).total


def _objective_usd(total_bps: np.ndarray, X: float, s0: float, lam: float) -> float:
    usd = total_bps / 1e4 * X * s0
    return float(usd.mean() + lam * usd.var(ddof=1))


def match_std_lam(
    target_std_bps: float,
    mp: MarketParams,
    X: float,
    mk,
    Y: float,
    G: float,
    *,
    lo: float = 1e-12,
    hi: float = 1e-2,
    iters: int = 60,
    fee_bps: float = 0.0,
    planner_exponent: float = 0.5,
) -> tuple[float, np.ndarray]:
    """Bisect ``lam`` (log scale) so the sqrt optimum's realised std hits the target.

    ``mp`` is the true market used for evaluation; the planner always uses
    ``planner_exponent`` (0.5 = the sqrt law), never ``mp.temp_exponent``.
    """

    def std_at(lam):
        tot = _evaluate(sqrt_schedule(mp, X, lam, Y, G, temp_exponent=planner_exponent), mk, mp, X, fee_bps)
        return float(tot.std(ddof=1)), tot

    s_lo, _ = std_at(lo)
    if target_std_bps >= s_lo:
        return lo, std_at(lo)[1]
    a, b = np.log(lo), np.log(hi)
    for _ in range(iters):
        m = 0.5 * (a + b)
        s, _ = std_at(np.exp(m))
        if s > target_std_bps:
            a = m
        else:
            b = m
    lam = float(np.exp(0.5 * (a + b)))
    return lam, std_at(lam)[1]


def realised_frontiers(
    market: MarketParams | None = None,
    q: float = 0.05,
    urgencies: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 4.0),
    n_paths: int = 4000,
    seed: int = 123,
    calibration: CalibrationResult | None = None,
    calibration_seed: int = 0,
    fee_bps: float = 0.0,
) -> list[FrontierPoint]:
    mp = market or MarketParams()
    if calibration is None:
        calibration = calibrate_impact(generate_metaorders(seed=calibration_seed, market=mp), seed=calibration_seed, market=mp)
    Y, G = calibration.temp_sqrt_Y, max(calibration.perm_coeff, 0.0)
    X = q * mp.adv
    mk = simulate_market(mp, n_paths, seed=seed)
    base = ac_params_from_market(mp, X, lam=0.0, temp_coeff=Y, perm_coeff=G)
    out: list[FrontierPoint] = []
    for k in urgencies:
        lam = lam_for_urgency(base, k)
        ac_tot = _evaluate(ac_trade_list(replace(base, lam=lam)), mk, mp, X, fee_bps)
        ac_std = float(ac_tot.std(ddof=1))
        lam_m, sq_tot = match_std_lam(ac_std, mp, X, mk, Y, G, fee_bps=fee_bps)
        matched = abs(float(sq_tot.std(ddof=1)) - ac_std) < 1e-6 * ac_std
        diff = ac_tot - sq_tot
        same = _evaluate(sqrt_schedule(mp, X, lam, Y, G, temp_exponent=0.5), mk, mp, X, fee_bps)
        out.append(
            FrontierPoint(
                kappa_T=float(k),
                lam=float(lam),
                ac_mean_bps=float(ac_tot.mean()),
                ac_std_bps=ac_std,
                sqrt_lam_matched=lam_m,
                sqrt_mean_bps=float(sq_tot.mean()),
                sqrt_std_bps=float(sq_tot.std(ddof=1)),
                paired_saving_bps=float(diff.mean()),
                paired_saving_se_bps=float(diff.std(ddof=1) / np.sqrt(len(diff))),
                ac_objective_usd=_objective_usd(ac_tot, X, mp.s0, lam),
                sqrt_objective_same_lam_usd=_objective_usd(same, X, mp.s0, lam),
                sqrt_std_same_lam_bps=float(same.std(ddof=1)),
                risk_matched=bool(matched),
            )
        )
    return out
