"""Slice 4 harness: static vs adaptive schedules on normal and event days.

Protocol (every number out of sample, every cost charged):

1. **Impact calibration** on synthetic metaorders (calibration seed 0), as
   in slice 2/3: sqrt temporary coefficient ``Y`` and permanent ``G``.
2. **Volume model** fitted on *training* days (seed 7, events at the same
   rate as evaluation): intraday profile ``u_hat`` and the nowcast shrinkage
   ``k0`` (:func:`execution_cost.volume_model.fit_nowcast`). No planner sees
   the simulator's true profile.
3. **Evaluation** on fresh days (seed 123, disjoint), common random numbers
   across schedules, true simulator parameters; 5 bps spread, sqrt
   temporary and linear permanent impact all charged; implementation
   shortfall in bps of arrival notional, reported for all / normal / event
   days.
4. Schedules: TWAP; VWAP on ``u_hat``; static SQRT-OPT on ``u_hat`` at the
   AC risk aversion ``lam`` for urgency ``kT``; ADAPTIVE (re-optimised every
   interval on the nowcast) at the same ``lam``; REPLAN (re-optimised with
   the day factor fixed at 1, a control that should equal static); POV-lag1,
   NOWCAST-POV (day-factor forecast) and SURGE-POV (plus the AR(1) surge
   term) at the slice-3 rate ``1.25 q``.

Paired differences vs static SQRT-OPT use the same days, so timing noise
cancels in their standard errors. ``utility`` is the realised mean-variance
objective ``E + lam Var`` of IS in price units, the quantity both static and
adaptive planners minimise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from execution_cost.adaptive import AdaptiveSqrtOpt, NowcastPOV
from execution_cost.calibration import CalibrationResult, calibrate_impact, generate_metaorders
from execution_cost.comparison import lam_for_urgency
from execution_cost.optimal_sqrt import SqrtExecutionProblem, solve_sqrt_schedule
from execution_cost.schedules import ac_params_from_market, twap_schedule, vwap_schedule
from execution_cost.shortfall import decompose_shortfall
from execution_cost.simulator import POV, EventParams, MarketParams, execute, simulate_market
from execution_cost.simulator import volume_profile
from execution_cost.volume_model import (
    fit_nowcast,
    fit_surge_nowcast,
    lag1_forecast,
    log_forecast_errors,
)

SPLITS = ("all", "normal", "event")
# (candidate, baseline): paired IS differences on the same days.
PAIRS = (
    ("ADAPTIVE", "SQRT-OPT"),
    ("REPLAN", "SQRT-OPT"),
    ("NOWCAST-POV", "POV-lag1"),
    ("SURGE-POV", "NOWCAST-POV"),
    ("SQRT-OPT", "TWAP"),
)


def static_sqrt_on_profile(mp, profile, X, lam, Y, G) -> np.ndarray:
    prob = SqrtExecutionProblem(
        X=float(X), profile=np.asarray(profile, float), adv=mp.adv, sigma_price=mp.sigma_price,
        temp_coeff=Y, perm_coeff=G, lam=lam, temp_exponent=0.5,
    )
    return solve_sqrt_schedule(prob).trades


def _masks(mk) -> dict[str, np.ndarray]:
    ev = mk.is_event if mk.is_event is not None else np.zeros(mk.n_paths, bool)
    return {"all": np.ones(mk.n_paths, bool), "normal": ~ev, "event": ev}


def _stats(total_bps, ref_bps, completion, mask, lam, notional) -> dict:
    t = total_bps[mask]
    d = (total_bps - ref_bps)[mask]
    n = int(mask.sum())
    cost = t / 1e4 * notional  # price units
    return {
        "n": n,
        "mean_bps": float(t.mean()),
        "std_bps": float(t.std(ddof=1)),
        "p95_bps": float(np.percentile(t, 95)),
        "diff_vs_static_bps": float(d.mean()),
        "diff_vs_static_se_bps": float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        "utility": float(cost.mean() + lam * cost.var(ddof=1)),
        "completion": float(completion[mask].mean()),
    }


@dataclass
class AdaptiveReport:
    q: float
    kT: float
    lam: float
    calibration: dict
    volume_model: dict
    forecast_rmse: dict
    n_event_days: int
    schedules: dict  # name -> split -> stats
    pairs: dict  # "A - B" -> split -> {mean_bps, se_bps}

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def evaluate_adaptive(
    market: MarketParams | None = None,
    q: float = 0.05,
    kT: float = 1.0,
    n_paths: int = 4000,
    seed: int = 123,
    train_seed: int = 7,
    n_train: int = 500,
    events: EventParams | None = EventParams(),
    calibration: CalibrationResult | None = None,
    calibration_seed: int = 0,
) -> AdaptiveReport:
    mp = market or MarketParams()
    if len({seed, train_seed, calibration_seed}) < 3:
        raise ValueError("evaluation, training and calibration seeds must differ")
    if calibration is None:
        orders = generate_metaorders(seed=calibration_seed, market=mp)
        calibration = calibrate_impact(orders, seed=calibration_seed, market=mp)
    Y, G = calibration.temp_sqrt_Y, max(calibration.perm_coeff, 0.0)
    X = q * mp.adv
    N = mp.n_intervals
    lam = lam_for_urgency(ac_params_from_market(mp, X, lam=0.0, temp_coeff=Y, perm_coeff=G), kT)
    rate = min(1.25 * q, 0.5)

    train = simulate_market(mp, n_train, seed=train_seed, events=events)
    nowcast, vm = fit_nowcast(train.volumes, mp.adv)
    surge, sm = fit_surge_nowcast(nowcast, train.volumes)
    u_hat = nowcast.profile
    u_true = volume_profile(N, mp.u_shape)

    mk = simulate_market(mp, n_paths, seed=seed, events=events)
    masks = _masks(mk)
    static = static_sqrt_on_profile(mp, u_hat, X, lam, Y, G)
    policies = {
        "TWAP": twap_schedule(X, N),
        "VWAP": vwap_schedule(X, u_hat),
        "SQRT-OPT": static,
        "REPLAN": AdaptiveSqrtOpt(mp, nowcast, X, lam, Y, G, adapt=False),
        "ADAPTIVE": AdaptiveSqrtOpt(mp, nowcast, X, lam, Y, G),
        "POV-lag1": POV(rate, lag=1),
        "NOWCAST-POV": NowcastPOV(rate, nowcast),
        "SURGE-POV": NowcastPOV(rate, surge),
    }
    totals = {name: decompose_shortfall(execute(pol, mk, mp, X)) for name, pol in policies.items()}
    ref = totals["SQRT-OPT"].total
    notional = X * mp.s0
    out = {}
    for name, br in totals.items():
        out[name] = {
            sp: {
                **_stats(br.total, ref, br.completion, masks[sp], lam, notional),
                "temporary_bps": float(br.temporary[masks[sp]].mean()),
                "timing_bps": float(br.timing[masks[sp]].mean()),
            }
            for sp in SPLITS
            if masks[sp].sum() > 1
        }

    pairs = {}
    for a, b in PAIRS:
        d = totals[a].total - totals[b].total
        pairs[f"{a} - {b}"] = {
            sp: {
                "mean_bps": float(d[m].mean()),
                "se_bps": float(d[m].std(ddof=1) / np.sqrt(m.sum())),
            }
            for sp, m in masks.items()
            if m.sum() > 1
        }
    e_now = log_forecast_errors(mk.volumes, nowcast.forecast)[:, 1:]
    e_lag = log_forecast_errors(mk.volumes, lambda o, j: lag1_forecast(o, j, u_hat, mp.adv))[:, 1:]
    e_srg = log_forecast_errors(mk.volumes, surge.forecast)[:, 1:]
    rmse = {
        sp: {
            "nowcast": float(np.sqrt(np.mean(e_now[m] ** 2))),
            "surge": float(np.sqrt(np.mean(e_srg[m] ** 2))),
            "lag1": float(np.sqrt(np.mean(e_lag[m] ** 2))),
        }
        for sp, m in masks.items()
        if m.sum() > 0
    }
    return AdaptiveReport(
        q=q, kT=kT, lam=lam, calibration=calibration.to_dict(),
        volume_model={
            "k0": vm["k0"],
            "k0_mse": vm["k0_mse"],
            "phi": sm["phi"],
            "n_train": n_train,
            "profile_max_rel_err": float(np.max(np.abs(u_hat / u_true - 1.0))),
        },
        forecast_rmse=rmse, n_event_days=int(masks["event"].sum()), schedules=out, pairs=pairs,
    )
