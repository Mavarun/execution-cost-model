"""Slice 5: walk-forward refit of the volume model across a regime break.

Slice 4 fitted the intraday profile ``u_hat``, the nowcast shrinkage ``k0``
and the surge slope ``phi`` **once**, on 500 training days, and listed "no
rolling refit across regimes" as a weakness. This module tests it with a
strictly causal walk-forward:

* **Days** arrive in blocks of ``block`` days. Blocks before ``break_block``
  use the pre-break market (``MarketParams`` / ``EventParams``), blocks from
  ``break_block`` on use the post-break one (e.g. a steeper U-shape and more
  frequent surges). Every block has its own seed, so the realised volumes of
  block ``b`` are the same whatever any model does.
* **FIXED**: fitted once on the ``n_train`` pre-break training days and never
  updated (the slice-4 practice).
* **ROLLING(W)**: before each block, refit ``u_hat``, ``k0`` and ``phi`` on the
  last ``W`` *realised* days (training days plus earlier evaluation blocks);
  nothing from the block itself or later is used.
* **EXPANDING**: the same on every realised day so far.

Every model drives the same implementable SURGE-POV policy (POV sized on the
day-factor nowcast plus the AR(1) surge term, rate ``1.25 q``) on the same days
(common random numbers). The full cost is charged: 5 bps spread, sqrt
temporary and linear permanent impact, plus opportunity cost of unfilled
shares at the close (``decompose_shortfall``). Reported per regime: forecast
log RMSE, mean IS in bps, completion, and paired IS differences vs FIXED with
standard errors on the same days. Impact is calibrated once on metaorders from
a disjoint seed, as in slices 2-4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from execution_cost.adaptive import NowcastPOV
from execution_cost.calibration import CalibrationResult, calibrate_impact, generate_metaorders
from execution_cost.shortfall import decompose_shortfall
from execution_cost.simulator import EventParams, MarketParams, execute, simulate_market
from execution_cost.volume_model import fit_nowcast, fit_surge_nowcast, log_forecast_errors

REGIMES = ("pre", "post")


@dataclass(frozen=True)
class WalkForwardSpec:
    pre_market: MarketParams = field(default_factory=MarketParams)
    post_market: MarketParams = field(default_factory=lambda: MarketParams(u_shape=3.0))
    pre_events: EventParams | None = field(default_factory=EventParams)
    post_events: EventParams | None = field(default_factory=lambda: EventParams(prob=0.30))
    n_train: int = 500
    block: int = 50
    n_blocks: int = 40
    break_block: int = 10
    windows: tuple[int, ...] = (100, 250)
    q: float = 0.05
    train_seed: int = 7
    eval_seed: int = 1000
    calibration_seed: int = 0

    def __post_init__(self) -> None:
        if self.pre_market.adv != self.post_market.adv or self.pre_market.n_intervals != self.post_market.n_intervals:
            raise ValueError("adv and n_intervals must not change at the break")
        if not 0 < self.break_block < self.n_blocks:
            raise ValueError("break_block must leave pre and post blocks")
        if min(self.windows) < 2 or self.block < 2:
            raise ValueError("windows and block need >= 2 days")
        if self.calibration_seed in (self.train_seed,) or self.eval_seed <= self.train_seed:
            raise ValueError("seeds must be disjoint (eval seeds are eval_seed + block)")


def block_market(spec: WalkForwardSpec, b: int):
    post = b >= spec.break_block
    mp = spec.post_market if post else spec.pre_market
    ev = spec.post_events if post else spec.pre_events
    return mp, simulate_market(mp, spec.block, seed=spec.eval_seed + b, events=ev)


def fit_models(history: np.ndarray, adv: float):
    """Day-factor + surge nowcast fitted on ``history`` (days, intervals) only."""
    base, vm = fit_nowcast(history, adv)
    surge, sm = fit_surge_nowcast(base, history)
    return surge, {"k0": vm["k0"], "phi": sm["phi"], "n_days": int(history.shape[0])}


@dataclass
class WalkForwardReport:
    spec: dict
    calibration: dict
    models: tuple[str, ...]
    per_regime: dict  # regime -> model -> stats
    pairs: dict  # "M - FIXED" -> regime -> {mean_bps, se_bps}
    fits: dict  # model -> list of (block, k0, phi, n_days)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def run_walk_forward(spec: WalkForwardSpec | None = None, calibration: CalibrationResult | None = None) -> WalkForwardReport:
    spec = spec or WalkForwardSpec()
    mp0 = spec.pre_market
    if calibration is None:
        orders = generate_metaorders(seed=spec.calibration_seed, market=mp0)
        calibration = calibrate_impact(orders, seed=spec.calibration_seed, market=mp0)
    X = spec.q * mp0.adv
    rate = min(1.25 * spec.q, 0.5)
    train = simulate_market(mp0, spec.n_train, seed=spec.train_seed, events=spec.pre_events).volumes
    models = ("FIXED", *[f"ROLLING-{w}" for w in spec.windows], "EXPANDING")
    fixed, info0 = fit_models(train, mp0.adv)
    history = train
    is_bps = {m: [] for m in models}
    sq_err = {m: [] for m in models}
    filled = {m: [] for m in models}
    regime = []
    fits = {m: [] for m in models}
    fits["FIXED"].append((-1, info0["k0"], info0["phi"], info0["n_days"]))
    for b in range(spec.n_blocks):
        fitted = {"FIXED": fixed}
        for w in spec.windows:
            fitted[f"ROLLING-{w}"], info = fit_models(history[-w:], mp0.adv)
            fits[f"ROLLING-{w}"].append((b, info["k0"], info["phi"], info["n_days"]))
        fitted["EXPANDING"], info = fit_models(history, mp0.adv)
        fits["EXPANDING"].append((b, info["k0"], info["phi"], info["n_days"]))
        mp, mk = block_market(spec, b)
        for m in models:
            br = decompose_shortfall(execute(NowcastPOV(rate, fitted[m]), mk, mp, X))
            is_bps[m].append(br.total)
            filled[m].append(br.completion)
            e = log_forecast_errors(mk.volumes, fitted[m].forecast)[:, 1:]
            sq_err[m].append((e**2).mean(axis=1))
        regime.append(np.full(spec.block, b >= spec.break_block))
        history = np.vstack([history, mk.volumes])  # realised only after the block is traded
    post = np.concatenate(regime)
    masks = {"pre": ~post, "post": post}
    cat = {m: (np.concatenate(is_bps[m]), np.concatenate(sq_err[m]), np.concatenate(filled[m])) for m in models}
    per_regime = {
        rg: {
            m: {
                "n_days": int(mk_.sum()),
                "mean_bps": float(cat[m][0][mk_].mean()),
                "std_bps": float(cat[m][0][mk_].std(ddof=1)),
                "forecast_log_rmse": float(np.sqrt(cat[m][1][mk_].mean())),
                "completion": float(cat[m][2][mk_].mean()),
            }
            for m in models
        }
        for rg, mk_ in masks.items()
    }
    pairs = {}
    for m in models[1:]:
        d = cat[m][0] - cat["FIXED"][0]
        pairs[f"{m} - FIXED"] = {
            rg: {"mean_bps": float(d[mk_].mean()), "se_bps": float(d[mk_].std(ddof=1) / np.sqrt(mk_.sum()))}
            for rg, mk_ in masks.items()
        }
    spec_d = {k: getattr(spec, k) for k in ("n_train", "block", "n_blocks", "break_block", "windows", "q",
                                            "train_seed", "eval_seed", "calibration_seed")}
    spec_d["pre_u_shape"], spec_d["post_u_shape"] = spec.pre_market.u_shape, spec.post_market.u_shape
    spec_d["pre_event_prob"] = spec.pre_events.prob if spec.pre_events else 0.0
    spec_d["post_event_prob"] = spec.post_events.prob if spec.post_events else 0.0
    return WalkForwardReport(spec=spec_d, calibration=calibration.to_dict(), models=models,
                             per_regime=per_regime, pairs=pairs, fits=fits)
