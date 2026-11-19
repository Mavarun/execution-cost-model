"""Slice 7 harness: does a sequential (CUSUM) gate save an alpha that dies, or flips?

Pre-registered protocol (frozen in :data:`CUSUM_PROTOCOL` and committed before any of its seeds
was run; development used synthetic normal draws in the unit tests and seed 11 on a 1,600-bar
path only):

* **Markets**: slice-5 defaults (rho 0.90, IC 0.10, 20 years). ``decay``: IC -> 0 at bar 2,520
  (slice 6's break); ``flip``: IC -> -0.05 at bar 2,520 (the alpha turns against you, a slice-6
  weakness); ``control``: no break. Costs: slice-1 defaults. Strategy: WF-PARTIAL, as in slice 6.
* **Arms**: ``ungated``, ``G252`` (slice 6's trailing gate, for reference) and ``CUSUM``
  (:func:`execution_cost.decay_gate.cusum_gate_states`, ARL target 2,520 bars, reference = the
  1,260 shadow bars before the first OOS bar).
* **Seeds**: 900-919 (fresh; slice 6 used 800-819), the same seeds for all three markets.
* **H8 (primary, one-sided paired t at 5%)**: on the post-break bars of ``decay``, CUSUM's mean
  net return (bps/day) beats ungated's.
* **H9 (secondary, one-sided paired t at 5%)**: the same on ``flip``.
* **Control (two-sided)**: CUSUM minus ungated in OOS net Sharpe on ``control``.
* **Decision**: the README recommends the CUSUM gate only if H8 passes **and** the control costs
  less than 0.10 net Sharpe. H9 is reported either way. Otherwise the gate ships off.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from execution_cost.alpha import AlphaParams, simulate_alpha_path, simulate_regime_path
from execution_cost.decay_gate import (
    CusumGateParams, GateParams, cusum_gate_states, gated_walk_forward, gated_walk_forward_states, shadow_record,
)
from execution_cost.metrics import annualized_sharpe
from execution_cost.walkforward import run_rules, walk_forward
from execution_cost.walkforward_eval import PROTOCOL, candidate_rules, cost_params, paired_seed_test

CUSUM_PROTOCOL = {
    "before": {"ic": 0.10, "rho": 0.90},
    "markets": {"decay": {"ic": 0.0, "rho": 0.90}, "flip": {"ic": -0.05, "rho": 0.90}, "control": None},
    "switch_at": 2520,
    "selector": "WF-PARTIAL",
    "cusum": {"arl0": 2520.0, "ref_bars": 1260},
    "seeds": tuple(range(900, 920)),
    "alpha_level": 0.05,
    "max_control_sharpe_loss": 0.10,
}
ARMS = ("ungated", "G252", "CUSUM")


def _path(seed: int, market: str, n: int | None, switch_at: int | None):
    base = AlphaParams(**PROTOCOL["market"])
    if n is not None:
        base = replace(base, n=int(n))
    sw = int(switch_at if switch_at is not None else CUSUM_PROTOCOL["switch_at"])
    after = CUSUM_PROTOCOL["markets"][market]
    if after is None:
        return simulate_alpha_path(base, seed=seed), sw
    return simulate_regime_path(replace(base, **CUSUM_PROTOCOL["before"]), replace(base, **after),
                                switch_at=sw, seed=seed), sw


def cusum_seed(seed: int, market: str = "decay", *, n: int | None = None, switch_at: int | None = None,
               min_train: int | None = None, test_len: int | None = None, ref_bars: int | None = None) -> dict:
    path, sw = _path(seed, market, n, switch_at)
    costs = cost_params()
    rules = candidate_rules()[CUSUM_PROTOCOL["selector"]]
    runs = run_rules(path, rules, costs)
    wf = walk_forward(path, rules, costs, min_train=int(min_train or PROTOCOL["min_train"]),
                      test_len=int(test_len or PROTOCOL["test_len"]), runs=runs)
    post = wf.oos_index >= sw
    cp = CusumGateParams(arl0=CUSUM_PROTOCOL["cusum"]["arl0"], ref_bars=int(ref_bars or CUSUM_PROTOCOL["cusum"]["ref_bars"]))
    on = cusum_gate_states(shadow_record(wf, runs), wf.fold_starts[0], cp)
    g = gated_walk_forward(path, wf, runs, costs, GateParams(lookback=252))
    c = gated_walk_forward_states(path, wf, runs, costs, on)
    res = {"ungated": (wf.net, wf.cost, np.ones(post.size)), "G252": (g.net, g.cost, g.on),
           "CUSUM": (c.net, c.cost, c.on)}
    out = {}
    for arm, (net, cost, state) in res.items():
        off_post = np.flatnonzero(state[post] == 0.0)
        out[arm] = {"post_bps": float(net[post].mean() * 1e4), "post_cost_bps": float(cost[post].mean() * 1e4),
                    "oos_sharpe": annualized_sharpe(net), "pre_sharpe": annualized_sharpe(net[~post]),
                    "share_on_pre": float(state[~post].mean()), "share_on_post": float(state[post].mean()),
                    "first_off_after_switch": int(off_post[0]) if off_post.size else None}
    return out


def run_cusum_study(seeds=None, **kw) -> dict:
    p = CUSUM_PROTOCOL
    seeds = tuple(seeds if seeds is not None else p["seeds"])
    rows = {m: [cusum_seed(s, m, **kw) for s in seeds] for m in p["markets"]}

    def arr(m, arm, key):
        return np.array([r[arm][key] for r in rows[m]], dtype=float)

    keys = ("post_bps", "post_cost_bps", "oos_sharpe", "pre_sharpe", "share_on_pre", "share_on_post")
    means = {m: {a: {k: float(np.mean(arr(m, a, k))) for k in keys} for a in ARMS} for m in p["markets"]}
    tests = {arm: {"decay_post_bps": paired_seed_test(arr("decay", arm, "post_bps"), arr("decay", "ungated", "post_bps"),
                                                      alternative="greater"),
                   "flip_post_bps": paired_seed_test(arr("flip", arm, "post_bps"), arr("flip", "ungated", "post_bps"),
                                                     alternative="greater"),
                   "control_sharpe": paired_seed_test(arr("control", arm, "oos_sharpe"),
                                                      arr("control", "ungated", "oos_sharpe"))}
             for arm in ("G252", "CUSUM")}
    t = tests["CUSUM"]
    h8 = bool(t["decay_post_bps"]["p"] < p["alpha_level"] and t["decay_post_bps"]["mean"] > 0)
    h9 = bool(t["flip_post_bps"]["p"] < p["alpha_level"] and t["flip_post_bps"]["mean"] > 0)
    ctl_ok = bool(t["control_sharpe"]["mean"] > -p["max_control_sharpe_loss"])
    delays = {m: [r["CUSUM"]["first_off_after_switch"] for r in rows[m]] for m in ("decay", "flip")}
    return {"seeds": list(seeds), "means": means, "tests": tests, "H8": h8, "H9": h9, "control_ok": ctl_ok,
            "recommend": bool(h8 and ctl_ok),
            "cusum_delay_bars": {m: {"median": float(np.median([d for d in v if d is not None])) if any(d is not None for d in v) else None,
                                     "never_off": int(sum(d is None for d in v))} for m, v in delays.items()}}


def reference_bias(seeds=None) -> dict:
    """Post-hoc reading (decides nothing): the CUSUM's alive reference vs what the alpha earns OOS.

    The reference is the last ``ref_bars`` shadow bars before the first OOS bar, which are the
    first fold's *in-sample* net returns of the rule chosen on them, so they are optimistic. On
    the no-break control this compares that reference mean with the OOS shadow mean, and reports
    the off-sum's realised drift per bar against the drift the ARL target was solved for.
    """
    p = CUSUM_PROTOCOL
    seeds = tuple(seeds if seeds is not None else p["seeds"])
    ref_m, oos_m, sds = [], [], []
    for s in seeds:
        path, _ = _path(s, "control", None, None)
        costs = cost_params()
        rules = candidate_rules()[p["selector"]]
        runs = run_rules(path, rules, costs)
        wf = walk_forward(path, rules, costs, min_train=PROTOCOL["min_train"], test_len=PROTOCOL["test_len"], runs=runs)
        sh = shadow_record(wf, runs)
        s0 = wf.fold_starts[0]
        ref = sh[s0 - p["cusum"]["ref_bars"]:s0]
        ref_m.append(float(ref.mean()))
        oos_m.append(float(sh[s0:].mean()))
        sds.append(float(ref.std(ddof=1)))
    ref_m, oos_m, sds = map(np.array, (ref_m, oos_m, sds))
    designed = 0.5 * ref_m / sds  # drift away from the off alarm the ARL target assumes
    realised = (oos_m - 0.5 * ref_m) / sds  # drift it actually gets while alive
    return {"ref_mean_bps": float(ref_m.mean() * 1e4), "oos_mean_bps": float(oos_m.mean() * 1e4),
            "seeds_ref_above_oos": int((ref_m > oos_m).sum()), "n": len(seeds),
            "median_designed_drift": float(np.median(designed)), "median_realised_drift": float(np.median(realised))}


# --------------------------------------------------------------------------------------------
# Slice 8 (frozen 2026-11-19, before any of its seeds was run).
#
# Slice 7's gate passed H8/H9 but cost 0.204 Sharpe on the control. The post-hoc reading
# (``reference_bias``, which selected nothing) found its in-sample reference overstated the OOS
# level: on slice 7's control the OOS/reference mean ratio was 0.77 (mean) / 0.75 (median). Slice 8
# changes exactly one thing, fixed from that reading before any slice-8 seed was run: the alive
# level is shrunk to 0.75 x the reference (``CusumGateParams.shrink``). Everything else is
# slice 7's protocol. Seeds 1000-1019 are disjoint from every earlier protocol (slice 7's
# 900-919 had in fact been used before, by slice 6's vol-target H7, which is disclosed).
#
# * H10 (the one that failed): CUSUM-S0.75 costs less than 0.10 OOS net Sharpe vs ungated on
#   the no-break control (point estimate, as in slice 7).
# * H11: CUSUM-S0.75 beats ungated on post-decay net bps/day, one-sided paired t at 5%.
# * Recommended only if H10 and H11 both pass; otherwise the gate stays off. The flip and the
#   unshrunk slice-7 gate are reported alongside and decide nothing.
# --------------------------------------------------------------------------------------------
SHRINK_PROTOCOL = {
    "shrink": 0.75,
    "seeds": tuple(range(1000, 1020)),
    "alpha_level": 0.05,
    "max_control_sharpe_loss": 0.10,
}
SHRINK_ARMS = ("ungated", "CUSUM", "CUSUM-S")


def shrink_seed(seed: int, market: str = "decay") -> dict:
    path, sw = _path(seed, market, None, None)
    costs = cost_params()
    rules = candidate_rules()[CUSUM_PROTOCOL["selector"]]
    runs = run_rules(path, rules, costs)
    wf = walk_forward(path, rules, costs, min_train=PROTOCOL["min_train"], test_len=PROTOCOL["test_len"], runs=runs)
    post = wf.oos_index >= sw
    sh, s0 = shadow_record(wf, runs), wf.fold_starts[0]
    c = CUSUM_PROTOCOL["cusum"]
    res = {"ungated": (wf.net, np.ones(post.size))}
    for arm, shrink in (("CUSUM", 1.0), ("CUSUM-S", SHRINK_PROTOCOL["shrink"])):
        on = cusum_gate_states(sh, s0, CusumGateParams(arl0=c["arl0"], ref_bars=c["ref_bars"], shrink=shrink))
        res[arm] = (gated_walk_forward_states(path, wf, runs, costs, on).net, on)
    out = {}
    for arm, (net, state) in res.items():
        off_post = np.flatnonzero(state[post] == 0.0)
        out[arm] = {"post_bps": float(net[post].mean() * 1e4), "oos_sharpe": annualized_sharpe(net),
                    "share_on_pre": float(state[~post].mean()), "share_on_post": float(state[post].mean()),
                    "first_off_after_switch": int(off_post[0]) if off_post.size else None}
    return out


def run_shrink_study(seeds=None) -> dict:
    p = SHRINK_PROTOCOL
    seeds = tuple(seeds if seeds is not None else p["seeds"])
    markets = tuple(CUSUM_PROTOCOL["markets"])
    rows = {m: [shrink_seed(s, m) for s in seeds] for m in markets}

    def arr(m, arm, key):
        return np.array([r[arm][key] for r in rows[m]], dtype=float)

    keys = ("post_bps", "oos_sharpe", "share_on_pre", "share_on_post")
    means = {m: {a: {k: float(np.mean(arr(m, a, k))) for k in keys} for a in SHRINK_ARMS} for m in markets}
    tests = {arm: {"decay_post_bps": paired_seed_test(arr("decay", arm, "post_bps"), arr("decay", "ungated", "post_bps"),
                                                      alternative="greater"),
                   "flip_post_bps": paired_seed_test(arr("flip", arm, "post_bps"), arr("flip", "ungated", "post_bps"),
                                                     alternative="greater"),
                   "control_sharpe": paired_seed_test(arr("control", arm, "oos_sharpe"),
                                                      arr("control", "ungated", "oos_sharpe"))}
             for arm in ("CUSUM", "CUSUM-S")}
    t = tests["CUSUM-S"]
    h10 = bool(t["control_sharpe"]["mean"] > -p["max_control_sharpe_loss"])
    h11 = bool(t["decay_post_bps"]["p"] < p["alpha_level"] and t["decay_post_bps"]["mean"] > 0)
    delays = {}
    for m in ("decay", "flip"):
        d = [r["CUSUM-S"]["first_off_after_switch"] for r in rows[m]]
        got = [x for x in d if x is not None]
        delays[m] = {"median": float(np.median(got)) if got else None, "never_off": int(sum(x is None for x in d))}
    return {"seeds": list(seeds), "means": means, "tests": tests, "H10": h10, "H11": h11,
            "recommend": bool(h10 and h11), "shrunk_delay_bars": delays}
