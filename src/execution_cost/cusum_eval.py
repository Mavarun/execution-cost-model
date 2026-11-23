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


def control_off_spells(shrink: float, seeds=None) -> dict:
    """Post-hoc slice-8 reading (decides nothing): false alarms and off-spells on the no-break control.

    Splits the control's off time into how often the gate falsely switches off (OOS bars per
    alarm, vs the 2,520 target) and how long it stays off each time (spells still open at the
    end are cut there, so lengths are a lower bound).
    """
    c = CUSUM_PROTOCOL["cusum"]
    seeds = tuple(seeds if seeds is not None else SHRINK_PROTOCOL["seeds"])
    spells, alarms, bars = [], 0, 0
    for s in seeds:
        path, _ = _path(s, "control", None, None)
        costs = cost_params()
        rules = candidate_rules()[CUSUM_PROTOCOL["selector"]]
        runs = run_rules(path, rules, costs)
        wf = walk_forward(path, rules, costs, min_train=PROTOCOL["min_train"], test_len=PROTOCOL["test_len"], runs=runs)
        on = cusum_gate_states(shadow_record(wf, runs), wf.fold_starts[0],
                               CusumGateParams(arl0=c["arl0"], ref_bars=c["ref_bars"], shrink=shrink))
        d = np.diff(np.r_[1.0, on, 1.0])
        starts, ends = np.flatnonzero(d == -1), np.flatnonzero(d == 1)
        spells += list(ends - starts)
        alarms += starts.size
        bars += on.size
    spells = np.array(spells, dtype=float)
    return {"alarms": int(alarms), "bars_per_alarm": float(bars / alarms) if alarms else float("inf"),
            "median_off_spell": float(np.median(spells)) if spells.size else 0.0,
            "mean_off_spell": float(spells.mean()) if spells.size else 0.0, "share_off": float(spells.sum() / bars)}


# --------------------------------------------------------------------------------------------
# Slice 9 (frozen 2026-11-20, before any of its seeds was run).
#
# Slice 8's post-hoc reading (``control_off_spells``) split the control cost into false alarms,
# already near target after shrinking (one per ~2,000 OOS bars), and off-spells of ~500 bars,
# because switching back on needs the full level h while the on-sum climbs only slowly. Slice 9
# changes exactly one thing on top of slice 8's gate: the re-entry level is 0.25 x h
# (``CusumGateParams.on_frac``). 0.25 is fixed a priori from that reading: the on-sum climbs
# roughly linearly, so spells should shrink about 4x, and slice 8's 25% off share times 0.25 is
# about 6%, inside the budget with room for the extra alarms that earlier re-entry allows. Not
# tuned on any seed. Seeds 1100-1119 are disjoint from every earlier protocol.
#
# * H12: CUSUM-SR (shrink 0.75, on_frac 0.25) costs less than 0.10 OOS net Sharpe vs ungated on
#   the no-break control (point estimate, as in slices 7 and 8).
# * H13: CUSUM-SR beats ungated on post-decay net bps/day, one-sided paired t at 5%.
# * Recommended (the gate's default turns on) only if both pass. The flip market and slice 8's
#   CUSUM-S are reported alongside and decide nothing.
# --------------------------------------------------------------------------------------------
REENTRY_PROTOCOL = {
    "shrink": 0.75,
    "on_frac": 0.25,
    "seeds": tuple(range(1100, 1120)),
    "alpha_level": 0.05,
    "max_control_sharpe_loss": 0.10,
}
REENTRY_ARMS = ("ungated", "CUSUM-S", "CUSUM-SR")


def _wf_for(seed: int, market: str):
    path, sw = _path(seed, market, None, None)
    costs = cost_params()
    rules = candidate_rules()[CUSUM_PROTOCOL["selector"]]
    runs = run_rules(path, rules, costs)
    wf = walk_forward(path, rules, costs, min_train=PROTOCOL["min_train"], test_len=PROTOCOL["test_len"], runs=runs)
    return path, sw, costs, runs, wf


def reentry_seed(seed: int, market: str = "decay") -> dict:
    path, sw, costs, runs, wf = _wf_for(seed, market)
    post = wf.oos_index >= sw
    sh, s0 = shadow_record(wf, runs), wf.fold_starts[0]
    c, p = CUSUM_PROTOCOL["cusum"], REENTRY_PROTOCOL
    res = {"ungated": (wf.net, np.ones(post.size))}
    for arm, on_frac in (("CUSUM-S", 1.0), ("CUSUM-SR", p["on_frac"])):
        gp = CusumGateParams(arl0=c["arl0"], ref_bars=c["ref_bars"], shrink=p["shrink"], on_frac=on_frac)
        on = cusum_gate_states(sh, s0, gp)
        res[arm] = (gated_walk_forward_states(path, wf, runs, costs, on).net, on)
    out = {}
    for arm, (net, state) in res.items():
        off_post = np.flatnonzero(state[post] == 0.0)
        switches = int(np.abs(np.diff(state)).sum())
        out[arm] = {"post_bps": float(net[post].mean() * 1e4), "oos_sharpe": annualized_sharpe(net),
                    "share_on_pre": float(state[~post].mean()), "share_on_post": float(state[post].mean()),
                    "switches": switches,
                    "first_off_after_switch": int(off_post[0]) if off_post.size else None}
    return out


def run_reentry_study(seeds=None) -> dict:
    p = REENTRY_PROTOCOL
    seeds = tuple(seeds if seeds is not None else p["seeds"])
    markets = tuple(CUSUM_PROTOCOL["markets"])
    rows = {m: [reentry_seed(s, m) for s in seeds] for m in markets}

    def arr(m, arm, key):
        return np.array([r[arm][key] for r in rows[m]], dtype=float)

    keys = ("post_bps", "oos_sharpe", "share_on_pre", "share_on_post", "switches")
    means = {m: {a: {k: float(np.mean(arr(m, a, k))) for k in keys} for a in REENTRY_ARMS} for m in markets}
    tests = {arm: {"decay_post_bps": paired_seed_test(arr("decay", arm, "post_bps"), arr("decay", "ungated", "post_bps"),
                                                      alternative="greater"),
                   "flip_post_bps": paired_seed_test(arr("flip", arm, "post_bps"), arr("flip", "ungated", "post_bps"),
                                                     alternative="greater"),
                   "control_sharpe": paired_seed_test(arr("control", arm, "oos_sharpe"),
                                                      arr("control", "ungated", "oos_sharpe"))}
             for arm in ("CUSUM-S", "CUSUM-SR")}
    t = tests["CUSUM-SR"]
    h12 = bool(t["control_sharpe"]["mean"] > -p["max_control_sharpe_loss"])
    h13 = bool(t["decay_post_bps"]["p"] < p["alpha_level"] and t["decay_post_bps"]["mean"] > 0)
    return {"seeds": list(seeds), "means": means, "tests": tests, "H12": h12, "H13": h13,
            "recommend": bool(h12 and h13)}


def decay_reentries(on_frac: float, seeds=None, market: str = "decay") -> dict:
    """Post-hoc slice-9 reading (decides nothing): what the gate does after a real decay.

    After the switch bar, counts re-entries (off -> on) per seed and the on-spells that follow
    them; with a dead alpha every re-entry is a false one.
    """
    c, p = CUSUM_PROTOCOL["cusum"], REENTRY_PROTOCOL
    seeds = tuple(seeds if seeds is not None else p["seeds"])
    reent, spells, bars = 0, [], 0
    for s in seeds:
        path, sw, costs, runs, wf = _wf_for(s, market)
        on = cusum_gate_states(shadow_record(wf, runs), wf.fold_starts[0],
                               CusumGateParams(arl0=c["arl0"], ref_bars=c["ref_bars"], shrink=p["shrink"], on_frac=on_frac))
        post = on[wf.oos_index >= sw]
        d = np.diff(np.r_[post[:1], post, 0.0])
        ups, downs = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        reent += ups.size
        # on-spells that start with a re-entry: each up is followed by the next down
        for u in ups:
            nxt = downs[downs >= u]
            spells.append(int(nxt[0] - u) if nxt.size else int(post.size - u))
        bars += post.size
    spells = np.array(spells, dtype=float)
    return {"reentries": int(reent), "per_seed": reent / len(seeds),
            "median_on_spell": float(np.median(spells)) if spells.size else 0.0,
            "mean_on_spell": float(spells.mean()) if spells.size else 0.0, "post_bars": int(bars)}


# --------------------------------------------------------------------------------------------
# Slice 10 (frozen 2026-11-20, before any of its seeds was run).
#
# Slice 9's post-hoc reading found slice 8's gate (CUSUM-S0.75, shrink 0.75, symmetric re-entry)
# cost 0.172 Sharpe on the control on seeds 1000-1019 but only 0.051 on 1100-1119: two 20-seed
# estimates of one quantity that disagree by more than their standard errors suggest. A single
# 20-seed run cannot settle whether that gate meets the budget. Slice 10 re-runs exactly slice 8's
# arms on 60 fresh seeds (1200-1259), three times the power, and decides on them alone (the
# earlier seed sets have been looked at; they are reported but not pooled into the decision).
#
# * H14: CUSUM-S0.75 costs less than 0.10 OOS net Sharpe vs ungated on the no-break control
#   (point estimate, as in slices 7-9); its 95% CI is reported.
# * H15: CUSUM-S0.75 beats ungated on post-decay net bps/day, one-sided paired t at 5%.
# * The shrunk gate is recommended only if both pass. ``CusumGateParams`` defaults do not change
#   in this slice either way; a pass is recorded as the recommended setting in the README.
# --------------------------------------------------------------------------------------------
POOLED_PROTOCOL = {
    "shrink": 0.75,
    "seeds": tuple(range(1200, 1260)),
    "alpha_level": 0.05,
    "max_control_sharpe_loss": 0.10,
}


def run_pooled_confirm(seeds=None) -> dict:
    """Slice 8's arms (ungated, CUSUM, CUSUM-S0.75) on the slice-10 seeds, decided by H14 / H15."""
    p = POOLED_PROTOCOL
    o = run_shrink_study(seeds if seeds is not None else p["seeds"])
    t = o["tests"]["CUSUM-S"]
    c = t["control_sharpe"]
    h14 = bool(c["mean"] > -p["max_control_sharpe_loss"])
    h15 = bool(t["decay_post_bps"]["p"] < p["alpha_level"] and t["decay_post_bps"]["mean"] > 0)
    ci = [float(c["mean"] - 1.96 * c["se"]), float(c["mean"] + 1.96 * c["se"])]
    return {**{k: o[k] for k in ("seeds", "means", "tests", "shrunk_delay_bars")},
            "H14": h14, "H15": h15, "recommend": bool(h14 and h15), "control_cost_ci": ci}


def control_cost_by_seed_set() -> dict:
    """Post-hoc slice-10 reading (decides nothing): CUSUM-S0.75's control cost on each seed set that
    has tested it (slices 8, 9 and 10), the 100-seed pooled estimate, and a one-way ANOVA p for
    whether the sets differ by more than chance."""
    from scipy import stats

    sets = {"slice 8 (1000-1019)": SHRINK_PROTOCOL["seeds"], "slice 9 (1100-1119)": REENTRY_PROTOCOL["seeds"],
            "slice 10 (1200-1259)": POOLED_PROTOCOL["seeds"]}
    diffs = {}
    for name, seeds in sets.items():
        rows = [shrink_seed(s, "control") for s in seeds]
        diffs[name] = np.array([r["CUSUM-S"]["oos_sharpe"] - r["ungated"]["oos_sharpe"] for r in rows])
    allv = np.concatenate(list(diffs.values()))
    return {"per_set": {k: {"cost": float(-v.mean()), "se": float(v.std(ddof=1) / np.sqrt(v.size)), "n": int(v.size)}
                        for k, v in diffs.items()},
            "pooled": {"cost": float(-allv.mean()), "se": float(allv.std(ddof=1) / np.sqrt(allv.size)), "n": int(allv.size)},
            "anova_p": float(stats.f_oneway(*diffs.values()).pvalue)}


# --------------------------------------------------------------------------------------------
# Slice 11 (frozen 2026-11-24, before any of its seeds was run).
#
# Slice 10 recommended CUSUM-S0.75 on a point estimate (0.096 Sharpe control cost) whose 95% CI
# (0.060-0.133) straddles the 0.10 budget, and the 100-seed pooled cost is 0.102. Its cost comes
# from false off-spells of a median 440 bars (slice 10 post-hoc). Slice 9 showed that lowering the
# re-entry level shortens them but gives up the decay protection. Slice 11 caps an off-spell
# instead (``CusumGateParams.max_off`` = 126 bars, about a quarter of the median spell, fixed a
# priori): after 126 bars off the gate switches on and must re-detect at the full h.
#
# * H16: CUSUM-S0.75-C126's control cost, as the *upper* end of its 95% CI, is below 0.10 Sharpe
#   (slice 10's weakness: bound the CI, not the mean).
# * H17: CUSUM-S0.75-C126 beats ungated on post-decay net bps/day, one-sided paired t at 5%.
# * H18 (reported, decides nothing alone): its post-decay gain is at least half of CUSUM-S0.75's on
#   the same seeds.
# * Recommended over CUSUM-S0.75 only if H16 and H17 pass. Defaults do not change either way.
#
# Seeds 1300-1359 (60, fresh). Dev check (never used for a decision): on a synthetic alive-then-dead
# shadow (seeds 1-5, tests/test_cusum_offspell_cap.py) the cap cuts the off share after the death
# from 0.46 to 0.11 on average; on the real pipeline with dev seeds 1-5, CUSUM-S0.75-C126's control
# cost CI was [-0.020, 0.082] and its post-decay gain +0.196 bps/day (p 0.042, 4/5), against +0.323
# for the uncapped gate. Expectation: H16 passes; H17 passes narrowly; H18 passes.
# --------------------------------------------------------------------------------------------
OFFCAP_PROTOCOL = {
    "shrink": 0.75,
    "max_off": 126,
    "seeds": tuple(range(1300, 1360)),
    "alpha_level": 0.05,
    "max_control_sharpe_loss": 0.10,
}
OFFCAP_ARMS = ("ungated", "CUSUM-S", "CUSUM-SC")


def offcap_seed(seed: int, market: str = "decay") -> dict:
    path, sw, costs, runs, wf = _wf_for(seed, market)
    post = wf.oos_index >= sw
    sh, s0 = shadow_record(wf, runs), wf.fold_starts[0]
    c, p = CUSUM_PROTOCOL["cusum"], OFFCAP_PROTOCOL
    res = {"ungated": (wf.net, np.ones(post.size))}
    for arm, cap in (("CUSUM-S", None), ("CUSUM-SC", p["max_off"])):
        gp = CusumGateParams(arl0=c["arl0"], ref_bars=c["ref_bars"], shrink=p["shrink"], max_off=cap)
        on = cusum_gate_states(sh, s0, gp)
        res[arm] = (gated_walk_forward_states(path, wf, runs, costs, on).net, on)
    out = {}
    for arm, (net, state) in res.items():
        out[arm] = {"post_bps": float(net[post].mean() * 1e4), "oos_sharpe": annualized_sharpe(net),
                    "share_on_pre": float(state[~post].mean()), "share_on_post": float(state[post].mean()),
                    "switches": int(np.abs(np.diff(state)).sum())}
    return out


def run_offcap_study(seeds=None) -> dict:
    p = OFFCAP_PROTOCOL
    seeds = tuple(seeds if seeds is not None else p["seeds"])
    markets = tuple(CUSUM_PROTOCOL["markets"])
    rows = {m: [offcap_seed(s, m) for s in seeds] for m in markets}

    def arr(m, arm, key):
        return np.array([r[arm][key] for r in rows[m]], dtype=float)

    keys = ("post_bps", "oos_sharpe", "share_on_pre", "share_on_post", "switches")
    means = {m: {a: {k: float(np.mean(arr(m, a, k))) for k in keys} for a in OFFCAP_ARMS} for m in markets}
    tests = {arm: {"decay_post_bps": paired_seed_test(arr("decay", arm, "post_bps"), arr("decay", "ungated", "post_bps"),
                                                      alternative="greater"),
                   "flip_post_bps": paired_seed_test(arr("flip", arm, "post_bps"), arr("flip", "ungated", "post_bps"),
                                                     alternative="greater"),
                   "control_sharpe": paired_seed_test(arr("control", arm, "oos_sharpe"),
                                                      arr("control", "ungated", "oos_sharpe"))}
             for arm in ("CUSUM-S", "CUSUM-SC")}
    cis = {arm: [float(-(t["control_sharpe"]["mean"] + 1.96 * t["control_sharpe"]["se"])),
                 float(-(t["control_sharpe"]["mean"] - 1.96 * t["control_sharpe"]["se"]))] for arm, t in tests.items()}
    t = tests["CUSUM-SC"]
    h16 = bool(cis["CUSUM-SC"][1] < p["max_control_sharpe_loss"])
    h17 = bool(t["decay_post_bps"]["p"] < p["alpha_level"] and t["decay_post_bps"]["mean"] > 0)
    h18 = bool(t["decay_post_bps"]["mean"] >= 0.5 * tests["CUSUM-S"]["decay_post_bps"]["mean"])
    return {"seeds": list(seeds), "means": means, "tests": tests, "control_cost_ci": cis,
            "H16": h16, "H17": h17, "H18": h18, "recommend": bool(h16 and h17)}
