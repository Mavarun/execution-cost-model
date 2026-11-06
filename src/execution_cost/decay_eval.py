"""Slice 6 harness: does a trailing-performance gate save an alpha that dies?

Pre-registered protocol (frozen in :data:`DECAY_PROTOCOL` and committed before any
decay seed was run; development used seed 3 on a 1,500-bar path in the unit tests
only):

* **Market**: slice-5 defaults (rho 0.90, IC 0.10, 20 years). In the *decay* market
  the IC drops to **0** at bar 2,520 (mid-OOS) and rho is unchanged; the *control*
  market has no decay. Costs: slice-1 defaults.
* **Strategy**: WF-PARTIAL (slice 5's recommended selector), expanding window,
  5 training years, annual folds.
* **Gates**: ``G252`` (primary; flat when the trailing 252-bar mean shadow net return
  is negative), ``G126`` and ``G504`` (secondary).
* **Seeds**: 800-819, the same seeds for the decay and control markets.
* **H6 (primary, one-sided paired t at 5%)**: on the post-decay bars ``[2520, 5040)``,
  G252's mean net return (bps/day) beats ungated WF-PARTIAL's. Sharpe is not used for
  H6 because a flat strategy has no Sharpe.
* **Control (two-sided)**: on the no-decay market, G252 minus ungated in OOS net Sharpe.
* **Decision**: the README recommends G252 only if H6 passes **and** the control
  costs less than ``max_control_sharpe_loss`` (0.10) net Sharpe. Otherwise the gate
  ships off (documented, not recommended).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from execution_cost.alpha import AlphaParams, simulate_alpha_path, simulate_regime_path
from execution_cost.decay_gate import GateParams, gated_walk_forward
from execution_cost.metrics import annualized_sharpe
from execution_cost.walkforward import run_rules, walk_forward
from execution_cost.walkforward_eval import PROTOCOL, candidate_rules, cost_params, paired_seed_test

DECAY_PROTOCOL = {
    "before": {"ic": 0.10, "rho": 0.90},
    "after": {"ic": 0.0, "rho": 0.90},
    "switch_at": 2520,
    "selector": "WF-PARTIAL",
    "gates": {"G252": 252, "G126": 126, "G504": 504},
    "primary": "G252",
    "seeds": tuple(range(800, 820)),
    "alpha_level": 0.05,
    "max_control_sharpe_loss": 0.10,
}


def _path(seed: int, decay: bool, n: int | None = None, switch_at: int | None = None):
    base = AlphaParams(**PROTOCOL["market"])
    if n is not None:
        base = replace(base, n=int(n))
    sw = int(switch_at if switch_at is not None else DECAY_PROTOCOL["switch_at"])
    if not decay:
        return simulate_alpha_path(base, seed=seed), sw
    p = DECAY_PROTOCOL
    return simulate_regime_path(replace(base, **p["before"]), replace(base, **p["after"]), switch_at=sw,
                                seed=seed), sw


def decay_seed(seed: int, *, decay: bool = True, n: int | None = None, switch_at: int | None = None,
               min_train: int | None = None, test_len: int | None = None) -> dict:
    """Ungated vs every gate on one path; post-switch mean net (bps/day) and Sharpe."""
    path, sw = _path(seed, decay, n, switch_at)
    costs = cost_params()
    rules = candidate_rules()[DECAY_PROTOCOL["selector"]]
    runs = run_rules(path, rules, costs)
    wf = walk_forward(path, rules, costs, min_train=int(min_train or PROTOCOL["min_train"]),
                      test_len=int(test_len or PROTOCOL["test_len"]), runs=runs)
    post = wf.oos_index >= sw
    out = {"ungated": {"post_bps": float(wf.net[post].mean() * 1e4),
                       "post_cost_bps": float(wf.cost[post].mean() * 1e4),
                       "oos_sharpe": annualized_sharpe(wf.net),
                       "pre_sharpe": annualized_sharpe(wf.net[~post]),
                       "share_on_pre": 1.0, "share_on_post": 1.0, "first_off_after_switch": None}}
    for name, lb in DECAY_PROTOCOL["gates"].items():
        g = gated_walk_forward(path, wf, runs, costs, GateParams(lookback=lb))
        off_post = np.flatnonzero(g.on[post] == 0.0)
        out[name] = {"post_bps": float(g.net[post].mean() * 1e4),
                     "post_cost_bps": float(g.cost[post].mean() * 1e4),
                     "oos_sharpe": annualized_sharpe(g.net),
                     "pre_sharpe": annualized_sharpe(g.net[~post]),
                     "share_on_pre": float(g.on[~post].mean()),
                     "share_on_post": float(g.on[post].mean()),
                     "first_off_after_switch": int(off_post[0]) if off_post.size else None}
    return out


def run_decay_study(seeds=None, **kw) -> dict:
    p = DECAY_PROTOCOL
    seeds = tuple(seeds if seeds is not None else p["seeds"])
    dec = [decay_seed(s, decay=True, **kw) for s in seeds]
    ctl = [decay_seed(s, decay=False, **kw) for s in seeds]
    arms = ["ungated", *p["gates"]]

    def arr(rows, arm, key):
        return np.array([r[arm][key] for r in rows], dtype=float)

    def mean_table(rows):
        keys = ("post_bps", "post_cost_bps", "oos_sharpe", "pre_sharpe", "share_on_pre", "share_on_post")
        return {a: {k: float(np.mean(arr(rows, a, k))) for k in keys} for a in arms}

    tests = {}
    for g in p["gates"]:
        tests[g] = {
            "decay_post_bps": paired_seed_test(arr(dec, g, "post_bps"), arr(dec, "ungated", "post_bps"),
                                               alternative="greater"),
            "control_sharpe": paired_seed_test(arr(ctl, g, "oos_sharpe"), arr(ctl, "ungated", "oos_sharpe")),
            "control_post_bps": paired_seed_test(arr(ctl, g, "post_bps"), arr(ctl, "ungated", "post_bps")),
        }
    prim = tests[p["primary"]]
    h6 = bool(prim["decay_post_bps"]["p"] < p["alpha_level"] and prim["decay_post_bps"]["mean"] > 0)
    ctl_ok = bool(prim["control_sharpe"]["mean"] > -p["max_control_sharpe_loss"])
    delays = [r[p["primary"]]["first_off_after_switch"] for r in dec]
    return {
        "seeds": list(seeds),
        "decay": mean_table(dec),
        "control": mean_table(ctl),
        "tests": tests,
        "H6": {**prim["decay_post_bps"], "pass": h6},
        "control_ok": ctl_ok,
        "recommend": h6 and ctl_ok,
        "detection_delay_bars": {"median": float(np.median([d for d in delays if d is not None]))
                                 if any(d is not None for d in delays) else None,
                                 "never_off": int(sum(d is None for d in delays))},
    }
