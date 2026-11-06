"""Slice 6, H7: band vs partial adjustment at train-chosen equal risk.

Pre-registered (frozen in :data:`VT_PROTOCOL` and committed before any H7 seed was
run). Slice 5's post-hoc equal-risk check found WF-PB - WF-PARTIAL = -0.021 Sharpe,
but it levered on the evaluation days. H7 repeats the question with leverage a
trader could have used.

* Market / costs / walk-forward: slice-5 defaults. Fresh seeds 900-919.
* Each selector (WF-PARTIAL, WF-BAND, WF-PB) is vol-targeted to 10% annualised on
  each fold's expanding training window (:mod:`execution_cost.vol_target`).
* **H7 (primary, two-sided paired t at 5%)**: vol-targeted WF-PB minus vol-targeted
  WF-PARTIAL in OOS net Sharpe. PARTIAL+BAND is recommended only if H7 passes with a
  positive mean; otherwise PARTIAL stays.
* Secondary: WF-BAND minus WF-PARTIAL (two-sided); realised OOS vol / 10% per
  selector (how well training-window targeting holds risk OOS); mean leverage.
"""

from __future__ import annotations

import numpy as np

from execution_cost.alpha import AlphaParams, simulate_alpha_path
from execution_cost.metrics import annualized_sharpe
from execution_cost.vol_target import TRADING_DAYS, vol_targeted
from execution_cost.walkforward import run_rules, walk_forward
from execution_cost.walkforward_eval import PROTOCOL, candidate_rules, cost_params, paired_seed_test

VT_PROTOCOL = {
    "target_annual_vol": 0.10,
    "selectors": ("WF-PARTIAL", "WF-BAND", "WF-PB"),
    "primary": ("WF-PB", "WF-PARTIAL"),
    "secondary": ("WF-BAND", "WF-PARTIAL"),
    "seeds": tuple(range(900, 920)),
    "alpha_level": 0.05,
}


def vt_seed(seed: int, *, n: int | None = None, min_train: int | None = None, test_len: int | None = None) -> dict:
    market = AlphaParams(**PROTOCOL["market"])
    if n is not None:
        market = AlphaParams(**{**PROTOCOL["market"], "n": int(n)})
    path = simulate_alpha_path(market, seed=seed)
    costs = cost_params()
    groups = candidate_rules()
    out = {}
    for name in VT_PROTOCOL["selectors"]:
        rules = groups[name]
        runs = run_rules(path, rules, costs)
        wf = walk_forward(path, rules, costs, min_train=int(min_train or PROTOCOL["min_train"]),
                          test_len=int(test_len or PROTOCOL["test_len"]), runs=runs)
        vt = vol_targeted(path, wf, runs, costs, VT_PROTOCOL["target_annual_vol"])
        out[name] = {
            "net_sharpe": annualized_sharpe(vt.net),
            "unlevered_net_sharpe": annualized_sharpe(wf.net),
            "realised_vol_ratio": float(np.std(vt.net, ddof=1) * np.sqrt(TRADING_DAYS) / VT_PROTOCOL["target_annual_vol"]),
            "mean_leverage": float(np.mean(vt.leverage)),
            "net_bps": float(vt.net.mean() * 1e4),
            "cost_bps": float(vt.cost.mean() * 1e4),
        }
    return out


def run_vt_study(seeds=None, **kw) -> dict:
    p = VT_PROTOCOL
    seeds = tuple(seeds if seeds is not None else p["seeds"])
    rows = [vt_seed(s, **kw) for s in seeds]

    def arr(name, key):
        return np.array([r[name][key] for r in rows], dtype=float)

    keys = ("net_sharpe", "unlevered_net_sharpe", "realised_vol_ratio", "mean_leverage", "net_bps", "cost_bps")
    tab = {nm: {k: float(arr(nm, k).mean()) for k in keys} for nm in p["selectors"]}
    a, b = p["primary"]
    h7 = paired_seed_test(arr(a, "net_sharpe"), arr(b, "net_sharpe"))
    c, d = p["secondary"]
    sec = paired_seed_test(arr(c, "net_sharpe"), arr(d, "net_sharpe"))
    passed = bool(h7["p"] < p["alpha_level"] and h7["mean"] > 0)
    return {"seeds": list(seeds), "table": tab, "H7": {**h7, "pass": passed}, "band_vs_partial": sec,
            "recommend": "PARTIAL+BAND" if passed else "PARTIAL"}
