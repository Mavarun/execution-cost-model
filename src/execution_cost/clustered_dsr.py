"""Post-hoc to slice 5 (run in slice 6): deflated Sharpe with a clustered trial count.

Slice 5 bracketed the right trial count: the raw 42 over-deflates (4 / 20 real
paths pass DSR > 0.95), the participation ratio (~1.3) lets 1 / 20 null paths
through. Rule fixed before running: ``clustered_trials(max_distance=0.5)`` on the
full-sample net returns of all 42 rules, the same seeds as slice 5 (500-519
signal, 600-619 null), the same trial variance. Passing bar: null paths with
DSR < 0.95 on >= 19 / 20 (slice 5's H4 bar); report how many signal paths clear.
"""

from __future__ import annotations

import numpy as np

from execution_cost.alpha import AlphaParams, simulate_alpha_path
from execution_cost.sharpe_stats import clustered_trials, deflated_sharpe, effective_trials, per_bar_sharpe
from execution_cost.walkforward import run_rules
from execution_cost.walkforward_eval import PROTOCOL, candidate_rules, cost_params, null_market

MAX_DISTANCE = 0.5


def seed_dsr(seed: int, market: AlphaParams | None = None) -> dict:
    market = market or AlphaParams(**PROTOCOL["market"])
    path = simulate_alpha_path(market, seed=seed)
    runs = run_rules(path, candidate_rules()["WF-ALL"], cost_params())
    x = np.column_stack([r.net[1:] for r in runs])
    sr = np.array([per_bar_sharpe(r.net[1:]) for r in runs])
    best = runs[int(np.nanargmax(sr))]
    var = float(np.nanvar(sr, ddof=1))
    n_cl = clustered_trials(x, max_distance=MAX_DISTANCE)
    n_eff = max(int(round(effective_trials(x))), 1)
    return {
        "n_raw": len(runs), "n_eff": n_eff, "n_clustered": n_cl,
        "dsr_raw": deflated_sharpe(best.net[1:], n_trials=len(runs), trial_var=var)["dsr"],
        "dsr_eff": deflated_sharpe(best.net[1:], n_trials=n_eff, trial_var=var)["dsr"],
        "dsr_clustered": deflated_sharpe(best.net[1:], n_trials=n_cl, trial_var=var)["dsr"],
    }


def run_clustered_study(signal_seeds=None, null_seeds=None) -> dict:
    sig = [seed_dsr(s) for s in (signal_seeds or PROTOCOL["eval_seeds"])]
    nul = [seed_dsr(s, null_market()) for s in (null_seeds or PROTOCOL["null_seeds"])]
    thr = PROTOCOL["dsr_threshold"]
    out = {"max_distance": MAX_DISTANCE}
    for kind in ("raw", "eff", "clustered"):
        out[kind] = {
            "mean_trials_signal": float(np.mean([r[f"n_{kind}"] for r in sig])),
            "mean_trials_null": float(np.mean([r[f"n_{kind}"] for r in nul])),
            "signal_pass": int(sum(r[f"dsr_{kind}"] > thr for r in sig)),
            "null_reject": int(sum(r[f"dsr_{kind}"] < thr for r in nul)),
            "n_signal": len(sig), "n_null": len(nul),
        }
    out["clustered_ok"] = out["clustered"]["null_reject"] >= PROTOCOL["null_dsr_min_seeds"]
    return out
