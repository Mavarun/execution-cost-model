#!/usr/bin/env python3
"""Slice 5: walk-forward refit of the volume model across a regime break.

Main scenario: 2,000 evaluation days in 40 blocks of 50; the break at block 10
steepens the intraday U-shape (1.5 -> 3.0) and doubles surge days (15% -> 30%).
Replicated over 5 disjoint (training, evaluation) seed pairs; controls with no
break; two one-factor breaks (profile only, events only). Every evaluation day
is out of sample for every fit; spread, impact and opportunity cost charged.
Writes artifacts/walk_forward_slice.{json,md}. Synthetic, seeded, offline.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from execution_cost.calibration import calibrate_impact, generate_metaorders  # noqa: E402
from execution_cost.simulator import EventParams, MarketParams  # noqa: E402
from execution_cost.walk_forward import WalkForwardSpec, run_walk_forward  # noqa: E402

SEEDS = ((7, 1000), (17, 2000), (27, 3000), (37, 4000), (47, 5000))
SCENARIOS = {
    "profile + events break": {},
    "no break (control)": {"post_market": MarketParams(), "post_events": EventParams()},
    "profile-only break": {"post_events": EventParams()},
    "events-only break": {"post_market": MarketParams()},
}


def pooled(reps: list[dict], key: str, regime: str) -> dict:
    """Mean and SE of a paired difference pooled over replications (equal days per rep)."""
    m = np.array([r["pairs"][key][regime]["mean_bps"] for r in reps])
    se = np.array([r["pairs"][key][regime]["se_bps"] for r in reps])
    return {"mean_bps": float(m.mean()), "se_bps": float(np.sqrt(np.sum(se**2)) / len(m)),
            "n_reps_better": int(np.sum(m < 0)), "per_rep": [round(float(x), 3) for x in m]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-blocks", type=int, default=40)
    ap.add_argument("--reps", type=int, default=len(SEEDS))
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts")
    args = ap.parse_args(argv)
    base = WalkForwardSpec(n_blocks=args.n_blocks, break_block=max(1, args.n_blocks // 4))
    cal = calibrate_impact(generate_metaorders(seed=0, market=base.pre_market), seed=0, market=base.pre_market)
    out = {"spec": None, "scenarios": {}}
    for name, kw in SCENARIOS.items():
        reps = [run_walk_forward(replace(base, train_seed=t, eval_seed=e, **kw), calibration=cal).to_dict()
                for t, e in SEEDS[: args.reps]]
        out["spec"] = out["spec"] or reps[0]["spec"]
        models = reps[0]["models"]
        out["scenarios"][name] = {
            "per_regime": {rg: {m: {k: float(np.mean([r["per_regime"][rg][m][k] for r in reps]))
                                    for k in ("mean_bps", "forecast_log_rmse", "completion")}
                                for m in models} for rg in ("pre", "post")},
            "pairs": {f"{m} - FIXED": {rg: pooled(reps, f"{m} - FIXED", rg) for rg in ("pre", "post")}
                      for m in models[1:]},
            "rolling250_fits_rep0": reps[0]["fits"]["ROLLING-250"] if "ROLLING-250" in models else None,
        }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "walk_forward_slice.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    s = out["spec"]
    lines = ["# Slice 5: walk-forward volume-model refit across a regime break", "",
             f"{s['n_blocks']} blocks x {s['block']} days, break at block {s['break_block']} "
             f"(post-break: U-shape {s['pre_u_shape']} -> {s['post_u_shape']}, surge days "
             f"{s['pre_event_prob']:.0%} -> {s['post_event_prob']:.0%} in the main scenario); "
             f"q = {s['q']:.0%} ADV SURGE-POV, {args.reps} seed pairs; spread, impact and opportunity cost charged.", ""]
    for name, sc in out["scenarios"].items():
        lines += [f"## {name}", "", "| model | pre IS bps | post IS bps | pre log RMSE | post log RMSE | post completion |",
                  "|---|---:|---:|---:|---:|---:|"]
        for m, pre in sc["per_regime"]["pre"].items():
            po = sc["per_regime"]["post"][m]
            lines.append(f"| {m} | {pre['mean_bps']:.2f} | {po['mean_bps']:.2f} | {pre['forecast_log_rmse']:.3f} | "
                         f"{po['forecast_log_rmse']:.3f} | {po['completion']:.3f} |")
        lines += ["", "| paired vs FIXED (bps, pooled ± SE) | pre | post | reps better (post) |", "|---|---:|---:|---:|"]
        for k, v in sc["pairs"].items():
            lines.append(f"| {k} | {v['pre']['mean_bps']:+.2f} ± {v['pre']['se_bps']:.2f} | "
                         f"{v['post']['mean_bps']:+.2f} ± {v['post']['se_bps']:.2f} | "
                         f"{v['post']['n_reps_better']}/{args.reps} |")
        lines.append("")
    (args.out / "walk_forward_slice.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
