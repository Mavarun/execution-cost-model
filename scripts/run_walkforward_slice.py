#!/usr/bin/env python3
"""Slice 5: walk-forward cost-aware rebalancing with deflated Sharpe.

Runs the pre-registered study in :mod:`execution_cost.walkforward_eval` and
writes ``artifacts/walkforward_slice.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from execution_cost.alpha import AlphaParams  # noqa: E402
from execution_cost.walkforward_eval import (  # noqa: E402
    PROTOCOL,
    SELECTORS,
    cost_params,
    decide,
    null_market,
    paired_seed_test,
    pooled_hac,
    run_study,
    table,
)

NAMES = list(SELECTORS) + ["FULL"]
SENSITIVITIES = {
    "costs x0.5": {"cost_scale": 0.5},
    "costs x2": {"cost_scale": 2.0},
    "IC 0.05": {"ic": 0.05},
    "rho 0.70 (half-life 1.9d)": {"rho": 0.70},
    "rho 0.97 (half-life 22.8d)": {"rho": 0.97},
}


def _ns(results, name):
    return np.array([r.oos[name]["net_sharpe"] for r in results])


def build(seeds, null_seeds, sens_seeds) -> dict:
    base = AlphaParams(**PROTOCOL["market"])
    res = run_study(seeds)
    null = run_study(null_seeds, market=null_market())
    out = {
        "protocol": {k: (list(v) if isinstance(v, tuple) else v) for k, v in PROTOCOL.items()},
        "seeds": list(seeds),
        "null_seeds": list(null_seeds),
        "table": table(res),
        "null_table": table(null),
        "decision": decide(res, null),
        "pooled_hac_bps_per_day": {
            f"{a} - {b}": pooled_hac(res, a, b)
            for a, b in (("WF-PB", "WF-PARTIAL"), ("WF-BAND", "WF-PARTIAL"), ("WF-ALL", "WF-PARTIAL"), ("WF-PARTIAL", "FULL"))
        },
        "insample_vs_oos": {
            "insample_best_mean": float(np.mean([r.insample_best["net_sharpe"] for r in res])),
            "same_rule_on_oos_window_mean": float(np.mean([r.insample_best["net_sharpe_oos_window"] for r in res])),
            "wf_all_oos_mean": float(_ns(res, "WF-ALL").mean()),
            "null_insample_best_mean": float(np.mean([r.insample_best["net_sharpe"] for r in null])),
            "null_wf_all_oos_mean": float(_ns(null, "WF-ALL").mean()),
            "eval_dsr": [r.insample_best["dsr"] for r in res],
            "null_dsr": [r.insample_best["dsr"] for r in null],
            "insample_best_labels": dict(Counter(r.insample_best["label"] for r in res)),
        },
        "choices_last_fold": {
            name: dict(Counter(r.choices[name][-1] for r in res)) for name in SELECTORS
        },
        "sensitivity": {},
    }
    for label, spec in SENSITIVITIES.items():
        market = replace(base, **{k: v for k, v in spec.items() if k in ("ic", "rho")})
        costs = cost_params(spec.get("cost_scale", 1.0))
        sres = run_study(sens_seeds, market=market, costs=costs)
        tab = table(sres)
        out["sensitivity"][label] = {
            "net_sharpe": {n: tab[n]["net_sharpe"] for n in NAMES},
            "PB_minus_PARTIAL": paired_seed_test(_ns(sres, "WF-PB"), _ns(sres, "WF-PARTIAL")),
            "BAND_minus_PARTIAL": paired_seed_test(_ns(sres, "WF-BAND"), _ns(sres, "WF-PARTIAL")),
        }
    return out


def markdown(rep: dict) -> str:
    L = []
    d = rep["decision"]
    L.append("# Slice 5: walk-forward cost-aware rebalancing\n")
    L.append(f"Evaluation seeds {rep['seeds'][0]}-{rep['seeds'][-1]} (IC 0.10), null seeds "
             f"{rep['null_seeds'][0]}-{rep['null_seeds'][-1]} (IC 0). OOS = 15 annual folds (3,780 bars) per seed.\n")
    L.append("## OOS means across seeds (annualised Sharpe; cost in bps/day)\n")
    L.append("| selector | gross Sharpe | net Sharpe (sd) | cost | turnover | mean abs weight | null net Sharpe |")
    L.append("|---|---|---|---|---|---|---|")
    for n in NAMES:
        t, z = rep["table"][n], rep["null_table"][n]
        L.append(f"| {n} | {t['gross_sharpe']:.2f} | {t['net_sharpe']:.2f} ({t['net_sharpe_sd']:.2f}) | "
                 f"{t['mean_cost_bps']:.2f} | {t['turnover']:.3f} | {t['mean_abs_weight']:.3f} | {z['net_sharpe']:.2f} |")
    L.append("\n## Pre-registered hypotheses\n")
    for h in ("H1", "H2", "H3"):
        x = d[h]
        L.append(f"- **{h}**: mean diff {x['mean']:+.3f} ± {x['se']:.3f}, t {x['t']:.2f}, p {x['p']:.4f}, "
                 f"wins {x['wins']}/{x['n']} -> {'PASS' if x['pass'] else 'FAIL'}")
    h4 = d["H4"]
    L.append(f"- **H4**: null WF-ALL OOS net Sharpe {h4['null_wf_all_mean_net_sharpe']:+.3f} ± {h4['null_wf_all_se']:.3f}; "
             f"null in-sample best {h4['null_insample_best_mean_net_sharpe']:+.3f}; DSR < 0.95 on "
             f"{h4['null_seeds_dsr_below']}/{h4['n_null']} null seeds; DSR > 0.95 on {h4['eval_seeds_dsr_above']}/"
             f"{h4['n_eval']} IC-0.10 seeds -> {'PASS' if h4['pass'] else 'FAIL'}")
    L.append(f"- Recommended family: **{d['recommended_family']}**\n")
    L.append("## Pooled Newey-West daily net-return differences (bps/day)\n")
    L.append("Raw return differences, not risk-matched: selectors hold different gross exposure (see mean abs weight), "
             "so these measure P&L per unit of capital, not Sharpe.\n")
    L.append("| pair | mean | SE | t |")
    L.append("|---|---|---|---|")
    for k, v in rep["pooled_hac_bps_per_day"].items():
        L.append(f"| {k} | {v['mean']:+.3f} | {v['se']:.3f} | {v['t']:.2f} |")
    io = rep["insample_vs_oos"]
    L.append("\n## Selection bias\n")
    L.append(f"Full-sample best of 42: net Sharpe {io['insample_best_mean']:.2f} in sample; the same rule on the OOS "
             f"window {io['same_rule_on_oos_window_mean']:.2f}; WF-ALL OOS {io['wf_all_oos_mean']:.2f}. "
             f"Null: in-sample best {io['null_insample_best_mean']:.2f}, WF-ALL OOS {io['null_wf_all_oos_mean']:.2f}.\n")
    L.append("## Sensitivity (secondary; same evaluation seeds)\n")
    L.append("| setting | FULL | WF-PARTIAL | WF-BAND | WF-PB | WF-ALL | PB - PARTIAL | BAND - PARTIAL |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k, v in rep["sensitivity"].items():
        s = v["net_sharpe"]
        pb, bd = v["PB_minus_PARTIAL"], v["BAND_minus_PARTIAL"]
        L.append(f"| {k} | {s['FULL']:.2f} | {s['WF-PARTIAL']:.2f} | {s['WF-BAND']:.2f} | {s['WF-PB']:.2f} | "
                 f"{s['WF-ALL']:.2f} | {pb['mean']:+.3f} ± {pb['se']:.3f} ({pb['wins']}/{pb['n']}) | "
                 f"{bd['mean']:+.3f} ± {bd['se']:.3f} ({bd['wins']}/{bd['n']}) |")
    return "\n".join(L) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=None, help="use only the first N seeds of each set (smoke runs)")
    p.add_argument("--out-dir", type=Path, default=ROOT / "artifacts")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    k = a.seeds
    seeds = PROTOCOL["eval_seeds"][:k] if k else PROTOCOL["eval_seeds"]
    null_seeds = PROTOCOL["null_seeds"][:k] if k else PROTOCOL["null_seeds"]
    rep = build(seeds, null_seeds, seeds)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "walkforward_slice.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    md = markdown(rep)
    (a.out_dir / "walkforward_slice.md").write_text(md, encoding="utf-8")
    print(json.dumps(rep, indent=2) if a.json else md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
