#!/usr/bin/env python3
"""Slice 5: price-adaptive (aggressive-in-the-money) execution vs the static plan.

Grid: buy q in {5, 10}% ADV x urgency kT in {1, 3} x three disjoint seed
triples (calibration / training / evaluation): (0, 7, 123), (1, 8, 124),
(2, 9, 125). 2,000 training days pick the AIM strength ``a``; 4,000 fresh
evaluation days score it. 15% event days throughout. Writes
artifacts/price_adaptive_slice.{json,md}. Synthetic, seeded, offline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from execution_cost.calibration import calibrate_impact, generate_metaorders  # noqa: E402
from execution_cost.price_adaptive_eval import A_GRID, evaluate_price_adaptive  # noqa: E402
from execution_cost.simulator import MarketParams  # noqa: E402

SEED_TRIPLES = ((0, 7, 123), (1, 8, 124), (2, 9, 125))
QS = (0.05, 0.10)
KTS = (1.0, 3.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-paths", type=int, default=4000)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-grid", type=int, default=40)
    ap.add_argument("--triples", type=int, default=len(SEED_TRIPLES))
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts")
    args = ap.parse_args(argv)
    mp = MarketParams()
    rows = []
    for cs, ts, es in SEED_TRIPLES[: args.triples]:
        cal = calibrate_impact(generate_metaorders(seed=cs, market=mp), seed=cs, market=mp)
        for q in QS:
            for k in KTS:
                rows.append(evaluate_price_adaptive(
                    mp, q=q, kT=k, n_paths=args.n_paths, n_train=args.n_train, seed=es,
                    train_seed=ts, calibration_seed=cs, calibration=cal, n_grid=args.n_grid,
                ).to_dict())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "price_adaptive_slice.json").write_text(
        json.dumps({"a_grid": list(A_GRID), "rows": rows}, indent=1) + "\n")

    L = ["# Slice 5: aggressive-in-the-money execution vs the static plan", "",
         f"{args.n_train} training days choose `a` from {list(A_GRID)} by realised E + lam Var; "
         f"{args.n_paths} disjoint evaluation days score it. 5 bps spread, sqrt temporary and "
         "linear permanent impact charged. Utility change is chosen - static (negative = better), "
         "with a paired bootstrap 95% CI over days.", "",
         "| seeds (cal/train/eval) | q | kT | chosen a | oracle a | static mean / std bps | chosen mean / std bps "
         "| IS diff bps ± SE | utility change % | utility CI (price units) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        s, c = r["static"], r["chosen"]
        sd = r["seeds"]
        L.append(
            f"| {sd['calibration']}/{sd['train']}/{sd['eval']} | {r['q']:.0%} | {r['kT']:g} | {r['chosen_a']:g} | "
            f"{r['oracle_a']:g} | {s['mean_bps']:.2f} / {s['std_bps']:.2f} | {c['mean_bps']:.2f} / {c['std_bps']:.2f} | "
            f"{r['diff_mean_bps']:+.2f} ± {r['diff_se_bps']:.2f} | {r['utility_change_pct']:+.2f}% | "
            f"[{r['utility_ci'][0]:+.0f}, {r['utility_ci'][1]:+.0f}] |")
    (args.out / "price_adaptive_slice.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
