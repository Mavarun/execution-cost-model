#!/usr/bin/env python3
"""Optimal-execution slice: impact calibration + TWAP/VWAP/POV/AC comparison.

1. Calibrate a square-root impact law on synthetic TWAP metaorders
   (train split), report exponent CI and out-of-sample fit vs linear impact.
2. Compare execution schedules for a buy of q x ADV on fresh simulated days
   (disjoint seed, common random numbers), net of spread, temporary and
   permanent impact; decompose implementation shortfall in bps.

Synthetic, seeded, offline. Not live PnL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from execution_cost.calibration import calibrate_impact, generate_metaorders  # noqa: E402
from execution_cost.comparison import compare_schedules  # noqa: E402
from execution_cost.simulator import MarketParams  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--q", type=float, nargs="+", default=[0.02, 0.05, 0.10], help="order size / ADV")
    p.add_argument("--n-paths", type=int, default=4000)
    p.add_argument("--seed", type=int, default=123, help="evaluation seed")
    p.add_argument("--calibration-seed", type=int, default=0)
    p.add_argument("--n-orders", type=int, default=20_000)
    p.add_argument("--spread-bps", type=float, default=5.0)
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def build_report(args) -> dict:
    mp = MarketParams(spread_bps=args.spread_bps)
    orders = generate_metaorders(n_orders=args.n_orders, seed=args.calibration_seed, market=mp)
    cal = calibrate_impact(orders, seed=args.calibration_seed, market=mp)
    comps = [
        compare_schedules(mp, q=q, n_paths=args.n_paths, seed=args.seed, calibration=cal).to_dict()
        for q in args.q
    ]
    return {"market": mp.__dict__, "calibration": cal.to_dict(), "comparisons": comps}


def print_report(rep: dict) -> None:
    c = rep["calibration"]
    print("## Impact calibration (synthetic TWAP metaorders, 70/30 split)")
    print(f"orders={c['n_orders']}  NLS delta={c['nls_delta']:.3f} (se {c['nls_delta_se']:.3f})  "
          f"binned delta={c['binned_delta']:.3f} CI=[{c['binned_delta_ci'][0]:.3f}, {c['binned_delta_ci'][1]:.3f}]")
    print(f"sqrt Y raw={c['sqrt_Y']:.3f}  temp-only Y={c['temp_sqrt_Y']:.3f}  "
          f"permanent={c['perm_coeff']:.3f} CI=[{c['perm_ci'][0]:.3f}, {c['perm_ci'][1]:.3f}]")
    print(f"OOS binned RMSE: power={c['oos_binned_rmse_power']:.4f} sqrt={c['oos_binned_rmse_sqrt']:.4f} "
          f"linear={c['oos_binned_rmse_linear']:.4f}  (order-level MSE sqrt={c['oos_mse_sqrt']:.4f} linear={c['oos_mse_linear']:.4f})")
    for comp in rep["comparisons"]:
        print()
        print(f"## Buy {comp['q']:.0%} of ADV, {comp['n_paths']} days (bps of arrival notional)")
        print("| schedule | mean IS | std | p95 | vs TWAP (paired) | spread | temporary | permanent | timing | opportunity | completion | pre-trade |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for s in comp["schedules"]:
            pre = "-" if s["pretrade_estimate_bps"] is None else f"{s['pretrade_estimate_bps']:.2f}"
            print(
                f"| {s['name']} | {s['mean_total_bps']:.2f} ± {s['se_total_bps']:.2f} | {s['std_total_bps']:.1f} | "
                f"{s['p95_total_bps']:.1f} | {s['paired_diff_vs_twap_bps']:+.2f} ± {s['paired_diff_se_bps']:.2f} | "
                f"{s['mean_spread_bps']:.2f} | {s['mean_temporary_bps']:.2f} | {s['mean_permanent_bps']:.2f} | "
                f"{s['mean_timing_bps']:.2f} | {s['mean_opportunity_bps']:.2f} | {s['mean_completion']:.3f} | {pre} |"
            )
    print("\nSynthetic simulation only; not live PnL.")


def main(argv=None) -> int:
    args = parse_args(argv)
    rep = build_report(args)
    if args.json:
        print(json.dumps(rep, indent=2))
    else:
        print_report(rep)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
