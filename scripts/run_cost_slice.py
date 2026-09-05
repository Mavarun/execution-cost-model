#!/usr/bin/env python3
"""Run execution-cost research slice: gross vs net Sharpe and turnover cost curve."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from execution_cost.costs import CostParams
from execution_cost.portfolio import run_cost_slice
from execution_cost.sensitivity import turnover_cost_curve


def _fmt(x: float) -> str:
    if x != x:
        return "nan"
    return f"{x:.6g}"


def build_report(args: argparse.Namespace) -> dict:
    params = CostParams(
        spread_bps=args.spread_bps,
        slippage_bps=args.slippage_bps,
        impact_coeff=args.impact_coeff,
        adv_notional=args.adv,
    )
    base = run_cost_slice(
        n=args.n,
        seed=args.seed,
        hold_bars=args.hold_bars,
        edge=args.edge,
        asset_vol=args.asset_vol,
        params=params,
    )
    grid = [int(x) for x in args.hold_grid.split(",") if x.strip()]
    curve = turnover_cost_curve(
        grid,
        n=args.n,
        seed=args.seed,
        edge=args.edge,
        asset_vol=args.asset_vol,
        params=params,
    )
    return {
        "hypothesis": {
            "h1": "Spread + slippage + square-root impact model participation costs",
            "h2": "Net Sharpe strictly below gross once costs apply",
            "h3": "Higher turnover destroys edge - net Sharpe falls as turnover rises",
        },
        "params": {
            "spread_bps": params.spread_bps,
            "slippage_bps": params.slippage_bps,
            "impact_coeff": params.impact_coeff,
            "adv_notional": params.adv_notional,
            "n": args.n,
            "seed": args.seed,
            "edge": args.edge,
            "asset_vol": args.asset_vol,
        },
        "baseline": base.to_dict(),
        "cost_curve": [p.to_dict() for p in curve],
        "checks": {
            "net_below_gross": base.net_sharpe < base.gross_sharpe,
            "curve_high_turnover_worse_net": (
                sorted(curve, key=lambda p: p.mean_turnover)[-1].net_sharpe
                <= sorted(curve, key=lambda p: p.mean_turnover)[0].net_sharpe + 1e-9
            ),
        },
    }


def print_report(report: dict) -> None:
    b = report["baseline"]
    print("## Baseline (synthetic signal, not live PnL)")
    print(
        f"gross_sharpe={_fmt(b['gross_sharpe'])}  net_sharpe={_fmt(b['net_sharpe'])}  "
        f"turnover={_fmt(b['mean_turnover'])}  mean_cost={_fmt(b['mean_daily_cost'])}  "
        f"hold_bars={b['hold_bars']}"
    )
    print()
    print("## Hypothesis checks")
    for k, v in report["checks"].items():
        print(f"{k}: {v}")
    print()
    print("## Turnover cost curve (larger hold_bars => lower turnover)")
    headers = ["hold", "turnover", "gross_sh", "net_sh", "drag", "mean_cost"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for p in report["cost_curve"]:
        print(
            "| "
            + " | ".join(
                [
                    str(p["hold_bars"]),
                    _fmt(p["mean_turnover"]),
                    _fmt(p["gross_sharpe"]),
                    _fmt(p["net_sharpe"]),
                    _fmt(p["sharpe_drag"]),
                    _fmt(p["mean_daily_cost"]),
                ]
            )
            + " |"
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hold-bars", type=int, default=5)
    p.add_argument("--hold-grid", type=str, default="1,2,3,5,8,13,21")
    p.add_argument("--edge", type=float, default=0.35)
    p.add_argument("--asset-vol", type=float, default=0.01)
    p.add_argument("--spread-bps", type=float, default=10.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--impact-coeff", type=float, default=0.25)
    p.add_argument("--adv", type=float, default=1.0)
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    report = build_report(args)
    if args.json:
        text = json.dumps(report, indent=2)
        print(text)
    else:
        print_report(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
