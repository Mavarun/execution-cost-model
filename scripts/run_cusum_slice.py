#!/usr/bin/env python3
"""Slice 7: sequential CUSUM gate under an IC decay and an IC flip -> artifacts/cusum_slice.{json,md}."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from execution_cost.cusum_eval import ARMS, run_cusum_study  # noqa: E402


def render(out: dict) -> str:
    lines = ["# Slice 7: CUSUM gate (ARL 2,520) under an IC decay (0.10 -> 0) and flip (0.10 -> -0.05) at bar 2,520", "",
             f"Seeds {out['seeds'][0]}-{out['seeds'][-1]}; WF-PARTIAL; costs: slice-1 defaults.", ""]
    for market in ("decay", "flip", "control"):
        lines += [f"## {market} market (means across seeds)", "",
                  "| arm | post bps/day | post cost bps/day | OOS net Sharpe | pre-switch net Sharpe | on pre | on post |",
                  "|---|---|---|---|---|---|---|"]
        for a in ARMS:
            m = out["means"][market][a]
            lines.append(f"| {a} | {m['post_bps']:.3f} | {m['post_cost_bps']:.3f} | {m['oos_sharpe']:.3f} | "
                         f"{m['pre_sharpe']:.3f} | {m['share_on_pre']:.3f} | {m['share_on_post']:.3f} |")
        lines.append("")
    lines += ["## Paired tests (gate - ungated)", "",
              "| gate | decay post bps/day (one-sided) | flip post bps/day (one-sided) | control OOS Sharpe |", "|---|---|---|---|"]
    cell = lambda d: f"{d['mean']:+.3f} ± {d['se']:.3f}, p = {d['p']:.3f}, {d['wins']}/{d['n']}"  # noqa: E731
    for g, t in out["tests"].items():
        lines.append(f"| {g} | {cell(t['decay_post_bps'])} | {cell(t['flip_post_bps'])} | {cell(t['control_sharpe'])} |")
    d = out["cusum_delay_bars"]
    lines += ["", f"H8 pass: **{out['H8']}**; H9 pass: **{out['H9']}**; control within budget: **{out['control_ok']}**; "
              f"recommend the CUSUM gate: **{out['recommend']}**.",
              f"CUSUM first off after the switch: decay median {d['decay']['median']} bars (never off on "
              f"{d['decay']['never_off']} seeds); flip median {d['flip']['median']} bars (never off on {d['flip']['never_off']})."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "cusum_slice.json"))
    args = ap.parse_args(argv)
    out = run_cusum_study()
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    md = render(out)
    Path(args.out).with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
