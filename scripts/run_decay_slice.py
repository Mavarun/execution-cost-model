#!/usr/bin/env python3
"""Slice 6: trailing-performance gate under an IC decay -> artifacts/decay_slice.{json,md}."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from execution_cost.decay_eval import DECAY_PROTOCOL, run_decay_study  # noqa: E402


def render(out: dict) -> str:
    arms = ["ungated", *DECAY_PROTOCOL["gates"]]
    lines = ["# Slice 6: trailing-performance gate under an IC decay (IC 0.10 -> 0 at bar 2,520)", "",
             f"Seeds {out['seeds'][0]}-{out['seeds'][-1]}; WF-PARTIAL; costs: slice-1 defaults.", ""]
    for market in ("decay", "control"):
        lines += [f"## {market} market (means across seeds)", "",
                  "| arm | post bps/day | post cost bps/day | OOS net Sharpe | pre-switch net Sharpe | on pre | on post |",
                  "|---|---|---|---|---|---|---|"]
        for a in arms:
            m = out[market][a]
            lines.append(f"| {a} | {m['post_bps']:.3f} | {m['post_cost_bps']:.3f} | {m['oos_sharpe']:.3f} | "
                         f"{m['pre_sharpe']:.3f} | {m['share_on_pre']:.3f} | {m['share_on_post']:.3f} |")
        lines.append("")
    lines += ["## Paired tests (gate - ungated)", "",
              "| gate | decay post bps/day (one-sided) | control OOS Sharpe | control post bps/day |", "|---|---|---|---|"]
    for g, t in out["tests"].items():
        cell = lambda d: f"{d['mean']:+.3f} ± {d['se']:.3f}, p = {d['p']:.3f}, {d['wins']}/{d['n']}"  # noqa: E731
        lines.append(f"| {g} | {cell(t['decay_post_bps'])} | {cell(t['control_sharpe'])} | {cell(t['control_post_bps'])} |")
    d = out["detection_delay_bars"]
    lines += ["", f"H6 pass: **{out['H6']['pass']}**; control within budget: **{out['control_ok']}**; "
              f"recommend G252: **{out['recommend']}**.",
              f"G252 detection delay after the switch: median {d['median']} bars; never off on {d['never_off']} seeds."]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "decay_slice.json"))
    args = ap.parse_args()
    out = run_decay_study()
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    md = render(out)
    Path(args.out).with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
