#!/usr/bin/env python3
"""Slice 9: the shrunk CUSUM gate with a faster re-entry level, 0.25 x h (frozen protocol).

Writes artifacts/reentry_cusum.{json,md}; the json is gitignored, the markdown is committed and a
test checks that a fresh run reproduces it. ~2 s.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from execution_cost.cusum_eval import REENTRY_ARMS, REENTRY_PROTOCOL, run_reentry_study  # noqa: E402

LABELS = {"ungated": "ungated WF-PARTIAL", "CUSUM-S": "CUSUM-S0.75 (slice 8)", "CUSUM-SR": "CUSUM-S0.75-R0.25"}


def _cell(t: dict) -> str:
    return f"{t['mean']:+.3f} ± {t['se']:.3f} (p = {t['p']:.3f}, {t['wins']}/{t['n']})"


def render(o: dict) -> str:
    m = o["means"]
    lines = ["# Fast re-entry CUSUM gate (slice 9)", "",
             f"Seeds {o['seeds'][0]}-{o['seeds'][-1]}; shrink {REENTRY_PROTOCOL['shrink']}, on_frac {REENTRY_PROTOCOL['on_frac']}; frozen in "
             "`cusum_eval.REENTRY_PROTOCOL`.", "",
             "| arm | decay: post bps/day | decay: on post | flip: post bps/day | control: OOS net Sharpe | control: on after bar 2,520 | control: switches |",
             "|---|---|---|---|---|---|---|"]
    for a in REENTRY_ARMS:
        lines.append(f"| {LABELS[a]} | {m['decay'][a]['post_bps']:.3f} | {m['decay'][a]['share_on_post']:.3f} | "
                     f"{m['flip'][a]['post_bps']:.3f} | {m['control'][a]['oos_sharpe']:.3f} | "
                     f"{m['control'][a]['share_on_post']:.3f} | {m['control'][a]['switches']:.1f} |")
    lines += ["", "| gate - ungated | decay post bps/day (one-sided) | flip post bps/day (one-sided) | control OOS Sharpe |",
              "|---|---|---|---|"]
    for a in ("CUSUM-S", "CUSUM-SR"):
        t = o["tests"][a]
        lines.append(f"| {LABELS[a]} | {_cell(t['decay_post_bps'])} | {_cell(t['flip_post_bps'])} | {_cell(t['control_sharpe'])} |")
    lines += ["", f"H12 (control cost < {REENTRY_PROTOCOL['max_control_sharpe_loss']}): **{o['H12']}**; "
              f"H13 (post-decay gain): **{o['H13']}**; recommend the fast re-entry gate: **{o['recommend']}**."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "reentry_cusum.json")
    a = ap.parse_args(argv)
    o = run_reentry_study()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(o, indent=2) + "\n", encoding="utf-8")
    md = render(o)
    a.out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
