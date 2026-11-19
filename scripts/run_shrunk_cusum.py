#!/usr/bin/env python3
"""Slice 8: the CUSUM decay gate with its in-sample reference shrunk to 0.75 (frozen protocol).

Writes artifacts/shrunk_cusum.{json,md}; the json is gitignored, the markdown is committed and a
test checks that a fresh run reproduces it. ~2 s.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from execution_cost.cusum_eval import SHRINK_ARMS, SHRINK_PROTOCOL, run_shrink_study  # noqa: E402

LABELS = {"ungated": "ungated WF-PARTIAL", "CUSUM": "CUSUM (slice 7)", "CUSUM-S": "CUSUM-S0.75"}


def _cell(t: dict) -> str:
    return f"{t['mean']:+.3f} ± {t['se']:.3f} (p = {t['p']:.3f}, {t['wins']}/{t['n']})"


def render(o: dict) -> str:
    m = o["means"]
    lines = ["# Shrunk-reference CUSUM gate (slice 8)", "",
             f"Seeds {o['seeds'][0]}-{o['seeds'][-1]}; shrink {SHRINK_PROTOCOL['shrink']}; protocol frozen in "
             "`cusum_eval.SHRINK_PROTOCOL`.", "",
             "| arm | decay: post bps/day | decay: on post | flip: post bps/day | control: OOS net Sharpe | control: on after bar 2,520 |",
             "|---|---|---|---|---|---|"]
    for a in SHRINK_ARMS:
        lines.append(f"| {LABELS[a]} | {m['decay'][a]['post_bps']:.3f} | {m['decay'][a]['share_on_post']:.3f} | "
                     f"{m['flip'][a]['post_bps']:.3f} | {m['control'][a]['oos_sharpe']:.3f} | "
                     f"{m['control'][a]['share_on_post']:.3f} |")
    lines += ["", "| gate - ungated | decay post bps/day (one-sided) | flip post bps/day (one-sided) | control OOS Sharpe |",
              "|---|---|---|---|"]
    for a in ("CUSUM", "CUSUM-S"):
        t = o["tests"][a]
        lines.append(f"| {LABELS[a]} | {_cell(t['decay_post_bps'])} | {_cell(t['flip_post_bps'])} | {_cell(t['control_sharpe'])} |")
    d = o["shrunk_delay_bars"]
    lines += ["", f"CUSUM-S0.75 first off after the switch: decay median {d['decay']['median']} bars "
              f"({d['decay']['never_off']} never), flip median {d['flip']['median']} bars ({d['flip']['never_off']} never).",
              f"H10 (control cost < {SHRINK_PROTOCOL['max_control_sharpe_loss']}): **{o['H10']}**; "
              f"H11 (post-decay gain): **{o['H11']}**; recommend the shrunk gate: **{o['recommend']}**."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "shrunk_cusum.json")
    a = ap.parse_args(argv)
    o = run_shrink_study()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(o, indent=2) + "\n", encoding="utf-8")
    md = render(o)
    a.out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
