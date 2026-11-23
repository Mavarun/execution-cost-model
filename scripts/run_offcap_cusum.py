#!/usr/bin/env python3
"""Slice 11: the shrunk CUSUM gate with an off-spell cap (126 bars), frozen protocol, 60 fresh seeds.

Writes artifacts/offcap_cusum.{json,md}; the json is gitignored, the markdown is committed and a
test checks that a fresh run reproduces it. ~3 s.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from execution_cost.cusum_eval import OFFCAP_ARMS, OFFCAP_PROTOCOL, run_offcap_study  # noqa: E402

LABELS = {"ungated": "ungated WF-PARTIAL", "CUSUM-S": "CUSUM-S0.75 (slice 8/10)", "CUSUM-SC": "CUSUM-S0.75-C126"}


def _cell(t: dict) -> str:
    return f"{t['mean']:+.3f} ± {t['se']:.3f} (p = {t['p']:.3f}, {t['wins']}/{t['n']})"


def render(o: dict) -> str:
    m = o["means"]
    lines = ["# Shrunk CUSUM gate with an off-spell cap (slice 11)", "",
             f"Seeds {o['seeds'][0]}-{o['seeds'][-1]}; shrink {OFFCAP_PROTOCOL['shrink']}; max_off "
             f"{OFFCAP_PROTOCOL['max_off']}; frozen in `cusum_eval.OFFCAP_PROTOCOL`.", "",
             "| arm | decay: post bps/day | decay: on post | flip: post bps/day | control: OOS net Sharpe | control: on after bar 2,520 | control: switches |",
             "|---|---|---|---|---|---|---|"]
    for a in OFFCAP_ARMS:
        lines.append(f"| {LABELS[a]} | {m['decay'][a]['post_bps']:.3f} | {m['decay'][a]['share_on_post']:.3f} | "
                     f"{m['flip'][a]['post_bps']:.3f} | {m['control'][a]['oos_sharpe']:.3f} | "
                     f"{m['control'][a]['share_on_post']:.3f} | {m['control'][a]['switches']:.1f} |")
    lines += ["", "| gate - ungated | decay post bps/day (one-sided) | flip post bps/day (one-sided) | control OOS Sharpe | control cost 95% CI |",
              "|---|---|---|---|---|"]
    for a in ("CUSUM-S", "CUSUM-SC"):
        t = o["tests"][a]
        lo, hi = o["control_cost_ci"][a]
        lines.append(f"| {LABELS[a]} | {_cell(t['decay_post_bps'])} | {_cell(t['flip_post_bps'])} | "
                     f"{_cell(t['control_sharpe'])} | [{lo:.3f}, {hi:.3f}] |")
    lines += ["", f"H16 (control cost CI upper end < {OFFCAP_PROTOCOL['max_control_sharpe_loss']}): **{o['H16']}**; "
              f"H17 (post-decay gain): **{o['H17']}**; H18 (keeps half the uncapped gain): **{o['H18']}**; "
              f"recommend the capped gate: **{o['recommend']}**."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "offcap_cusum.json")
    a = ap.parse_args(argv)
    o = run_offcap_study()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(o, indent=2) + "\n", encoding="utf-8")
    md = render(o)
    a.out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
