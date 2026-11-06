#!/usr/bin/env python3
"""Clustered-trial deflated Sharpe (post-hoc to slice 5) -> artifacts/clustered_dsr.md."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from execution_cost.clustered_dsr import run_clustered_study  # noqa: E402


def render(out: dict) -> str:
    lines = [f"# Deflated Sharpe with a clustered trial count (average linkage, cut {out['max_distance']})", "",
             "| trial count | mean trials (signal) | signal seeds DSR > 0.95 | null seeds DSR < 0.95 |",
             "|---|---|---|---|"]
    for kind, label in (("raw", "raw 42"), ("eff", "participation ratio"), ("clustered", "clustered")):
        m = out[kind]
        lines.append(f"| {label} | {m['mean_trials_signal']:.1f} | {m['signal_pass']} / {m['n_signal']} | "
                     f"{m['null_reject']} / {m['n_null']} |")
    lines += ["", f"Clustered count meets the null bar (>= 19 / 20): **{out['clustered_ok']}**."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    md = render(run_clustered_study())
    (ROOT / "artifacts" / "clustered_dsr.md").write_text(md, encoding="utf-8")
    print(md)
