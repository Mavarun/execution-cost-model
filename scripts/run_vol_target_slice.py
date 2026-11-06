#!/usr/bin/env python3
"""Slice 6, H7: band vs partial at train-chosen equal risk -> artifacts/vol_target_slice.{json,md}."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from execution_cost.vol_target_eval import VT_PROTOCOL, run_vt_study  # noqa: E402


def render(out: dict) -> str:
    lines = ["# Slice 6, H7: vol-targeted selectors (10% annualised, leverage solved on training bars)", "",
             f"Seeds {out['seeds'][0]}-{out['seeds'][-1]}; costs re-charged on levered trades.", "",
             "| selector | vol-targeted net Sharpe | unlevered net Sharpe | realised vol / target | mean leverage | net bps/day | cost bps/day |",
             "|---|---|---|---|---|---|---|"]
    for nm in VT_PROTOCOL["selectors"]:
        m = out["table"][nm]
        lines.append(f"| {nm} | {m['net_sharpe']:.3f} | {m['unlevered_net_sharpe']:.3f} | {m['realised_vol_ratio']:.3f} | "
                     f"{m['mean_leverage']:.2f} | {m['net_bps']:.3f} | {m['cost_bps']:.3f} |")
    h, s = out["H7"], out["band_vs_partial"]
    lines += ["", f"H7 (WF-PB - WF-PARTIAL, two-sided): {h['mean']:+.3f} ± {h['se']:.3f}, p = {h['p']:.3f}, "
              f"{h['wins']}/{h['n']} seeds; pass: **{h['pass']}**.",
              f"WF-BAND - WF-PARTIAL: {s['mean']:+.3f} ± {s['se']:.3f}, p = {s['p']:.3f}, {s['wins']}/{s['n']}.",
              f"Recommended rule: **{out['recommend']}**."]
    return "\n".join(lines) + "\n"


def main() -> int:
    out = run_vt_study()
    target = ROOT / "artifacts" / "vol_target_slice.json"
    target.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    md = render(out)
    target.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
