#!/usr/bin/env python3
"""Slice 4: adaptive re-optimisation, fitted volume nowcasts, event days.

Main grid: buy q in {2, 5, 10}% ADV x urgency kT in {1, 3}, 4,000 evaluation
days (seed 123) with 15% event days. Sensitivity (q = 5%, kT = 3): day-volume
noise 0.2 / 0.5 and event surge 4x / 8x, i.e. when should adapting pay?
Writes artifacts/adaptive_slice.{json,md}. Synthetic, seeded, offline.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from execution_cost.adaptive_eval import evaluate_adaptive  # noqa: E402
from execution_cost.calibration import calibrate_impact, generate_metaorders  # noqa: E402
from execution_cost.simulator import EventParams, MarketParams  # noqa: E402

SHOW = ("TWAP", "VWAP", "SQRT-OPT", "ADAPTIVE", "POV-lag1", "NOWCAST-POV", "SURGE-POV")


def pm(x):
    return f"{x['mean_bps']:+.2f} ± {x['se_bps']:.2f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-paths", type=int, default=4000)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts")
    args = ap.parse_args(argv)
    mp = MarketParams()
    cal = calibrate_impact(generate_metaorders(seed=0, market=mp), seed=0, market=mp)
    grid = [
        evaluate_adaptive(mp, q=q, kT=k, n_paths=args.n_paths, calibration=cal).to_dict()
        for q in (0.02, 0.05, 0.10)
        for k in (1.0, 3.0)
    ]
    sens = []
    for dn in (0.2, 0.5):
        for surge in (4.0, 8.0):
            m = replace(mp, day_volume_noise=dn)
            c = cal if dn == mp.day_volume_noise else calibrate_impact(
                generate_metaorders(seed=0, market=m), seed=0, market=m
            )
            r = evaluate_adaptive(
                m, q=0.05, kT=3.0, n_paths=args.n_paths, calibration=c, events=EventParams(surge=surge)
            ).to_dict()
            r["day_volume_noise"], r["surge"] = dn, surge
            sens.append(r)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "adaptive_slice.json").write_text(json.dumps({"grid": grid, "sensitivity": sens}, indent=1) + "\n")

    L = ["# Slice 4: adaptive execution on normal and event days", ""]
    r0 = grid[0]
    L += [
        f"Volume model (training seed 7, {r0['volume_model']['n_train']} days): k0 = {r0['volume_model']['k0']}, "
        f"surge phi = {r0['volume_model']['phi']:.3f}, fitted-profile max rel. error "
        f"{r0['volume_model']['profile_max_rel_err']:.3f}. Evaluation: {args.n_paths} days, "
        f"{r0['n_event_days']} event days.",
        "",
        "Forecast log RMSE (evaluation days, intervals 1..N-1):",
        "",
        "| split | lag-1 (slice 3) | day-factor nowcast | + surge AR(1) |",
        "|---|---|---|---|",
    ]
    for sp, v in r0["forecast_rmse"].items():
        L.append(f"| {sp} | {v['lag1']:.3f} | {v['nowcast']:.3f} | {v['surge']:.3f} |")
    L += ["", "Mean IS (bps), all days / event days; paired differences ± SE on the same days:", "",
          "| q | kT | " + " | ".join(SHOW) + " |", "|---|---|" + "---|" * len(SHOW)]
    for r in grid:
        cells = [f"{r['schedules'][n]['all']['mean_bps']:.2f} / {r['schedules'][n]['event']['mean_bps']:.2f}" for n in SHOW]
        L.append(f"| {r['q']:.0%} | {r['kT']:g} | " + " | ".join(cells) + " |")
    L += ["", "| q | kT | ADAPTIVE - SQRT-OPT (all) | (event) | utility change | NOWCAST-POV - POV-lag1 (all) | (event) | SURGE-POV - NOWCAST-POV (all) | (event) | POV-lag1 / NOWCAST completion |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in grid:
        p = r["pairs"]
        s = r["schedules"]
        du = s["ADAPTIVE"]["all"]["utility"] / s["SQRT-OPT"]["all"]["utility"] - 1.0
        L.append(
            f"| {r['q']:.0%} | {r['kT']:g} | {pm(p['ADAPTIVE - SQRT-OPT']['all'])} | {pm(p['ADAPTIVE - SQRT-OPT']['event'])} | "
            f"{du:+.2%} | {pm(p['NOWCAST-POV - POV-lag1']['all'])} | {pm(p['NOWCAST-POV - POV-lag1']['event'])} | "
            f"{pm(p['SURGE-POV - NOWCAST-POV']['all'])} | {pm(p['SURGE-POV - NOWCAST-POV']['event'])} | "
            f"{s['POV-lag1']['all']['completion']:.3f} / {s['NOWCAST-POV']['all']['completion']:.3f} |"
        )
    L += ["", "Sensitivity (q = 5%, kT = 3): when does adapting pay?", "",
          "| day-volume noise | surge | ADAPTIVE - SQRT-OPT (all) | (event) | utility change | ADAPTIVE std / SQRT-OPT std |",
          "|---|---|---|---|---|---|"]
    for r in sens:
        p, s = r["pairs"]["ADAPTIVE - SQRT-OPT"], r["schedules"]
        du = s["ADAPTIVE"]["all"]["utility"] / s["SQRT-OPT"]["all"]["utility"] - 1.0
        L.append(
            f"| {r['day_volume_noise']} | {r['surge']:g}x | {pm(p['all'])} | {pm(p['event'])} | {du:+.2%} | "
            f"{s['ADAPTIVE']['all']['std_bps']:.1f} / {s['SQRT-OPT']['all']['std_bps']:.1f} |"
        )
    (args.out / "adaptive_slice.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
