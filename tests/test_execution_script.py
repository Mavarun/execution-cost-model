"""Smoke test: execution comparison CLI runs offline on a small config."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def test_execution_comparison_script_small():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_execution_comparison.py"
    spec = importlib.util.spec_from_file_location("exec_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = mod.parse_args(
        ["--q", "0.05", "--n-paths", "300", "--n-orders", "4000", "--urgencies", "1", "3", "--true-deltas", "0.4", "0.7"]
    )
    rep = mod.build_report(args)
    assert set(rep) == {"market", "calibration", "comparisons", "frontiers", "misspecification"}
    names = [s["name"] for s in rep["comparisons"][0]["schedules"]]
    assert names[:2] == ["TWAP", "VWAP"] and any(n.startswith("POV") for n in names)
    assert {"SQRT-OPT(kT=1)", "SQRT-OPT(kT=3)"} <= set(names) and any(n.startswith("POV-lag1") for n in names)
    assert [p["kappa_T"] for p in rep["frontiers"][0]["points"]] == [1.0, 3.0]
    assert [r["true_delta"] for r in rep["misspecification"]["rows"]] == [0.4, 0.4, 0.7, 0.7]
    mod.print_report(rep)


def test_execution_comparison_script_can_skip_slow_sections(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_execution_comparison.py"
    spec = importlib.util.spec_from_file_location("exec_script2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "rep.json"
    rc = mod.main(["--q", "0.05", "--n-paths", "200", "--n-orders", "4000", "--no-frontier", "--no-misspec", "--out", str(out)])
    assert rc == 0
    rep = json.loads(out.read_text())
    assert set(rep) == {"market", "calibration", "comparisons"}
