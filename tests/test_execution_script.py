"""Smoke test: execution comparison CLI runs offline on a small config."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def test_execution_comparison_script_small():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_execution_comparison.py"
    spec = importlib.util.spec_from_file_location("exec_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = mod.parse_args(["--q", "0.05", "--n-paths", "300", "--n-orders", "4000"])
    rep = mod.build_report(args)
    assert set(rep) == {"market", "calibration", "comparisons"}
    names = [s["name"] for s in rep["comparisons"][0]["schedules"]]
    assert names[:2] == ["TWAP", "VWAP"] and any(n.startswith("POV") for n in names)
    mod.print_report(rep)
