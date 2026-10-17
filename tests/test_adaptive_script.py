"""Smoke test: slice-4 CLI runs offline and writes its json + markdown."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def test_adaptive_slice_script_small(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_adaptive_slice.py"
    spec = importlib.util.spec_from_file_location("adaptive_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--n-paths", "150", "--out", str(tmp_path)]) == 0
    rep = json.loads((tmp_path / "adaptive_slice.json").read_text())
    assert [(r["q"], r["kT"]) for r in rep["grid"]] == [
        (q, k) for q in (0.02, 0.05, 0.10) for k in (1.0, 3.0)
    ]
    assert [(r["day_volume_noise"], r["surge"]) for r in rep["sensitivity"]] == [
        (0.2, 4.0), (0.2, 8.0), (0.5, 4.0), (0.5, 8.0)
    ]
    md = (tmp_path / "adaptive_slice.md").read_text()
    assert "ADAPTIVE - SQRT-OPT" in md and "Forecast log RMSE" in md
