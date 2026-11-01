"""Slice-5 CLI smoke test and README / artifact consistency."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("wf_script", ROOT / "scripts" / "run_walkforward_slice.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_walkforward_script_small(tmp_path, capsys):
    mod = _load()
    assert mod.main(["--seeds", "2", "--out-dir", str(tmp_path)]) == 0
    rep = json.loads((tmp_path / "walkforward_slice.json").read_text())
    assert rep["seeds"] == [500, 501] and rep["null_seeds"] == [600, 601]
    assert set(rep["table"]) == {"WF-PARTIAL", "WF-BAND", "WF-PB", "WF-ALL", "FULL"}
    assert set(rep["decision"]) == {"H1", "H2", "H3", "H4", "recommended_family"}
    assert len(rep["sensitivity"]) == 5
    md = (tmp_path / "walkforward_slice.md").read_text()
    assert "Pre-registered hypotheses" in md and "Selection bias" in md
    assert "Slice 5" in capsys.readouterr().out


def test_readme_headline_numbers_match_the_committed_artifact():
    md = (ROOT / "artifacts" / "walkforward_slice.md").read_text()
    readme = (ROOT / "README.md").read_text()
    h1 = re.search(r"\*\*H1\*\*: mean diff ([+-]\d+\.\d+) ± (\d+\.\d+), t [-\d.]+, p (\d+\.\d+)", md)
    assert h1, "H1 line missing from artifact"
    mean, se, p = float(h1.group(1)), float(h1.group(2)), float(h1.group(3))
    assert f"{mean:+.3f} ± {se:.3f}, p = {p:.2f}" in readme
    assert "Recommended family: **PARTIAL**" in md
    for row in ("| WF-PARTIAL | 1.34 | 1.02 (0.31)", "| WF-PB | 1.32 | 1.03 (0.29)", "| FULL | 1.44 | -1.16 (0.36)"):
        assert row in md


def test_readme_post_hoc_effective_trials_match_the_artifact():
    md = (ROOT / "artifacts" / "walkforward_slice.md").read_text()
    readme = (ROOT / "README.md").read_text()
    m = re.search(r"Effective trials .*?: (\d+\.\d+) on IC-0.10 seeds", md)
    assert m and f"**{m.group(1)}** (IC 0.10)" in readme
    assert "DSR > 0.95 on IC-0.10 seeds: 4/20 -> 20/20; null seeds with DSR < 0.95: 19/20" in md


def test_readme_equal_risk_rows_match_the_artifact():
    md = (ROOT / "artifacts" / "walkforward_slice.md").read_text()
    readme = (ROOT / "README.md").read_text()
    for pair in ("WF-PB vs WF-PARTIAL", "WF-BAND vs WF-PARTIAL"):
        m = re.search(re.escape(pair) + r" \| (\d+\.\d+) \| ([+-]\d+\.\d+) ± (\d+\.\d+)", md)
        assert m, pair
        assert f"| {pair} | {m.group(1)} |" in readme
        assert f"{m.group(2)} ± {m.group(3)}" in readme
