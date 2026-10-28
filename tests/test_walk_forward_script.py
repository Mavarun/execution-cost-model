"""Slice-5 CLI smoke test: small run writes the scenarios, pooled pairs and markdown."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cli_writes_all_scenarios(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "scripts" / "run_walk_forward_slice.py"), "--n-blocks", "4",
                    "--reps", "2", "--out", str(tmp_path)], check=True, capture_output=True, text=True)
    out = json.loads((tmp_path / "walk_forward_slice.json").read_text())
    assert set(out["scenarios"]) == {"profile + events break", "no break (control)", "profile-only break",
                                     "events-only break"}
    assert out["spec"]["break_block"] == 1 and out["spec"]["n_blocks"] == 4
    p = out["scenarios"]["no break (control)"]["pairs"]["EXPANDING - FIXED"]["post"]
    assert len(p["per_rep"]) == 2 and p["se_bps"] >= 0
    md = (tmp_path / "walk_forward_slice.md").read_text()
    assert "## no break (control)" in md and "ROLLING-250 - FIXED" in md


def _paired_row(md: str, scenario: str, pair: str) -> list[str]:
    section = md.split(f"## {scenario}\n", 1)[1].split("\n## ", 1)[0]
    row = next(line for line in section.splitlines() if line.startswith(f"| {pair} |"))
    return [c.strip() for c in row.strip("|").split("|")]


def test_committed_artifact_matches_the_readme():
    # artifacts/*.json is gitignored in this repo, so the committed markdown is the record
    md = (ROOT / "artifacts" / "walk_forward_slice.md").read_text()
    _, _, post, better = _paired_row(md, "profile + events break", "ROLLING-250 - FIXED")
    assert f"**{post}** ({better})" in (ROOT / "README.md").read_text()
    _, _, ctrl, _ = _paired_row(md, "no break (control)", "ROLLING-250 - FIXED")
    mean, se = (float(x) for x in ctrl.split(" ± "))
    assert abs(mean) < 2 * se + 0.01
