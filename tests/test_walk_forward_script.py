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


def test_committed_artifact_matches_the_readme():
    art = json.loads((ROOT / "artifacts" / "walk_forward_slice.json").read_text())
    main = art["scenarios"]["profile + events break"]["pairs"]["ROLLING-250 - FIXED"]["post"]
    assert f"**{main['mean_bps']:+.2f} ± {main['se_bps']:.2f}** ({main['n_reps_better']}/5)" in \
        (ROOT / "README.md").read_text()
    ctrl = art["scenarios"]["no break (control)"]["pairs"]["ROLLING-250 - FIXED"]["post"]
    assert abs(ctrl["mean_bps"]) < 2 * ctrl["se_bps"] + 0.01
