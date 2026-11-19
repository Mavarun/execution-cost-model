"""The slice-9 script reproduces its committed markdown, and the decision is the pre-registered one."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_script_reproduces_the_committed_markdown(tmp_path):
    spec = importlib.util.spec_from_file_location("sc", ROOT / "scripts" / "run_reentry_cusum.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--out", str(tmp_path / "reentry_cusum.json")]) == 0
    fresh = (tmp_path / "reentry_cusum.md").read_text(encoding="utf-8")
    assert fresh == (ROOT / "artifacts" / "reentry_cusum.md").read_text(encoding="utf-8")
    assert "H12 (control cost < 0.1): **True**; H13 (post-decay gain): **False**" in fresh and "recommend the fast re-entry gate: **False**" in fresh
