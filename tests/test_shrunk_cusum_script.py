"""The slice-8 script reproduces its committed markdown, and the decision is the pre-registered one."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_script_reproduces_the_committed_markdown(tmp_path):
    spec = importlib.util.spec_from_file_location("sc", ROOT / "scripts" / "run_shrunk_cusum.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--out", str(tmp_path / "shrunk_cusum.json")]) == 0
    fresh = (tmp_path / "shrunk_cusum.md").read_text(encoding="utf-8")
    assert fresh == (ROOT / "artifacts" / "shrunk_cusum.md").read_text(encoding="utf-8")
    assert "H10 (control cost < 0.1): **False**" in fresh and "recommend the shrunk gate: **False**" in fresh
