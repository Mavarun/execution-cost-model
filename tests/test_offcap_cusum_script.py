"""The slice-11 script reproduces its committed markdown, and the decision is the pre-registered one."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_script_reproduces_the_committed_markdown(tmp_path):
    spec = importlib.util.spec_from_file_location("oc", ROOT / "scripts" / "run_offcap_cusum.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--out", str(tmp_path / "offcap_cusum.json")]) == 0
    fresh = (tmp_path / "offcap_cusum.md").read_text(encoding="utf-8")
    assert fresh == (ROOT / "artifacts" / "offcap_cusum.md").read_text(encoding="utf-8")
    assert "H16 (control cost CI upper end < 0.1): **True**; H17 (post-decay gain): **False**" in fresh
    assert "H18 (keeps half the uncapped gain): **False**; recommend the capped gate: **False**" in fresh
