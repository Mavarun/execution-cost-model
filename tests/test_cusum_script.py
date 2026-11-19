"""The slice-7 script renders every table, and the committed markdown matches a fresh run."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_script_reproduces_the_committed_markdown(tmp_path):
    spec = importlib.util.spec_from_file_location("cs", ROOT / "scripts" / "run_cusum_slice.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--out", str(tmp_path / "cusum_slice.json")]) == 0
    fresh = (tmp_path / "cusum_slice.md").read_text(encoding="utf-8")
    committed = (ROOT / "artifacts" / "cusum_slice.md").read_text(encoding="utf-8")
    assert fresh == committed
    assert "recommend the CUSUM gate: **False**" in committed and "H8 pass: **True**" in committed
