import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("run_price_adaptive_slice", ROOT / "scripts" / "run_price_adaptive_slice.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_smoke_writes_json_and_markdown(tmp_path):
    rc = _load().main(["--n-paths", "60", "--n-train", "40", "--n-grid", "8", "--triples", "1", "--out", str(tmp_path)])
    assert rc == 0
    data = json.loads((tmp_path / "price_adaptive_slice.json").read_text())
    assert len(data["rows"]) == 4 and 0.0 in data["a_grid"]
    assert {(r["q"], r["kT"]) for r in data["rows"]} == {(0.05, 1.0), (0.05, 3.0), (0.1, 1.0), (0.1, 3.0)}
    md = (tmp_path / "price_adaptive_slice.md").read_text()
    assert md.count("| 0/7/123 |") == 4


def test_readme_slice5_table_matches_the_committed_artifact():
    rows = [l.split("|")[1:-1] for l in (ROOT / "artifacts" / "price_adaptive_slice.md").read_text().splitlines()
            if re.match(r"^\| \d/\d/\d+ \|", l)]
    assert len(rows) == 12
    readme = (ROOT / "README.md").read_text()
    sec = readme.split("## Slice 5 (2026-10-25)")[1].split("\n## ")[0]
    for q in ("5%", "10%"):
        for k in ("1", "3"):
            cells = [r for r in rows if r[1].strip() == q and r[2].strip() == k]
            chosen = ", ".join(r[3].strip() for r in cells)
            util = " / ".join(r[8].strip() for r in cells)
            line = [l for l in sec.splitlines() if l.startswith(f"| {q} | {k} |")]
            assert len(line) == 1, (q, k)
            assert f"| {chosen} |" in line[0] and util in line[0].replace("**", "")
