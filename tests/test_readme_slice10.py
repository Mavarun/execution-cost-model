"""Every slice-10 number the README quotes is recomputed here."""

from pathlib import Path

import pytest

from execution_cost.cusum_eval import POOLED_PROTOCOL, control_cost_by_seed_set, control_off_spells, run_pooled_confirm

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
SECTION = README[README.index("## Slice 10 (2026-11-20)"): README.index("## Slice 9 (2026-11-20)")]


@pytest.fixture(scope="module")
def study():
    return run_pooled_confirm()


def test_tables_are_the_artifact():
    for line in (ROOT / "artifacts" / "pooled_cusum.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("|"):
            assert line in SECTION, line


def test_decision_and_prose(study):
    assert study["H14"] and study["H15"] and study["recommend"]
    c = study["tests"]["CUSUM-S"]
    lo, hi = study["control_cost_ci"]
    assert f"control cost is {-c['control_sharpe']['mean']:.3f} Sharpe" in SECTION
    assert f"95% CI ({-hi:.3f}-{-lo:.3f}) contains the 0.10 budget" in SECTION and -hi < 0.10 < -lo
    d = c["decay_post_bps"]
    assert f"+{d['mean']:.3f} bps/day (p < 0.001, {d['wins']}/{d['n']})" in SECTION and d["p"] < 0.001


def test_posthoc_numbers():
    r = control_cost_by_seed_set()
    for k, v in r["per_set"].items():
        assert f"| {k} | {v['cost']:.3f} ± {v['se']:.3f} | {v['n']} |" in SECTION
    assert f"the cost is {r['pooled']['cost']:.3f}, just over" in SECTION and r["pooled"]["cost"] > 0.10
    assert f"p = {r['anova_p']:.3f})" in SECTION
    s = control_off_spells(0.75, seeds=POOLED_PROTOCOL["seeds"])
    assert f"off {100 * s['share_off']:.0f}% of control bars: {s['alarms']} false alarms" in SECTION
    assert f"median off-spell of {s['median_off_spell']:.0f} bars" in SECTION
