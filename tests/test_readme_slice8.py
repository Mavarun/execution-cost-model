"""Every slice-8 number the README quotes is recomputed here."""

from pathlib import Path

import pytest

from execution_cost.cusum_eval import control_off_spells, run_shrink_study

README = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
SECTION = README[README.index("## Slice 8 (2026-11-19)"): README.index("## Slice 7 (2026-11-19)")]


@pytest.fixture(scope="module")
def study():
    return run_shrink_study()


def test_tables_match(study):
    labels = {"ungated": "ungated WF-PARTIAL", "CUSUM": "CUSUM (slice 7)", "CUSUM-S": "CUSUM-S0.75"}
    m = study["means"]
    for a, label in labels.items():
        row = (f"| {label} | {m['decay'][a]['post_bps']:.3f} | {m['decay'][a]['share_on_post']:.3f} | "
               f"{m['flip'][a]['post_bps']:.3f} | {m['control'][a]['oos_sharpe']:.3f} | {m['control'][a]['share_on_post']:.3f} |")
        assert row in SECTION, row
    for a in ("CUSUM", "CUSUM-S"):
        t = study["tests"][a]["control_sharpe"]
        assert f"{t['mean']:+.3f} ± {t['se']:.3f}" in SECTION


def test_decision_and_prose_match(study):
    assert not study["H10"] and study["H11"] and not study["recommend"]
    t = study["tests"]
    assert f"from\n{-t['CUSUM']['control_sharpe']['mean']:.3f} to {-t['CUSUM-S']['control_sharpe']['mean']:.3f} Sharpe" in SECTION
    assert f"+{t['CUSUM-S']['decay_post_bps']['mean']:.3f} bps/day" in SECTION and "ships off" in SECTION


def test_off_spell_numbers_match():
    f, s = control_off_spells(1.0), control_off_spells(0.75)
    assert f"from {f['alarms']} to {s['alarms']}" in SECTION
    assert f"{f['bars_per_alarm']:,.0f} -> {s['bars_per_alarm']:,.0f}" in SECTION
    assert f"mean {s['mean_off_spell']:.0f} bars (median {s['median_off_spell']:.0f})" in SECTION
    assert f"off {100 * s['share_off']:.0f}% of the time" in SECTION
