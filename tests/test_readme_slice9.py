"""Every slice-9 number the README quotes is recomputed here."""

from pathlib import Path

import pytest

from execution_cost.cusum_eval import REENTRY_PROTOCOL, control_off_spells, decay_reentries, run_reentry_study

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
SECTION = README[README.index("## Slice 9 (2026-11-20)"): README.index("## Slice 8 (2026-11-19)")]


@pytest.fixture(scope="module")
def study():
    return run_reentry_study()


def test_tables_are_the_committed_artifact():
    md = (ROOT / "artifacts" / "reentry_cusum.md").read_text(encoding="utf-8")
    for line in md.splitlines():
        if line.startswith("|"):
            assert line in SECTION, line


def test_decision_and_prose(study):
    assert study["H12"] and not study["H13"] and not study["recommend"]
    t, m = study["tests"], study["means"]
    assert f"falls to {-t['CUSUM-SR']['control_sharpe']['mean']:.3f}" in SECTION
    d = t["CUSUM-SR"]["decay_post_bps"]
    assert f"{d['mean']:+.3f} bps/day (p = {d['p']:.3f}, {d['wins']}/{d['n']})" in SECTION
    assert f"{100 * m['decay']['CUSUM-SR']['share_on_post']:.0f}% of post-decay bars" in SECTION
    assert f"against {100 * m['decay']['CUSUM-S']['share_on_post']:.0f}% for slice 8's" in SECTION
    assert (f"from {t['CUSUM-S']['flip_post_bps']['mean']:+.3f} to\n{t['CUSUM-SR']['flip_post_bps']['mean']:+.3f}"
            in SECTION)
    assert f"costs only\n  {-t['CUSUM-S']['control_sharpe']['mean']:.3f} Sharpe" in SECTION


def test_posthoc_numbers():
    fast, full = decay_reentries(0.25), decay_reentries(1.0)
    assert f"on\n  {fast['reentries']} times" in SECTION
    assert f"({fast['per_seed']:.2f} per seed, against {full['per_seed']:.2f} at full h)" in SECTION
    assert f"median {fast['median_on_spell']:.0f} bars" in SECTION
    here, s8 = control_off_spells(0.75, seeds=REENTRY_PROTOCOL["seeds"]), control_off_spells(0.75)
    assert f"fall to {here['alarms']} (one per {here['bars_per_alarm']:,.0f} OOS bars, against {s8['alarms']}" in SECTION
    assert f"off share is {100 * here['share_off']:.0f}% ({100 * s8['share_off']:.0f}% there)" in SECTION


def test_cross_seed_set_comparison(study):
    from execution_cost.cusum_eval import run_shrink_study

    s8 = run_shrink_study()["tests"]["CUSUM-S"]["control_sharpe"]
    s9 = study["tests"]["CUSUM-S"]["control_sharpe"]
    assert f"differ by {abs(s8['mean'] - s9['mean']):.2f} Sharpe" in SECTION
    assert f"({-s8['mean']:.3f} ± {s8['se']:.3f} vs {-s9['mean']:.3f} ± {s9['se']:.3f})" in SECTION
