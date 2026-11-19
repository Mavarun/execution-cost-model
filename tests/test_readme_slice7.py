"""Every slice-7 number the README quotes is recomputed here, so the prose cannot drift from the code."""

from pathlib import Path

import pytest

from execution_cost.cusum_eval import reference_bias, run_cusum_study

README = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
SECTION = README[README.index("## Slice 7 (2026-11-19)"): README.index("## Slice 6 (2026-11-06)")]


@pytest.fixture(scope="module")
def study():
    return run_cusum_study()


def test_arm_rows_match(study):
    m = study["means"]
    for arm, label in (("ungated", "ungated WF-PARTIAL"), ("G252", "G252 (slice 6)"), ("CUSUM", "CUSUM (ARL 2,520)")):
        d, f, c = m["decay"][arm], m["flip"][arm], m["control"][arm]
        row = (f"| {label} | {d['post_bps']:.3f} | {d['share_on_post']:.3f} | {f['post_bps']:.3f} | "
               f"{f['share_on_post']:.3f} | {c['oos_sharpe']:.3f} | {c['share_on_post']:.3f} |")
        assert row in SECTION, row


def test_paired_tests_and_decision_match(study):
    t = study["tests"]["CUSUM"]
    assert f"+{t['decay_post_bps']['mean']:.3f} ± {t['decay_post_bps']['se']:.3f}" in SECTION
    assert f"+{t['flip_post_bps']['mean']:.3f} ± {t['flip_post_bps']['se']:.3f}" in SECTION
    assert f"gives up **{-t['control_sharpe']['mean']:.3f}** net Sharpe" in SECTION
    assert study["H8"] and study["H9"] and not study["control_ok"] and not study["recommend"]
    assert "ships off" in SECTION
    assert f"(p = {study['tests']['G252']['decay_post_bps']['p']:.3f})" in SECTION


def test_reference_bias_numbers_match():
    b = reference_bias()
    assert f"({b['ref_mean_bps']:.2f} vs {b['oos_mean_bps']:.2f} bps/day" in SECTION
    assert f"higher on {b['seeds_ref_above_oos']} of 20 seeds" in SECTION
    assert f"median {b['median_realised_drift']:.3f} per bar, not the {b['median_designed_drift']:.3f}" in SECTION
