"""README slice-6 numbers recomputed from the frozen studies (all run in seconds)."""

from pathlib import Path

import pytest

from execution_cost.clustered_dsr import run_clustered_study
from execution_cost.decay_eval import run_decay_study
from execution_cost.vol_target_eval import run_vt_study

README = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
SEC = README[README.index("## Slice 6 (2026-11-06)"): README.index("## Slice 5 (2026-11-01)")]


@pytest.fixture(scope="module")
def decay():
    return run_decay_study()


def test_decay_table_and_verdict(decay):
    for arm, label in (("ungated", "ungated WF-PARTIAL"), ("G252", "G252 (primary)"), ("G126", "G126"), ("G504", "G504")):
        d, c = decay["decay"][arm], decay["control"][arm]
        row = f"| {label} | {d['post_bps']:.3f} | {d['post_cost_bps']:.3f} | {d['share_on_post']:.3f} | {c['oos_sharpe']:.3f} |"
        assert row in SEC, row
    h = decay["H6"]
    assert not h["pass"] and not decay["recommend"]
    assert f"{h['mean']:+.3f} ± {h['se']:.3f} bps/day (one-sided p = {h['p']:.2f}, {h['wins']} / 20 seeds)".replace("+", "") in SEC
    ctl = decay["tests"]["G252"]["control_sharpe"]
    assert f"**{-ctl['mean']:.3f}** Sharpe" in SEC


def test_vol_target_table_and_verdict():
    out = run_vt_study()
    for nm in ("WF-PARTIAL", "WF-BAND", "WF-PB"):
        m = out["table"][nm]
        assert (f"| {nm} | {m['net_sharpe']:.3f} | {m['unlevered_net_sharpe']:.3f} | {m['realised_vol_ratio']:.3f} | "
                f"{m['mean_leverage']:.2f} |") in SEC
    h, b = out["H7"], out["band_vs_partial"]
    assert out["recommend"] == "PARTIAL"
    assert f"{h['mean']:+.3f} ± {h['se']:.3f} Sharpe (p = {h['p']:.2f}, {h['wins']} / 20 seeds)".replace("+", "") in SEC
    assert f"**{-b['mean']:.3f} ± {b['se']:.3f}**" in SEC


def test_clustered_table():
    out = run_clustered_study()
    assert out["clustered_ok"]
    assert f"| clustered, cut 0.5 | {out['clustered']['mean_trials_signal']:.1f} | {out['clustered']['signal_pass']} / 20 | {out['clustered']['null_reject']} / 20 |" in SEC
    assert f"| raw 42 | 42.0 | {out['raw']['signal_pass']} / 20 | {out['raw']['null_reject']} / 20 |" in SEC
