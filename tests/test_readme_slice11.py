"""Every slice-11 number in the README comes from the committed artifact or a recomputation."""

from pathlib import Path

from execution_cost.cusum_eval import capped_redetection, decay_reentries, run_offcap_study

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
SEC = README[README.index("## Slice 11 (2026-11-24)"): README.index("## Slice 10 (2026-11-20)")]
MD = (ROOT / "artifacts" / "offcap_cusum.md").read_text(encoding="utf-8")


def test_tables_are_the_artifact_tables():
    for line in MD.splitlines():
        if line.startswith("| ") and "---" not in line:
            assert line in SEC, line


def test_prose():
    o = run_offcap_study()
    t, u = o["tests"]["CUSUM-SC"], o["tests"]["CUSUM-S"]
    lo, hi = o["control_cost_ci"]["CUSUM-SC"]
    assert f"falls to {-t['control_sharpe']['mean']:.3f} Sharpe (CI {lo:.3f}-{hi:.3f})" in SEC
    d = t["decay_post_bps"]
    assert f"+{d['mean']:.3f} bps/day (p = {d['p']:.3f}, {d['wins']}/{d['n']})" in SEC
    assert f"uncapped gate's +{u['decay_post_bps']['mean']:.3f}" in SEC and 0.18 < d["mean"] / u["decay_post_bps"]["mean"] < 0.25
    ulo, uhi = o["control_cost_ci"]["CUSUM-S"]
    assert f"(CI {ulo:.3f}-{uhi:.3f})" in SEC and uhi > 0.10
    assert o["H16"] and not o["H17"] and not o["H18"]
    c, r = capped_redetection(), decay_reentries(1.0, seeds=range(1300, 1360))
    assert f"re-enters {c['per_seed']:.2f} times" in SEC and f"({r['per_seed']:.2f} uncapped)" in SEC
    assert f"median {c['median_on_spell']:.0f} bars" in SEC
    assert f"on {o['means']['decay']['CUSUM-SC']['share_on_post']:.0%} of post-decay bars" in SEC
