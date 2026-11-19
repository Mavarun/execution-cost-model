"""decay_reentries: false re-entries after a real decay (slice 9 post-hoc reading)."""

from execution_cost.cusum_eval import decay_reentries


def test_counts_are_consistent_on_dev_seeds():
    full, fast = decay_reentries(1.0, seeds=(5, 6)), decay_reentries(0.25, seeds=(5, 6))
    for r in (full, fast):
        assert r["post_bars"] > 0 and r["reentries"] >= 0
        assert r["per_seed"] == r["reentries"] / 2
        assert r["median_on_spell"] >= 0
    assert fast["reentries"] >= full["reentries"]
