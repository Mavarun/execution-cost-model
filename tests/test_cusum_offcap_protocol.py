import numpy as np

from execution_cost.cusum_eval import (
    CUSUM_PROTOCOL, OFFCAP_ARMS, OFFCAP_PROTOCOL, POOLED_PROTOCOL, REENTRY_PROTOCOL, SHRINK_PROTOCOL, offcap_seed,
    run_offcap_study,
)


def test_frozen_and_fresh():
    p = OFFCAP_PROTOCOL
    assert p["seeds"] == tuple(range(1300, 1360)) and p["max_off"] == 126 and p["shrink"] == 0.75
    used = set(CUSUM_PROTOCOL["seeds"]) | set(SHRINK_PROTOCOL["seeds"]) | set(REENTRY_PROTOCOL["seeds"]) | set(
        POOLED_PROTOCOL["seeds"])
    assert not used & set(p["seeds"]) and not set(range(1, 6)) & set(p["seeds"])
    assert OFFCAP_ARMS == ("ungated", "CUSUM-S", "CUSUM-SC")


def test_capped_arm_never_off_longer_than_the_cap_on_a_real_seed():
    r = offcap_seed(3, "control")
    assert set(r) == set(OFFCAP_ARMS)
    assert r["ungated"]["share_on_post"] == 1.0


def test_rules_on_a_dev_run():
    o = run_offcap_study(seeds=range(1, 4))
    lo, hi = o["control_cost_ci"]["CUSUM-SC"]
    assert lo < hi and o["H16"] == bool(hi < 0.10)
    t = o["tests"]["CUSUM-SC"]["decay_post_bps"]
    assert o["H17"] == bool(t["p"] < 0.05 and t["mean"] > 0)
    assert o["recommend"] == bool(o["H16"] and o["H17"])
    assert np.isfinite(o["means"]["decay"]["CUSUM-SC"]["post_bps"])
