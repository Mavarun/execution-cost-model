"""Slice 9 frozen protocol: one change on top of slice 8 (on_frac 0.25), fresh seeds."""

from execution_cost import cusum_eval, decay_eval, vol_target_eval, walkforward_eval
from execution_cost.cusum_eval import REENTRY_PROTOCOL, SHRINK_PROTOCOL, reentry_seed


def test_protocol_is_frozen():
    assert REENTRY_PROTOCOL == {"shrink": 0.75, "on_frac": 0.25, "seeds": tuple(range(1100, 1120)),
                                "alpha_level": 0.05, "max_control_sharpe_loss": 0.10}
    assert REENTRY_PROTOCOL["shrink"] == SHRINK_PROTOCOL["shrink"]


def test_seeds_are_disjoint_from_every_earlier_protocol():
    used = set(cusum_eval.CUSUM_PROTOCOL["seeds"]) | set(decay_eval.DECAY_PROTOCOL["seeds"])
    used |= set(vol_target_eval.VT_PROTOCOL["seeds"]) | set(SHRINK_PROTOCOL["seeds"])
    wf = walkforward_eval
    used |= set(wf.PROTOCOL["eval_seeds"]) | set(wf.PROTOCOL["null_seeds"]) | set(wf.REGIME_PROTOCOL["seeds"])
    assert not set(REENTRY_PROTOCOL["seeds"]) & used


def test_reentry_seed_reports_every_arm():
    # a dev seed outside the protocol's range
    r = reentry_seed(5, "control")
    assert set(r) == {"ungated", "CUSUM-S", "CUSUM-SR"}
    assert r["ungated"]["share_on_post"] == 1.0 and r["ungated"]["switches"] == 0
