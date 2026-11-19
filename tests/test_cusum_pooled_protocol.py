"""Slice 10 frozen protocol: slice 8's arms on 60 fresh seeds."""

from execution_cost import cusum_eval, decay_eval, vol_target_eval, walkforward_eval
from execution_cost.cusum_eval import POOLED_PROTOCOL, REENTRY_PROTOCOL, SHRINK_PROTOCOL, run_pooled_confirm


def test_protocol_is_frozen():
    assert POOLED_PROTOCOL == {"shrink": 0.75, "seeds": tuple(range(1200, 1260)), "alpha_level": 0.05,
                               "max_control_sharpe_loss": 0.10}
    assert POOLED_PROTOCOL["shrink"] == SHRINK_PROTOCOL["shrink"]


def test_seeds_are_disjoint_from_every_earlier_protocol():
    used = set(cusum_eval.CUSUM_PROTOCOL["seeds"]) | set(decay_eval.DECAY_PROTOCOL["seeds"])
    used |= set(vol_target_eval.VT_PROTOCOL["seeds"]) | set(SHRINK_PROTOCOL["seeds"]) | set(REENTRY_PROTOCOL["seeds"])
    wf = walkforward_eval
    used |= set(wf.PROTOCOL["eval_seeds"]) | set(wf.PROTOCOL["null_seeds"]) | set(wf.REGIME_PROTOCOL["seeds"])
    assert not set(POOLED_PROTOCOL["seeds"]) & used


def test_runs_on_dev_seeds():
    o = run_pooled_confirm(seeds=(5, 6))
    lo, hi = o["control_cost_ci"]
    assert lo <= o["tests"]["CUSUM-S"]["control_sharpe"]["mean"] <= hi
    assert set(o) >= {"H14", "H15", "recommend"}
