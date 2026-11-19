"""Slice 8 frozen protocol: one change (shrink 0.75), seeds disjoint from every earlier protocol."""

from execution_cost import cusum_eval, decay_eval, vol_target_eval, walkforward_eval
from execution_cost.cusum_eval import SHRINK_PROTOCOL, shrink_seed


def test_protocol_is_frozen():
    assert SHRINK_PROTOCOL == {"shrink": 0.75, "seeds": tuple(range(1000, 1020)), "alpha_level": 0.05,
                               "max_control_sharpe_loss": 0.10}


def test_seeds_are_disjoint_from_every_earlier_protocol():
    used = set(cusum_eval.CUSUM_PROTOCOL["seeds"]) | set(decay_eval.DECAY_PROTOCOL["seeds"])
    used |= set(vol_target_eval.VT_PROTOCOL["seeds"])
    wf = walkforward_eval
    used |= set(wf.PROTOCOL["eval_seeds"]) | set(wf.PROTOCOL["null_seeds"]) | set(wf.REGIME_PROTOCOL["seeds"])
    assert not set(SHRINK_PROTOCOL["seeds"]) & used
    # Slice 7's seeds were not fresh: slice 6's vol-target study had used them.
    assert set(cusum_eval.CUSUM_PROTOCOL["seeds"]) == set(vol_target_eval.VT_PROTOCOL["seeds"])


def test_shrink_seed_reports_every_arm():
    r = shrink_seed(1000, "control")
    assert set(r) == {"ungated", "CUSUM", "CUSUM-S"}
    assert r["ungated"]["share_on_post"] == 1.0
