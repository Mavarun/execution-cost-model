"""Post-hoc slice-7 reading: the CUSUM's in-sample reference overstates the alive level."""

from execution_cost.cusum_eval import reference_bias
from execution_cost.decay_gate import siegmund_arl, siegmund_h


def test_reference_overstates_the_oos_level_and_halves_the_drift():
    b = reference_bias()
    assert b["ref_mean_bps"] > 1.2 * b["oos_mean_bps"] and b["seeds_ref_above_oos"] == 13
    assert b["median_realised_drift"] < 0.5 * b["median_designed_drift"]
    # With the drift it really gets, the level solved for a 2520-bar ARL false-alarms far sooner.
    h = siegmund_h(b["median_designed_drift"], 2520)
    assert 1200 < siegmund_arl(h, b["median_realised_drift"]) < 0.6 * 2520
