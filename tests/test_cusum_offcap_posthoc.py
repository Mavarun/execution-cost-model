"""Slice 11 post-hoc (decides nothing): cap re-entries after a real decay and how long they last."""

from execution_cost.cusum_eval import capped_redetection, decay_reentries


def test_capped_gate_re_enters_often_and_stays_on_for_hundreds_of_bars():
    c = capped_redetection()
    assert c["reentries"] == 190 and round(c["per_seed"], 2) == 3.17
    assert c["median_on_spell"] == 428.0 and round(c["share_on_spell_over_500"], 2) == 0.41
    u = decay_reentries(1.0, seeds=range(1300, 1360))
    assert u["reentries"] == 41 and round(u["per_seed"], 2) == 0.68 and u["median_on_spell"] == 322.0
    # the cap's 126 off bars against a ~428-bar re-detection: off about a fifth of post-decay bars
    assert 0.15 < 126 / (126 + c["median_on_spell"]) < 0.25
