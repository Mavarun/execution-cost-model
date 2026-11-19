"""Post-hoc slice-8 reading: the shrunk gate false-alarms near its target, but each alarm is slow to undo."""

from execution_cost.cusum_eval import control_off_spells


def test_shrinking_fixes_the_alarm_rate_not_the_spell_length():
    full, shr = control_off_spells(1.0), control_off_spells(0.75)
    assert (full["alarms"], shr["alarms"]) == (50, 37)
    assert full["bars_per_alarm"] < 1600 and 2000 < shr["bars_per_alarm"] < 2520
    # Each false alarm keeps the gate off for about two years of trading days on average (~500 bars).
    assert shr["mean_off_spell"] > 450 and shr["median_off_spell"] > 300
    assert 0.15 < shr["share_off"] < full["share_off"]
