import numpy as np
import pytest

from execution_cost.adaptive_eval import PAIRS, evaluate_adaptive
from execution_cost.simulator import EventParams


@pytest.fixture(scope="module")
def rep():
    return evaluate_adaptive(q=0.05, kT=1.0, n_paths=600, n_train=200)


def test_seeds_must_be_disjoint():
    with pytest.raises(ValueError):
        evaluate_adaptive(seed=7, train_seed=7, n_paths=10)


def test_every_schedule_reported_on_every_split(rep):
    assert rep.n_event_days > 0
    for name, splits in rep.schedules.items():
        assert set(splits) == {"all", "normal", "event"}, name
        assert splits["all"]["n"] == 600
        assert splits["normal"]["n"] + splits["event"]["n"] == 600
    assert set(rep.pairs) == {f"{a} - {b}" for a, b in PAIRS}


def test_static_like_schedules_complete_and_costs_are_charged(rep):
    for name in ("TWAP", "VWAP", "SQRT-OPT", "REPLAN", "ADAPTIVE"):
        assert rep.schedules[name]["all"]["completion"] == pytest.approx(1.0)
        # spread alone is 2.5 bps; temporary impact must be positive
        assert rep.schedules[name]["all"]["temporary_bps"] > 0
        assert rep.schedules[name]["all"]["mean_bps"] > 2.5


def test_replan_control_equals_static(rep):
    d = rep.pairs["REPLAN - SQRT-OPT"]["all"]
    assert abs(d["mean_bps"]) < 0.02


def test_profile_is_fitted_not_given(rep):
    assert 0 < rep.volume_model["profile_max_rel_err"] < 0.2


def test_reproducible():
    a = evaluate_adaptive(q=0.02, n_paths=200, n_train=100)
    b = evaluate_adaptive(q=0.02, n_paths=200, n_train=100)
    assert a.schedules == b.schedules


def test_without_events_event_split_is_absent():
    r = evaluate_adaptive(q=0.02, n_paths=200, n_train=100, events=None)
    assert r.n_event_days == 0
    assert "event" not in r.schedules["TWAP"]
    assert np.isfinite(r.schedules["ADAPTIVE"]["all"]["mean_bps"])
