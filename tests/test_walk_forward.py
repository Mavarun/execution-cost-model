"""Slice 5 walk-forward: causal refits, disjoint seeds, a no-break control, reproducibility."""

from dataclasses import replace

import numpy as np
import pytest

import execution_cost.walk_forward as wf
from execution_cost.simulator import EventParams, MarketParams
from execution_cost.walk_forward import WalkForwardSpec, block_market, run_walk_forward

SMALL = WalkForwardSpec(n_train=120, block=20, n_blocks=6, break_block=3, windows=(40,))


def test_spec_validation():
    with pytest.raises(ValueError):
        WalkForwardSpec(break_block=0)
    with pytest.raises(ValueError):
        WalkForwardSpec(post_market=MarketParams(adv=2e6))
    with pytest.raises(ValueError):
        WalkForwardSpec(eval_seed=7)


def test_block_days_do_not_depend_on_any_model_and_switch_at_the_break():
    mp_a, a = block_market(SMALL, 2)
    mp_b, b = block_market(SMALL, 3)
    assert mp_a.u_shape == 1.5 and mp_b.u_shape == 3.0
    assert np.array_equal(a.volumes, block_market(SMALL, 2)[1].volumes)
    assert not np.allclose(a.volumes, b.volumes)


def test_every_refit_uses_only_days_realised_before_its_block(monkeypatch):
    seen = []
    orig = wf.fit_models

    def spy(history, adv):
        seen.append(history.shape[0])
        return orig(history, adv)

    monkeypatch.setattr(wf, "fit_models", spy)
    run_walk_forward(SMALL)
    # FIXED once, then per block: ROLLING-40 and EXPANDING
    assert seen[0] == SMALL.n_train
    for b in range(SMALL.n_blocks):
        avail = SMALL.n_train + b * SMALL.block
        assert seen[1 + 2 * b] == min(40, avail) and seen[2 + 2 * b] == avail


def test_no_break_control_rolling_matches_fixed_within_noise():
    same = replace(SMALL, post_market=MarketParams(), post_events=EventParams(), n_blocks=8, break_block=4)
    r = run_walk_forward(same).to_dict()
    for rg in ("pre", "post"):
        d = r["pairs"]["EXPANDING - FIXED"][rg]
        assert abs(d["mean_bps"]) < 3 * d["se_bps"] + 0.05


def test_report_shape_costs_and_reproducibility():
    a, b = run_walk_forward(SMALL).to_dict(), run_walk_forward(SMALL).to_dict()
    assert a["per_regime"] == b["per_regime"] and a["pairs"] == b["pairs"]
    assert a["models"] == ("FIXED", "ROLLING-40", "EXPANDING")
    for rg in ("pre", "post"):
        for m in a["models"]:
            s = a["per_regime"][rg][m]
            assert s["n_days"] == 60 and s["mean_bps"] > 0  # spread + impact are always charged
            assert 0.9 < s["completion"] <= 1.0
    assert a["fits"]["FIXED"] == [(-1, a["fits"]["FIXED"][0][1], a["fits"]["FIXED"][0][2], 120)]


def test_refit_tracks_a_steeper_profile_after_the_break():
    r = run_walk_forward(replace(SMALL, n_blocks=8, break_block=3)).to_dict()
    pr = r["per_regime"]["post"]
    assert pr["ROLLING-40"]["forecast_log_rmse"] < pr["FIXED"]["forecast_log_rmse"]
