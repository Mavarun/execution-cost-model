from execution_cost.vol_target_eval import VT_PROTOCOL, run_vt_study, vt_seed


def test_protocol_is_frozen():
    assert VT_PROTOCOL["seeds"] == tuple(range(900, 920))
    assert VT_PROTOCOL["primary"] == ("WF-PB", "WF-PARTIAL") and VT_PROTOCOL["target_annual_vol"] == 0.10


def test_small_study_shape_and_determinism():
    kw = dict(n=1500, min_train=500, test_len=250)
    assert vt_seed(4, **kw) == vt_seed(4, **kw)
    out = run_vt_study(seeds=(1, 2, 3), **kw)
    assert out["H7"]["n"] == 3 and out["recommend"] in ("PARTIAL", "PARTIAL+BAND")
    for nm in VT_PROTOCOL["selectors"]:
        assert out["table"][nm]["mean_leverage"] > 0
