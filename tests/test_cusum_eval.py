from execution_cost.cusum_eval import ARMS, CUSUM_PROTOCOL, cusum_seed, run_cusum_study
from execution_cost.decay_eval import DECAY_PROTOCOL


def test_protocol_is_frozen_and_seeds_fresh():
    p = CUSUM_PROTOCOL
    assert p["markets"]["decay"] == {"ic": 0.0, "rho": 0.90} and p["markets"]["flip"] == {"ic": -0.05, "rho": 0.90}
    assert p["markets"]["control"] is None and p["switch_at"] == 2520
    assert p["cusum"] == {"arl0": 2520.0, "ref_bars": 1260} and p["max_control_sharpe_loss"] == 0.10
    assert p["seeds"] == tuple(range(900, 920)) and not set(p["seeds"]) & set(DECAY_PROTOCOL["seeds"])


KW = dict(n=1600, switch_at=1000, min_train=600, test_len=250, ref_bars=500)


def test_seed_runs_deterministically_on_a_small_path():
    a = cusum_seed(11, "flip", **KW)
    assert a == cusum_seed(11, "flip", **KW) and set(a) == set(ARMS)
    assert a["ungated"]["share_on_post"] == 1.0 and 0.0 <= a["CUSUM"]["share_on_post"] <= 1.0


def test_study_shape_on_small_paths():
    out = run_cusum_study(seeds=(1, 2, 3), **KW)
    assert set(out["means"]) == {"decay", "flip", "control"} and out["tests"]["CUSUM"]["decay_post_bps"]["n"] == 3
    assert isinstance(out["recommend"], bool)
