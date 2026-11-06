from execution_cost.decay_eval import DECAY_PROTOCOL, decay_seed, run_decay_study


def test_protocol_is_frozen():
    p = DECAY_PROTOCOL
    assert p["after"] == {"ic": 0.0, "rho": 0.90} and p["switch_at"] == 2520
    assert p["primary"] == "G252" and p["gates"] == {"G252": 252, "G126": 126, "G504": 504}
    assert p["seeds"] == tuple(range(800, 820)) and p["max_control_sharpe_loss"] == 0.10


def test_decay_seed_small_path_runs_and_is_deterministic():
    kw = dict(n=1600, switch_at=1000, min_train=600, test_len=250)
    a = decay_seed(11, **kw)
    b = decay_seed(11, **kw)
    assert a == b
    assert set(a) == {"ungated", "G252", "G126", "G504"}
    assert 0.0 <= a["G252"]["share_on_post"] <= 1.0


def test_study_shape_on_small_paths():
    out = run_decay_study(seeds=(1, 2, 3), n=1600, switch_at=1000, min_train=600, test_len=250)
    assert set(out["tests"]) == {"G252", "G126", "G504"}
    assert out["H6"]["n"] == 3
    assert isinstance(out["recommend"], bool)
