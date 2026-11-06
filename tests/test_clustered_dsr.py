from execution_cost.clustered_dsr import MAX_DISTANCE, run_clustered_study, seed_dsr


def test_rule_is_frozen():
    assert MAX_DISTANCE == 0.5


def test_clustered_count_sits_between_eff_and_raw():
    r = seed_dsr(500)
    assert r["n_eff"] <= r["n_clustered"] <= r["n_raw"] == 42
    assert r["dsr_raw"] <= r["dsr_clustered"] + 1e-12


def test_study_shape():
    out = run_clustered_study(signal_seeds=(500,), null_seeds=(600,))
    assert set(out) >= {"raw", "eff", "clustered", "clustered_ok"}
    assert out["raw"]["n_signal"] == 1
