"""control_cost_by_seed_set: the shrunk gate's control cost on every seed set that tested it."""

import pytest

from execution_cost.cusum_eval import control_cost_by_seed_set


@pytest.fixture(scope="module")
def r():
    return control_cost_by_seed_set()


def test_sets_and_pooling(r):
    ps = r["per_set"]
    assert [v["n"] for v in ps.values()] == [20, 20, 60] and r["pooled"]["n"] == 100
    pooled = sum(v["cost"] * v["n"] for v in ps.values()) / 100
    assert r["pooled"]["cost"] == pytest.approx(pooled, abs=1e-12)


def test_reading(r):
    ps = r["per_set"]
    assert ps["slice 10 (1200-1259)"]["cost"] < 0.10 < r["pooled"]["cost"]
    assert r["anova_p"] < 0.05
