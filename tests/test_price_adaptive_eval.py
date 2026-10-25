import numpy as np
import pytest

from execution_cost.price_adaptive_eval import evaluate_price_adaptive, paired_bootstrap_ci, utility
from execution_cost.simulator import MarketParams


def test_utility_and_bootstrap_ci():
    rng = np.random.default_rng(0)
    a = rng.normal(10, 5, 500)
    assert utility(a, 1e4, 0.0) == pytest.approx(a.mean())
    assert utility(a, 1e4, 2.0) == pytest.approx(a.mean() + 2.0 * a.var(ddof=1))
    lo, hi = paired_bootstrap_ci(a, a, 1e4, 1.0)
    assert lo == hi == 0.0
    lo, hi = paired_bootstrap_ci(a + 1.0, a, 1e4, 1.0)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)


@pytest.fixture(scope="module")
def small():
    return evaluate_price_adaptive(MarketParams(), q=0.05, kT=3.0, n_paths=300, n_train=200,
                                   a_grid=(-0.5, 0.0, 0.5), n_grid=16)


def test_choice_is_made_on_training_days_only(small):
    r = small
    assert r.chosen_a == min(r.train_utility, key=r.train_utility.get)
    assert r.chosen["utility"] == pytest.approx(r.eval_utility[r.chosen_a])
    assert r.utility_change == pytest.approx(r.chosen["utility"] - r.static["utility"])
    assert r.utility_ci[0] <= r.utility_change <= r.utility_ci[1] or abs(r.utility_change) < 1e-9
    assert set(r.to_dict()["train_utility"]) == {"-0.5", "0", "0.5"}


def test_a_zero_tracks_the_static_plan(small):
    assert small.eval_utility[0.0] == pytest.approx(small.static["utility"], rel=0.01)


def test_rejects_shared_seeds_and_grids_without_zero():
    with pytest.raises(ValueError):
        evaluate_price_adaptive(seed=7, train_seed=7, n_paths=10, n_train=10)
    with pytest.raises(ValueError):
        evaluate_price_adaptive(a_grid=(0.5,), n_paths=10, n_train=10)
