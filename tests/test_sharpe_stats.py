"""PSR / expected-max / DSR / HAC: closed forms and Monte Carlo checks."""

from __future__ import annotations

import numpy as np
import pytest

from execution_cost.sharpe_stats import (
    deflated_sharpe,
    expected_max_sharpe,
    hac_mean_test,
    per_bar_sharpe,
    probabilistic_sharpe,
    sharpe_se,
)


def test_sharpe_se_normal_case():
    assert sharpe_se(0.0, 101) == pytest.approx(0.1)
    assert sharpe_se(0.1, 1001) == pytest.approx(np.sqrt((1 + 0.5 * 0.01) / 1000))


def test_psr_is_half_at_the_benchmark_and_monotone():
    assert probabilistic_sharpe(0.05, 1000, benchmark=0.05) == pytest.approx(0.5)
    assert probabilistic_sharpe(0.08, 1000) > probabilistic_sharpe(0.04, 1000) > 0.5
    # negative skew and fat tails widen the SE, lowering PSR for a positive SR
    assert probabilistic_sharpe(0.08, 1000, skew=-1, kurt=8) < probabilistic_sharpe(0.08, 1000)


def test_sharpe_se_matches_monte_carlo_under_normal_returns():
    rng = np.random.default_rng(0)
    srs = [per_bar_sharpe(rng.normal(0.05, 1.0, 500)) for _ in range(4000)]
    assert np.std(srs) == pytest.approx(sharpe_se(0.05, 500), rel=0.05)


def test_expected_max_matches_monte_carlo():
    rng = np.random.default_rng(1)
    n_trials, t = 50, 500
    maxes = []
    srs_all = []
    for _ in range(400):
        srs = np.array([per_bar_sharpe(rng.normal(0.0, 1.0, t)) for _ in range(n_trials)])
        maxes.append(srs.max())
        srs_all.append(srs)
    v = np.var(np.concatenate(srs_all))
    assert expected_max_sharpe(n_trials, v) == pytest.approx(np.mean(maxes), rel=0.06)


def test_expected_max_grows_with_trials_and_is_zero_for_one():
    assert expected_max_sharpe(1, 0.01) == 0.0
    vals = [expected_max_sharpe(n, 0.001) for n in (2, 10, 100, 1000)]
    assert all(a < b for a, b in zip(vals, vals[1:]))
    with pytest.raises(ValueError):
        expected_max_sharpe(0, 0.01)


def test_deflated_sharpe_is_psr_at_expected_max():
    rng = np.random.default_rng(2)
    r = rng.normal(0.0006, 0.01, 2520)
    out = deflated_sharpe(r, n_trials=40, trial_var=2.5e-4)
    assert out["benchmark"] == pytest.approx(expected_max_sharpe(40, 2.5e-4))
    assert out["dsr"] < out["psr0"]
    one = deflated_sharpe(r, n_trials=1, trial_var=2.5e-4)
    assert one["dsr"] == pytest.approx(one["psr0"])
    assert out["sharpe_annual"] == pytest.approx(out["sharpe"] * np.sqrt(252))


def test_hac_mean_test_recovers_mean_and_widens_se_under_autocorrelation():
    rng = np.random.default_rng(3)
    e = rng.normal(0, 1, 20_000)
    iid = hac_mean_test(e + 0.05, maxlags=5)
    assert iid["mean"] == pytest.approx(0.05 + e.mean())
    assert iid["se"] == pytest.approx(1 / np.sqrt(20_000), rel=0.1)
    ma = np.convolve(e, np.ones(5), mode="same")
    assert hac_mean_test(ma, maxlags=10)["se"] > 1.8 * ma.std() / np.sqrt(ma.size)


def test_effective_trials_bounds():
    from execution_cost.sharpe_stats import effective_trials

    rng = np.random.default_rng(4)
    iid = rng.normal(size=(5000, 10))
    assert effective_trials(iid) == pytest.approx(10, rel=0.03)
    common = rng.normal(size=(5000, 1))
    same = common + 1e-6 * rng.normal(size=(5000, 10))
    assert effective_trials(same) == pytest.approx(1.0, abs=1e-3)
    # two independent blocks of five near-identical trials -> about 2
    blocks = np.hstack([np.repeat(rng.normal(size=(5000, 1)), 5, axis=1) + 1e-6 * rng.normal(size=(5000, 5)) for _ in range(2)])
    assert effective_trials(blocks) == pytest.approx(2.0, rel=0.03)
    assert effective_trials(iid[:, :1]) == 1.0
