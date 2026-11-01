"""Persistent-alpha generator: moments, alignment, no look-ahead, determinism."""

from __future__ import annotations

import numpy as np
import pytest

from execution_cost.alpha import AlphaParams, ar1_factor, simulate_alpha_path


def test_ar1_factor_is_unit_variance_with_lag1_autocorrelation_rho():
    rng = np.random.default_rng(3)
    f = ar1_factor(200_000, 0.9, rng)
    assert f.var() == pytest.approx(1.0, rel=0.03)
    ac1 = np.corrcoef(f[1:], f[:-1])[0, 1]
    assert ac1 == pytest.approx(0.9, abs=0.005)


def test_ar1_factor_rejects_nonstationary_rho():
    with pytest.raises(ValueError):
        ar1_factor(10, 1.0, np.random.default_rng(0))


def test_half_life_matches_rho():
    assert AlphaParams(rho=0.5).half_life() == pytest.approx(1.0)
    assert AlphaParams(rho=0.9).half_life() == pytest.approx(6.579, abs=1e-3)
    assert AlphaParams(rho=0.0).half_life() == 0.0


def test_factor_predicts_next_return_with_correlation_ic():
    p = AlphaParams(n=200_000, ic=0.1, rho=0.9)
    path = simulate_alpha_path(p, seed=11)
    corr = np.corrcoef(path.factor[:-1], path.returns[1:])[0, 1]
    assert corr == pytest.approx(0.1, abs=0.006)
    assert path.returns.std() == pytest.approx(0.01, rel=0.02)


def test_target_uses_only_the_previous_signal():
    path = simulate_alpha_path(AlphaParams(n=500), seed=5)
    assert path.target[0] == 0.0
    np.testing.assert_allclose(path.target[1:], np.tanh(path.signal[:-1]))
    # Same-bar factor is uncorrelated with the same-bar return shock beyond the
    # one-bar predictive link: target[t] never sees returns[t].
    assert np.all(np.abs(path.target) < 1.0)


def test_simulation_is_deterministic_in_seed():
    a = simulate_alpha_path(AlphaParams(n=300), seed=9)
    b = simulate_alpha_path(AlphaParams(n=300), seed=9)
    c = simulate_alpha_path(AlphaParams(n=300), seed=10)
    np.testing.assert_array_equal(a.returns, b.returns)
    np.testing.assert_array_equal(a.target, b.target)
    assert not np.array_equal(a.returns, c.returns)


def test_zero_ic_is_a_null_market():
    path = simulate_alpha_path(AlphaParams(n=100_000, ic=0.0), seed=1)
    corr = np.corrcoef(path.target[1:], path.returns[1:])[0, 1]
    assert abs(corr) < 0.01


def test_regime_path_switches_persistence_and_keeps_unit_variance():
    from execution_cost.alpha import simulate_regime_path

    a, b = AlphaParams(n=200_000, rho=0.97), AlphaParams(n=200_000, rho=0.70)
    p = simulate_regime_path(a, b, switch_at=100_000, seed=2)
    f = p.factor
    assert np.corrcoef(f[1:100_000], f[:99_999])[0, 1] == pytest.approx(0.97, abs=0.01)
    assert np.corrcoef(f[100_001:], f[100_000:-1])[0, 1] == pytest.approx(0.70, abs=0.01)
    assert f[100_000:].var() == pytest.approx(1.0, rel=0.03)
    np.testing.assert_allclose(p.target[1:], np.tanh(p.signal[:-1]))


def test_regime_path_validates_inputs():
    from execution_cost.alpha import simulate_regime_path

    with pytest.raises(ValueError):
        simulate_regime_path(AlphaParams(n=100), AlphaParams(n=200), switch_at=50)
    with pytest.raises(ValueError):
        simulate_regime_path(AlphaParams(n=100), AlphaParams(n=100), switch_at=100)
