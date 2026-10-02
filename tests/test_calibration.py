"""Square-root impact calibration on synthetic metaorders."""

from __future__ import annotations

import numpy as np
import pytest

from execution_cost.calibration import (
    calibrate_impact,
    fit_binned_loglog,
    fit_fixed_exponent,
    fit_power_law_nls,
    generate_metaorders,
    twap_effective_sqrt_coefficient,
)


def test_fits_recover_exact_power_law_without_noise():
    q = np.exp(np.linspace(np.log(0.005), np.log(0.2), 400))
    y = 0.7 * q**0.5
    nls = fit_power_law_nls(q, y)
    assert nls["Y"] == pytest.approx(0.7, rel=1e-6)
    assert nls["delta"] == pytest.approx(0.5, abs=1e-6)
    assert fit_fixed_exponent(q, y, 0.5) == pytest.approx(0.7)
    x = q[:, None] ** 0.5
    assert fit_fixed_exponent(q, y + 0.01 * np.sin(q), 0.5) == pytest.approx(
        np.linalg.lstsq(x, y + 0.01 * np.sin(q), rcond=None)[0][0]
    )
    rng = np.random.default_rng(0)
    b = fit_binned_loglog(q, y * np.exp(0.01 * rng.normal(size=q.size)), n_bins=8)
    assert b["delta"] == pytest.approx(0.5, abs=0.02)


def test_metaorders_are_seeded_and_in_range():
    a = generate_metaorders(n_orders=800, seed=3)
    b = generate_metaorders(n_orders=800, seed=3)
    np.testing.assert_array_equal(a.y, b.y)
    assert a.q.min() >= 0.005 and a.q.max() <= 0.2
    assert set(np.unique(a.sigma)) == {0.01, 0.02, 0.03, 0.04}


def test_calibration_recovers_square_root_exponent_and_rejects_linear():
    r = calibrate_impact(seed=0)
    assert r.n_orders == 20_000
    lo, hi = r.binned_delta_ci
    assert lo < 0.5 < hi  # true interval-level exponent
    assert hi < 0.9  # linear impact (delta = 1) rejected
    assert abs(r.nls_delta - 0.5) < 0.15
    # sqrt-coefficient ~ temporary TWAP coefficient + small permanent spill-over
    eff = twap_effective_sqrt_coefficient()
    assert eff - 0.03 < r.sqrt_Y < eff + 0.08
    # permanent impact is noisy but its CI covers the true 0.25
    assert r.perm_ci[0] < 0.25 < r.perm_ci[1]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sqrt_law_beats_linear_out_of_sample(seed):
    r = calibrate_impact(seed=seed)
    assert r.oos_binned_rmse_sqrt < r.oos_binned_rmse_linear
    assert r.oos_mse_sqrt <= r.oos_mse_linear
