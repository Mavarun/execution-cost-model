"""Execution schedules: TWAP, VWAP, POV, and Almgren-Chriss.

* **TWAP** - equal shares per interval.
* **VWAP** - shares proportional to the *expected* intraday volume profile
  (pre-trade; does not see realised volume).
* **POV** - a fixed fraction of *realised* interval volume
  (:class:`execution_cost.simulator.POV`); adapts to liquidity but may not
  finish, leaving opportunity cost.
* **Almgren-Chriss** - risk-averse front-loaded schedule from
  :mod:`execution_cost.almgren_chriss`, parameterised from the market's
  (or a calibrated) impact model via :func:`ac_params_from_market`.
"""

from __future__ import annotations

import numpy as np

from execution_cost.almgren_chriss import ACParams, ac_trade_list
from execution_cost.simulator import MarketParams, volume_profile


def twap_schedule(target: float, n_intervals: int) -> np.ndarray:
    return np.full(n_intervals, target / n_intervals)


def vwap_schedule(target: float, profile: np.ndarray) -> np.ndarray:
    prof = np.asarray(profile, dtype=float)
    return target * prof / prof.sum()


def ac_params_from_market(
    params: MarketParams,
    target: float,
    lam: float,
    temp_coeff: float | None = None,
    perm_coeff: float | None = None,
) -> ACParams:
    """Map the square-root market model onto AC's linear-impact inputs.

    AC needs linear temporary impact ``h(v) = epsilon + eta v``. The sqrt law
    is linearised at the TWAP rate ``v0 = target / T`` (T = 1 day) so both
    give the same per-share temporary cost there:
    ``eta = Y sigma s0 sqrt(v0 / ADV) / v0``. Away from ``v0`` the linear
    model over-charges fast trading and under-charges slow trading - a
    deliberate, documented approximation. ``temp_coeff`` / ``perm_coeff`` may
    be calibrated estimates instead of the true simulator values.
    """
    Y = params.temp_coeff if temp_coeff is None else float(temp_coeff)
    G = params.perm_coeff if perm_coeff is None else float(perm_coeff)
    v0 = target / 1.0
    eta = Y * params.sigma_price * np.sqrt(v0 / params.adv) / v0
    gamma = G * params.sigma_price / params.adv
    return ACParams(
        X=float(target),
        T=1.0,
        N=params.n_intervals,
        sigma=params.sigma_price,
        eta=float(eta),
        gamma=float(gamma),
        epsilon=params.half_spread,
        lam=float(lam),
    )


def ac_schedule(
    params: MarketParams,
    target: float,
    lam: float,
    temp_coeff: float | None = None,
    perm_coeff: float | None = None,
) -> np.ndarray:
    return ac_trade_list(ac_params_from_market(params, target, lam, temp_coeff, perm_coeff))


def expected_profile(params: MarketParams) -> np.ndarray:
    return volume_profile(params.n_intervals, params.u_shape)
