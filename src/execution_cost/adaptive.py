"""Adaptive sqrt-law execution: re-optimise the rest of the order each interval.

Slice 3's SQRT-OPT is a *static* schedule planned on the expected profile.
On a day whose volume runs 30% above normal, impact is cheaper than planned
and a risk-averse trader should speed up; on a thin day, slow down. This
policy re-solves the slice-3 mean-variance problem for the shares still to
trade over the intervals still to come, every interval, with the volume
forecast of :mod:`execution_cost.volume_model` (fitted profile x day-factor
nowcast from volume seen so far), and trades the first leg.

Scaling (planner exponent 0.5). For ``R`` shares left, day factor ``D``,
weights ``w`` over the remaining intervals: temporary cost scales as
``R^1.5 D^-0.5``, permanent and timing variance as ``R^2``. Only their
ratio, ``sqrt(R D)``, moves the optimal ``w``, so ``argmin w(R, D)`` equals
``argmin w(R D, 1)``. The first-leg fraction ``w_0(j, z)`` with
``z = sqrt(R D)`` is precomputed on a log grid per interval and
interpolated, so 4,000 days x 26 intervals cost ~26 x 40 small solves.
Timing variance per interval stays ``sigma_p^2 / N`` of the full day
(``lam`` is rescaled for the shorter sub-problem).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from execution_cost.optimal_sqrt import SqrtExecutionProblem, solve_sqrt_schedule
from execution_cost.simulator import MarketParams
from execution_cost.volume_model import DayFactorNowcast


def first_leg_fraction(
    params: MarketParams,
    profile: np.ndarray,
    j: int,
    shares_left: float,
    lam: float,
    temp_coeff: float,
    perm_coeff: float,
) -> float:
    """Optimal fraction of ``shares_left`` to trade in interval ``j`` (D = 1)."""
    N = params.n_intervals
    sub = np.asarray(profile[j:], dtype=float)
    M = len(sub)
    if M == 1:
        return 1.0
    prob = SqrtExecutionProblem(
        X=float(shares_left),
        profile=sub,  # un-normalised: expected volume = ADV * u_k
        adv=params.adv,
        sigma_price=params.sigma_price,
        temp_coeff=temp_coeff,
        perm_coeff=perm_coeff,
        lam=lam * M / N,  # keep sigma_p^2 / N per interval
        temp_exponent=0.5,
    )
    sol = solve_sqrt_schedule(prob)
    return float(sol.trades[0] / shares_left)


@dataclass
class AdaptiveSqrtOpt:
    """Re-optimising sqrt-law schedule (an adaptive policy for ``execute``)."""

    params: MarketParams
    nowcast: DayFactorNowcast
    target: float
    lam: float
    temp_coeff: float
    perm_coeff: float
    n_grid: int = 40
    adapt: bool = True  # False: always D_hat = 1 (re-solve on the plan only)
    grid_span: float = 1.0  # widen the z grid by this factor each side (price-adaptive use)
    _grid: np.ndarray = field(init=False, repr=False)
    _w0: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        X = float(self.target)
        if self.grid_span < 1.0:
            raise ValueError("grid_span must be >= 1")
        lo = np.sqrt(1e-4 * X * 0.2) / self.grid_span
        hi = np.sqrt(X * 5.0) * self.grid_span
        self._grid = np.geomspace(lo, hi, self.n_grid)
        N = self.params.n_intervals
        self._w0 = np.ones((N, self.n_grid))
        for j in range(N - 1):
            for i, z in enumerate(self._grid):
                self._w0[j, i] = first_leg_fraction(
                    self.params, self.nowcast.profile, j, z**2, self.lam, self.temp_coeff, self.perm_coeff
                )

    def fraction(self, j: int, shares_left: np.ndarray, day_factor: np.ndarray) -> np.ndarray:
        z = np.sqrt(np.maximum(shares_left, 0.0) * np.maximum(day_factor, 1e-6))
        lz = np.log(np.clip(z, self._grid[0], self._grid[-1]))
        return np.clip(np.interp(lz, np.log(self._grid), self._w0[j]), 0.0, 1.0)

    def shares(self, j, observed, remaining, params):
        if j == params.n_intervals - 1:
            return remaining
        d = self.nowcast.day_factor(observed) if self.adapt else np.ones(len(remaining))
        return remaining * self.fraction(j, remaining, d)


@dataclass
class PriceAdaptiveSqrtOpt(AdaptiveSqrtOpt):
    """Aggressive-in-the-money sqrt-law policy (Almgren & Lorenz 2007, AIM).

    Under a mean-variance objective on *realised* cost, a static schedule is
    not optimal among adaptive ones: after a favourable move (the trader is
    "in the money") the cost budget can absorb more impact, so trading faster
    to lock the gain in lowers variance at a small expected cost, and the
    reverse after an adverse move. This policy re-solves the rest of the order
    with risk aversion ``lam * m`` where

        z_j = -(mid_move_j - G sigma_p (X - R_j) / ADV) / (sigma_p sqrt(j / N))
        m_j = clip(exp(a z_j), m_min, m_max)

    ``z`` is the standardised *exogenous* move (the policy's own known
    permanent impact under the calibrated ``G`` is removed), positive when the
    price moved in the order's favour. ``a > 0`` is AIM, ``a < 0`` the passive
    (PIM) mirror, ``a = 0`` re-plans the static schedule. ``a`` is a free
    parameter: it is chosen on training days, never on evaluation days.

    Scaling: with the 0.5 planner exponent the first-leg fraction depends on
    ``lam`` and ``R D`` only through ``lam sqrt(R D)``, so ``lam m`` equals
    ``R D m^2`` at ``lam`` and the parent's precomputed grid is reused at
    ``z' = sqrt(R D) m`` (``grid_span`` widens it to cover ``m`` up to
    ``m_max``). ``adapt`` (volume nowcast) defaults to off so the price
    reaction is measured on its own.
    """

    a: float = 0.0
    m_min: float = float(np.exp(-2.0))
    m_max: float = float(np.exp(2.0))
    adapt: bool = False
    grid_span: float = float(np.exp(2.0))
    uses_prices = True

    def multiplier(self, j: int, mid_move: np.ndarray, remaining: np.ndarray) -> np.ndarray:
        mp = self.params
        if j == 0 or self.a == 0.0:
            return np.ones(len(remaining))
        own = self.perm_coeff * mp.sigma_price * (self.target - remaining) / mp.adv
        z = -(np.asarray(mid_move, float) - own) / (mp.sigma_price * np.sqrt(j / mp.n_intervals))
        return np.clip(np.exp(np.clip(self.a * z, -50.0, 50.0)), self.m_min, self.m_max)

    def shares(self, j, observed, remaining, params, mid_move=None):
        if j == params.n_intervals - 1:
            return remaining
        d = self.nowcast.day_factor(observed) if self.adapt else np.ones(len(remaining))
        mv = np.zeros(len(remaining)) if mid_move is None else mid_move
        m = self.multiplier(j, mv, remaining)
        z = np.sqrt(np.maximum(remaining, 0.0) * np.maximum(d, 1e-6)) * m
        lz = np.log(np.clip(z, self._grid[0], self._grid[-1]))
        w = np.clip(np.interp(lz, np.log(self._grid), self._w0[j]), 0.0, 1.0)
        return remaining * w


@dataclass
class NowcastPOV:
    """Implementable POV sized on the day-factor nowcast instead of lag-1 volume.

    Like :class:`execution_cost.simulator.POV` it never forces completion:
    shares it does not reach by the close are opportunity cost.
    """

    rate: float
    nowcast: DayFactorNowcast

    def __post_init__(self) -> None:
        if not 0.0 < self.rate < 1.0:
            raise ValueError("POV rate must be in (0, 1)")

    def shares(self, j, observed, remaining, params):
        return self.rate * self.nowcast.forecast(observed, j)
