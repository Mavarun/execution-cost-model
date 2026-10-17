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
    _grid: np.ndarray = field(init=False, repr=False)
    _w0: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        X = float(self.target)
        lo, hi = np.sqrt(1e-4 * X * 0.2), np.sqrt(X * 5.0)
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
