"""Intraday volume forecasting fitted on training days (no look-ahead).

Slice 3's POV-lag1 forecast the current interval from the previous
interval's realised volume (``V_{j-1} u_j / u_{j-1}``): one noisy draw, so
its participation errors are large (interval noise 30%). The simulator's
volume is ``ADV u_j D Z_j`` with a day factor ``D`` shared by every
interval of the day. The standard practical model therefore is

* a **fitted profile** ``u_hat``: mean intraday volume share over training
  days (the planner never sees the simulator's ``u``);
* a **day-factor nowcast** from the volume seen so far today,
  shrunk toward 1 with ``k0`` pseudo-intervals of prior volume:

      D_hat_j = (sum_{k<j} V_k + k0 ADV / N) / (ADV sum_{k<j} u_hat_k + k0 ADV / N)

  and the forecast ``V_hat_j = ADV u_hat_j D_hat_j``.

``k0`` is chosen on the training days by minimising the squared log
forecast error; evaluation uses disjoint days. Event days (volume surges)
violate the profile: the nowcast only learns about a surge after it starts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

K0_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def fit_profile(volumes: np.ndarray) -> np.ndarray:
    """Mean per-day volume share of each interval (sums to 1)."""
    v = np.asarray(volumes, dtype=float)
    if v.ndim != 2 or v.shape[0] < 1:
        raise ValueError("volumes must be (days, intervals)")
    shares = v / v.sum(axis=1, keepdims=True)
    u = shares.mean(axis=0)
    return u / u.sum()


@dataclass(frozen=True)
class DayFactorNowcast:
    profile: np.ndarray  # fitted u_hat (N,)
    adv: float
    k0: float = 2.0

    @property
    def n_intervals(self) -> int:
        return int(len(self.profile))

    def day_factor(self, observed: np.ndarray) -> np.ndarray:
        """Shrunk day-factor estimate from ``observed`` = volumes[:, :j]."""
        obs = np.asarray(observed, dtype=float)
        j = obs.shape[1]
        prior = self.k0 * self.adv / self.n_intervals
        seen = obs.sum(axis=1)
        expected = self.adv * float(np.sum(self.profile[:j]))
        return (seen + prior) / (expected + prior)

    def forecast(self, observed: np.ndarray, j: int) -> np.ndarray:
        """Forecast of interval ``j``'s volume from ``observed`` = volumes[:, :j]."""
        return self.adv * self.profile[j] * self.day_factor(observed)

    def remaining_volume(self, observed: np.ndarray) -> np.ndarray:
        """(P, N - j) forecasts of every interval still to come."""
        j = np.asarray(observed).shape[1]
        return self.adv * self.profile[None, j:] * self.day_factor(observed)[:, None]


def lag1_forecast(observed: np.ndarray, j: int, profile: np.ndarray, adv: float) -> np.ndarray:
    """Slice-3 POV-lag1 forecast (previous interval rescaled by the profile)."""
    obs = np.asarray(observed, dtype=float)
    if j == 0:
        return np.full(obs.shape[0], adv * profile[0])
    return obs[:, j - 1] * profile[j] / profile[j - 1]


def log_forecast_errors(volumes: np.ndarray, forecaster) -> np.ndarray:
    """(P, N) log(V_j / V_hat_j) for ``forecaster(observed, j)``."""
    v = np.asarray(volumes, dtype=float)
    out = np.empty_like(v)
    for j in range(v.shape[1]):
        out[:, j] = np.log(v[:, j] / forecaster(v[:, :j], j))
    return out


def fit_nowcast(train_volumes: np.ndarray, adv: float, k0_grid=K0_GRID) -> tuple[DayFactorNowcast, dict]:
    """Fit ``u_hat`` and choose ``k0`` on training days only."""
    u = fit_profile(train_volumes)
    scores = {}
    for k0 in k0_grid:
        m = DayFactorNowcast(u, adv, float(k0))
        e = log_forecast_errors(train_volumes, m.forecast)
        scores[float(k0)] = float(np.mean(e**2))
    best = min(scores, key=lambda k: (scores[k], k))
    return DayFactorNowcast(u, adv, best), {"k0_mse": scores, "k0": best}
