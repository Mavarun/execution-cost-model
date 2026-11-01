"""Cost-aware rebalancing rules that map a desired position to a held one.

All rules are causal: ``w[t]`` depends on ``target[: t + 1]`` and the
position held before the segment (``w0``), never on returns. ``target[t]``
already uses information through ``t - 1`` only (see :mod:`execution_cost.alpha`).

* ``FULL`` - hold the target (trade the whole gap every bar).
* ``PARTIAL(a)`` - ``w_t = w_{t-1} + a (x_t - w_{t-1})``: partial adjustment
  (Garleanu & Pedersen 2013 for quadratic costs). With ``w0 = 0`` this is an
  EWMA of the target, i.e. exactly slice 1's ``hold_bars`` smoother
  (``a = 2 / (span + 1)``), so it is the slice-1 turnover knob.
* ``BAND(b)`` - no-trade region: do nothing while ``|x_t - w_{t-1}| <= b``,
  otherwise trade to the *nearest edge* of the band (optimal for
  proportional costs; Davis & Norman 1990, Leland 2000).
* ``PARTIAL+BAND(a, b)`` - skip trades while the gap is inside the band,
  otherwise close a fraction ``a`` of the distance to the nearest band edge:
  ``w_t = w_{t-1} + a (gap - sign(gap) b)``. It nests both (``a = 1`` is
  ``BAND(b)``, ``b = 0`` is ``PARTIAL(a)``) and, like them, its only state is
  the held position, so a run split at any bar and continued from the held
  position is identical to one pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from execution_cost.costs import CostParams, total_unit_cost

FAMILIES = ("FULL", "PARTIAL", "BAND", "PARTIAL+BAND")


def partial_adjust(target: np.ndarray, rate: float, *, w0: float = 0.0) -> np.ndarray:
    """``w_t = (1 - a) w_{t-1} + a x_t`` started from ``w_{-1} = w0``."""
    a = float(rate)
    if not 0.0 < a <= 1.0:
        raise ValueError("rate must lie in (0, 1]")
    x = np.asarray(target, dtype=float)
    if x.size == 0:
        return x.copy()
    zi = np.array([(1.0 - a) * float(w0)])
    out, _ = lfilter([a], [1.0, -(1.0 - a)], x, zi=zi)
    return out


def no_trade_band(
    target: np.ndarray, width: float, *, w0: float = 0.0, rate: float = 1.0
) -> np.ndarray:
    """Trade only when the gap exceeds ``width``; close ``rate`` of the excess."""
    b = float(width)
    a = float(rate)
    if b < 0.0:
        raise ValueError("width must be non-negative")
    if not 0.0 < a <= 1.0:
        raise ValueError("rate must lie in (0, 1]")
    x = np.asarray(target, dtype=float)
    out = np.empty_like(x)
    w = float(w0)
    for t in range(x.size):
        gap = x[t] - w
        if gap > b:
            w += a * (gap - b)
        elif gap < -b:
            w += a * (gap + b)
        out[t] = w
    return out


@dataclass(frozen=True)
class Rule:
    """A rebalancing rule: family plus its parameters."""

    family: str
    rate: float = 1.0
    width: float = 0.0

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family {self.family!r}")

    @property
    def label(self) -> str:
        if self.family == "FULL":
            return "FULL"
        if self.family == "PARTIAL":
            return f"PARTIAL(a={self.rate:g})"
        if self.family == "BAND":
            return f"BAND(b={self.width:g})"
        return f"PARTIAL+BAND(a={self.rate:g},b={self.width:g})"

    def positions(self, target: np.ndarray, *, w0: float = 0.0) -> np.ndarray:
        x = np.asarray(target, dtype=float)
        if self.family == "FULL":
            return x.copy()
        if self.family == "PARTIAL":
            return partial_adjust(x, self.rate, w0=w0)
        if self.family == "BAND":
            return no_trade_band(x, self.width, w0=w0)
        return no_trade_band(x, self.width, w0=w0, rate=self.rate)


def trades_from_positions(weights: np.ndarray, *, w0: float = 0.0) -> np.ndarray:
    """``dw_t = w_t - w_{t-1}`` with ``w_{-1} = w0``."""
    w = np.asarray(weights, dtype=float)
    return np.diff(np.concatenate([[float(w0)], w]))


def net_returns(
    returns: np.ndarray,
    weights: np.ndarray,
    params: CostParams,
    *,
    daily_vol: np.ndarray | float,
    w0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gross ``w_t r_t``, cost ``|dw_t| * unit_cost(dw_t)`` and net, per bar.

    The full slice-1 cost model is charged: half-spread, linear slippage and
    square-root impact on participation ``|dw| / ADV``.
    """
    r = np.asarray(returns, dtype=float)
    w = np.asarray(weights, dtype=float)
    if r.shape != w.shape:
        raise ValueError("returns and weights must share shape")
    dw = np.abs(trades_from_positions(w, w0=w0))
    unit = total_unit_cost(dw, params, daily_vol=daily_vol)
    cost = np.where(dw > 0.0, dw * unit, 0.0)
    gross = w * r
    return gross, cost, gross - cost
