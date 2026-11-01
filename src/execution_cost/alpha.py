"""Persistent-alpha synthetic market for the slice-5 rebalancing study.

Slice 1 used an i.i.d. factor: a signal observed at ``t`` predicted only
``r_{t+1}``, so any smoothing threw information away and "trade less" could
only be bought by losing edge. Real alphas decay over several days. Here the
latent factor is AR(1),

    f_t = rho f_{t-1} + sqrt(1 - rho^2) u_t,       Var(f_t) = 1,
    r_{t+1} = vol * (ic * f_t + sqrt(1 - ic^2) e_{t+1}),
    s_t = f_t + noise_sd * z_t,

so ``ic`` is the correlation of the latent factor with the next return and
``rho`` sets the alpha half-life ``log(0.5) / log(rho)`` bars. The trader
sees only ``s_t``; the unconstrained target position is ``tanh(s_t)``.

Timing convention (same as :mod:`execution_cost.portfolio`): position
``w[t]`` is chosen with information through ``t - 1`` and earns
``returns[t]``. Arrays are aligned so that ``target[t]`` is the position the
trader would like to hold over bar ``t`` (it uses ``s_{t-1}``) and
``target[0] = 0``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import lfilter


@dataclass(frozen=True)
class AlphaParams:
    """Parameters of the persistent-alpha market (per bar = per trading day)."""

    n: int = 5040
    ic: float = 0.10
    rho: float = 0.90
    noise_sd: float = 0.50
    asset_vol: float = 0.01

    def half_life(self) -> float:
        """Alpha half-life in bars (``inf`` for a constant factor)."""
        if self.rho <= 0.0:
            return 0.0
        if self.rho >= 1.0:
            return float("inf")
        return float(np.log(0.5) / np.log(self.rho))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AlphaPath:
    """One simulated path: aligned returns, desired positions and the factor."""

    returns: np.ndarray
    target: np.ndarray
    factor: np.ndarray
    signal: np.ndarray
    daily_vol: np.ndarray
    params: AlphaParams
    seed: int


def ar1_factor(n: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Stationary unit-variance AR(1) path (first value drawn from N(0, 1))."""
    if not -1.0 < rho < 1.0:
        raise ValueError("rho must lie in (-1, 1)")
    shocks = rng.normal(0.0, 1.0, size=n)
    scale = np.sqrt(1.0 - rho**2)
    drive = shocks * scale
    drive[0] = shocks[0]  # stationary start
    return lfilter([1.0], [1.0, -rho], drive)


def simulate_alpha_path(params: AlphaParams | None = None, *, seed: int = 0) -> AlphaPath:
    """Simulate one aligned (returns, target) path. Deterministic in ``seed``."""
    p = params or AlphaParams()
    if p.n < 3:
        raise ValueError("n must be at least 3")
    if not 0.0 <= abs(p.ic) < 1.0:
        raise ValueError("|ic| must lie in [0, 1)")
    rng = np.random.default_rng(seed)
    factor = ar1_factor(p.n, p.rho, rng)
    eps = rng.normal(0.0, 1.0, size=p.n)
    noise = rng.normal(0.0, 1.0, size=p.n)
    signal = factor + p.noise_sd * noise

    returns = np.zeros(p.n, dtype=float)
    resid = np.sqrt(1.0 - p.ic**2)
    returns[1:] = p.asset_vol * (p.ic * factor[:-1] + resid * eps[1:])
    target = np.zeros(p.n, dtype=float)
    target[1:] = np.tanh(signal[:-1])
    return AlphaPath(
        returns=returns,
        target=target,
        factor=factor,
        signal=signal,
        daily_vol=np.full(p.n, p.asset_vol, dtype=float),
        params=p,
        seed=int(seed),
    )
