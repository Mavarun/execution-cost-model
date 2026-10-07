"""Intraday execution simulator with square-root temporary impact.

Market (per simulated day, ``N`` equal intervals):

* Expected volume share per interval follows a U-shaped profile
  ``u_j ∝ 1 + u_shape * ((j - c) / c)^2`` (open/close heavier than midday).
  Realised volume ``V_j = ADV u_j D Z_j`` with log-normal day factor ``D``
  and interval noise ``Z_j`` (both mean one).
* Unaffected mid moves by arithmetic Brownian increments
  ``s0 sigma_daily sqrt(1/N) xi_j``.

Execution of ``n_j`` shares in interval ``j`` (``side=+1`` buy, ``-1`` sell):

* fill price ``p_j = m_j + side (half_spread + temp_j)`` where
  ``temp_j = temp_coeff * sigma_daily * s0 * (n_j / V_j)^temp_exponent``.
  For exponent 0.5 this is exactly
  :func:`execution_cost.costs.square_root_impact_cost` applied to the
  interval participation rate.
* permanent impact (linear in shares, Almgren-Chriss/Kyle style) shifts all
  later mids by ``side * perm_coeff * sigma_daily * s0 * n_j / ADV``.

Market randomness is generated once (:func:`simulate_market`) and reused for
every schedule, so schedule comparisons use common random numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from execution_cost.costs import square_root_impact_cost


@dataclass(frozen=True)
class MarketParams:
    s0: float = 100.0
    sigma_daily: float = 0.02
    adv: float = 1_000_000.0
    n_intervals: int = 26
    spread_bps: float = 5.0
    temp_coeff: float = 0.5
    temp_exponent: float = 0.5
    perm_coeff: float = 0.25
    u_shape: float = 1.5
    volume_noise: float = 0.3
    day_volume_noise: float = 0.2

    @property
    def half_spread(self) -> float:
        return 0.5 * self.spread_bps / 1e4 * self.s0

    @property
    def sigma_price(self) -> float:
        """Daily volatility in price units."""
        return self.sigma_daily * self.s0


def volume_profile(n_intervals: int, u_shape: float = 1.5) -> np.ndarray:
    """Expected fraction of daily volume per interval (sums to 1)."""
    j = np.arange(n_intervals, dtype=float)
    c = (n_intervals - 1) / 2.0
    w = 1.0 + u_shape * ((j - c) / max(c, 1.0)) ** 2
    return w / w.sum()


@dataclass
class MarketPaths:
    """Exogenous randomness for ``n_paths`` days (shared across schedules)."""

    noise_increments: np.ndarray  # (P, N) unaffected mid changes, price units
    volumes: np.ndarray  # (P, N) realised interval volumes, shares
    profile: np.ndarray  # (N,) expected volume fractions

    @property
    def n_paths(self) -> int:
        return int(self.volumes.shape[0])


def simulate_market(params: MarketParams, n_paths: int, seed: int = 0) -> MarketPaths:
    rng = np.random.default_rng(seed)
    N = params.n_intervals
    prof = volume_profile(N, params.u_shape)
    xi = rng.standard_normal((n_paths, N))
    noise = params.sigma_price * np.sqrt(1.0 / N) * xi
    sd, sv = params.day_volume_noise, params.volume_noise
    day = np.exp(sd * rng.standard_normal((n_paths, 1)) - 0.5 * sd**2)
    intr = np.exp(sv * rng.standard_normal((n_paths, N)) - 0.5 * sv**2)
    vols = params.adv * prof[None, :] * day * intr
    return MarketPaths(noise_increments=noise, volumes=vols, profile=prof)


@dataclass
class POV:
    """Percentage-of-volume: trade ``rate`` x interval volume.

    ``lag=0`` is the idealised slice-2 POV: it sees the *current* interval's
    realised volume before trading in it. ``lag=1`` is implementable: it
    targets ``rate`` x a forecast of the current interval's volume made from
    the previous interval's realised volume, rescaled by the expected
    profile (``V_{j-1} u_j / u_{j-1}``; the first interval uses its
    pre-trade expectation ``ADV u_0``). The realised participation, and so
    the impact, is then ``rate x forecast / V_j``.
    """

    rate: float
    lag: int = 0

    def __post_init__(self):
        if not 0.0 < self.rate < 1.0:
            raise ValueError("POV rate must be in (0, 1)")
        if self.lag not in (0, 1):
            raise ValueError("POV lag must be 0 (idealised) or 1 (previous interval)")

    def volume_signal(self, market: "MarketPaths", params: "MarketParams", j: int) -> np.ndarray:
        """Volume the POV order sizes interval ``j`` on (shape ``(P,)``)."""
        if self.lag == 0:
            return market.volumes[:, j]
        if j == 0:
            return np.full(market.n_paths, params.adv * market.profile[0])
        return market.volumes[:, j - 1] * market.profile[j] / market.profile[j - 1]


@dataclass
class ExecutionResult:
    side: int
    target: float | np.ndarray  # scalar or (P,) per-path order size
    trades: np.ndarray  # (P, N) shares (>= 0)
    fill_prices: np.ndarray  # (P, N)
    mids: np.ndarray  # (P, N) mid just before each interval's trading
    perm_before: np.ndarray  # (P, N) cumulative permanent impact in mid
    noise_before: np.ndarray  # (P, N) cumulative unaffected move in mid
    final_mid: np.ndarray  # (P,)
    final_noise: np.ndarray  # (P,)
    final_perm: np.ndarray  # (P,)
    arrival: float
    half_spread: float
    temp_impact: np.ndarray  # (P, N) temporary impact per share, price units

    @property
    def executed(self) -> np.ndarray:
        return self.trades.sum(axis=1)

    @property
    def unexecuted(self) -> np.ndarray:
        return self.target - self.executed

    @property
    def completion(self) -> np.ndarray:
        return self.executed / self.target


def temporary_impact(participation: np.ndarray, params: MarketParams) -> np.ndarray:
    """Per-share temporary impact (price units) at an interval participation."""
    part = np.maximum(np.asarray(participation, dtype=float), 0.0)
    if params.temp_exponent == 0.5:
        frac = square_root_impact_cost(
            part, impact_coeff=params.temp_coeff, daily_vol=params.sigma_daily
        )
    else:
        frac = params.temp_coeff * params.sigma_daily * part**params.temp_exponent
    return frac * params.s0


def execute(
    schedule: np.ndarray | POV,
    market: MarketPaths,
    params: MarketParams,
    target: float | np.ndarray,
    side: int = 1,
) -> ExecutionResult:
    """Run a schedule against every simulated day.

    ``schedule`` is a static trade list of length ``N`` (or ``(P, N)``) that
    should sum to ``target``, or a :class:`POV` instance. Static trades are
    capped so cumulative shares never exceed ``target``. ``target`` may be a
    ``(P,)`` array to give every simulated day its own order size (used for
    metaorder calibration).
    """
    if side not in (1, -1):
        raise ValueError("side must be +1 (buy) or -1 (sell)")
    P, N = market.volumes.shape
    trades = np.zeros((P, N))
    fills = np.zeros((P, N))
    mids = np.zeros((P, N))
    permb = np.zeros((P, N))
    noiseb = np.zeros((P, N))
    temps = np.zeros((P, N))
    tgt = np.asarray(target, dtype=float)
    if tgt.ndim not in (0, 1) or (tgt.ndim == 1 and tgt.shape != (P,)):
        raise ValueError("target must be a scalar or have shape (n_paths,)")
    remaining = np.broadcast_to(tgt, (P,)).astype(float).copy()
    perm = np.zeros(P)
    noise = np.zeros(P)
    static = None
    if not isinstance(schedule, POV):
        static = np.broadcast_to(np.asarray(schedule, dtype=float), (P, N))
        if (static < -1e-12).any():
            raise ValueError("static schedules must be non-negative")
    for j in range(N):
        vol = market.volumes[:, j]
        if static is not None:
            n = np.minimum(static[:, j], remaining)
        else:
            n = np.minimum(schedule.rate * schedule.volume_signal(market, params, j), remaining)
        n = np.maximum(n, 0.0)
        mid = params.s0 + noise + perm
        temp = temporary_impact(n / vol, params)
        trades[:, j] = n
        mids[:, j] = mid
        permb[:, j] = perm
        noiseb[:, j] = noise
        temps[:, j] = temp
        fills[:, j] = mid + side * (params.half_spread + temp)
        remaining = remaining - n
        perm = perm + side * params.perm_coeff * params.sigma_price * n / params.adv
        noise = noise + market.noise_increments[:, j]
    return ExecutionResult(
        side=side,
        target=float(tgt) if tgt.ndim == 0 else tgt.copy(),
        trades=trades,
        fill_prices=fills,
        mids=mids,
        perm_before=permb,
        noise_before=noiseb,
        final_mid=params.s0 + noise + perm,
        final_noise=noise,
        final_perm=perm,
        arrival=params.s0,
        half_spread=params.half_spread,
        temp_impact=temps,
    )
