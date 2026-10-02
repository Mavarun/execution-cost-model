"""Almgren-Chriss (2000) optimal execution with linear impact.

Reference: Almgren, R., Chriss, N. (2000). "Optimal execution of portfolio
transactions." *Journal of Risk* 3(2), 5-39.

Model (liquidating ``X`` shares over ``[0, T]`` in ``N`` intervals of length
``tau = T/N``; a buy program is the mirror image):

* holdings ``x_0 = X, ..., x_N = 0``; trades ``n_k = x_{k-1} - x_k``;
* mid price ``S_k = S_{k-1} + sigma sqrt(tau) xi_k - tau g(n_k / tau)`` with
  linear permanent impact ``g(v) = gamma v``;
* execution price ``S_{k-1} - h(n_k / tau)`` with linear temporary impact
  ``h(v) = epsilon sign(v) + eta v``.

Implementation shortfall ``X S_0 - sum n_k (S_{k-1} - h)`` has

    E = 1/2 gamma X^2 + epsilon sum |n_k| + (eta_tilde / tau) sum n_k^2
    V = sigma^2 tau sum_{k=1..N} x_k^2,    eta_tilde = eta - gamma tau / 2

and the minimiser of ``E + lambda V`` is

    x_j = X sinh(kappa (T - t_j)) / sinh(kappa T),
    (2 / tau^2) (cosh(kappa tau) - 1) = lambda sigma^2 / eta_tilde.

``lambda -> 0`` gives the linear (TWAP) schedule; larger ``lambda``
front-loads trading to cut timing risk at the price of more impact.

Units: ``sigma`` in price per sqrt(time unit), ``eta`` in price per
(share / time unit), ``gamma`` in price per share, ``epsilon`` in price,
``lam`` in 1 / price.

Default ``ACParams`` (one day, 26 x 15-min bins, $100 stock, 2% daily vol,
100k shares): temporary impact ``eta * X/T`` = $0.25/share (25 bps) at the
TWAP rate, total permanent impact ``gamma * X`` = 2.5 bps, half-spread
``epsilon`` = 2.5 bps.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class ACParams:
    X: float = 100_000.0
    T: float = 1.0
    N: int = 26
    sigma: float = 2.0
    eta: float = 2.5e-6
    gamma: float = 2.5e-7
    epsilon: float = 0.025
    lam: float = 1e-6

    @property
    def tau(self) -> float:
        return self.T / self.N

    @property
    def eta_tilde(self) -> float:
        return self.eta - 0.5 * self.gamma * self.tau

    def validate(self) -> None:
        if self.N < 1 or self.T <= 0 or self.X < 0:
            raise ValueError("need N >= 1, T > 0, X >= 0")
        if self.sigma < 0 or self.lam < 0 or self.gamma < 0 or self.epsilon < 0:
            raise ValueError("sigma, lam, gamma, epsilon must be >= 0")
        if self.eta_tilde <= 0:
            raise ValueError("eta must exceed gamma * tau / 2 (eta_tilde > 0)")


def ac_kappa(p: ACParams) -> float:
    """Urgency ``kappa`` from the exact discrete-time relation."""
    p.validate()
    kt2 = p.lam * p.sigma**2 / p.eta_tilde
    return float(np.arccosh(0.5 * kt2 * p.tau**2 + 1.0) / p.tau)


def ac_trajectory(p: ACParams) -> np.ndarray:
    """Optimal holdings ``x_0..x_N`` (length ``N + 1``)."""
    kappa = ac_kappa(p)
    t = np.arange(p.N + 1) * p.tau
    if kappa * p.T < 1e-10:
        return p.X * (1.0 - t / p.T)
    # sinh ratio, computed stably for large kappa*T
    num = np.exp(-kappa * t) - np.exp(-kappa * (2.0 * p.T - t))
    den = 1.0 - np.exp(-2.0 * kappa * p.T)
    x = p.X * num / den
    x[-1] = 0.0
    return x


def ac_trade_list(p: ACParams) -> np.ndarray:
    """Shares traded in each of the ``N`` intervals (sums to ``X``)."""
    return -np.diff(ac_trajectory(p))


def expected_cost(p: ACParams, holdings: np.ndarray | None = None) -> float:
    """``E[IS]`` of a holdings path under the AC linear-impact model."""
    x = ac_trajectory(p) if holdings is None else np.asarray(holdings, dtype=float)
    n = -np.diff(x)
    return float(
        0.5 * p.gamma * p.X**2 + p.epsilon * np.abs(n).sum() + p.eta_tilde / p.tau * (n**2).sum()
    )


def cost_variance(p: ACParams, holdings: np.ndarray | None = None) -> float:
    """``Var[IS]`` of a holdings path: ``sigma^2 tau sum x_k^2``."""
    x = ac_trajectory(p) if holdings is None else np.asarray(holdings, dtype=float)
    return float(p.sigma**2 * p.tau * (x[1:] ** 2).sum())


def optimal_holdings_by_linear_solve(p: ACParams) -> np.ndarray:
    """Independent check: minimise ``E + lam V`` as a tridiagonal linear system.

    For a one-directional schedule the objective is quadratic in the
    interior holdings ``x_1..x_{N-1}``; setting the gradient to zero gives
    ``(2 + c) x_j - x_{j-1} - x_{j+1} = 0`` with ``c = lam sigma^2 tau^2 / eta_tilde``.
    """
    p.validate()
    m = p.N - 1
    if m <= 0:
        return np.array([p.X, 0.0])
    c = p.lam * p.sigma**2 * p.tau**2 / p.eta_tilde
    A = np.diag(np.full(m, 2.0 + c)) - np.diag(np.ones(m - 1), 1) - np.diag(np.ones(m - 1), -1)
    b = np.zeros(m)
    b[0] = p.X
    interior = np.linalg.solve(A, b)
    return np.concatenate([[p.X], interior, [0.0]])


def efficient_frontier(p: ACParams, lams) -> list[dict]:
    """``(lam, E, sqrt(V))`` along a grid of risk aversions."""
    out = []
    for lam in lams:
        q = replace(p, lam=float(lam))
        out.append(
            {
                "lam": float(lam),
                "expected_cost": expected_cost(q),
                "std_cost": float(np.sqrt(cost_variance(q))),
                "kappa": ac_kappa(q),
            }
        )
    return out


def simulate_ac_shortfall(
    p: ACParams,
    trades: np.ndarray | None = None,
    s0: float = 100.0,
    n_paths: int = 20_000,
    seed: int = 0,
) -> np.ndarray:
    """Monte Carlo IS (price units) of a sell schedule under the AC dynamics."""
    n = ac_trade_list(p) if trades is None else np.asarray(trades, dtype=float)
    rng = np.random.default_rng(seed)
    xi = rng.standard_normal((n_paths, p.N))
    v = n / p.tau
    # mid before each trade: S_{k-1}
    perm = p.tau * p.gamma * v  # permanent drop per interval
    moves = p.sigma * np.sqrt(p.tau) * xi - perm[None, :]
    mids_before = s0 + np.concatenate([np.zeros((n_paths, 1)), np.cumsum(moves, 1)[:, :-1]], 1)
    exec_px = mids_before - (p.epsilon * np.sign(n) + p.eta * v)[None, :]
    return p.X * s0 - exec_px @ n
