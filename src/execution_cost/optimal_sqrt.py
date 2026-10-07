"""Profile-aware optimal execution under the square-root impact law.

The Almgren-Chriss schedule in :mod:`execution_cost.schedules` is optimal for
*linear* temporary impact on a *flat* volume curve. Mapping the sqrt law onto
it at the TWAP rate (``ac_params_from_market``) ignores both the concavity of
impact and the U-shaped intraday volume. That is why AC(kT=1) cost no less
than TWAP in slice 2 while VWAP was cheaper. This module solves the
mean-variance problem directly in the simulator's own cost model.

Buying ``X`` shares in ``N`` intervals with trades ``n_j >= 0``,
``sum n_j = X``, expected interval volume ``v_j = ADV u_j`` (pre-trade
profile), price-unit vol ``sigma_p = sigma_daily s0``:

* temporary  ``T(n) = Y sigma_p sum_j n_j (n_j / v_j)^delta``;
* permanent  ``P(n) = (G sigma_p / ADV) sum_j n_j C_{j-1}
  = (G sigma_p / 2 ADV) (X^2 - sum_j n_j^2)``, where ``C_{j-1}`` is the
  number of shares bought before interval ``j``;
* timing variance ``V(n) = (sigma_p^2 / N) sum_j r_j^2``, where
  ``r_j = X - C_j`` is the number of shares still to buy after interval ``j``
  (the mid increment after interval ``j`` moves the price of those);
* spread ``half_spread * X``, a constant.

The schedule minimises ``E = T + P`` plus ``lam V`` (``lam`` in 1 / price,
the same units as :class:`execution_cost.almgren_chriss.ACParams`, so the
same ``lam`` means the same risk aversion in both models). With ``lam = 0``
and ``G = 0`` the first-order condition ``(1 + delta) Y sigma_p
(n_j / v_j)^delta = mu`` gives ``n_j ∝ v_j``, which is VWAP exactly. Risk
aversion front-loads trading, but it is cheapest to do so in the heavy-volume
opening bins, which AC does not know about.

Solved by SLSQP on the simplex with the analytic gradient, starting from
VWAP. The problem is convex for the defaults (the concave ``-sum n^2`` term
of ``P`` is dominated by the ``n^{1.5}`` term at these sizes); a KKT residual
is returned so callers and tests can check optimality.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from execution_cost.simulator import MarketParams, volume_profile


@dataclass(frozen=True)
class SqrtExecutionProblem:
    X: float
    profile: np.ndarray  # (N,) expected volume fractions, sums to 1
    adv: float
    sigma_price: float
    temp_coeff: float
    perm_coeff: float
    lam: float = 0.0
    temp_exponent: float = 0.5

    @property
    def n_intervals(self) -> int:
        return int(len(self.profile))

    @property
    def expected_volume(self) -> np.ndarray:
        return self.adv * np.asarray(self.profile, dtype=float)

    @classmethod
    def from_market(
        cls,
        market: MarketParams,
        target: float,
        lam: float = 0.0,
        temp_coeff: float | None = None,
        perm_coeff: float | None = None,
        temp_exponent: float | None = None,
    ) -> "SqrtExecutionProblem":
        """Problem for ``market`` (optionally with calibrated coefficients)."""
        return cls(
            X=float(target),
            profile=volume_profile(market.n_intervals, market.u_shape),
            adv=market.adv,
            sigma_price=market.sigma_price,
            temp_coeff=market.temp_coeff if temp_coeff is None else float(temp_coeff),
            perm_coeff=market.perm_coeff if perm_coeff is None else float(perm_coeff),
            lam=float(lam),
            temp_exponent=market.temp_exponent if temp_exponent is None else float(temp_exponent),
        )

    # --- model -----------------------------------------------------------------

    def temporary_cost(self, n: np.ndarray) -> float:
        n = np.maximum(np.asarray(n, dtype=float), 0.0)
        d = self.temp_exponent
        return float(self.temp_coeff * self.sigma_price * np.sum(n ** (1.0 + d) / self.expected_volume**d))

    def permanent_cost(self, n: np.ndarray) -> float:
        n = np.asarray(n, dtype=float)
        before = np.concatenate([[0.0], np.cumsum(n)[:-1]])
        return float(self.perm_coeff * self.sigma_price / self.adv * (n @ before))

    def remaining(self, n: np.ndarray) -> np.ndarray:
        return self.X - np.cumsum(np.asarray(n, dtype=float))

    def variance(self, n: np.ndarray) -> float:
        r = self.remaining(n)
        return float(self.sigma_price**2 / self.n_intervals * np.sum(r**2))

    def expected_cost(self, n: np.ndarray) -> float:
        return self.temporary_cost(n) + self.permanent_cost(n)

    def objective(self, n: np.ndarray) -> float:
        return self.expected_cost(n) + self.lam * self.variance(n)

    def gradient(self, n: np.ndarray) -> np.ndarray:
        n = np.maximum(np.asarray(n, dtype=float), 0.0)
        d = self.temp_exponent
        g_temp = self.temp_coeff * self.sigma_price * (1.0 + d) * n**d / self.expected_volume**d
        # d/dn_j of sum_k n_k C_{k-1} = C_{j-1} + sum_{k>j} n_k = X - n_j
        g_perm = self.perm_coeff * self.sigma_price / self.adv * (self.X - n)
        r = self.remaining(n)
        tail = np.cumsum(r[::-1])[::-1]  # sum_{k >= j} r_k
        g_var = -2.0 * self.sigma_price**2 / self.n_intervals * tail
        return g_temp + g_perm + self.lam * g_var


@dataclass
class SqrtSolution:
    trades: np.ndarray
    objective: float
    expected_cost: float
    std_cost: float
    kkt_residual: float
    success: bool
    n_iter: int


def kkt_residual(problem: SqrtExecutionProblem, n: np.ndarray, tol_share: float = 1e-6) -> float:
    """Spread of the gradient over active bins, relative to its mean.

    At an optimum of a smooth objective on the simplex, every bin that trades
    has the same marginal cost and idle bins have a marginal cost no lower.
    """
    g = problem.gradient(n)
    active = n > tol_share * problem.X
    if not active.any():
        return float("inf")
    ga = g[active]
    scale = max(abs(float(ga.mean())), 1e-300)
    res = float(ga.max() - ga.min()) / scale
    if (~active).any():
        res = max(res, max(0.0, float(ga.mean() - g[~active].min())) / scale)
    return res


def solve_sqrt_schedule(problem: SqrtExecutionProblem, *, tol: float = 1e-12) -> SqrtSolution:
    """Mean-variance optimal static schedule under the sqrt law (SLSQP)."""
    X, N = problem.X, problem.n_intervals
    if X <= 0:
        raise ValueError("target must be positive")
    scale = problem.temp_coeff * problem.sigma_price * X  # objective scale ~ cost of X at unit participation

    def f(w):
        return problem.objective(X * w) / scale

    def g(w):
        return problem.gradient(X * w) * X / scale

    w0 = np.asarray(problem.profile, dtype=float) / np.sum(problem.profile)
    res = minimize(
        f,
        w0,
        jac=g,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * N,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones_like(w)}],
        options={"ftol": tol, "maxiter": 1000},
    )
    w = np.clip(res.x, 0.0, None)
    w = w / w.sum()
    n = X * w
    return SqrtSolution(
        trades=n,
        objective=problem.objective(n),
        expected_cost=problem.expected_cost(n),
        std_cost=float(np.sqrt(problem.variance(n))),
        kkt_residual=kkt_residual(problem, n),
        success=bool(res.success),
        n_iter=int(res.nit),
    )


def sqrt_schedule(
    market: MarketParams,
    target: float,
    lam: float,
    temp_coeff: float | None = None,
    perm_coeff: float | None = None,
    temp_exponent: float | None = None,
) -> np.ndarray:
    """Trade list (sums to ``target``) for ``market`` at risk aversion ``lam``.

    ``temp_exponent`` is the exponent the *planner* assumes; it defaults to
    ``market.temp_exponent``. Pass it explicitly when the planner should not
    see the simulator's true exponent (see :mod:`execution_cost.misspecification`).
    """
    prob = SqrtExecutionProblem.from_market(market, target, lam, temp_coeff, perm_coeff, temp_exponent)
    return solve_sqrt_schedule(prob).trades
