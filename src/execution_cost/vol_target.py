"""Walk-forward volatility targeting with leverage chosen on training bars only.

Slice 5's equal-risk check levered each selector *on the evaluation days* (a
diagnostic, not something a trader could do). Here the leverage of fold ``k`` is
solved on that fold's training window: the chosen rule's in-sample positions are
scaled by ``k`` and the full cost model is re-charged on the scaled trades until the
training net-return std equals the target. OOS positions are the walk-forward's held
positions times the fold's ``k``; costs on the whole levered OOS path are recomputed,
so leverage changes at fold boundaries pay for their trades.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from execution_cost.alpha import AlphaPath
from execution_cost.costs import CostParams
from execution_cost.rebalance import net_returns
from execution_cost.walkforward import RuleRun, WalkForwardResult

TRADING_DAYS = 252


@dataclass
class VolTargetResult:
    leverage: list[float]
    weights: np.ndarray
    gross: np.ndarray
    cost: np.ndarray
    net: np.ndarray


def _train_std(path: AlphaPath, w: np.ndarray, lo: int, hi: int, params: CostParams, k: float) -> float:
    _, _, net = net_returns(path.returns[lo:hi], k * w[lo:hi], params, daily_vol=path.daily_vol[lo:hi],
                            w0=float(k * w[lo - 1]) if lo > 0 else 0.0)
    return float(np.std(net, ddof=1))


def solve_leverage(path: AlphaPath, run: RuleRun, lo: int, hi: int, params: CostParams, target_daily: float,
                   *, k_lo: float = 0.05, k_hi: float = 20.0, tol: float = 1e-6) -> float:
    """``k`` with training net std on ``[lo, hi)`` equal to ``target_daily`` (clipped to the bracket)."""
    f = lambda k: _train_std(path, run.weights, lo, hi, params, k) - target_daily  # noqa: E731
    if f(k_lo) >= 0:
        return k_lo
    if f(k_hi) <= 0:
        return k_hi
    a, b = k_lo, k_hi
    while b - a > tol * max(1.0, a):
        m = 0.5 * (a + b)
        a, b = (a, m) if f(m) > 0 else (m, b)
    return 0.5 * (a + b)


def vol_targeted(path: AlphaPath, wf: WalkForwardResult, runs: list[RuleRun], params: CostParams,
                 target_annual_vol: float = 0.10) -> VolTargetResult:
    by_label = {r.rule.label: r for r in runs}
    target_daily = target_annual_vol / np.sqrt(TRADING_DAYS)
    s0, n = wf.fold_starts[0], path.returns.size
    ks: list[float] = []
    lev = np.empty(n - s0)
    for i, s in enumerate(wf.fold_starts):
        e = wf.fold_starts[i + 1] if i + 1 < len(wf.fold_starts) else n
        k = solve_leverage(path, by_label[wf.choices[i]], 1, s, params, target_daily)
        ks.append(k)
        lev[s - s0 : e - s0] = k
    w = lev * wf.weights
    g, c, nn = net_returns(path.returns[s0:], w, params, daily_vol=path.daily_vol[s0:])
    return VolTargetResult(leverage=ks, weights=w, gross=g, cost=c, net=nn)
