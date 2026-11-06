"""Trailing-performance gate for an alpha that may die (slice 6).

Slice 5 tested one regime break (a persistence drop) and listed an IC decay to
zero, the break that matters most for live alphas, as untested. When the IC goes
to zero the walk-forward selector keeps trading: every candidate's training
Sharpe is still dominated by the years when the alpha worked, so it pays costs on
noise until the expanding window forgets, which takes years.

The gate is an overlay on any walk-forward record. Its input is the **shadow**
record: the net returns the ungated strategy would have earned (before the first
fold, the in-sample net of the first fold's chosen rule, which is known at the
fold start). At bar ``t`` the gate is on when the mean shadow net return over
``[t - lookback, t)`` is at least ``threshold`` (per bar). When it is off the
target is zero, and the same rule that was chosen walk-forward trades toward it,
so exits and re-entries pay the full cost model. The gate reads only bars before
``t``, and the shadow never depends on the gate, so there is no feedback loop and
no look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from execution_cost.alpha import AlphaPath
from execution_cost.costs import CostParams
from execution_cost.rebalance import Rule, net_returns, trades_from_positions
from execution_cost.walkforward import RuleRun, WalkForwardResult


@dataclass(frozen=True)
class GateParams:
    lookback: int = 252
    threshold: float = 0.0  # per-bar mean net return

    def label(self) -> str:
        return f"G{self.lookback}"


@dataclass
class GatedResult:
    on: np.ndarray  # gate state per OOS bar (1 = trading)
    weights: np.ndarray
    gross: np.ndarray
    cost: np.ndarray
    net: np.ndarray

    @property
    def turnover(self) -> float:
        return float(np.mean(np.abs(trades_from_positions(self.weights))))

    @property
    def share_on(self) -> float:
        return float(self.on.mean())


def shadow_record(wf: WalkForwardResult, runs: list[RuleRun]) -> np.ndarray:
    """Ungated net returns over the whole path (in-sample head + OOS record)."""
    s0 = wf.fold_starts[0]
    first = next(r for r in runs if r.rule.label == wf.choices[0])
    return np.concatenate([first.net[:s0], wf.net])


def gate_states(shadow: np.ndarray, start: int, gate: GateParams) -> np.ndarray:
    """Gate state for bars ``[start, len(shadow))`` using shadow bars before each."""
    if gate.lookback < 2:
        raise ValueError("lookback must be >= 2")
    if start < gate.lookback:
        raise ValueError("need lookback bars of shadow history before start")
    c = np.concatenate([[0.0], np.cumsum(shadow)])
    t = np.arange(start, shadow.size)
    trailing = (c[t] - c[t - gate.lookback]) / gate.lookback
    return (trailing >= gate.threshold).astype(float)


def gated_walk_forward(
    path: AlphaPath,
    wf: WalkForwardResult,
    runs: list[RuleRun],
    params: CostParams,
    gate: GateParams,
) -> GatedResult:
    """Replay ``wf``'s fold choices on a gated target from the held position."""
    by_label: dict[str, Rule] = {r.rule.label: r.rule for r in runs}
    shadow = shadow_record(wf, runs)
    s0 = wf.fold_starts[0]
    on = gate_states(shadow, s0, gate)
    n = path.returns.size
    w_held = 0.0
    parts = {"w": [], "g": [], "c": [], "n": []}
    for k, s in enumerate(wf.fold_starts):
        e = wf.fold_starts[k + 1] if k + 1 < len(wf.fold_starts) else n
        rule = by_label[wf.choices[k]]
        tgt = path.target[s:e] * on[s - s0 : e - s0]
        w = rule.positions(tgt, w0=w_held)
        g, c, nn = net_returns(path.returns[s:e], w, params, daily_vol=path.daily_vol[s:e], w0=w_held)
        for key, v in zip("wgcn", (w, g, c, nn)):
            parts[key].append(v)
        w_held = float(w[-1])
    return GatedResult(on=on, weights=np.concatenate(parts["w"]), gross=np.concatenate(parts["g"]),
                       cost=np.concatenate(parts["c"]), net=np.concatenate(parts["n"]))
