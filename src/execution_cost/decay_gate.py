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


# --- Slice 7: a sequential (CUSUM) gate with hysteresis ------------------------------------------
# Slice 6's trailing-mean gate flips on noise: a 252-bar mean of daily net returns with a daily
# Sharpe of ~0.07 is below zero a good share of the time even when the alpha is alive (it was on
# only 85% of pre-decay bars), and it switches back on just as easily. A two-sided Page CUSUM
# needs accumulated evidence in each direction, and the two directions use separate sums, so
# switching back on costs as much evidence as switching off.


@dataclass(frozen=True)
class CusumGateParams:
    arl0: float = 2520.0  # target bars between false "off" alarms while the alpha is alive
    ref_bars: int = 1260  # shadow bars before ``start`` used for the alive mean and sd
    h: float | None = None  # fixed alarm level; None derives it from ``arl0`` (``siegmund_h``)
    shrink: float = 1.0  # alive level used = shrink x the reference mean (slice 8; 1.0 = slice 7)
    on_frac: float = 1.0  # re-entry level = on_frac x h (slice 9; 1.0 = the symmetric slice 7/8 gate)
    max_off: int | None = None  # slice 11: after this many bars off, switch back on and re-check (None = never)

    def label(self) -> str:
        base = f"C{self.h:g}" if self.h is not None else f"CARL{self.arl0:g}"
        base = base if self.shrink == 1.0 else f"{base}s{self.shrink:g}"
        base = base if self.on_frac == 1.0 else f"{base}r{self.on_frac:g}"
        return base if self.max_off is None else f"{base}c{self.max_off}"


def siegmund_arl(h: float, delta: float) -> float:
    """Siegmund's approximation to a one-sided CUSUM's average run length.

    Increments have mean ``-delta`` (drift away from the alarm) and unit variance; ``h`` is the
    alarm level. Uses the usual 1.166 boundary correction.
    """
    b = h + 1.166
    x = 2.0 * delta * b
    return float((np.exp(x) - x - 1.0) / (2.0 * delta**2))


def siegmund_h(delta: float, arl0: float) -> float:
    """Smallest alarm level whose in-control ARL (``siegmund_arl``) reaches ``arl0``."""
    if delta <= 0 or arl0 <= 1:
        raise ValueError("need delta > 0 and arl0 > 1")
    lo, hi = 0.0, 1.0
    while siegmund_arl(hi, delta) < arl0:
        hi *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if siegmund_arl(mid, delta) < arl0 else (lo, mid)
    return hi


def cusum_gate_states(shadow: np.ndarray, start: int, gate: CusumGateParams) -> np.ndarray:
    """Gate state for bars ``[start, len(shadow))`` from a two-sided CUSUM on the shadow net return.

    Reference: the shadow's mean ``mu`` and sd ``sd`` over the ``ref_bars`` bars before ``start``
    (the alive level, known at the start). The test is between "alive" (mean ``mu``) and "dead"
    (mean 0), so each bar adds ``(m - x) / sd`` to the off-sum and ``(x - m) / sd`` to the on-sum,
    with ``m = mu / 2``; while alive the off-sum drifts down by ``delta = mu / (2 sd)`` per bar.
    The alarm level is ``gate.h`` if set, else ``siegmund_h(delta, gate.arl0)``: a daily
    alpha Sharpe of ~0.07 gives delta ~0.03, and with that drift an h of 5 false-alarms within
    weeks, so h has to come from the ARL target, not a textbook constant. The gate turns off when
    the off-sum exceeds h and back on when the on-sum does; each switch resets both sums. State at
    bar ``t`` uses shadow bars before ``t``. If the reference mean is not positive the gate stays on.

    ``gate.shrink`` scales the alive level to ``shrink * mu`` before anything else (slice 8). The
    reference bars are the first fold's in-sample net, which overstates the OOS level; shrinking
    it moves both the midpoint ``m`` and the drift the alarm level is solved for, so the ARL target
    holds at the level the alpha actually earns, at the cost of a slower reaction to a decay.

    ``gate.on_frac`` sets the re-entry level to ``on_frac * h`` (slice 9). With 1.0 the gate needs
    the full alarm level to switch back on, so one false alarm keeps it off for hundreds of bars
    (slice 8's post-hoc reading); a lower level re-enters sooner, at the price of re-entering
    falsely more often after a real decay.

    ``gate.max_off`` (slice 11) caps an off-spell: after ``max_off`` bars off, the gate switches
    back on with both sums reset, so it has to re-detect a decay at the full level ``h``. Unlike a
    lower ``on_frac`` it never lowers the bar for switching on on evidence; it only limits how long
    one false alarm can keep the book flat.
    """
    if gate.ref_bars < 20 or (gate.h is not None and gate.h <= 0):
        raise ValueError("need ref_bars >= 20 and h > 0 when given")
    if not 0.0 < gate.shrink <= 1.0:
        raise ValueError("need 0 < shrink <= 1")
    if not 0.0 < gate.on_frac <= 1.0:
        raise ValueError("need 0 < on_frac <= 1")
    if gate.max_off is not None and gate.max_off < 1:
        raise ValueError("need max_off >= 1 when given")
    if start < gate.ref_bars:
        raise ValueError("need ref_bars bars of shadow history before start")
    ref = shadow[start - gate.ref_bars:start]
    mu, sd = gate.shrink * float(ref.mean()), float(ref.std(ddof=1))
    n = shadow.size - start
    if mu <= 0 or sd <= 0:
        return np.ones(n)
    m = 0.5 * mu
    h = gate.h if gate.h is not None else siegmund_h(m / sd, gate.arl0)
    on = np.empty(n)
    state, s_off, s_on, off_for = 1.0, 0.0, 0.0, 0
    for i in range(n):
        on[i] = state
        x = shadow[start + i]  # observed after the decision for bar start + i
        s_off = max(0.0, s_off + (m - x) / sd)
        s_on = max(0.0, s_on + (x - m) / sd)
        off_for = off_for + 1 if state == 0.0 else 0
        if state == 1.0 and s_off > h:
            state, s_off, s_on = 0.0, 0.0, 0.0
        elif state == 0.0 and (s_on > gate.on_frac * h or (gate.max_off is not None and off_for >= gate.max_off)):
            state, s_off, s_on = 1.0, 0.0, 0.0
    return on


def gated_walk_forward_states(
    path: AlphaPath, wf: WalkForwardResult, runs: list[RuleRun], params: CostParams, on: np.ndarray
) -> GatedResult:
    """``gated_walk_forward`` with precomputed OOS gate states (any gate, e.g. the CUSUM one)."""
    by_label: dict[str, Rule] = {r.rule.label: r.rule for r in runs}
    s0 = wf.fold_starts[0]
    n = path.returns.size
    if on.size != n - s0:
        raise ValueError("gate states must cover every OOS bar")
    w_held = 0.0
    parts = {"w": [], "g": [], "c": [], "n": []}
    for k, s in enumerate(wf.fold_starts):
        e = wf.fold_starts[k + 1] if k + 1 < len(wf.fold_starts) else n
        rule = by_label[wf.choices[k]]
        w = rule.positions(path.target[s:e] * on[s - s0:e - s0], w0=w_held)
        g, c, nn = net_returns(path.returns[s:e], w, params, daily_vol=path.daily_vol[s:e], w0=w_held)
        for key, v in zip("wgcn", (w, g, c, nn)):
            parts[key].append(v)
        w_held = float(w[-1])
    return GatedResult(on=on, weights=np.concatenate(parts["w"]), gross=np.concatenate(parts["g"]),
                       cost=np.concatenate(parts["c"]), net=np.concatenate(parts["n"]))
