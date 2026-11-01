"""Expanding-window walk-forward selection of a rebalancing rule.

At each fold start ``s_k`` (``min_train, min_train + test_len, ...``) every
candidate rule is scored on bars ``[0, s_k)`` only, by **net** Sharpe after the
full cost model; the winner trades bars ``[s_k, s_k + test_len)`` starting
from the position actually held at ``s_k``. The out-of-sample (OOS) record is
the concatenation of those test segments, so every reported OOS number comes
from a rule chosen without seeing the bars it is scored on.

Rules are run once over the whole path to score training windows. Because
every rule is causal and its only state is the held position, the slice
``[0, s_k)`` of that run is exactly what a trader would have had at ``s_k``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from execution_cost.alpha import AlphaPath
from execution_cost.costs import CostParams
from execution_cost.rebalance import Rule, net_returns, trades_from_positions
from execution_cost.sharpe_stats import per_bar_sharpe


@dataclass(frozen=True)
class RuleRun:
    """A rule run over the whole path from flat."""

    rule: Rule
    weights: np.ndarray
    gross: np.ndarray
    cost: np.ndarray
    net: np.ndarray


@dataclass
class WalkForwardResult:
    """OOS record of a walk-forward selection."""

    fold_starts: list[int]
    choices: list[str]
    oos_index: np.ndarray
    weights: np.ndarray
    gross: np.ndarray
    cost: np.ndarray
    net: np.ndarray
    train_scores: list[dict[str, float]] = field(default_factory=list)

    @property
    def turnover(self) -> float:
        """Mean one-way |dw| over the OOS bars (the first fold enters from flat)."""
        return float(np.mean(np.abs(trades_from_positions(self.weights, w0=0.0))))


def fold_starts(n: int, *, min_train: int, test_len: int) -> list[int]:
    if min_train < 2 or test_len < 1:
        raise ValueError("min_train must be >= 2 and test_len >= 1")
    if min_train >= n:
        raise ValueError("min_train must be smaller than the path length")
    return list(range(int(min_train), int(n), int(test_len)))


def run_rules(path: AlphaPath, rules: list[Rule], params: CostParams) -> list[RuleRun]:
    runs = []
    for rule in rules:
        w = rule.positions(path.target)
        g, c, n = net_returns(path.returns, w, params, daily_vol=path.daily_vol)
        runs.append(RuleRun(rule=rule, weights=w, gross=g, cost=c, net=n))
    return runs


def select_on_window(runs: list[RuleRun], end: int, *, start: int = 1) -> tuple[int, dict[str, float]]:
    """Index of the best net Sharpe on bars ``[start, end)``; ties keep grid order."""
    scores = {r.rule.label: per_bar_sharpe(r.net[start:end]) for r in runs}
    vals = np.array([scores[r.rule.label] for r in runs], dtype=float)
    vals = np.where(np.isfinite(vals), vals, -np.inf)
    return int(np.argmax(vals)), scores


def walk_forward(
    path: AlphaPath,
    rules: list[Rule],
    params: CostParams,
    *,
    min_train: int,
    test_len: int,
    runs: list[RuleRun] | None = None,
    window: int | None = None,
) -> WalkForwardResult:
    """Walk-forward selection over ``rules`` (see module docstring).

    ``window=None`` scores on the expanding window ``[1, s_k)``; an integer
    scores on the rolling window ``[max(1, s_k - window), s_k)``.
    """
    if window is not None and window < 2:
        raise ValueError("window must be >= 2")
    if not rules:
        raise ValueError("need at least one rule")
    n = path.returns.size
    starts = fold_starts(n, min_train=min_train, test_len=test_len)
    runs = runs if runs is not None else run_rules(path, rules, params)
    w_held = 0.0
    parts_w, parts_g, parts_c, parts_n, choices, all_scores = [], [], [], [], [], []
    for s in starts:
        e = min(s + test_len, n)
        lo = 1 if window is None else max(1, s - int(window))
        k, scores = select_on_window(runs, s, start=lo)
        rule = runs[k].rule
        w = rule.positions(path.target[s:e], w0=w_held)
        g, c, nn = net_returns(path.returns[s:e], w, params, daily_vol=path.daily_vol[s:e], w0=w_held)
        parts_w.append(w)
        parts_g.append(g)
        parts_c.append(c)
        parts_n.append(nn)
        choices.append(rule.label)
        all_scores.append(scores)
        w_held = float(w[-1])
    res = WalkForwardResult(
        fold_starts=starts,
        choices=choices,
        oos_index=np.arange(starts[0], n),
        weights=np.concatenate(parts_w),
        gross=np.concatenate(parts_g),
        cost=np.concatenate(parts_c),
        net=np.concatenate(parts_n),
        train_scores=all_scores,
    )
    return res


def fixed_rule_oos(run: RuleRun, oos_start: int) -> dict[str, np.ndarray]:
    """A rule held fixed for the whole path, read on the OOS bars only."""
    return {
        "weights": run.weights[oos_start:],
        "gross": run.gross[oos_start:],
        "cost": run.cost[oos_start:],
        "net": run.net[oos_start:],
    }
