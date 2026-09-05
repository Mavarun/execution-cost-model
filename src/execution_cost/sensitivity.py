"""Turnover sensitivity: show how higher trading destroys net Sharpe."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from execution_cost.costs import CostParams
from execution_cost.portfolio import run_cost_slice


@dataclass(frozen=True)
class CurvePoint:
    hold_bars: int
    mean_turnover: float
    gross_sharpe: float
    net_sharpe: float
    mean_daily_cost: float
    sharpe_drag: float

    def to_dict(self) -> dict:
        return asdict(self)


def turnover_cost_curve(
    hold_bars_grid: list[int] | tuple[int, ...] | None = None,
    *,
    n: int = 2000,
    seed: int = 42,
    edge: float = 0.35,
    asset_vol: float = 0.01,
    params: CostParams | None = None,
) -> list[CurvePoint]:
    """Sweep holding horizons (inverse turnover) and record gross/net Sharpe.

    Smaller ``hold_bars`` => higher turnover => larger cost drag. The curve
    supports the hypothesis that aggressive turnover destroys edge after costs.
    """
    params = params or CostParams()
    grid = list(hold_bars_grid) if hold_bars_grid is not None else [1, 2, 3, 5, 8, 13, 21]
    points: list[CurvePoint] = []
    for hb in grid:
        res = run_cost_slice(
            n=n, seed=seed, hold_bars=int(hb), edge=edge, asset_vol=asset_vol, params=params
        )
        points.append(
            CurvePoint(
                hold_bars=int(hb),
                mean_turnover=res.mean_turnover,
                gross_sharpe=res.gross_sharpe,
                net_sharpe=res.net_sharpe,
                mean_daily_cost=res.mean_daily_cost,
                sharpe_drag=res.gross_sharpe - res.net_sharpe,
            )
        )
    return points


def curve_drag_rises_with_turnover(points: list[CurvePoint], *, atol: float = 1e-9) -> bool:
    """True if Sharpe drag is non-decreasing as mean turnover rises."""
    if len(points) < 2:
        return True
    ordered = sorted(points, key=lambda p: p.mean_turnover)
    drags = np.array([p.sharpe_drag for p in ordered], dtype=float)
    return bool(np.all(np.diff(drags) >= -atol))
