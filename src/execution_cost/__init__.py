"""Execution cost model: spread, slippage, square-root impact, net vs gross Sharpe."""

from execution_cost.costs import (
    CostParams,
    half_spread_cost,
    linear_slippage_cost,
    participation_rate,
    square_root_impact_cost,
    total_unit_cost,
)
from execution_cost.metrics import annualized_sharpe, mean_turnover
from execution_cost.portfolio import (
    PortfolioResult,
    apply_costs_to_returns,
    run_cost_slice,
    synthetic_signal_backtest,
)
from execution_cost.sensitivity import (
    CurvePoint,
    curve_drag_rises_with_turnover,
    turnover_cost_curve,
)

__all__ = [
    "CostParams",
    "CurvePoint",
    "PortfolioResult",
    "annualized_sharpe",
    "apply_costs_to_returns",
    "curve_drag_rises_with_turnover",
    "half_spread_cost",
    "linear_slippage_cost",
    "mean_turnover",
    "participation_rate",
    "run_cost_slice",
    "square_root_impact_cost",
    "synthetic_signal_backtest",
    "total_unit_cost",
    "turnover_cost_curve",
]

__version__ = "0.1.0"
