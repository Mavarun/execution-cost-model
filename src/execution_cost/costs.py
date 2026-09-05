"""Participation-aware execution costs: spread, slippage, square-root impact.

Unit costs are expressed as a fraction of notional (return units). Trading
one unit of weight change incurs ``total_unit_cost(...)`` in return drag.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostParams:
    """Market microstructure parameters for the cost model.

    Attributes
    ----------
    spread_bps:
        Bid-ask spread in basis points. Half-spread is paid on each one-way trade.
    slippage_bps:
        Linear temporary slippage coefficient in bps at full ADV participation.
    impact_coeff:
        Square-root permanent/temporary impact coefficient (return units at
        participation=1 when ``daily_vol`` is scaled in).
    adv_notional:
        Average daily volume in the same notional units as absolute trade size.
        Used only to form participation = |trade| / ADV.
    """

    spread_bps: float = 10.0
    slippage_bps: float = 5.0
    impact_coeff: float = 0.25
    adv_notional: float = 1.0


def participation_rate(trade_notional: float | np.ndarray, adv_notional: float) -> np.ndarray:
    """Participation = |trade| / ADV, clipped to a sane upper bound for numerics."""
    adv = max(float(adv_notional), 1e-12)
    p = np.abs(np.asarray(trade_notional, dtype=float)) / adv
    return np.clip(p, 0.0, 5.0)


def half_spread_cost(spread_bps: float) -> float:
    """One-way half-spread cost as a fraction of notional."""
    return 0.5 * float(spread_bps) / 10_000.0


def linear_slippage_cost(participation: float | np.ndarray, slippage_bps: float) -> np.ndarray:
    """Temporary slippage linear in participation rate."""
    p = np.asarray(participation, dtype=float)
    return p * (float(slippage_bps) / 10_000.0)


def square_root_impact_cost(
    participation: float | np.ndarray,
    *,
    impact_coeff: float,
    daily_vol: float | np.ndarray = 0.01,
) -> np.ndarray:
    """Square-root market impact: ``eta * sigma * sqrt(participation)`` (Almgren-style)."""
    p = np.asarray(participation, dtype=float)
    sigma = np.asarray(daily_vol, dtype=float)
    return float(impact_coeff) * sigma * np.sqrt(np.maximum(p, 0.0))


def total_unit_cost(
    trade_notional: float | np.ndarray,
    params: CostParams,
    *,
    daily_vol: float | np.ndarray = 0.01,
) -> np.ndarray:
    """Combined one-way unit cost (fraction of notional) for a trade size."""
    part = participation_rate(trade_notional, params.adv_notional)
    spread = half_spread_cost(params.spread_bps)
    slip = linear_slippage_cost(part, params.slippage_bps)
    impact = square_root_impact_cost(
        part, impact_coeff=params.impact_coeff, daily_vol=daily_vol
    )
    return spread + slip + impact
