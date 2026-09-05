"""Performance metrics for the execution-cost research slice."""

from __future__ import annotations

import numpy as np


def annualized_sharpe(
    returns: np.ndarray,
    *,
    periods_per_year: float = 252.0,
    risk_free: float = 0.0,
) -> float:
    """Annualized Sharpe of a return series (sample std, population ddof=1)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    excess = r - risk_free / periods_per_year
    vol = excess.std(ddof=1)
    if vol <= 0.0:
        return float("nan")
    return float(np.sqrt(periods_per_year) * excess.mean() / vol)


def mean_turnover(weights: np.ndarray) -> float:
    """Average one-way turnover: mean |dw_t| over the path."""
    w = np.asarray(weights, dtype=float)
    if w.size < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(w))))
