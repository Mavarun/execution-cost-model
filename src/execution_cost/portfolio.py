"""Apply execution costs to a synthetic predictive signal and compare Sharpes."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from execution_cost.costs import CostParams, total_unit_cost
from execution_cost.metrics import annualized_sharpe, mean_turnover


@dataclass(frozen=True)
class PortfolioResult:
    """Gross vs net performance for one synthetic path."""

    gross_sharpe: float
    net_sharpe: float
    mean_turnover: float
    mean_daily_cost: float
    n_obs: int
    seed: int
    hold_bars: int

    def to_dict(self) -> dict:
        return asdict(self)


def _ewma(x: np.ndarray, span: int) -> np.ndarray:
    """Causal EWMA used as a smooth signal transform."""
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = alpha * x[t] + (1.0 - alpha) * out[t - 1]
    return out


def synthetic_signal_backtest(
    *,
    n: int = 2000,
    seed: int = 42,
    hold_bars: int = 5,
    edge: float = 0.35,
    asset_vol: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build synthetic returns, lagged predictive signal, and target weights.

    Latent factor ``f_t`` drives ``r_{t+1}``. A noisy signal of ``f_t`` is
    EWMA-smoothed; ``hold_bars`` controls turnover (larger => slower trading).

    Timing: signal observed at ``t`` sets weight that earns ``r_{t+1}`` (no peek).

    Returns
    -------
    asset_returns, weights, daily_vol_proxy
    """
    rng = np.random.default_rng(seed)
    factor = rng.normal(0.0, 1.0, size=n)
    innov = rng.normal(0.0, 1.0, size=n)
    # factor[t] predicts return[t+1]
    asset_returns = np.zeros(n, dtype=float)
    load = float(edge)
    resid = float(np.sqrt(max(1.0 - load**2, 1e-8)))
    asset_returns[1:] = asset_vol * (load * factor[:-1] + resid * innov[1:])
    signal = factor + rng.normal(0.0, 0.50, size=n)
    raw = np.tanh(signal)
    smooth = _ewma(raw, span=max(int(hold_bars), 1))
    target = np.clip(smooth, -1.0, 1.0)
    # Position for period t+1 decided from signal through t.
    weights = np.zeros(n, dtype=float)
    weights[1:] = target[:-1]
    daily_vol = np.full(n, asset_vol, dtype=float)
    return asset_returns, weights, daily_vol


def apply_costs_to_returns(
    asset_returns: np.ndarray,
    weights: np.ndarray,
    params: CostParams,
    *,
    daily_vol: np.ndarray | float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return gross returns, per-period costs, and net returns.

    Cost at t is charged on |dw_t| using participation-aware unit costs.
    """
    r = np.asarray(asset_returns, dtype=float)
    w = np.asarray(weights, dtype=float)
    if r.shape != w.shape:
        raise ValueError("asset_returns and weights must share shape")
    trades = np.zeros_like(w)
    trades[0] = w[0]
    trades[1:] = np.diff(w)
    unit = total_unit_cost(np.abs(trades), params, daily_vol=daily_vol)
    costs = np.abs(trades) * unit
    gross = w * r
    net = gross - costs
    return gross, costs, net


def run_cost_slice(
    *,
    n: int = 2000,
    seed: int = 42,
    hold_bars: int = 5,
    edge: float = 0.35,
    asset_vol: float = 0.01,
    params: CostParams | None = None,
) -> PortfolioResult:
    """End-to-end synthetic backtest with execution costs (not live PnL)."""
    params = params or CostParams()
    asset_returns, weights, daily_vol = synthetic_signal_backtest(
        n=n, seed=seed, hold_bars=hold_bars, edge=edge, asset_vol=asset_vol
    )
    gross, costs, net = apply_costs_to_returns(
        asset_returns, weights, params, daily_vol=daily_vol
    )
    return PortfolioResult(
        gross_sharpe=annualized_sharpe(gross),
        net_sharpe=annualized_sharpe(net),
        mean_turnover=mean_turnover(weights),
        mean_daily_cost=float(np.mean(costs)),
        n_obs=int(len(gross)),
        seed=seed,
        hold_bars=hold_bars,
    )
