"""Implementation shortfall (IS) and its decomposition.

References: Perold, A. F. (1988). "The implementation shortfall: paper
versus reality." *Journal of Portfolio Management* 14(3). Kissell, R.,
Glantz, M. (2003), *Optimal Trading Strategies*, ch. 2 (delay / trading /
opportunity split).

For side ``s`` (+1 buy, -1 sell), target ``X``, fills ``n_j`` at ``p_j``,
unexecuted ``U = X - sum n_j``, decision price ``d``, arrival price ``a`` and
end-of-horizon price ``p_f``:

    IS = s [ sum n_j (p_j - d) + U (p_f - d) ] + fees
       = s X (a - d)                         delay
       + s sum n_j (p_j - a)                 execution
       + s U (p_f - a)                       opportunity
       + fees

With simulator internals (mid before each trade ``m_j = a + perm_j +
noise_j``) the execution term splits exactly into

    spread       sum n_j * half_spread
    temporary    sum n_j * temp_impact_j
    permanent    s sum n_j * perm_j     (own earlier trades moving the mid)
    timing       s sum n_j * noise_j    (unaffected price drift / volatility)

All components are reported in basis points of arrival notional ``X a``;
positive = cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from execution_cost.simulator import ExecutionResult

COMPONENTS = ("delay", "spread", "temporary", "permanent", "timing", "opportunity", "fees")


def perold_shortfall(
    fill_qty: np.ndarray,
    fill_px: np.ndarray,
    *,
    target: float,
    side: int,
    decision_price: float,
    arrival_price: float,
    final_price: float,
    fees: float = 0.0,
) -> dict:
    """Model-free Perold split from raw fills (price units, not bps)."""
    q = np.asarray(fill_qty, dtype=float)
    p = np.asarray(fill_px, dtype=float)
    unexec = target - q.sum()
    delay = side * target * (arrival_price - decision_price)
    execution = side * float(q @ (p - arrival_price))
    opportunity = side * unexec * (final_price - arrival_price)
    total = side * (float(q @ (p - decision_price)) + unexec * (final_price - decision_price)) + fees
    return {
        "total": total,
        "delay": delay,
        "execution": execution,
        "opportunity": opportunity,
        "fees": fees,
        "unexecuted": unexec,
    }


@dataclass
class ShortfallBreakdown:
    """Per-path IS components in bps of arrival notional."""

    total: np.ndarray
    delay: np.ndarray
    spread: np.ndarray
    temporary: np.ndarray
    permanent: np.ndarray
    timing: np.ndarray
    opportunity: np.ndarray
    fees: np.ndarray
    completion: np.ndarray

    def components_sum(self) -> np.ndarray:
        return sum(getattr(self, c) for c in COMPONENTS)

    def summary(self) -> dict:
        n = len(self.total)
        out = {
            "mean_total_bps": float(self.total.mean()),
            "std_total_bps": float(self.total.std(ddof=1)) if n > 1 else 0.0,
            "se_total_bps": float(self.total.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "mean_completion": float(self.completion.mean()),
        }
        for c in COMPONENTS:
            out[f"mean_{c}_bps"] = float(getattr(self, c).mean())
        return out


def decompose_shortfall(
    res: ExecutionResult,
    decision_price: float | None = None,
    fee_bps: float = 0.0,
) -> ShortfallBreakdown:
    """Exact IS decomposition for every simulated path."""
    s = res.side
    a = res.arrival
    d = a if decision_price is None else float(decision_price)
    n = res.trades
    notional = res.target * a
    to_bps = 1e4 / notional
    unexec = res.unexecuted

    delay = np.broadcast_to(s * np.asarray(res.target) * (a - d), (n.shape[0],)).astype(float)
    spread = n.sum(1) * res.half_spread
    temporary = (n * res.temp_impact).sum(1)
    permanent = s * (n * res.perm_before).sum(1)
    timing = s * (n * res.noise_before).sum(1)
    opportunity = s * unexec * (res.final_mid - a)
    fees = fee_bps / 1e4 * (n * res.fill_prices).sum(1)
    total = s * ((n * (res.fill_prices - d)).sum(1) + unexec * (res.final_mid - d)) + fees

    return ShortfallBreakdown(
        total=total * to_bps,
        delay=delay * to_bps,
        spread=spread * to_bps,
        temporary=temporary * to_bps,
        permanent=permanent * to_bps,
        timing=timing * to_bps,
        opportunity=opportunity * to_bps,
        fees=fees * to_bps,
        completion=res.completion,
    )
