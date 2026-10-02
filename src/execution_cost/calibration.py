"""Square-root impact law calibrated on synthetic metaorders.

Reference for the law: Tóth, B. et al. (2011). "Anomalous price impact and
the critical nature of liquidity in financial markets." *Physical Review X*
1, 021006; Almgren, R. et al. (2005). "Direct estimation of equity market
impact." *Risk* 18(7).

Design
------
Each metaorder is one simulated day (:mod:`execution_cost.simulator`): a TWAP
buy of ``q = Q / ADV`` (log-uniform) in a stock with daily vol ``sigma``
(uniform). What an analyst would observe is the arrival-to-average-fill
slippage net of the known half-spread, normalised by daily vol:

    I = (VWAP_fills - arrival) / arrival - half_spread / arrival,   y = I / sigma

and fits ``y = Y q^delta``. Price noise during execution (std ~ 0.58 in
sigma units for a full-day TWAP) dwarfs impact for small orders, so:

* **NLS** on all orders (``scipy.optimize.curve_fit``), and
* **binned log-log WLS** (``statsmodels``): mean ``y`` in quantile bins of
  ``log q``, regress ``log mean`` on ``log q`` with inverse-variance weights.

Permanent impact is estimated from the post-trade mid move
``(final_mid - arrival) / (sigma s0)`` regressed on ``q`` (OLS through the
origin, HC1 errors) and is expected to be imprecise - a full day of price
noise sits on top of it.

Out-of-sample: orders are split 70/30 (seeded). Power-law, fixed-sqrt and
linear (``delta = 1``) fits from the train split are scored on the test
split by order-level MSE and by RMSE of binned means.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import statsmodels.api as sm
from scipy.optimize import curve_fit

from execution_cost.simulator import MarketParams, execute, simulate_market


@dataclass
class Metaorders:
    q: np.ndarray  # Q / ADV
    sigma: np.ndarray  # daily vol
    y: np.ndarray  # normalised impact I / sigma
    perm_move: np.ndarray  # (final_mid - arrival) / (sigma s0)


def generate_metaorders(
    n_orders: int = 20_000,
    q_range: tuple[float, float] = (0.005, 0.2),
    sigma_levels: tuple[float, ...] = (0.01, 0.02, 0.03, 0.04),
    market: MarketParams | None = None,
    seed: int = 0,
) -> Metaorders:
    """Simulate TWAP buy metaorders with log-uniform sizes."""
    base = market or MarketParams()
    rng = np.random.default_rng(seed)
    per = n_orders // len(sigma_levels)
    qs, sig, ys, perms = [], [], [], []
    for k, s in enumerate(sigma_levels):
        mp = replace(base, sigma_daily=float(s))
        q = np.exp(rng.uniform(np.log(q_range[0]), np.log(q_range[1]), size=per))
        Q = q * mp.adv
        mk = simulate_market(mp, per, seed=seed * 1000 + k + 1)
        sched = Q[:, None] / mp.n_intervals * np.ones((1, mp.n_intervals))
        res = execute(sched, mk, mp, Q, side=1)
        vwap = (res.trades * res.fill_prices).sum(1) / res.trades.sum(1)
        impact = (vwap - res.arrival - mp.half_spread) / res.arrival
        qs.append(q)
        sig.append(np.full(per, s))
        ys.append(impact / s)
        perms.append((res.final_mid - res.arrival) / (s * mp.s0))
    return Metaorders(np.concatenate(qs), np.concatenate(sig), np.concatenate(ys), np.concatenate(perms))


def twap_effective_sqrt_coefficient(market: MarketParams | None = None) -> float:
    """Expected metaorder-level sqrt coefficient of a full-day TWAP (temporary part).

    Interval participation is ``(q / N) / (u_j D Z_j)``, so the share-weighted
    mean temporary impact in sigma units is
    ``Y * mean_j (N u_j)^-1/2 * E[D^-1/2] * E[Z^-1/2] * sqrt(q)``; for a
    mean-one log-normal with log-sd ``s``, ``E[Z^-1/2] = exp(3 s^2 / 8)``.
    Permanent impact adds a further ``~perm_coeff * q / 2`` (linear) term.
    """
    from execution_cost.simulator import volume_profile

    mp = market or MarketParams()
    u = volume_profile(mp.n_intervals, mp.u_shape)
    prof = float(np.mean((mp.n_intervals * u) ** -0.5))
    jensen = np.exp(3 * mp.volume_noise**2 / 8) * np.exp(3 * mp.day_volume_noise**2 / 8)
    return float(mp.temp_coeff * prof * jensen)


def _power(q, Y, delta):
    return Y * q**delta


def fit_power_law_nls(q: np.ndarray, y: np.ndarray) -> dict:
    popt, pcov = curve_fit(_power, q, y, p0=(0.5, 0.5), maxfev=20_000)
    se = np.sqrt(np.diag(pcov))
    return {"Y": float(popt[0]), "delta": float(popt[1]), "Y_se": float(se[0]), "delta_se": float(se[1])}


def fit_fixed_exponent(q: np.ndarray, y: np.ndarray, delta: float) -> float:
    """Least-squares ``Y`` for a fixed exponent (closed form)."""
    x = q**delta
    return float((x @ y) / (x @ x))


def binned_means(q: np.ndarray, y: np.ndarray, n_bins: int = 10):
    edges = np.quantile(np.log(q), np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, np.log(q), side="right") - 1, 0, n_bins - 1)
    qm, ym, se = [], [], []
    for b in range(n_bins):
        m = idx == b
        qm.append(np.exp(np.log(q[m]).mean()))
        ym.append(y[m].mean())
        se.append(y[m].std(ddof=1) / np.sqrt(m.sum()))
    return np.array(qm), np.array(ym), np.array(se)


def fit_binned_loglog(q: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict:
    qm, ym, se = binned_means(q, y, n_bins)
    ok = ym > 0
    X = sm.add_constant(np.log(qm[ok]))
    # delta-method variance of log(mean): (se / mean)^2
    w = 1.0 / (se[ok] / ym[ok]) ** 2
    fit = sm.WLS(np.log(ym[ok]), X, weights=w).fit()
    lo, hi = fit.conf_int()[1]
    return {
        "delta": float(fit.params[1]),
        "delta_ci": (float(lo), float(hi)),
        "Y": float(np.exp(fit.params[0])),
        "n_bins_used": int(ok.sum()),
    }


def fit_permanent(q: np.ndarray, perm_move: np.ndarray) -> dict:
    fit = sm.OLS(perm_move, q[:, None]).fit(cov_type="HC1")
    lo, hi = fit.conf_int()[0]
    return {"perm_coeff": float(fit.params[0]), "perm_ci": (float(lo), float(hi))}


@dataclass
class CalibrationResult:
    n_orders: int
    nls_Y: float
    nls_delta: float
    nls_delta_se: float
    binned_delta: float
    binned_delta_ci: tuple
    sqrt_Y: float  # Y with delta fixed at 0.5 (used for scheduling)
    perm_coeff: float
    perm_ci: tuple
    oos_mse_power: float
    oos_mse_sqrt: float
    oos_mse_linear: float
    oos_binned_rmse_power: float
    oos_binned_rmse_sqrt: float
    oos_binned_rmse_linear: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["binned_delta_ci"] = list(self.binned_delta_ci)
        d["perm_ci"] = list(self.perm_ci)
        return d


def calibrate_impact(
    orders: Metaorders | None = None,
    train_frac: float = 0.7,
    seed: int = 0,
    n_bins: int = 10,
) -> CalibrationResult:
    """Fit the impact law on a train split and score it out of sample."""
    m = orders or generate_metaorders(seed=seed)
    rng = np.random.default_rng(seed + 7)
    perm = rng.permutation(len(m.q))
    cut = int(train_frac * len(perm))
    tr, te = perm[:cut], perm[cut:]

    nls = fit_power_law_nls(m.q[tr], m.y[tr])
    binned = fit_binned_loglog(m.q[tr], m.y[tr], n_bins)
    y_sqrt = fit_fixed_exponent(m.q[tr], m.y[tr], 0.5)
    y_lin = fit_fixed_exponent(m.q[tr], m.y[tr], 1.0)
    permfit = fit_permanent(m.q[tr], m.perm_move[tr])

    preds = {
        "power": _power(m.q[te], nls["Y"], nls["delta"]),
        "sqrt": _power(m.q[te], y_sqrt, 0.5),
        "linear": _power(m.q[te], y_lin, 1.0),
    }
    mse = {k: float(np.mean((m.y[te] - v) ** 2)) for k, v in preds.items()}
    qb, yb, _ = binned_means(m.q[te], m.y[te], n_bins)
    brmse = {
        "power": float(np.sqrt(np.mean((yb - _power(qb, nls["Y"], nls["delta"])) ** 2))),
        "sqrt": float(np.sqrt(np.mean((yb - _power(qb, y_sqrt, 0.5)) ** 2))),
        "linear": float(np.sqrt(np.mean((yb - _power(qb, y_lin, 1.0)) ** 2))),
    }
    return CalibrationResult(
        n_orders=len(m.q),
        nls_Y=nls["Y"],
        nls_delta=nls["delta"],
        nls_delta_se=nls["delta_se"],
        binned_delta=binned["delta"],
        binned_delta_ci=binned["delta_ci"],
        sqrt_Y=y_sqrt,
        perm_coeff=permfit["perm_coeff"],
        perm_ci=permfit["perm_ci"],
        oos_mse_power=mse["power"],
        oos_mse_sqrt=mse["sqrt"],
        oos_mse_linear=mse["linear"],
        oos_binned_rmse_power=brmse["power"],
        oos_binned_rmse_sqrt=brmse["sqrt"],
        oos_binned_rmse_linear=brmse["linear"],
    )
