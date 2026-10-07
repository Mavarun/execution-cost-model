"""Does the sqrt-law optimum survive a wrong impact exponent?

The sqrt optimum assumes temporary impact ``∝ (n / v)^0.5``. Empirical
exponents scatter roughly between 0.4 and 0.7 by market and period, so this
module makes the simulator's true exponent ``delta`` differ from 0.5 and
asks, out of sample and net of all costs:

* **SQRT-OPT** plans with delta = 0.5 and the coefficient fitted at 0.5;
* **POWER-OPT** plans with the exponent estimated by NLS on the train
  metaorders (``delta_hat``) and the coefficient refitted at ``delta_hat``;
* **AC** is the sqrt-linearised Almgren-Chriss baseline.

Metaorders are generated in the *true* market on the calibration seed; the
evaluation days use a disjoint seed. Schedules are compared at equal realised
IS standard deviation (the sqrt/power optimum's ``lam`` is bisected to AC's
realised std), so the paired saving is a pure cost difference at equal risk.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from execution_cost.almgren_chriss import ac_trade_list
from execution_cost.calibration import calibrate_impact, generate_metaorders, temp_coeff_at_exponent
from execution_cost.comparison import lam_for_urgency
from execution_cost.frontier import _evaluate, match_std_lam
from execution_cost.schedules import ac_params_from_market
from execution_cost.simulator import MarketParams, simulate_market


@dataclass
class MisspecRow:
    true_delta: float
    delta_hat: float
    delta_hat_se: float
    kappa_T: float
    ac_mean_bps: float
    ac_std_bps: float
    sqrt_saving_bps: float  # AC - SQRT-OPT at equal realised std
    sqrt_saving_se_bps: float
    power_saving_bps: float  # AC - POWER-OPT at equal realised std
    power_saving_se_bps: float
    power_vs_sqrt_bps: float  # SQRT - POWER (> 0: knowing delta helps)
    power_vs_sqrt_se_bps: float
    sqrt_matched: bool
    power_matched: bool

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = a - b
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def exponent_sensitivity(
    true_deltas: tuple[float, ...] = (0.35, 0.5, 0.65, 0.8),
    urgencies: tuple[float, ...] = (1.0, 3.0),
    q: float = 0.05,
    n_paths: int = 4000,
    seed: int = 123,
    calibration_seed: int = 0,
    n_orders: int = 20_000,
    market: MarketParams | None = None,
) -> list[MisspecRow]:
    base_mp = market or MarketParams()
    rows: list[MisspecRow] = []
    for d in true_deltas:
        mp = replace(base_mp, temp_exponent=float(d))
        orders = generate_metaorders(n_orders=n_orders, seed=calibration_seed, market=mp)
        cal = calibrate_impact(orders, seed=calibration_seed, market=mp)
        G = max(cal.perm_coeff, 0.0)
        Y5 = cal.temp_sqrt_Y
        d_hat = float(np.clip(cal.nls_delta, 0.05, 1.5))
        Yd = temp_coeff_at_exponent(orders, d_hat, cal.perm_coeff, seed=calibration_seed, market=mp)
        X = q * mp.adv
        mk = simulate_market(mp, n_paths, seed=seed)
        ac_base = ac_params_from_market(mp, X, lam=0.0, temp_coeff=Y5, perm_coeff=G)
        for k in urgencies:
            lam = lam_for_urgency(ac_base, k)
            ac = _evaluate(ac_trade_list(replace(ac_base, lam=lam)), mk, mp, X, 0.0)
            s = float(ac.std(ddof=1))
            _, sq = match_std_lam(s, mp, X, mk, Y5, G, planner_exponent=0.5)
            _, pw = match_std_lam(s, mp, X, mk, Yd, G, planner_exponent=d_hat)
            sq_m, sq_se = _paired(ac, sq)
            pw_m, pw_se = _paired(ac, pw)
            pv_m, pv_se = _paired(sq, pw)
            rows.append(
                MisspecRow(
                    true_delta=float(d),
                    delta_hat=float(cal.nls_delta),
                    delta_hat_se=float(cal.nls_delta_se),
                    kappa_T=float(k),
                    ac_mean_bps=float(ac.mean()),
                    ac_std_bps=s,
                    sqrt_saving_bps=sq_m,
                    sqrt_saving_se_bps=sq_se,
                    power_saving_bps=pw_m,
                    power_saving_se_bps=pw_se,
                    power_vs_sqrt_bps=pv_m,
                    power_vs_sqrt_se_bps=pv_se,
                    sqrt_matched=bool(abs(sq.std(ddof=1) - s) < 1e-6 * s),
                    power_matched=bool(abs(pw.std(ddof=1) - s) < 1e-6 * s),
                )
            )
    return rows
