"""Slice 5 harness: does reacting to the realised price path beat the static plan?

Protocol (every reported number out of sample, every cost charged):

1. **Impact calibration** on synthetic metaorders (calibration seed), as in
   slices 2-4: sqrt temporary ``Y`` and permanent ``G``.
2. **Training days** (``train_seed``): fit the volume profile ``u_hat`` and,
   for each candidate ``a`` in :data:`A_GRID`, run the price-adaptive policy
   (:class:`execution_cost.adaptive.PriceAdaptiveSqrtOpt`) and score the
   realised mean-variance utility ``E + lam Var`` of IS in price units. The
   ``a`` with the lowest training utility is chosen. ``a = 0`` (the static
   plan) is in the grid, so "do not adapt" can win.
3. **Evaluation days** (``seed``, disjoint, common random numbers): the chosen
   policy vs static SQRT-OPT on ``u_hat`` at the same ``lam``. 5 bps spread,
   sqrt temporary and linear permanent impact all charged. Paired IS
   difference ± SE; utility difference with a seeded paired bootstrap 95% CI
   (resampling days); std and 95th percentile of IS. The oracle ``a`` (best
   on evaluation days) is reported only to measure selection loss.

Utility, not mean IS, is the objective here. By construction AIM pays more
on average to cut variance (Almgren & Lorenz 2007), so a lower mean is not
the claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from execution_cost.adaptive import PriceAdaptiveSqrtOpt
from execution_cost.adaptive_eval import static_sqrt_on_profile
from execution_cost.calibration import CalibrationResult, calibrate_impact, generate_metaorders
from execution_cost.comparison import lam_for_urgency
from execution_cost.schedules import ac_params_from_market
from execution_cost.shortfall import decompose_shortfall
from execution_cost.simulator import EventParams, MarketParams, execute, simulate_market
from execution_cost.volume_model import fit_nowcast

A_GRID = (-0.5, -0.25, 0.0, 0.25, 0.5, 1.0)
N_BOOT = 2000


def utility(total_bps: np.ndarray, notional: float, lam: float) -> float:
    """Realised ``E + lam Var`` of IS in price units."""
    c = np.asarray(total_bps, float) / 1e4 * notional
    return float(c.mean() + lam * c.var(ddof=1))


def paired_bootstrap_ci(a_bps, b_bps, notional, lam, n_boot=N_BOOT, seed=0) -> tuple[float, float]:
    """95% CI of ``utility(a) - utility(b)`` resampling days jointly."""
    a, b = np.asarray(a_bps, float), np.asarray(b_bps, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    ca, cb = a[idx] / 1e4 * notional, b[idx] / 1e4 * notional
    d = (ca.mean(1) + lam * ca.var(1, ddof=1)) - (cb.mean(1) + lam * cb.var(1, ddof=1))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi)


def _summary(t: np.ndarray, notional: float, lam: float) -> dict:
    return {
        "mean_bps": float(t.mean()),
        "std_bps": float(t.std(ddof=1)),
        "p95_bps": float(np.percentile(t, 95)),
        "utility": utility(t, notional, lam),
    }


@dataclass
class PriceAdaptiveReport:
    q: float
    kT: float
    lam: float
    seeds: dict
    train_utility: dict  # a -> utility on training days
    chosen_a: float
    eval_utility: dict  # a -> utility on evaluation days (oracle diagnostics only)
    oracle_a: float
    static: dict
    chosen: dict
    diff_mean_bps: float
    diff_se_bps: float
    utility_change: float  # chosen - static, price units
    utility_change_pct: float
    utility_ci: tuple
    std_change_bps: float

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["train_utility"] = {f"{k:g}": v for k, v in self.train_utility.items()}
        d["eval_utility"] = {f"{k:g}": v for k, v in self.eval_utility.items()}
        d["utility_ci"] = list(self.utility_ci)
        return d


def evaluate_price_adaptive(
    market: MarketParams | None = None,
    q: float = 0.05,
    kT: float = 3.0,
    n_paths: int = 4000,
    n_train: int = 2000,
    seed: int = 123,
    train_seed: int = 7,
    calibration_seed: int = 0,
    events: EventParams | None = EventParams(),
    calibration: CalibrationResult | None = None,
    a_grid=A_GRID,
    n_grid: int = 40,
) -> PriceAdaptiveReport:
    mp = market or MarketParams()
    if len({seed, train_seed, calibration_seed}) < 3:
        raise ValueError("evaluation, training and calibration seeds must differ")
    if 0.0 not in a_grid:
        raise ValueError("a_grid must contain 0 (the static plan)")
    if calibration is None:
        orders = generate_metaorders(seed=calibration_seed, market=mp)
        calibration = calibrate_impact(orders, seed=calibration_seed, market=mp)
    Y, G = calibration.temp_sqrt_Y, max(calibration.perm_coeff, 0.0)
    X = q * mp.adv
    notional = X * mp.s0
    lam = lam_for_urgency(ac_params_from_market(mp, X, lam=0.0, temp_coeff=Y, perm_coeff=G), kT)

    train = simulate_market(mp, n_train, seed=train_seed, events=events)
    nowcast, _ = fit_nowcast(train.volumes, mp.adv)
    base = PriceAdaptiveSqrtOpt(mp, nowcast, X, lam, Y, G, n_grid=n_grid)

    def policy(a: float) -> PriceAdaptiveSqrtOpt:
        pol = object.__new__(PriceAdaptiveSqrtOpt)
        pol.__dict__.update(base.__dict__)  # share the precomputed grid
        pol.a = float(a)
        return pol

    def run(mk, a):
        return decompose_shortfall(execute(policy(a), mk, mp, X)).total

    train_u = {a: utility(run(train, a), notional, lam) for a in a_grid}
    chosen_a = min(a_grid, key=lambda a: (train_u[a], abs(a)))

    mk = simulate_market(mp, n_paths, seed=seed, events=events)
    static = decompose_shortfall(execute(static_sqrt_on_profile(mp, nowcast.profile, X, lam, Y, G), mk, mp, X)).total
    totals = {a: run(mk, a) for a in a_grid}
    eval_u = {a: utility(t, notional, lam) for a, t in totals.items()}
    ch = totals[chosen_a]
    d = ch - static
    s_u, c_u = utility(static, notional, lam), eval_u[chosen_a]
    return PriceAdaptiveReport(
        q=q, kT=kT, lam=lam,
        seeds={"calibration": calibration_seed, "train": train_seed, "eval": seed},
        train_utility=train_u, chosen_a=chosen_a, eval_utility=eval_u,
        oracle_a=min(a_grid, key=lambda a: eval_u[a]),
        static=_summary(static, notional, lam), chosen=_summary(ch, notional, lam),
        diff_mean_bps=float(d.mean()), diff_se_bps=float(d.std(ddof=1) / np.sqrt(len(d))),
        utility_change=c_u - s_u, utility_change_pct=100.0 * (c_u - s_u) / s_u,
        utility_ci=paired_bootstrap_ci(ch, static, notional, lam, seed=seed),
        std_change_bps=float(ch.std(ddof=1) - static.std(ddof=1)),
    )
