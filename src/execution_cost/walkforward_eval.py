"""Slice 5 harness: cost-aware rebalancing chosen walk-forward, with deflation.

Pre-registered protocol (frozen in :data:`PROTOCOL` before any evaluation
seed was scored; only development seed 42 was looked at, to check that the
cost level and grids are in a sensible range):

* **Market**: :class:`execution_cost.alpha.AlphaParams` defaults - 5,040
  daily bars (20 years), IC 0.10, AR(1) alpha with rho 0.90 (half-life 6.6
  days), signal noise 0.5, 1% daily vol. Target position ``tanh(signal)``.
* **Costs**: slice-1 defaults, all charged on every trade - 10 bps spread
  (half paid), 5 bps linear slippage at 100% ADV, sqrt impact
  ``0.25 * sigma * sqrt(|dw| / ADV)``, ADV = 1 unit of weight.
* **Candidates** (42 trials): FULL; PARTIAL over 8 rates; BAND over 8
  widths; PARTIAL+BAND over a 5 x 5 grid.
* **Walk-forward**: expanding window, first 1,260 bars (5 years) training
  only, then 15 annual test folds of 252 bars; the selector maximises
  training net Sharpe. Selectors: WF-PARTIAL (the slice-1 turnover knob, the
  baseline), WF-BAND, WF-PB (PARTIAL+BAND grid) and WF-ALL (all 42).
* **Seeds**: evaluation 500-519 (IC 0.10), null control 600-619 (IC 0).
* **Hypotheses and decision rules**
  - H1 (primary): WF-PB beats WF-PARTIAL on OOS net Sharpe; one-sided
    paired t-test across the 20 evaluation seeds at 5%. If H1 passes the
    recommended rule family is PARTIAL+BAND, otherwise PARTIAL stays.
  - H2: WF-BAND vs WF-PARTIAL, two-sided at 5%, reported either way.
  - H3: the full-sample best of 42 overstates WF-ALL's OOS net Sharpe
    (mean in-sample minus OOS > 0, one-sided at 5%).
  - H4 (null): with IC 0, WF-ALL's mean OOS net Sharpe is <= 0 and the
    deflated Sharpe of the full-sample best of 42 is < 0.95 on at least
    19 / 20 null seeds; with IC 0.10 report how many seeds have DSR > 0.95.

Secondary sensitivities (cost x0.5 / x2, IC 0.05, rho 0.7 / 0.97) reuse the
evaluation seeds and do not change the decision.

Regime-break extension (H5, frozen in :data:`REGIME_PROTOCOL` after H1-H4
were scored and before any regime seed was run): the alpha's half-life
drops from 22.8 to 1.9 days (rho 0.97 -> 0.70) at bar 2,520, mid-way through
the OOS years. The slow-alpha regime favours slow PARTIAL rules and the fast
one favours PARTIAL+BAND (sensitivity table), so the best rule changes.

* H5 (one-sided, paired over seeds 700-719): WF-ALL with a **rolling
  3-year** training window beats the expanding window on OOS net Sharpe
  over the post-break bars ``[2520, 5040)``.
* Control (two-sided, same seeds, no break, rho 0.90): rolling vs expanding
  over the same bars; a cost of the shorter window is expected.
* Decision: if H5 passes, the README recommends a rolling window when a
  regime change is suspected; the default stays expanding either way.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import stats

from execution_cost.alpha import AlphaParams, simulate_alpha_path
from execution_cost.costs import CostParams
from execution_cost.metrics import annualized_sharpe
from execution_cost.rebalance import Rule, net_returns, trades_from_positions
from execution_cost.sharpe_stats import (
    deflated_sharpe,
    effective_trials,
    hac_mean_test,
    per_bar_sharpe,
)
from execution_cost.walkforward import fixed_rule_oos, run_rules, walk_forward

PARTIAL_RATES = (0.5, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03)
BAND_WIDTHS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
PB_RATES = (0.7, 0.5, 0.3, 0.2, 0.1)
PB_WIDTHS = (0.1, 0.2, 0.3, 0.5, 0.7)

PROTOCOL = {
    "market": AlphaParams().to_dict(),
    "costs": {"spread_bps": 10.0, "slippage_bps": 5.0, "impact_coeff": 0.25, "adv_notional": 1.0},
    "min_train": 1260,
    "test_len": 252,
    "eval_seeds": tuple(range(500, 520)),
    "null_seeds": tuple(range(600, 620)),
    "dev_seed": 42,
    "n_trials": 42,
    "alpha_level": 0.05,
    "dsr_threshold": 0.95,
    "null_dsr_min_seeds": 19,
    "primary": ("WF-PB", "WF-PARTIAL"),
}

SELECTORS = ("WF-PARTIAL", "WF-BAND", "WF-PB", "WF-ALL")

REGIME_PROTOCOL = {
    "before": {"rho": 0.97, "ic": 0.10},
    "after": {"rho": 0.70, "ic": 0.10},
    "switch_at": 2520,
    "rolling_window": 756,
    "seeds": tuple(range(700, 720)),
    "selector": "WF-ALL",
    "alpha_level": 0.05,
}


def candidate_rules() -> dict[str, list[Rule]]:
    """The pre-registered grids, grouped by selector family."""
    partial = [Rule("PARTIAL", rate=a) for a in PARTIAL_RATES]
    band = [Rule("BAND", width=b) for b in BAND_WIDTHS]
    pb = [Rule("PARTIAL+BAND", rate=a, width=b) for a in PB_RATES for b in PB_WIDTHS]
    return {
        "WF-PARTIAL": partial,
        "WF-BAND": band,
        "WF-PB": pb,
        "WF-ALL": [Rule("FULL")] + partial + band + pb,
    }


def cost_params(scale: float = 1.0) -> CostParams:
    c = PROTOCOL["costs"]
    return CostParams(
        spread_bps=c["spread_bps"] * scale,
        slippage_bps=c["slippage_bps"] * scale,
        impact_coeff=c["impact_coeff"] * scale,
        adv_notional=c["adv_notional"],
    )


@dataclass(frozen=True)
class SeedResult:
    seed: int
    oos: dict[str, dict[str, float]]
    oos_net: dict[str, np.ndarray]
    oos_weights: dict[str, np.ndarray]
    oos_returns: np.ndarray
    oos_vol: np.ndarray
    costs: CostParams
    choices: dict[str, list[str]]
    insample_best: dict[str, float]


def _summary(gross: np.ndarray, cost: np.ndarray, net: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    return {
        "gross_sharpe": annualized_sharpe(gross),
        "net_sharpe": annualized_sharpe(net),
        "mean_cost_bps": float(cost.mean() * 1e4),
        "turnover": float(np.mean(np.abs(trades_from_positions(weights)))),
        "mean_abs_weight": float(np.mean(np.abs(weights))),
    }


def evaluate_seed(
    seed: int,
    *,
    market: AlphaParams | None = None,
    costs: CostParams | None = None,
    min_train: int | None = None,
    test_len: int | None = None,
) -> SeedResult:
    """All selectors, the FULL baseline and the full-sample best on one path."""
    market = market or AlphaParams(**PROTOCOL["market"])
    costs = costs or cost_params()
    min_train = int(min_train or PROTOCOL["min_train"])
    test_len = int(test_len or PROTOCOL["test_len"])
    path = simulate_alpha_path(market, seed=seed)
    groups = candidate_rules()
    all_runs = run_rules(path, groups["WF-ALL"], costs)
    by_label = {r.rule.label: r for r in all_runs}

    oos, oos_net, oos_w, choices = {}, {}, {}, {}
    for name in SELECTORS:
        runs = [by_label[r.label] for r in groups[name]]
        wf = walk_forward(path, groups[name], costs, min_train=min_train, test_len=test_len, runs=runs)
        oos[name] = _summary(wf.gross, wf.cost, wf.net, wf.weights)
        oos_net[name] = wf.net
        oos_w[name] = wf.weights
        choices[name] = wf.choices
    full = fixed_rule_oos(by_label["FULL"], min_train)
    oos["FULL"] = _summary(full["gross"], full["cost"], full["net"], full["weights"])
    oos_net["FULL"] = full["net"]
    oos_w["FULL"] = full["weights"]

    # Full-sample selection (the slice-1 practice) and its deflated Sharpe.
    trial_sr = np.array([per_bar_sharpe(r.net[1:]) for r in all_runs])
    best = all_runs[int(np.nanargmax(trial_sr))]
    dsr = deflated_sharpe(best.net[1:], n_trials=len(all_runs), trial_var=float(np.nanvar(trial_sr, ddof=1)))
    # Post-hoc (added after the pre-registered run): deflate with the
    # effective number of trials instead of the raw 42 correlated ones.
    n_eff = effective_trials(np.column_stack([r.net[1:] for r in all_runs]))
    dsr_eff = deflated_sharpe(
        best.net[1:], n_trials=max(int(round(n_eff)), 1), trial_var=float(np.nanvar(trial_sr, ddof=1))
    )
    best_oos_part = annualized_sharpe(best.net[min_train:])
    insample = {
        "label": best.rule.label,
        "net_sharpe": annualized_sharpe(best.net[1:]),
        "net_sharpe_oos_window": best_oos_part,
        "dsr": dsr["dsr"],
        "psr0": dsr["psr0"],
        "benchmark_annual": dsr["benchmark_annual"],
        "n_eff": n_eff,
        "dsr_eff": dsr_eff["dsr"],
        "benchmark_eff_annual": dsr_eff["benchmark_annual"],
    }
    return SeedResult(
        seed=seed,
        oos=oos,
        oos_net=oos_net,
        oos_weights=oos_w,
        oos_returns=path.returns[min_train:],
        oos_vol=path.daily_vol[min_train:],
        costs=costs,
        choices=choices,
        insample_best=insample,
    )


def levered_net(r: SeedResult, name: str, k: float) -> np.ndarray:
    """OOS net returns of selector ``name`` with every position scaled by ``k``.

    Costs are recomputed on the scaled trades, so leverage pays the
    superlinear slippage and impact terms; nothing is rescaled ex post.
    """
    w = k * r.oos_weights[name]
    _, _, net = net_returns(r.oos_returns, w, r.costs, daily_vol=r.oos_vol)
    return net


def equal_risk_leverage(
    r: SeedResult, name: str, target_std: float, *, lo: float = 0.05, hi: float = 20.0, tol: float = 1e-6
) -> float:
    """Leverage ``k`` at which ``name``'s OOS net-return std equals ``target_std``.

    Post-hoc diagnostic (like slice 3's equal-risk frontier): ``k`` is solved
    on the evaluation days themselves, so it answers "at the same realised
    risk, which rule earns more after costs", not "which rule would a trader
    pick".
    """
    f = lambda k: float(np.std(levered_net(r, name, k), ddof=1)) - target_std  # noqa: E731
    if f(lo) > 0 or f(hi) < 0:
        raise ValueError("target risk not bracketed")
    a, b = lo, hi
    while b - a > tol * max(1.0, a):
        m = 0.5 * (a + b)
        if f(m) > 0:
            b = m
        else:
            a = m
    return 0.5 * (a + b)


def equal_risk_comparison(results: list[SeedResult], cand: str, base: str) -> dict[str, float]:
    """Lever ``cand`` to ``base``'s realised OOS risk; paired net Sharpe and P&L gap."""
    lev, d_sharpe, d_bps = [], [], []
    for r in results:
        target = float(np.std(r.oos_net[base], ddof=1))
        k = equal_risk_leverage(r, cand, target)
        net = levered_net(r, cand, k)
        lev.append(k)
        d_sharpe.append(annualized_sharpe(net) - annualized_sharpe(r.oos_net[base]))
        d_bps.append(float((net.mean() - r.oos_net[base].mean()) * 1e4))
    d = np.array(d_sharpe)
    t = stats.ttest_1samp(d, 0.0)
    return {
        "leverage_mean": float(np.mean(lev)),
        "sharpe_diff": float(d.mean()),
        "sharpe_diff_se": float(d.std(ddof=1) / np.sqrt(d.size)),
        "p": float(t.pvalue),
        "wins": int((d > 0).sum()),
        "n": int(d.size),
        "pnl_diff_bps_per_day": float(np.mean(d_bps)),
    }


def paired_seed_test(a: np.ndarray, b: np.ndarray, *, alternative: str = "two-sided") -> dict[str, float]:
    """Paired t-test of per-seed metrics ``a - b`` plus a sign count."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    t = stats.ttest_1samp(d, 0.0, alternative=alternative)
    return {
        "mean": float(d.mean()),
        "se": float(d.std(ddof=1) / np.sqrt(d.size)),
        "t": float(t.statistic),
        "p": float(t.pvalue),
        "wins": int((d > 0).sum()),
        "n": int(d.size),
    }


def pooled_hac(results: list[SeedResult], cand: str, base: str) -> dict[str, float]:
    """Newey-West test on the stacked daily net-return differences (bps/day)."""
    d = np.concatenate([r.oos_net[cand] - r.oos_net[base] for r in results]) * 1e4
    return hac_mean_test(d, maxlags=10)


def run_study(
    seeds: tuple[int, ...] | list[int],
    *,
    market: AlphaParams | None = None,
    costs: CostParams | None = None,
) -> list[SeedResult]:
    return [evaluate_seed(int(s), market=market, costs=costs) for s in seeds]


def table(results: list[SeedResult]) -> dict[str, dict[str, float]]:
    """Mean OOS metrics per selector across seeds."""
    names = list(SELECTORS) + ["FULL"]
    out = {}
    for name in names:
        rows = [r.oos[name] for r in results]
        out[name] = {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}
        out[name]["net_sharpe_sd"] = float(np.std([row["net_sharpe"] for row in rows], ddof=1))
    return out


def decide(results: list[SeedResult], null_results: list[SeedResult]) -> dict:
    """Apply the pre-registered decision rules."""
    alpha = PROTOCOL["alpha_level"]
    ns = {n: np.array([r.oos[n]["net_sharpe"] for r in results]) for n in list(SELECTORS) + ["FULL"]}
    h1 = paired_seed_test(ns["WF-PB"], ns["WF-PARTIAL"], alternative="greater")
    h2 = paired_seed_test(ns["WF-BAND"], ns["WF-PARTIAL"])
    ins = np.array([r.insample_best["net_sharpe"] for r in results])
    h3 = paired_seed_test(ins, ns["WF-ALL"], alternative="greater")
    null_wf = np.array([r.oos["WF-ALL"]["net_sharpe"] for r in null_results])
    null_dsr = np.array([r.insample_best["dsr"] for r in null_results])
    null_ins = np.array([r.insample_best["net_sharpe"] for r in null_results])
    eval_dsr = np.array([r.insample_best["dsr"] for r in results])
    thr = PROTOCOL["dsr_threshold"]
    h4_pass = bool(null_wf.mean() <= 0.0 and int((null_dsr < thr).sum()) >= PROTOCOL["null_dsr_min_seeds"])
    return {
        "H1": {**h1, "pass": bool(h1["p"] < alpha and h1["mean"] > 0)},
        "H2": {**h2, "pass": bool(h2["p"] < alpha)},
        "H3": {**h3, "pass": bool(h3["p"] < alpha and h3["mean"] > 0)},
        "H4": {
            "null_wf_all_mean_net_sharpe": float(null_wf.mean()),
            "null_wf_all_se": float(null_wf.std(ddof=1) / np.sqrt(null_wf.size)),
            "null_insample_best_mean_net_sharpe": float(null_ins.mean()),
            "null_seeds_dsr_below": int((null_dsr < thr).sum()),
            "eval_seeds_dsr_above": int((eval_dsr > thr).sum()),
            "n_null": int(null_wf.size),
            "n_eval": int(eval_dsr.size),
            "pass": h4_pass,
        },
        "recommended_family": "PARTIAL+BAND" if (h1["p"] < alpha and h1["mean"] > 0) else "PARTIAL",
    }


def null_market() -> AlphaParams:
    return replace(AlphaParams(**PROTOCOL["market"]), ic=0.0)


def regime_compare(seed: int, *, break_: bool = True, n: int | None = None, switch_at: int | None = None,
                   min_train: int | None = None, test_len: int | None = None,
                   window: int | None = None) -> dict[str, float]:
    """Expanding vs rolling WF-ALL on one path, scored on the post-switch bars."""
    from execution_cost.alpha import simulate_regime_path

    rp = REGIME_PROTOCOL
    base = AlphaParams(**PROTOCOL["market"])
    if n is not None:
        base = replace(base, n=int(n))
    sw = int(switch_at if switch_at is not None else rp["switch_at"])
    min_train = int(min_train or PROTOCOL["min_train"])
    test_len = int(test_len or PROTOCOL["test_len"])
    window = int(window or rp["rolling_window"])
    if break_:
        path = simulate_regime_path(replace(base, **rp["before"]), replace(base, **rp["after"]), switch_at=sw, seed=seed)
    else:
        path = simulate_alpha_path(base, seed=seed)
    costs = cost_params()
    rules = candidate_rules()[rp["selector"]]
    runs = run_rules(path, rules, costs)
    out = {}
    for name, win in (("expanding", None), ("rolling", window)):
        wf = walk_forward(path, rules, costs, min_train=min_train, test_len=test_len, runs=runs, window=win)
        post = wf.oos_index >= sw
        out[f"{name}_net_sharpe"] = annualized_sharpe(wf.net[post])
        out[f"{name}_pre_net_sharpe"] = annualized_sharpe(wf.net[~post]) if (~post).sum() > 1 else float("nan")
        out[f"{name}_choices"] = wf.choices
    return out


def run_regime_study(seeds=None, **kw) -> dict:
    """H5 and its no-break control over the pre-registered seeds."""
    seeds = tuple(seeds if seeds is not None else REGIME_PROTOCOL["seeds"])
    brk = [regime_compare(s, break_=True, **kw) for s in seeds]
    ctl = [regime_compare(s, break_=False, **kw) for s in seeds]
    def arr(rows, key):
        return np.array([r[key] for r in rows])
    h5 = paired_seed_test(arr(brk, "rolling_net_sharpe"), arr(brk, "expanding_net_sharpe"), alternative="greater")
    control = paired_seed_test(arr(ctl, "rolling_net_sharpe"), arr(ctl, "expanding_net_sharpe"))
    return {
        "seeds": list(seeds),
        "break": {
            "expanding": float(arr(brk, "expanding_net_sharpe").mean()),
            "rolling": float(arr(brk, "rolling_net_sharpe").mean()),
            "expanding_pre": float(np.nanmean(arr(brk, "expanding_pre_net_sharpe"))),
            "rolling_pre": float(np.nanmean(arr(brk, "rolling_pre_net_sharpe"))),
        },
        "control": {
            "expanding": float(arr(ctl, "expanding_net_sharpe").mean()),
            "rolling": float(arr(ctl, "rolling_net_sharpe").mean()),
        },
        "H5": {**h5, "pass": bool(h5["p"] < REGIME_PROTOCOL["alpha_level"] and h5["mean"] > 0)},
        "control_test": control,
        "last_fold_choices": {
            "expanding": [r["expanding_choices"][-1] for r in brk],
            "rolling": [r["rolling_choices"][-1] for r in brk],
        },
    }
