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
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import stats

from execution_cost.alpha import AlphaParams, simulate_alpha_path
from execution_cost.costs import CostParams
from execution_cost.metrics import annualized_sharpe
from execution_cost.rebalance import Rule, trades_from_positions
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

    oos, oos_net, choices = {}, {}, {}
    for name in SELECTORS:
        runs = [by_label[r.label] for r in groups[name]]
        wf = walk_forward(path, groups[name], costs, min_train=min_train, test_len=test_len, runs=runs)
        oos[name] = _summary(wf.gross, wf.cost, wf.net, wf.weights)
        oos_net[name] = wf.net
        choices[name] = wf.choices
    full = fixed_rule_oos(by_label["FULL"], min_train)
    oos["FULL"] = _summary(full["gross"], full["cost"], full["net"], full["weights"])
    oos_net["FULL"] = full["net"]

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
    return SeedResult(seed=seed, oos=oos, oos_net=oos_net, choices=choices, insample_best=insample)


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
