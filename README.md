# execution-cost-model

Research slice: **spread + slippage + square-root impact** costs applied to a
synthetic predictive signal. Compares **gross vs net Sharpe** and shows a
**turnover cost curve**. Not live PnL.

## Hypotheses

1. **Participation costs** - Half-spread, linear temporary slippage, and
   Almgren-style square-root impact (`eta * sigma * sqrt(participation)`) jointly price
   one-way trading costs as a fraction of notional.
2. **Net < gross** - Once any turnover occurs, net Sharpe is strictly below
   gross Sharpe for positive cost parameters.
3. **Turnover destroys edge** - Faster rebalancing (higher mean `|dw|`) raises
   mean daily cost and lowers net Sharpe along the cost curve.

## Model

| Component | Formula (one-way, return units) |
|-----------|----------------------------------|
| Half-spread | `0.5 * spread_bps / 1e4` |
| Slippage | `participation * slippage_bps / 1e4` |
| Square-root impact | `impact_coeff * sigma_daily * sqrt(participation)` |
| Participation | `|trade| / ADV` |

Gross return: `w_t * r_t`. Cost: `|dw_t| * unit_cost(dw_t)`. Net: gross - cost.

Synthetic path: latent factor drives next-bar returns; a noisy lagged signal is
EWMA-smoothed with `hold_bars` controlling turnover.

Default cost params: `spread_bps=10`, `slippage_bps=5`, `impact_coeff=0.25`, `adv_notional=1`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# or: pip install -r requirements.txt
```

## Run

```bash
pytest
python scripts/run_cost_slice.py
python scripts/run_cost_slice.py --json --out artifacts/cost_slice.json
python scripts/run_execution_comparison.py          # calibration + schedule comparison tables
python scripts/run_execution_comparison.py --json --out artifacts/execution.json
python scripts/run_execution_comparison.py --no-frontier --no-misspec   # slice-2/3 tables only (fast)
python scripts/run_walkforward_slice.py             # slice 5: walk-forward rebalancing + deflated Sharpe (~10 s)
```

## Layout

- `src/execution_cost/costs.py` - cost primitives
- `src/execution_cost/portfolio.py` - synthetic signal + cost application
- `src/execution_cost/sensitivity.py` - turnover / hold-bars sweep
- `src/execution_cost/metrics.py` - Sharpe, turnover helpers
- `src/execution_cost/almgren_chriss.py` - AC optimal trajectory, frontier
- `src/execution_cost/simulator.py` - intraday execution simulator
- `src/execution_cost/schedules.py` - TWAP / VWAP / POV / AC schedules
- `src/execution_cost/shortfall.py` - implementation shortfall decomposition
- `src/execution_cost/calibration.py` - sqrt impact calibration on metaorders
- `src/execution_cost/comparison.py` - calibrated, out-of-sample schedule comparison
- `src/execution_cost/optimal_sqrt.py` - profile-aware mean-variance optimum under the sqrt law (SLSQP)
- `src/execution_cost/frontier.py` - SQRT-OPT vs AC at equal realised risk
- `src/execution_cost/misspecification.py` - true impact exponent != 0.5
- `src/execution_cost/volume_model.py` - fitted profile, day-factor and surge nowcasts
- `src/execution_cost/adaptive.py` - re-optimising sqrt-law policy, nowcast POV
- `src/execution_cost/adaptive_eval.py` - slice-4 harness (normal vs event days)
- `src/execution_cost/alpha.py` - persistent AR(1) alpha market (slice 5)
- `src/execution_cost/rebalance.py` - FULL / PARTIAL / no-trade BAND / PARTIAL+BAND rules
- `src/execution_cost/sharpe_stats.py` - PSR, expected max Sharpe, deflated Sharpe, effective trials, Newey-West
- `src/execution_cost/walkforward.py` - expanding-window rule selection, OOS stitching
- `src/execution_cost/walkforward_eval.py` - slice-5 pre-registered protocol and harness
- `scripts/run_walkforward_slice.py` - CLI for slice 5
- `scripts/run_adaptive_slice.py` - CLI for slice 4
- `scripts/run_cost_slice.py` - CLI for baseline + cost curve
- `scripts/run_execution_comparison.py` - CLI for the execution slice
- `.github/workflows/tests.yml` - pytest CI (py3.11 / 3.12)
- `tests/` - unit and hypothesis checks

## Slice 2 (2026-10-03): optimal execution - Almgren-Chriss, sqrt impact, IS, schedules

### What was added

| Module | Content |
|---|---|
| `almgren_chriss.py` | AC (2000) optimal trajectory (exact discrete kappa, stable sinh form), E[IS], Var[IS], efficient frontier, Monte Carlo check |
| `simulator.py` | Intraday simulator: 26 bins, U-shaped stochastic volume, ABM mid, **square-root temporary impact on interval participation** (reuses `costs.square_root_impact_cost`), linear permanent impact, common random numbers |
| `schedules.py` | TWAP, VWAP (expected profile), POV (realised volume), AC (sqrt law linearised at the TWAP rate) |
| `shortfall.py` | Perold implementation shortfall: delay / spread / temporary / permanent / timing / opportunity / fees, exact per-path identity, and a model-free split from raw fills |
| `calibration.py` | Square-root law fitted on synthetic metaorders: NLS and binned log-log WLS (statsmodels), permanent impact from post-trade moves, 70/30 out-of-sample scoring |
| `comparison.py` | Calibrate -> schedule -> evaluate on disjoint-seed days; paired differences vs TWAP; calibrated pre-trade estimate vs realised |

### Test design

- **AC correctness**: `lam=0` gives TWAP. The closed form equals an independent tridiagonal solve of `min E + lam V` and beats 200 random feasible perturbations. A 40k-path Monte Carlo matches the analytic mean (within 4 SE) and variance (3%).
- **Calibration** (ground truth: interval-level exponent 0.5, `temp_coeff=0.5`, `perm_coeff=0.25`): 20,000 full-day TWAP buys, `q = Q/ADV` log-uniform in [0.5%, 20%], daily vol in {1, 2, 3, 4}%. Observed slippage is arrival-to-VWAP net of the half-spread, normalised by vol. Analytic target for the metaorder-level temporary coefficient: `0.543` (Jensen effects of volume noise and the U-profile).
- **Comparison**: buy 2 / 5 / 10% of ADV, 4,000 simulated days, eval seed 123 vs calibration seed 0. Costs: 5 bps spread + sqrt temporary + linear permanent, all charged. AC uses the *calibrated* coefficients at urgency `kappa*T` = 1 and 3. POV rate is 1.25 x q.

### Results (`python scripts/run_execution_comparison.py`)

Calibration (train 14k / test 6k orders):

| NLS delta | binned delta (95% CI) | Y raw | Y temporary-only | permanent (95% CI) | OOS binned RMSE sqrt / power / linear |
|---|---|---|---|---|---|
| 0.550 (se 0.047) | 0.519 [0.417, 0.621] | 0.573 | 0.540 (truth 0.543) | 0.232 [0.006, 0.458] (truth 0.25) | 0.029 / 0.031 / 0.056 |

Schedules, buy 5% ADV (bps of arrival notional; paired = same simulated days):

| schedule | mean IS | std | p95 | vs TWAP (paired) | temporary | permanent | completion | pre-trade est. |
|---|---|---|---|---|---|---|---|---|
| TWAP | 29.22 ± 1.78 | 112.3 | 208.8 | 0 | 24.39 | 1.20 | 1.000 | 27.74 |
| VWAP | 28.38 ± 1.72 | 108.5 | 202.8 | **-0.83 ± 0.15** | 23.56 | 1.20 | 1.000 | 26.92 |
| AC(kT=1) | 29.14 ± 1.66 | 105.2 | 196.5 | -0.08 ± 0.13 | 24.34 | 1.20 | 1.000 | 27.70 |
| AC(kT=3) | 31.67 ± 1.21 | **76.2** | **154.1** | +2.45 ± 0.69 | 27.06 | 1.18 | 1.000 | 30.36 |
| POV(6.25%) | 29.13 ± 1.54 | 97.1 | 186.3 | -0.09 ± 0.47 | 24.42 | 1.14 | 0.982 | - |

Spread is 2.50 bps for every completed schedule. The timing component averages about +1 bps here, which is noise (SE ~1.8). At 2% / 10% ADV the VWAP saving vs TWAP is -0.53 / -1.18 bps, and the AC(kT=3) premium is +1.49 / +3.53 bps.

Reading:
- **Timing risk dominates.** The std of IS (~110 bps) is about 4x the mean cost for a full-day order in a 2%-vol stock. Mean-cost rankings are only resolvable with paired (common-random-number) comparisons.
- **VWAP beats TWAP slightly but significantly** under U-shaped volume. Concave impact rewards trading where liquidity is.
- **Urgent AC buys risk reduction.** At kT=3 it pays +2.45 bps of impact to cut IS std by 32% and p95 by 55 bps. That is the AC efficient-frontier trade-off, reproduced under the (non-linear) sqrt simulator.
- **POV** adapts to realised volume and finishes early (lower std). It left 1.8% unfilled on low-volume days, which is charged as opportunity cost.
- **The calibrated pre-trade model is accurate out of sample.** Estimates sit within 0.6 bps (under 1.5%) of the realised spread + temporary + permanent cost at all three sizes. Without netting out permanent impact, the raw sqrt coefficient double-counted about 1.3 bps.

### Weaknesses (honest)

- Everything is a synthetic simulator, and the "true" impact law is the one the simulator implements. Calibration recovering it is a consistency check, not evidence about real markets. There is no real tape, queue dynamics, venue fees or adverse selection.
- **Permanent impact is barely identified.** Its CI spans roughly 0 to 2x the truth, because a full day of price noise sits on the post-trade move. Order-level OOS MSE barely separates sqrt from linear (0.310 vs 0.313); only binned means do.
- AC is solved for *linear* impact and mapped from the sqrt law at the TWAP rate. It does not know the intraday volume profile, which is why AC(kT=1) is not costlier than TWAP. *(Slice 3 adds the nonlinear, profile-aware optimiser.)*
- POV is idealised: it sees current-interval volume without lag. *(Slice 3 adds an implementable lag-1 POV.)*
- Single stock, single day horizon, constant intraday vol, no intraday vol smile, no cross-impact.
- The first-slice limits still apply to the portfolio-level Sharpe analysis (full-sample stats, synthetic factor).

## Slice 5 (2026-11-01): cost-aware rebalancing chosen walk-forward, with deflated Sharpe

Slice 1 left two weaknesses at the portfolio level: the turnover knob was picked on **full-sample** Sharpe (no out-of-sample test, no multiple-testing correction), and its factor was i.i.d., so slowing down could only throw edge away. This slice asks a standard portfolio-construction question under the slice-1 cost model: **does a no-trade band beat plain partial adjustment once both are tuned walk-forward?**

### What was added

| Module | Content |
|---|---|
| `alpha.py` | AR(1) latent alpha (`rho` 0.90, half-life 6.6 days), IC 0.10 to the next return, noisy signal (noise sd 0.5), target position `tanh(signal)`, 1% daily vol, 20 years of daily bars. |
| `rebalance.py` | `FULL`; `PARTIAL(a)` (`w += a (x - w)`, identical to slice 1's EWMA knob); `BAND(b)` (no trade inside `|x - w| <= b`, else trade to the nearest edge); `PARTIAL+BAND(a, b)` (close a fraction `a` of the excess beyond the band; nests both). Every rule's only state is the held position. Costs: the full slice-1 model (half-spread, linear slippage, sqrt impact on `|dw| / ADV`). |
| `sharpe_stats.py` | Sharpe SE under skew / kurtosis, probabilistic Sharpe, expected max Sharpe of N trials, deflated Sharpe (Bailey & Lopez de Prado), Newey-West mean test (statsmodels). |
| `walkforward.py` | Expanding window: 5 training years, then 15 annual folds; at each fold the rule with the best **training net Sharpe** trades the next year from the position actually held. |
| `walkforward_eval.py` | The pre-registered protocol (grids, seeds, hypotheses, decision rules) frozen in `PROTOCOL` and committed before any evaluation seed was scored. 42 candidate rules; selectors WF-PARTIAL (baseline), WF-BAND, WF-PB, WF-ALL. |

### Test design

- Generator: AR(1) variance 1 and lag-1 autocorrelation `rho`; factor-to-next-return correlation equals IC; targets use only the previous signal; IC 0 is a null.
- Rules: `PARTIAL` equals slice 1's EWMA; every trade of `BAND` lands on a band edge; wider bands trade strictly less; `PARTIAL+BAND` nests both; causality (changing later targets never changes earlier positions); a run split at any bar and continued from the held position equals one pass; cost accounting matches slice 1 exactly.
- Statistics: Sharpe SE and expected max of 50 null trials match Monte Carlo (5% / 6%); DSR equals PSR for one trial; HAC SE widens under autocorrelation.
- Walk-forward: folds cover exactly the bars after the training window; scrambling returns after a fold start cannot change that fold's choice; each choice is the training argmax; a one-rule walk-forward equals the fixed rule; costs at fold boundaries are charged from the held position.

### Results (`python scripts/run_walkforward_slice.py`, 20 evaluation seeds x 3,780 OOS days)

OOS means across seeds (annualised; cost in bps/day of capital):

| selector | gross Sharpe | net Sharpe (sd across seeds) | cost | turnover | mean abs weight | null (IC 0) net Sharpe |
|---|---|---|---|---|---|---|
| FULL (trade to target) | 1.44 | **-1.16** (0.36) | 10.93 | 0.382 | 0.589 | -2.62 |
| WF-PARTIAL (baseline) | 1.34 | 1.02 (0.31) | 0.94 | 0.069 | 0.388 | -0.11 |
| WF-BAND | 1.26 | 0.92 (0.34) | 0.51 | 0.026 | 0.199 | -0.23 |
| WF-PB | 1.32 | 1.03 (0.29) | 0.61 | 0.041 | 0.283 | -0.11 |
| WF-ALL (all 42) | 1.31 | 1.02 (0.29) | 0.61 | 0.041 | 0.283 | -0.11 |

Pre-registered hypotheses (paired across the 20 seeds):

| | test | result | verdict |
|---|---|---|---|
| H1 (primary) | WF-PB > WF-PARTIAL net Sharpe, one-sided | +0.015 ± 0.018, p = 0.21, 11/20 seeds | **FAIL** - PARTIAL stays the recommended rule |
| H2 | WF-BAND vs WF-PARTIAL, two-sided | **-0.096 ± 0.041**, p = 0.031, 4/20 | the band alone is *worse* |
| H3 | full-sample best of 42 > WF-ALL OOS | +0.053 ± 0.028, p = 0.035 | pass, but small |
| H4 | null: WF-ALL OOS <= 0 and DSR < 0.95 on >= 19/20 | WF-ALL -0.11 ± 0.05; DSR < 0.95 on 20/20 | pass |

Sensitivity (secondary, same seeds; PB - PARTIAL and BAND - PARTIAL in net Sharpe, ± SE, seeds won):

| setting | FULL | WF-PARTIAL | WF-PB | PB - PARTIAL | BAND - PARTIAL |
|---|---|---|---|---|---|
| costs x0.5 | 0.12 | 1.19 | 1.19 | -0.005 ± 0.011 (8/20) | -0.074 ± 0.027 (5/20) |
| costs x2 | -3.50 | 0.77 | 0.83 | **+0.053 ± 0.022** (14/20) | +0.015 ± 0.030 (9/20) |
| IC 0.05 | -1.84 | 0.39 | 0.41 | +0.020 ± 0.017 (10/20) | -0.021 ± 0.033 (7/20) |
| rho 0.70 (half-life 1.9 d) | -2.15 | 0.53 | 0.68 | **+0.150 ± 0.032** (17/20) | +0.082 ± 0.044 (14/20) |
| rho 0.97 (half-life 22.8 d) | -0.80 | 1.28 | 1.26 | -0.020 ± 0.017 (7/20) | -0.120 ± 0.036 (4/20) |

Reading:
- **Costs decide everything.** Trading to the target every day has gross Sharpe 1.44 and net -1.16 (11 bps/day of cost). Any walk-forward-tuned slow-down recovers net Sharpe ~1.0 at a cost of 0.5-0.9 bps/day.
- **The no-trade band is not a free upgrade (H1 failed, H2 reversed).** A band is the optimum for *proportional* costs (the half-spread). Under the slice-1 model the total cost of a trade is `|dw| (c0 + c1 |dw| + c2 sqrt|dw|)`, superlinear in trade size because of slippage and sqrt impact, and superlinear costs reward splitting trades, which is what partial adjustment does (Garleanu-Pedersen). A pure band trades in lumps, carries a large tracking error to the target (mean |w| halves) and loses 0.10 Sharpe to partial adjustment. Adding partial adjustment inside the band only ties PARTIAL (+0.015, not significant). The band earns its keep only when trading is expensive relative to alpha decay: fast alpha (half-life 1.9 d, +0.15, 17/20) and doubled costs (+0.05, 14/20). Those are secondary cells, not a pre-registered claim.
- **Mean P&L and Sharpe disagree, and Sharpe is the pre-registered metric.** The raw daily net-return gap WF-PB - WF-PARTIAL is -0.78 bps/day (Newey-West t -11.9) because WF-PB holds about 0.28 vs 0.39 of gross exposure; per unit of risk it is level (see the equal-risk check below).
- **Full-sample tuning overstates only modestly here** (+0.05 Sharpe vs walk-forward), because the grid is smooth and the market is stationary. Note that the full-sample winner applied to the same OOS years scores 1.12 vs walk-forward's 1.02: that 0.10 is the look-ahead value of knowing the best setting.
- **Deflation works on the null and is conservative on the real signal.** With IC 0 every seed's best of 42 has DSR < 0.95 (20/20) and walk-forward loses money (-0.11), so no fake edge. With IC 0.10 only 4/20 seeds reach DSR > 0.95 even though the true net Sharpe is about 1: 20 years of a 1-Sharpe strategy picked from 42 correlated trials is weak evidence under DSR, partly because the trial variance includes the clearly dominated FULL rule.

### Post-hoc: equal realised risk (added after the pre-registered run)

Sharpe is scale-free only if leverage is free, and here it is not: costs grow faster than linearly in trade size. `equal_risk_comparison` levers every position of the candidate by `k`, recomputes all costs on the levered trades, and bisects `k` until the candidate's OOS net-return std equals WF-PARTIAL's (on the evaluation days, as in slice 3's equal-risk frontier).

| pair | mean leverage | net Sharpe diff at equal risk | p | seeds won | net P&L diff (bps/day) |
|---|---|---|---|---|---|
| WF-PB vs WF-PARTIAL | 1.43 | -0.021 ± 0.018 | 0.25 | 8 / 20 | -0.07 |
| WF-BAND vs WF-PARTIAL | 2.19 | **-0.208 ± 0.042** | < 0.001 | 2 / 20 | -0.61 |

At the same risk the band's small edge in unlevered Sharpe disappears (PB +0.015 -> -0.021), and the pure band falls further behind, because levering a lumpy trade path pays the superlinear impact. This confirms the H1 verdict: partial adjustment stays the recommended rule.

### Post-hoc: effective number of trials (added after the pre-registered run)

The 42 rules are all smoothings of one signal, so their net returns are highly correlated. `sharpe_stats.effective_trials` takes the eigenvalue participation ratio of the trial correlation matrix, `(sum l)^2 / sum l^2`.

| | N = 42 (pre-registered) | N_eff (post-hoc) |
|---|---|---|
| trials | 42 | **1.31** (IC 0.10), 1.32 (null) |
| expected-max benchmark, annualised | 0.99 | 0.00 |
| IC-0.10 seeds with DSR > 0.95 | 4 / 20 | 20 / 20 |
| null seeds with DSR < 0.95 | 20 / 20 | 19 / 20 (null seed 615: 0.958) |

The two corrections bracket the truth. The raw count over-deflates, so a real 1-Sharpe edge looks unproven on 16 of 20 paths. The participation ratio says the grid is about one bet; that removes the multiple-testing penalty, recognises every real path and lets 1 of 20 null paths through, which is the nominal 5% rate. Neither is the right answer for a grid whose trials are one dominant factor plus small differences. A clustering-based count is the next step.

### Weaknesses (honest)

- Single synthetic asset; the cost model is slice 1's (participation in units of ADV, no fees, no borrow). Real alphas are not AR(1) with constant IC, so walk-forward never faces a regime break here.
- The pre-registered DSR uses the raw count of 42 highly correlated trials and over-deflates; the post-hoc participation-ratio count (1.3) removes the penalty and admits the nominal 5% of null paths. A clustered trial count sits between them and is untested.
- Selection is on unlevered net Sharpe. The equal-risk check levers ex post on the evaluation days; a walk-forward volatility target chosen on training data only is untested.
- 20 seeds give ~0.02 SE on paired Sharpe differences; the favourable PB cells (fast alpha, high costs) are post-hoc sensitivities and would need their own pre-registration.

## Slice 4 (2026-10-17): adaptive re-optimisation, fitted volume nowcasts, event days

Slice 3 listed three weaknesses: schedules were **static** (no re-optimisation on realised volume), POV-lag1 used **one naive forecast** rather than a fitted volume model, and a **wrong volume profile** (event days) was untested. This slice tests all three.

### What was added

| Module | Content |
|---|---|
| `simulator.py` | `EventParams`: 15% of days get an unplanned volume surge (4x, decaying 0.6 per interval, 4 intervals, random start). Events use their own random stream, so a seed's normal days are identical with and without events. `execute` also runs *adaptive policies* that see only the volumes realised before each interval. |
| `volume_model.py` | Fitted intraday profile (mean daily share on training days; no planner sees the simulator's profile), a **day-factor nowcast** from today's volume so far, shrunk toward 1 with `k0` pseudo-intervals (`k0` chosen on training days by log-forecast MSE), and a **surge nowcast** adding an AR(1) term on the last interval's log surprise (`phi` by least squares on training days). |
| `adaptive.py` | `AdaptiveSqrtOpt`: every interval, re-solve the slice-3 mean-variance sqrt-law problem for the shares left over the intervals left, with volume = fitted profile x nowcast day factor, and trade the first leg. Uses the scaling `argmin w(R, D) = argmin w(R D, 1)` to precompute first-leg fractions on a log grid. `NowcastPOV`: POV sized on a nowcast forecast. |
| `adaptive_eval.py` | Harness: impact calibrated on metaorders (seed 0), volume model fitted on 500 training days (seed 7, events at the same rate), evaluation on 4,000 fresh days (seed 123), common random numbers, 5 bps spread + sqrt temporary + linear permanent impact charged, IS in bps by all / normal / event days, paired differences, realised `E + lam Var`. |
| `scripts/run_adaptive_slice.py` | Grid q in {2, 5, 10}% ADV x kT in {1, 3}; sensitivity to day-volume noise (0.2 / 0.5) and surge size (4x / 8x). Writes `artifacts/adaptive_slice.{json,md}`. |

### Test design

- Events: normal days byte-identical with and without events; surge multipliers exact inside the window and 1 outside; event share matches `prob`.
- No look-ahead: a policy's trades up to interval `j` do not change when later volumes change; forecasts at `j` ignore volumes at `j` and later.
- Adaptive correctness: the first leg at interval 0 equals the static optimum's; on a noise-free day the adaptive policy **replays the static schedule** (mean-variance with known volume is time consistent); it speeds up on heavy days and slows on thin ones; it always completes.
- Volume model: fitted profile within 3% of the true U-shape; nowcast exact on noise-free days; `phi` is larger when surges exist; nowcasts beat lag-1 out of sample.
- Harness: disjoint seeds enforced; `REPLAN` (re-optimise with day factor fixed at 1) equals static (control, |diff| < 0.02 bps); reproducible.

### Results (`python scripts/run_adaptive_slice.py`, 4,000 days, 566 event days)

Forecast log RMSE on evaluation days: lag-1 **0.441**, day-factor nowcast **0.352**, + surge AR(1) **0.347** (event days 0.522 / 0.496 / 0.471). Fitted `k0` = 4, `phi` = 0.15.

Paired differences in bps (± SE, same days):

| q | kT | ADAPTIVE - SQRT-OPT (all / event) | utility | NOWCAST-POV - POV-lag1 (all / event) | SURGE-POV - NOWCAST-POV (all / event) |
|---|---|---|---|---|---|
| 2% | 1 | -0.01 ± 0.02 / -0.03 ± 0.05 | -0.03% | -0.24 ± 0.16 / -0.40 ± 0.64 | -0.07 ± 0.03 / -0.17 ± 0.09 |
| 5% | 1 | -0.01 ± 0.02 / -0.03 ± 0.05 | -0.03% | **-0.39 ± 0.16** / -0.95 ± 0.65 | **-0.11 ± 0.03** / -0.22 ± 0.09 |
| 5% | 3 | -0.02 ± 0.02 / -0.01 ± 0.05 | -0.05% | -0.39 ± 0.16 / -0.95 ± 0.65 | -0.11 ± 0.03 / -0.22 ± 0.09 |
| 10% | 1 | -0.01 ± 0.02 / -0.03 ± 0.05 | -0.02% | **-0.55 ± 0.17** / -1.55 ± 0.65 | **-0.16 ± 0.03** / -0.28 ± 0.09 |
| 10% | 3 | -0.02 ± 0.02 / -0.00 ± 0.05 | -0.08% | -0.55 ± 0.17 / -1.55 ± 0.65 | -0.16 ± 0.03 / -0.28 ± 0.09 |

(POV rows do not depend on kT.) Sensitivity at q = 5%, kT = 3:

| day-volume noise | surge | ADAPTIVE - SQRT-OPT (all / event) | utility | std ADAPTIVE / static |
|---|---|---|---|---|
| 0.2 (default) | 4x | -0.02 ± 0.02 / -0.01 ± 0.05 | -0.05% | 46.2 / 46.2 |
| 0.2 | 8x | -0.00 ± 0.01 / -0.02 ± 0.05 | -0.06% | 46.5 / 46.6 |
| 0.5 | 4x | **-0.52 ± 0.07** / -0.47 ± 0.18 | -0.38% | 47.8 / 47.2 |
| 0.5 | 8x | **-0.48 ± 0.07** / -0.48 ± 0.18 | -0.40% | 48.0 / 47.5 |

Reading:
- **Re-optimising on realised volume is worth ~0.01 bps at the default day-volume noise (20%)**, within 1 SE at every size and urgency, even on event days. Under the sqrt law, the optimal schedule hardly moves when the whole day's volume scales by D: temporary cost scales as D^-0.5 and only `sqrt(R D)` enters the trade-off. A negative result for "adaptive beats static" in this market.
- **It starts to pay when day volume is very uncertain**: at 50% day-volume noise the adaptive policy saves 0.5 bps (7 SE) and lowers realised `E + lam Var` by 0.4%, with slightly *higher* IS std (it trades faster on heavy days).
- **The fitted volume model is the real gain for POV**: the day-factor nowcast cuts forecast error 20% and POV cost 0.24 / 0.39 / 0.55 bps at 2 / 5 / 10% ADV (1.5-3.2 SE), and lifts completion 0.987 -> 0.997. The surge term adds 0.07-0.16 bps (2-5 SE). POV still costs 1.2-1.9 bps more than static SQRT-OPT at kT = 1.
- **Event days do not break the static schedules here**: static SQRT-OPT is no worse on event days (extra volume lowers participation). They hurt POV-lag1 most (+3.1 bps vs static at 5% ADV on event days, vs +1.5 on normal days), because a surge inflates its next-interval forecast after the surge has decayed.

### Weaknesses (honest)

- Same simulator as slices 2-3: the planner's cost model has the simulator's functional form, so real adaptive gains (queue dynamics, price-volume feedback) are untested. Event days are a parametric surge, not real news-day tape.
- The adaptive policy re-optimises on volume only. It ignores the realised price path; with arithmetic Brownian prices and mean-variance utility that is close to optimal, but a price-adaptive (aggressive-in-the-money) policy is untested.
- The adaptive policy uses the day-factor nowcast, not the surge nowcast, because the precomputed grid assumes a uniform scaling of the remaining profile. A per-path re-solve with the surge shape is the obvious next step.
- One seed triple (0 / 7 / 123); 566 event days give event-split SEs of 0.05-0.65 bps. The 50%-noise result is a single sensitivity cell.
- `k0` and `phi` are fitted once on 500 training days; there is no rolling refit across regimes.

## Slice 3 (2026-10-07): profile-aware sqrt-law optimum, equal-risk frontiers, exponent misspecification

### What was added

| Module | Content |
|---|---|
| `optimal_sqrt.py` | Mean-variance optimal static schedule in the simulator's own cost model: `min T(n) + P(n) + lam V(n)` over `n_j >= 0, sum n_j = X`, with `T = Y sigma_p sum n_j (n_j / v_j)^delta` on the expected U-profile `v_j`, permanent `P = (G sigma_p / 2 ADV)(X^2 - sum n_j^2)` and timing variance `V = (sigma_p^2 / N) sum r_j^2`. SLSQP with the analytic gradient, started from VWAP, KKT residual returned. `lam` has the same units as AC, so the same `lam` means the same risk aversion. |
| `simulator.py` | `POV(rate, lag=1)`: sizes interval `j` on the previous interval's realised volume rescaled by the profile (`V_{j-1} u_j / u_{j-1}`), so it is implementable; `lag=0` is the old idealised POV. |
| `comparison.py` | Adds `SQRT-OPT(kT=1, 3)` at AC's `lam` (calibrated coefficients) and `POV-lag1` to the out-of-sample table. |
| `frontier.py` | Equal-risk comparison: bisect SQRT-OPT's `lam` until its *realised* IS std on the evaluation days equals AC's, then report the paired mean saving (same days). Also the realised `E + lam Var` at AC's own `lam`. |
| `misspecification.py` | The simulator's true exponent is set to 0.35 / 0.5 / 0.65 / 0.8; metaorders are regenerated in that market and recalibrated. SQRT-OPT (plans with 0.5) and POWER-OPT (plans with the NLS estimate) are compared with AC at equal risk. The planner exponent is always an explicit argument, so no planner sees the true exponent. |

### Test design

- **Optimiser correctness**: with `lam = 0, G = 0` the KKT condition gives `n_j ∝ v_j` and the solver returns VWAP exactly; on a flat profile it returns TWAP; the analytic gradient matches finite differences; the closed-form permanent cost equals the interval sum; the KKT residual is ~0 and flags a non-optimal schedule; on its own model it beats TWAP, VWAP, the linearised AC schedule and random feasible perturbations; raising `lam` front-loads and trades expected cost for variance.
- **Evaluation**: unchanged from slice 2 - calibration seed 0, evaluation seed 123 (4,000 days), every cost charged (5 bps spread, sqrt temporary, linear permanent), paired differences on common random numbers.
- **Equal risk**: the bisection hits AC's realised std to 1e-6 relative; when even the risk-neutral SQRT-OPT is less risky than AC (AC kT=0.5), the row is flagged "lower risk" rather than forced.

### Results (`python scripts/run_execution_comparison.py`)

Same `lam` as AC, buy 5% ADV (bps; paired vs TWAP on the same days):

| schedule | mean IS | std | p95 | vs TWAP (paired) | temporary | pre-trade est. |
|---|---|---|---|---|---|---|
| AC(kT=1) | 29.14 | 105.2 | 196.5 | -0.08 ± 0.13 | 24.34 | 27.70 |
| SQRT-OPT(kT=1) | 29.27 | **88.0** | 170.0 | +0.06 ± 0.42 | 24.61 | 27.95 |
| AC(kT=3) | 31.67 | 76.2 | 154.1 | +2.45 ± 0.69 | 27.06 | 30.36 |
| SQRT-OPT(kT=3) | 39.78 | **45.7** | 114.8 | +10.56 ± 1.18 | 35.69 | 38.76 |
| POV(6.25%), idealised | 29.13 | 97.1 | 186.3 | -0.09 ± 0.47 | 24.42 | - |
| POV-lag1(6.25%) | 30.83 | 96.4 | 185.4 | **+1.61 ± 0.44** | 26.15 | - |

At the same `lam` the two models sit at *different* frontier points, so this table alone does not say which schedule is better. Equal realised risk does:

| buy | AC kT | AC std | AC mean | SQRT-OPT mean | saving at equal risk (paired) | realised E + lam Var, AC -> SQRT-OPT |
|---|---|---|---|---|---|---|
| 2% | 1 | 105.2 | 19.47 | 19.00 | +0.47 ± 0.15 | -5% |
| 2% | 4 | 65.5 | 22.42 | 21.79 | +0.62 ± 0.15 | -23% |
| 5% | 1 | 105.2 | 29.14 | 28.39 | +0.75 ± 0.16 | -5% |
| 5% | 3 | 76.2 | 31.67 | 30.80 | +0.87 ± 0.15 | -19% |
| 5% | 4 | 65.5 | 33.92 | 32.98 | +0.95 ± 0.15 | -23% |
| 10% | 1 | 105.3 | 40.42 | 39.35 | +1.07 ± 0.16 | -5% |
| 10% | 4 | 65.6 | 47.27 | 45.96 | +1.31 ± 0.15 | -23% |

AC(kT=0.5) is dominated outright: the risk-neutral SQRT-OPT has both lower std (108.5 vs 110.4) and lower cost (+0.50 / +0.79 / +1.12 bps at 2 / 5 / 10%).

Exponent misspecification, buy 5% ADV, saving vs AC at equal realised risk:

| true delta | NLS delta_hat | kT | SQRT-OPT saving | POWER-OPT saving | SQRT - POWER |
|---|---|---|---|---|---|
| 0.35 | 0.380 ± 0.029 | 1 / 3 | +0.73 / +0.94 | +0.73 / +0.95 | 0.000 / +0.005 |
| 0.50 | 0.550 ± 0.047 | 1 / 3 | +0.75 / +0.87 | +0.75 / +0.87 | 0.000 / -0.001 |
| 0.65 | 0.727 ± 0.076 | 1 / 3 | +0.70 / +0.73 | +0.70 / +0.72 | 0.000 / -0.003 |
| 0.80 | 0.911 ± 0.118 | 1 / 3 | +0.61 / +0.57 | +0.61 / +0.57 | 0.000 / -0.001 |

(Savings SE ~0.15 bps throughout.)

Reading:
- **At equal risk the profile-aware optimum is cheaper than AC by 0.47-1.31 bps (3-8.5 SE), growing with order size and urgency.** The saving is about the size of the VWAP-vs-TWAP saving (0.53 / 0.83 / 1.18 bps), so most of it comes from *where* in the day it trades (the U-profile), not from the concavity of impact.
- **The exponent barely matters for the schedule.** Planning with the estimated exponent instead of 0.5 changes cost by at most 0.005 bps across true exponents 0.35-0.8, even though NLS over-estimates the exponent by 0.03-0.11. The saving vs AC survives every misspecification tested.
- **Same `lam` is not the same urgency.** AC's linearised impact over-charges fast trading, so at a given `lam` it under-reacts; SQRT-OPT at AC's `lam` cuts std 16% (kT=1) or 40% (kT=3) and pays for it in impact. Matching on realised risk needs SQRT-OPT's `lam` 4.6-7.9x smaller (5% ADV).
- **Implementable POV costs 1.1-2.4 bps more than the idealised one** (paired vs TWAP +0.98 / +1.61 / +2.32 vs -0.09 / -0.09 / -0.10 at 2 / 5 / 10%). Profile-scaled previous-interval volume is a noisy forecast (interval noise 30%), and participation errors are charged through the concave impact. The idealised POV's apparent parity with TWAP was look-ahead.
- **Pre-trade estimates stay accurate for SQRT-OPT** (within 0.75 bps of realised spread + temporary + permanent at every size and urgency).

### Weaknesses (honest)

- The "truth" is still the simulator's own model, and the planner's cost model is that same functional form; real books have queue dynamics, resilience and adverse selection, which would shrink or reverse a ~1 bps edge.
- Static schedules only: no re-optimisation on realised volume or price (an adaptive / dynamic-programming policy is the natural next slice). POV-lag1 uses one naive forecast, not a fitted volume model.
- Equal-risk savings are ~5 SE with 4,000 paired days; a single seed pair. The kT grid is coarse (0.5-4).
- The misspecification study varies only the exponent; a wrong volume profile (e.g. event days) is the more likely real-world failure and is untested.
- NLS exponent estimates are biased upwards at every true exponent; the binned estimator was not used for planning.

## Weaknesses / next slices (slice 1)

- Synthetic factor model only - no real tape, venue fees, or overnight gaps.
- ADV and vol treated as constants; no intraday schedule or liquidity regime.
- Impact is temporary in the PnL path (charged on trade) without permanent
  price trajectory feedback.
- Single-name weight path; no multi-asset capacity or cross-impact.
- Sharpe uses full-sample stats (no purged CV / deflated Sharpe). *(Slice 5 adds walk-forward selection and the deflated Sharpe.)*

## References

- Almgren, R., Chriss, N. (2000). Optimal execution of portfolio transactions. *Journal of Risk* 3(2).
- Almgren, R., Thum, C., Hauptmann, E., Li, H. (2005). Direct estimation of equity market impact. *Risk* 18(7).
- Tóth, B. et al. (2011). Anomalous price impact and the critical nature of liquidity in financial markets. *Physical Review X* 1, 021006.
- Perold, A. F. (1988). The implementation shortfall: paper versus reality. *Journal of Portfolio Management* 14(3).
