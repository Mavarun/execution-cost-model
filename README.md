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
- Sharpe uses full-sample stats (no purged CV / deflated Sharpe).

## References

- Almgren, R., Chriss, N. (2000). Optimal execution of portfolio transactions. *Journal of Risk* 3(2).
- Almgren, R., Thum, C., Hauptmann, E., Li, H. (2005). Direct estimation of equity market impact. *Risk* 18(7).
- Tóth, B. et al. (2011). Anomalous price impact and the critical nature of liquidity in financial markets. *Physical Review X* 1, 021006.
- Perold, A. F. (1988). The implementation shortfall: paper versus reality. *Journal of Portfolio Management* 14(3).
