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
- AC is solved for *linear* impact and mapped from the sqrt law at the TWAP rate. It does not know the intraday volume profile, which is why AC(kT=1) is not costlier than TWAP. A nonlinear, profile-aware optimiser is the next step.
- POV is idealised: it sees current-interval volume without lag.
- Single stock, single day horizon, constant intraday vol, no intraday vol smile, no cross-impact.
- The first-slice limits still apply to the portfolio-level Sharpe analysis (full-sample stats, synthetic factor).

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
