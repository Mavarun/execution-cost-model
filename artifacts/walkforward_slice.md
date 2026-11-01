# Slice 5: walk-forward cost-aware rebalancing

Evaluation seeds 500-519 (IC 0.10), null seeds 600-619 (IC 0). OOS = 15 annual folds (3,780 bars) per seed.

## OOS means across seeds (annualised Sharpe; cost in bps/day)

| selector | gross Sharpe | net Sharpe (sd) | cost | turnover | mean abs weight | null net Sharpe |
|---|---|---|---|---|---|---|
| WF-PARTIAL | 1.34 | 1.02 (0.31) | 0.94 | 0.069 | 0.388 | -0.11 |
| WF-BAND | 1.26 | 0.92 (0.34) | 0.51 | 0.026 | 0.199 | -0.23 |
| WF-PB | 1.32 | 1.03 (0.29) | 0.61 | 0.041 | 0.283 | -0.11 |
| WF-ALL | 1.31 | 1.02 (0.29) | 0.61 | 0.041 | 0.283 | -0.11 |
| FULL | 1.44 | -1.16 (0.36) | 10.93 | 0.382 | 0.589 | -2.62 |

## Pre-registered hypotheses

- **H1**: mean diff +0.015 ± 0.018, t 0.84, p 0.2067, wins 11/20 -> FAIL
- **H2**: mean diff -0.096 ± 0.041, t -2.32, p 0.0314, wins 4/20 -> PASS
- **H3**: mean diff +0.053 ± 0.028, t 1.92, p 0.0352, wins 12/20 -> PASS
- **H4**: null WF-ALL OOS net Sharpe -0.110 ± 0.053; null in-sample best -0.008; DSR < 0.95 on 20/20 null seeds; DSR > 0.95 on 4/20 IC-0.10 seeds -> PASS
- Recommended family: **PARTIAL**

## Pooled Newey-West daily net-return differences (bps/day)

Raw return differences, not risk-matched: selectors hold different gross exposure (see mean abs weight), so these measure P&L per unit of capital, not Sharpe.

| pair | mean | SE | t |
|---|---|---|---|
| WF-PB - WF-PARTIAL | -0.779 | 0.065 | -11.91 |
| WF-BAND - WF-PARTIAL | -1.664 | 0.110 | -15.15 |
| WF-ALL - WF-PARTIAL | -0.793 | 0.067 | -11.88 |
| WF-PARTIAL - FULL | +7.887 | 0.171 | 46.23 |

## Selection bias

Full-sample best of 42: net Sharpe 1.08 in sample; the same rule on the OOS window 1.12; WF-ALL OOS 1.02. Null: in-sample best -0.01, WF-ALL OOS -0.11.

## Sensitivity (secondary; same evaluation seeds)

| setting | FULL | WF-PARTIAL | WF-BAND | WF-PB | WF-ALL | PB - PARTIAL | BAND - PARTIAL |
|---|---|---|---|---|---|---|---|
| costs x0.5 | 0.12 | 1.19 | 1.12 | 1.19 | 1.18 | -0.005 ± 0.011 (8/20) | -0.074 ± 0.027 (5/20) |
| costs x2 | -3.50 | 0.77 | 0.79 | 0.83 | 0.82 | +0.053 ± 0.022 (14/20) | +0.015 ± 0.030 (9/20) |
| IC 0.05 | -1.84 | 0.39 | 0.37 | 0.41 | 0.39 | +0.020 ± 0.017 (10/20) | -0.021 ± 0.033 (7/20) |
| rho 0.70 (half-life 1.9d) | -2.15 | 0.53 | 0.62 | 0.68 | 0.67 | +0.150 ± 0.032 (17/20) | +0.082 ± 0.044 (14/20) |
| rho 0.97 (half-life 22.8d) | -0.80 | 1.28 | 1.16 | 1.26 | 1.26 | -0.020 ± 0.017 (7/20) | -0.120 ± 0.036 (4/20) |
