# Slice 7: CUSUM gate (ARL 2,520) under an IC decay (0.10 -> 0) and flip (0.10 -> -0.05) at bar 2,520

Seeds 900-919; WF-PARTIAL; costs: slice-1 defaults.

## decay market (means across seeds)

| arm | post bps/day | post cost bps/day | OOS net Sharpe | pre-switch net Sharpe | on pre | on post |
|---|---|---|---|---|---|---|
| ungated | -0.822 | 0.637 | 0.072 | 0.820 | 1.000 | 1.000 |
| G252 | -0.390 | 0.287 | 0.140 | 0.661 | 0.789 | 0.407 |
| CUSUM | -0.254 | 0.123 | 0.170 | 0.563 | 0.749 | 0.190 |

## flip market (means across seeds)

| arm | post bps/day | post cost bps/day | OOS net Sharpe | pre-switch net Sharpe | on pre | on post |
|---|---|---|---|---|---|---|
| ungated | -2.073 | 0.539 | -0.247 | 0.820 | 1.000 | 1.000 |
| G252 | -0.465 | 0.148 | 0.135 | 0.661 | 0.789 | 0.248 |
| CUSUM | -0.211 | 0.065 | 0.201 | 0.563 | 0.749 | 0.080 |

## control market (means across seeds)

| arm | post bps/day | post cost bps/day | OOS net Sharpe | pre-switch net Sharpe | on pre | on post |
|---|---|---|---|---|---|---|
| ungated | 2.459 | 0.811 | 0.863 | 0.820 | 1.000 | 1.000 |
| G252 | 2.022 | 0.681 | 0.767 | 0.661 | 0.789 | 0.810 |
| CUSUM | 1.686 | 0.578 | 0.659 | 0.563 | 0.749 | 0.701 |

## Paired tests (gate - ungated)

| gate | decay post bps/day (one-sided) | flip post bps/day (one-sided) | control OOS Sharpe |
|---|---|---|---|
| G252 | +0.432 ± 0.180, p = 0.014, 16/20 | +1.609 ± 0.229, p = 0.000, 19/20 | -0.096 ± 0.023, p = 0.001, 2/20 |
| CUSUM | +0.568 ± 0.200, p = 0.005, 14/20 | +1.863 ± 0.246, p = 0.000, 19/20 | -0.204 ± 0.055, p = 0.002, 1/20 |

H8 pass: **True**; H9 pass: **True**; control within budget: **False**; recommend the CUSUM gate: **False**.
CUSUM first off after the switch: decay median 266.5 bars (never off on 0 seeds); flip median 164.0 bars (never off on 0).
