# Slice 6: trailing-performance gate under an IC decay (IC 0.10 -> 0 at bar 2,520)

Seeds 800-819; WF-PARTIAL; costs: slice-1 defaults.

## decay market (means across seeds)

| arm | post bps/day | post cost bps/day | OOS net Sharpe | pre-switch net Sharpe | on pre | on post |
|---|---|---|---|---|---|---|
| ungated | -0.269 | 0.567 | 0.295 | 1.068 | 1.000 | 1.000 |
| G252 | -0.377 | 0.295 | 0.263 | 0.963 | 0.852 | 0.501 |
| G126 | -0.221 | 0.295 | 0.288 | 0.920 | 0.768 | 0.486 |
| G504 | -0.275 | 0.280 | 0.322 | 0.995 | 0.927 | 0.497 |

## control market (means across seeds)

| arm | post bps/day | post cost bps/day | OOS net Sharpe | pre-switch net Sharpe | on pre | on post |
|---|---|---|---|---|---|---|
| ungated | 2.956 | 0.746 | 1.058 | 1.068 | 1.000 | 1.000 |
| G252 | 2.396 | 0.641 | 0.955 | 0.963 | 0.852 | 0.833 |
| G126 | 2.193 | 0.589 | 0.925 | 0.920 | 0.768 | 0.748 |
| G504 | 2.807 | 0.704 | 1.025 | 0.995 | 0.927 | 0.927 |

## Paired tests (gate - ungated)

| gate | decay post bps/day (one-sided) | control OOS Sharpe | control post bps/day |
|---|---|---|---|
| G252 | -0.108 ± 0.111, p = 0.828, 7/20 | -0.104 ± 0.021, p = 0.000, 3/20 | -0.560 ± 0.076, p = 0.000, 0/20 |
| G126 | +0.048 ± 0.113, p = 0.337, 9/20 | -0.133 ± 0.026, p = 0.000, 3/20 | -0.763 ± 0.092, p = 0.000, 1/20 |
| G504 | -0.006 ± 0.099, p = 0.523, 7/20 | -0.033 ± 0.010, p = 0.004, 5/20 | -0.149 ± 0.032, p = 0.000, 2/20 |

H6 pass: **False**; control within budget: **False**; recommend G252: **False**.
G252 detection delay after the switch: median 94.5 bars; never off on 0 seeds.
