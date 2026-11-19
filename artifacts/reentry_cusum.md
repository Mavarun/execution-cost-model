# Fast re-entry CUSUM gate (slice 9)

Seeds 1100-1119; shrink 0.75, on_frac 0.25; frozen in `cusum_eval.REENTRY_PROTOCOL`.

| arm | decay: post bps/day | decay: on post | flip: post bps/day | control: OOS net Sharpe | control: on after bar 2,520 | control: switches |
|---|---|---|---|---|---|---|
| ungated WF-PARTIAL | -0.669 | 1.000 | -1.903 | 0.930 | 1.000 | 0.0 |
| CUSUM-S0.75 (slice 8) | -0.317 | 0.320 | -0.378 | 0.879 | 0.857 | 1.1 |
| CUSUM-S0.75-R0.25 | -0.643 | 0.833 | -1.262 | 0.916 | 0.980 | 1.6 |

| gate - ungated | decay post bps/day (one-sided) | flip post bps/day (one-sided) | control OOS Sharpe |
|---|---|---|---|
| CUSUM-S0.75 (slice 8) | +0.352 ± 0.212 (p = 0.057, 13/20) | +1.526 ± 0.262 (p = 0.000, 20/20) | -0.051 ± 0.017 (p = 0.009, 1/20) |
| CUSUM-S0.75-R0.25 | +0.026 ± 0.095 (p = 0.393, 8/20) | +0.641 ± 0.216 (p = 0.004, 14/20) | -0.014 ± 0.005 (p = 0.018, 1/20) |

H12 (control cost < 0.1): **True**; H13 (post-decay gain): **False**; recommend the fast re-entry gate: **False**.
