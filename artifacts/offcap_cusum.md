# Shrunk CUSUM gate with an off-spell cap (slice 11)

Seeds 1300-1359; shrink 0.75; max_off 126; frozen in `cusum_eval.OFFCAP_PROTOCOL`.

| arm | decay: post bps/day | decay: on post | flip: post bps/day | control: OOS net Sharpe | control: on after bar 2,520 | control: switches |
|---|---|---|---|---|---|---|
| ungated WF-PARTIAL | -0.481 | 1.000 | -1.679 | 0.958 | 1.000 | 0.0 |
| CUSUM-S0.75 (slice 8/10) | -0.197 | 0.264 | -0.296 | 0.873 | 0.815 | 1.9 |
| CUSUM-S0.75-C126 | -0.419 | 0.839 | -1.305 | 0.935 | 0.958 | 2.4 |

| gate - ungated | decay post bps/day (one-sided) | flip post bps/day (one-sided) | control OOS Sharpe | control cost 95% CI |
|---|---|---|---|---|
| CUSUM-S0.75 (slice 8/10) | +0.283 ± 0.095 (p = 0.002, 37/60) | +1.383 ± 0.124 (p = 0.000, 56/60) | -0.085 ± 0.015 (p = 0.000, 2/60) | [0.054, 0.115] |
| CUSUM-S0.75-C126 | +0.061 ± 0.043 (p = 0.081, 33/60) | +0.374 ± 0.059 (p = 0.000, 49/60) | -0.023 ± 0.008 (p = 0.007, 14/60) | [0.007, 0.039] |

H16 (control cost CI upper end < 0.1): **True**; H17 (post-decay gain): **False**; H18 (keeps half the uncapped gain): **False**; recommend the capped gate: **False**.
