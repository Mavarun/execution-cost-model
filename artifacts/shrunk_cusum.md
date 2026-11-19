# Shrunk-reference CUSUM gate (slice 8)

Seeds 1000-1019; shrink 0.75; protocol frozen in `cusum_eval.SHRINK_PROTOCOL`.

| arm | decay: post bps/day | decay: on post | flip: post bps/day | control: OOS net Sharpe | control: on after bar 2,520 |
|---|---|---|---|---|---|
| ungated WF-PARTIAL | -0.571 | 1.000 | -1.665 | 0.878 | 1.000 |
| CUSUM (slice 7) | -0.213 | 0.189 | -0.225 | 0.665 | 0.641 |
| CUSUM-S0.75 | -0.247 | 0.217 | -0.216 | 0.706 | 0.698 |

| gate - ungated | decay post bps/day (one-sided) | flip post bps/day (one-sided) | control OOS Sharpe |
|---|---|---|---|
| CUSUM (slice 7) | +0.358 ± 0.150 (p = 0.014, 13/20) | +1.440 ± 0.166 (p = 0.000, 19/20) | -0.213 ± 0.044 (p = 0.000, 2/20) |
| CUSUM-S0.75 | +0.324 ± 0.140 (p = 0.016, 13/20) | +1.449 ± 0.169 (p = 0.000, 19/20) | -0.172 ± 0.039 (p = 0.000, 2/20) |

CUSUM-S0.75 first off after the switch: decay median 228.0 bars (1 never), flip median 150.0 bars (1 never).
H10 (control cost < 0.1): **False**; H11 (post-decay gain): **True**; recommend the shrunk gate: **False**.
