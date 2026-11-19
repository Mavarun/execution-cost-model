# Shrunk CUSUM gate on 60 fresh seeds (slice 10)

Seeds 1200-1259; shrink 0.75; frozen in `cusum_eval.POOLED_PROTOCOL`.

| arm | decay: post bps/day | decay: on post | flip: post bps/day | control: OOS net Sharpe | control: on after bar 2,520 |
|---|---|---|---|---|---|
| ungated WF-PARTIAL | -0.691 | 1.000 | -1.966 | 0.919 | 1.000 |
| CUSUM (slice 7) | -0.155 | 0.206 | -0.263 | 0.790 | 0.673 |
| CUSUM-S0.75 (slice 8) | -0.170 | 0.261 | -0.301 | 0.823 | 0.771 |

| gate - ungated | decay post bps/day (one-sided) | flip post bps/day (one-sided) | control OOS Sharpe |
|---|---|---|---|
| CUSUM (slice 7) | +0.535 ± 0.102 (p = 0.000, 44/60) | +1.703 ± 0.118 (p = 0.000, 57/60) | -0.129 ± 0.019 (p = 0.000, 5/60) |
| CUSUM-S0.75 (slice 8) | +0.521 ± 0.097 (p = 0.000, 39/60) | +1.665 ± 0.119 (p = 0.000, 56/60) | -0.096 ± 0.018 (p = 0.000, 5/60) |

CUSUM-S0.75 control Sharpe change 95% CI [-0.133, -0.060].
H14 (control cost < 0.1): **True**; H15 (post-decay gain): **True**; recommend the shrunk gate: **True**.
