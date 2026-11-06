# Slice 6, H7: vol-targeted selectors (10% annualised, leverage solved on training bars)

Seeds 900-919; costs re-charged on levered trades.

| selector | vol-targeted net Sharpe | unlevered net Sharpe | realised vol / target | mean leverage | net bps/day | cost bps/day |
|---|---|---|---|---|---|---|
| WF-PARTIAL | 0.837 | 0.863 | 0.996 | 1.46 | 3.280 | 1.249 |
| WF-BAND | 0.691 | 0.845 | 1.008 | 4.09 | 2.754 | 1.560 |
| WF-PB | 0.829 | 0.879 | 1.002 | 2.62 | 3.276 | 1.161 |

H7 (WF-PB - WF-PARTIAL, two-sided): -0.008 ± 0.016, p = 0.619, 8/20 seeds; pass: **False**.
WF-BAND - WF-PARTIAL: -0.146 ± 0.038, p = 0.001, 5/20.
Recommended rule: **PARTIAL**.
