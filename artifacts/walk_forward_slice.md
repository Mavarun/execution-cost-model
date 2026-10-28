# Slice 5: walk-forward volume-model refit across a regime break

40 blocks x 50 days, break at block 10 (post-break: U-shape 1.5 -> 3.0, surge days 15% -> 30% in the main scenario); q = 5% ADV SURGE-POV, 5 seed pairs; spread, impact and opportunity cost charged.

## profile + events break

| model | pre IS bps | post IS bps | pre log RMSE | post log RMSE | post completion |
|---|---:|---:|---:|---:|---:|
| FIXED | 29.17 | 31.43 | 0.348 | 0.406 | 0.996 |
| ROLLING-100 | 29.16 | 30.66 | 0.351 | 0.375 | 0.996 |
| ROLLING-250 | 29.16 | 30.70 | 0.349 | 0.375 | 0.996 |
| EXPANDING | 29.17 | 31.04 | 0.348 | 0.387 | 0.996 |

| paired vs FIXED (bps, pooled ± SE) | pre | post | reps better (post) |
|---|---:|---:|---:|
| ROLLING-100 - FIXED | -0.01 ± 0.03 | -0.77 ± 0.07 | 5/5 |
| ROLLING-250 - FIXED | -0.01 ± 0.01 | -0.73 ± 0.06 | 5/5 |
| EXPANDING - FIXED | +0.00 ± 0.00 | -0.39 ± 0.03 | 5/5 |

## no break (control)

| model | pre IS bps | post IS bps | pre log RMSE | post log RMSE | post completion |
|---|---:|---:|---:|---:|---:|
| FIXED | 29.17 | 30.66 | 0.348 | 0.347 | 0.996 |
| ROLLING-100 | 29.16 | 30.68 | 0.351 | 0.349 | 0.995 |
| ROLLING-250 | 29.16 | 30.68 | 0.349 | 0.348 | 0.995 |
| EXPANDING | 29.17 | 30.66 | 0.348 | 0.347 | 0.996 |

| paired vs FIXED (bps, pooled ± SE) | pre | post | reps better (post) |
|---|---:|---:|---:|
| ROLLING-100 - FIXED | -0.01 ± 0.03 | +0.02 ± 0.02 | 1/5 |
| ROLLING-250 - FIXED | -0.01 ± 0.01 | +0.01 ± 0.02 | 1/5 |
| EXPANDING - FIXED | +0.00 ± 0.00 | -0.00 ± 0.00 | 2/5 |

## profile-only break

| model | pre IS bps | post IS bps | pre log RMSE | post log RMSE | post completion |
|---|---:|---:|---:|---:|---:|
| FIXED | 29.17 | 31.47 | 0.348 | 0.387 | 0.996 |
| ROLLING-100 | 29.16 | 30.65 | 0.351 | 0.352 | 0.995 |
| ROLLING-250 | 29.16 | 30.71 | 0.349 | 0.352 | 0.995 |
| EXPANDING | 29.17 | 31.07 | 0.348 | 0.366 | 0.995 |

| paired vs FIXED (bps, pooled ± SE) | pre | post | reps better (post) |
|---|---:|---:|---:|
| ROLLING-100 - FIXED | -0.01 ± 0.03 | -0.81 ± 0.07 | 5/5 |
| ROLLING-250 - FIXED | -0.01 ± 0.01 | -0.76 ± 0.07 | 5/5 |
| EXPANDING - FIXED | +0.00 ± 0.00 | -0.40 ± 0.03 | 5/5 |

## events-only break

| model | pre IS bps | post IS bps | pre log RMSE | post log RMSE | post completion |
|---|---:|---:|---:|---:|---:|
| FIXED | 29.17 | 30.62 | 0.348 | 0.371 | 0.996 |
| ROLLING-100 | 29.16 | 30.66 | 0.351 | 0.373 | 0.996 |
| ROLLING-250 | 29.16 | 30.65 | 0.349 | 0.372 | 0.996 |
| EXPANDING | 29.17 | 30.63 | 0.348 | 0.370 | 0.996 |

| paired vs FIXED (bps, pooled ± SE) | pre | post | reps better (post) |
|---|---:|---:|---:|
| ROLLING-100 - FIXED | -0.01 ± 0.03 | +0.04 ± 0.02 | 2/5 |
| ROLLING-250 - FIXED | -0.01 ± 0.01 | +0.03 ± 0.02 | 1/5 |
| EXPANDING - FIXED | +0.00 ± 0.00 | +0.01 ± 0.01 | 1/5 |
