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
```

## Layout

- `src/execution_cost/costs.py` - cost primitives
- `src/execution_cost/portfolio.py` - synthetic signal + cost application
- `src/execution_cost/sensitivity.py` - turnover / hold-bars sweep
- `src/execution_cost/metrics.py` - Sharpe, turnover helpers
- `scripts/run_cost_slice.py` - CLI for baseline + cost curve
- `tests/` - unit and hypothesis checks

## Weaknesses / next slices

- Synthetic factor model only - no real tape, venue fees, or overnight gaps.
- ADV and vol treated as constants; no intraday schedule or liquidity regime.
- Impact is temporary in the PnL path (charged on trade) without permanent
  price trajectory feedback.
- Single-name weight path; no multi-asset capacity or cross-impact.
- Sharpe uses full-sample stats (no purged CV / deflated Sharpe).
