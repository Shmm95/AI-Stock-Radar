# Forward Return Statistics V1 Decision

## Status and approved source

Forward Return Statistics V1 is complete. This is observational portfolio
research based on the following approved, provenance-validated lineage:

- Forward Return Statistics stamp: `20260802_134341`
- Entry Statistics stamp: `20260802_103007`
- Trade Timing Attribution stamp: `20260802_085312`
- Source stop-walk-forward stamp: `20260802_081449`
- Frozen research snapshot: `20260802_081049_d93145cb1dcf`
- Model: `FIXED_BASELINE`
- Population: 256 completed trades across 13 independently reset-capital
  out-of-sample windows
- Controlled basket: AAPL, AMZN, GOOGL, META, MSFT, NVDA, TSLA, BTC-USD, and
  ETH-USD

Horizon availability was H1=256, H3=255, H5=251, and H10=239. The primary
population includes `FORCE_CLOSE_END`; the named sensitivity excludes only
`FORCE_CLOSE_END` (42 excluded, 214 remaining).

## Approved methodology

- Entry occurs at the Open of the ticker-local entry row.
- `terminal_index = entry_index + horizon - 1`.
- H1 uses the entry bar.
- Horizons are 1, 3, 5, and 10 ticker-local completed bars.
- Returns use the actual simulated entry fill as the denominator.
- High and Low results are observational OHLC excursions, not executable
  MFE/MAE.
- Post-exit market outcomes are counterfactual observations, not realized
  trade returns.
- Test-window boundaries are never crossed.
- Unavailable horizons remain in the population with missing outcomes.
- The primary population includes `FORCE_CLOSE_END`; sensitivity excludes only
  `FORCE_CLOSE_END`.
- The portfolio engine is not replayed.

## Principal observational findings

- H1 mean return was approximately `-0.36%`; median approximately `-0.16%`.
- H3 mean was approximately `-0.28%`; median approximately `+0.04%`.
- H5 mean was approximately `-0.30%`; median approximately `+0.29%`.
- H10 mean was approximately `-0.24%`; median approximately `+0.81%`.
- `FORCE_CLOSE_END`-excluded means remained negative at all four horizons.
- Final winners and losers begin separating early, but winner/loser labels are
  future outcome labels and cannot be used as entry rules.
- Most stop exits do not rapidly recover above entry in the observed H10
  counterfactual path; this does not support loosening the stop.
- Absolute entry gaps of at least 1% show materially weaker forward outcomes
  than gaps below 1%, but this is post-hoc and requires a controlled ablation.
- High ATR observations appear weaker, but the evidence is less robust than
  the absolute-gap finding.
- Results show ticker and year dependence, including a weak 2022 period;
  therefore no ticker removal or broad rule change is authorized.

## Approved baseline decisions

1. Current next-Open entry execution remains unchanged.
2. Baseline entry logic remains unchanged.
3. Stop loss must not be loosened based on these results.
4. Trailing-stop and exit rules remain unchanged pending MFE/MAE and exit
   attribution.
5. No positive-gap filter is approved.
6. Signal score must not be converted into a confidence threshold.
7. MACD remains diagnostic-only.
8. No ticker is removed.
9. No regime, execution, risk, sizing, or portfolio rule changes are
   authorized.
10. Forward Return Statistics is observational and does not establish
    causality.
11. The approved baseline remains frozen.

## Future research candidates — not implemented

- `ABSOLUTE_ENTRY_GAP_SHOCK` is the strongest future controlled-ablation
  candidate.
- `ENTRY_VOLATILITY_ATR` remains a secondary research candidate.

These candidates do not authorize an entry filter, parameter change,
ticker-specific behavior, or a change to the approved development pathway.
