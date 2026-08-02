# Entry Statistics V1 Decision

## Status and approved source

Entry Statistics V1 is complete. This decision is based on the official,
provenance-validated observational analysis with the following lineage:

- Entry Statistics stamp: `20260802_103007`
- Trade Timing Attribution stamp: `20260802_085312`
- Source stop-walk-forward stamp: `20260802_081449`
- Frozen research snapshot: `20260802_081049_d93145cb1dcf`
- Population: 256 completed `FIXED_BASELINE` trades across 13 independently
  reset out-of-sample portfolio windows

The controlled development basket remains AAPL, AMZN, GOOGL, META, MSFT,
NVDA, TSLA, BTC-USD, and ETH-USD. It is not the final production universe.

## Approved conclusions

1. Entry Statistics V1 is complete.
2. No active strategy, entry, execution, risk, fee, or portfolio rule changes
   result from this analysis.
3. Mean raw entry gap was approximately `+0.241558%`; median raw entry gap was
   approximately `+0.017782%`.
4. Winner and loser entry-gap distributions did not provide useful general
   separation.
5. Do not introduce a directional positive-gap filter.
6. Absolute large entry gaps are a future research candidate only.
7. The below `-2%` and at-or-above `+2%` samples were weak, but their samples
   were small and must not be converted directly into a baseline rule.
8. ATR-based entry volatility is a future diagnostic candidate only.
9. `signal_score` remains a deterministic ranking field, not a calibrated
   success probability or an entry threshold.
10. MACD remains diagnostic only.
11. `FORCE_CLOSE_END` materially affects outcome summaries and must remain
    separately identified.
12. Stocks and crypto have structurally different gap behavior.
13. The approved active baseline remains unchanged.

## Future research candidates — not implemented

- `ABSOLUTE_ENTRY_GAP_SHOCK`
- `ENTRY_VOLATILITY_ATR`

These are observational research candidates only. They do not authorize an
entry filter, parameter change, ticker-specific behavior, or alteration to the
active portfolio baseline.

## Scope boundary

Entry Statistics V1 is attribution and diagnostic research. It does not
reorder the approved research-to-paper-trading pathway, advance paper trading,
or authorize real-money execution.
