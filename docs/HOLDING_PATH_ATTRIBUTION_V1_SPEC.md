# Portfolio Holding-Path Attribution V1 Specification

## Status and purpose

Portfolio Holding-Path Attribution V1 is an observational research analyzer.
It measures the ticker-local price path while each already-completed
`FIXED_BASELINE` trade was exposed. It does not replay the portfolio engine and
does not alter signals, entries, exits, event ordering, costs, risk, position
limits, stop parameters, or the active baseline.

This specification authorizes only the analyzer, its tests, this document, and
the corresponding Current Stage update. A passing result is not authority for
a strategy or production change.

## Locked source lineage

The only approved source chain is:

- Forward Return Statistics: `20260802_134341`
- Entry Statistics: `20260802_103007`
- Trade Timing Attribution: `20260802_085312`
- Stop Walk-Forward: `20260802_081449`
- Snapshot: `20260802_081049_d93145cb1dcf`
- Snapshot fingerprint:
  `d93145cb1dcf3f837415f5b28ac29eeb51c13bc8ce5b8b1187aa34ebfc01ad44`
- Population: 256 completed `FIXED_BASELINE` trades in 13 independently
  reset out-of-sample windows

The analyzer verifies all Forward Return Statistics result hashes, all
upstream hashes declared by its provenance, the approved Forward Return
Statistics code and test hashes, and the frozen snapshot. It rechecks every
consumed source hash immediately before saving.

## Exposure-aware exit semantics

### Open exits

`EXIT_SIGNAL_NEXT_OPEN` and `GAP_STOP_LOSS` include the raw exit Open. The exit
bar High, Low, and Close occur after the modeled exit and are excluded from the
trade path.

### Intrabar stops

`STOP_LOSS` includes the raw exit Open and the known configured stop touch as
the certain/censored path. Daily OHLC does not reveal whether the exit-bar High
or Low occurred before the stop. Therefore:

- certain/censored MFE and MAE use completed exposed bars, exit Open, and the
  known stop touch;
- possible MFE may use the exit-bar High as an upper bound;
- possible MAE may use the exit-bar Low as a lower bound;
- the exit bar is marked `INTRABAR_STOP_BOUNDED` and intrabar ordering remains
  explicitly uncertain;
- the bounded values must never be described as exact executable MFE or MAE.

### End-of-window closes

`FORCE_CLOSE_END` can carry a union-calendar timestamp that is not a ticker
bar. It maps to the last real ticker bar at or before that timestamp inside the
test window and includes the complete bar through its Close. No ticker bar is
fabricated.

## Path and metric definitions

The entry fill anchors both excursions at zero, so censored MFE cannot be
negative and censored MAE cannot be positive.

For an extreme price `P` and entry fill `E`:

```text
excursion_percent = (P / E - 1) * 100
excursion_R       = excursion_percent / initial_stop_percent
```

Trade outputs include:

- certain/censored and possible MFE/MAE prices, percentages, and R multiples;
- explicit bound widths and exactness flags;
- ticker bars and timestamps to censored MFE and MAE;
- `MFE_BEFORE_MAE`, `MAE_BEFORE_MFE`, or `AMBIGUOUS_SAME_BAR` ordering;
- net realized return and gross/net realized R;
- giveback from censored MFE, MFE capture ratio, recovery from censored MAE,
  and exit location in the censored range;
- maximum/minimum close excursion, maximum close drawdown, and underwater
  close-bar frequency;
- descriptive holding buckets `0`, `1_2`, `3_5`, `6_10`, `11_20`, `21_40`,
  and `41_PLUS`.

The path-row output records each ticker bar from entry through effective exit,
raw OHLC, inclusion flags, the known stop touch where applicable, and
cumulative certain and possible excursions.

## Populations and aggregations

The primary population contains all 256 completed baseline trades. The sole
sensitivity population excludes raw `FORCE_CLOSE_END` trades and keeps all
other trades unchanged.

Long-form population statistics use population standard deviation (`ddof=0`)
and linear-interpolated percentiles. Tables cover overall results, asset
classes, tickers, windows, exit reasons, exit categories, outcomes, holding
buckets, extreme order, exit semantics, and the force-close-excluded
sensitivity.

Holding buckets are fixed descriptive strata. They are not tested thresholds
and cannot select a strategy parameter.

## Required artifacts

A saving run emits:

- trade-level attribution CSV;
- ticker-bar path CSV;
- quality-screen CSV;
- all aggregation CSVs;
- machine-readable JSON methodology/results;
- provenance JSON hashing every other generated artifact and every consumed
  source (the provenance file cannot self-hash).

Output filenames are stamp-specific and existing files are never overwritten.

## Acceptance criteria

The implementation must prove that:

1. open exits exclude exit-bar High/Low/Close;
2. intrabar stops report certain values and possible bounds separately;
3. same-bar extremes remain ambiguous;
4. union-calendar force closes map to the last real ticker bar;
5. ticker-bar holding counts and path rows reconcile exactly;
6. MFE/MAE anchor and bound invariants hold;
7. primary and force-close-excluded populations remain separate;
8. source order and identifiers remain unchanged;
9. saving requires strict official provenance and rejects mutation/overwrite;
10. `--no-save` creates no output;
11. the quality screen always states that it does not authorize strategy
    change;
12. the official integration reconciles 256 trades, 13 windows, 42
    `FORCE_CLOSE_END`, and 114 bounded intrabar `STOP_LOSS` trades.

## Scientific boundary and next decision

The output can diagnose opportunity capture, adverse path, giveback, and exit
timing. It cannot by itself justify a trailing-stop, replacement, holding-time,
or exit-rule change. Any candidate mechanism arising from this attribution
requires a separately specified, pre-registered, out-of-sample experiment.

After this stage is run and reviewed, the next roadmap item is the
specification of Execution and Slippage Stress Testing. That later stage is
not authorized by this document.
