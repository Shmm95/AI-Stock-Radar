# Portfolio Execution and Slippage Stress Testing V1 Specification

## Status and authorization boundary

This stage is a pre-registered historical robustness test of the frozen
`FIXED_BASELINE`. It changes execution costs only on copied backtest configs.
It does not change the portfolio engine, signals, stops, trailing exits, risk,
position limits, ticker universe, window definitions, baseline configuration,
paper trading, or production behavior.

Passing any result or gate requires human review and does not authorize a
baseline, paper-trading, production, or automation change.

## Locked lineage

The sole approved chain is:

- Holding-Path Attribution: `20260802_155749`
- Forward Return Statistics: `20260802_134341`
- Entry Statistics: `20260802_103007`
- Trade Timing Attribution: `20260802_085312`
- Stop Walk-Forward: `20260802_081449`
- Snapshot: `20260802_081049_d93145cb1dcf`
- Snapshot fingerprint:
  `d93145cb1dcf3f837415f5b28ac29eeb51c13bc8ce5b8b1187aa34ebfc01ad44`
- Test population: 13 independently reset six-month OOS windows
- Baseline population: 256 completed trades

Every declared Holding-Path result, Holding-Path upstream source, Stop
Walk-Forward result, snapshot artifact, approved code file, and test file must
pass hash verification. Every consumed hash is checked again immediately
before saving.

## Baseline execution assumptions

The frozen config contains:

```text
commission_rate = 0.0005 (0.05%)
minimum_fee     = 1.00
slippage_bps    = 5.0
```

All non-cost config fields are protected and must be identical in every run.
Stress configs are created with `dataclasses.replace`; the source config is
never mutated.

## Fixed stress grid

The complete Cartesian grid is fixed before observation:

```text
commission/minimum-fee multiplier: 1x, 2x, 4x, 8x
slippage multiplier:               1x, 2x, 4x, 8x
total scenarios:                   16
```

Commission-rate and minimum-fee multipliers always move together. The
resulting absolute values are:

| Multiplier | Commission rate | Minimum fee |
|---:|---:|---:|
| 1x | 0.05% | 1 |
| 2x | 0.10% | 2 |
| 4x | 0.20% | 4 |
| 8x | 0.40% | 8 |

| Multiplier | Slippage |
|---:|---:|
| 1x | 5 bps |
| 2x | 10 bps |
| 4x | 20 bps |
| 8x | 40 bps |

Scenario identifiers use `Cxx_Sxx`; `C01_S01` is the exact baseline replay and
`C02_S02` is the sole primary robustness scenario. No scenario is trained,
ranked, selected, or promoted. The 4x and 8x scenarios are diagnostic boundary
observations only.

## Replay and benchmark rules

Each scenario runs on each of the 13 frozen OOS test windows. Each window starts
with the frozen initial capital. Scenario equity is then stitched by scaling
successive reset-window curves, matching the approved walk-forward aggregate
method.

`C01_S01` must reproduce every official `FIXED_BASELINE` window for ending
equity, return, drawdown, trade counts, win rate, profit factor, fees,
exposure, rejections, matched-benchmark return, and excess return within
`1e-8`. Any mismatch fails closed.

The matched benchmark is recalculated for each scenario using that scenario's
costs and realized strategy exposure. Benchmark excess is reported but is not
a primary gate because the approved research question is execution-cost
survival, not benchmark selection.

## Trade pairing

Trades are outer-matched to baseline within each window using ticker, entry
timestamp, exit timestamp, exit reason, and deterministic duplicate
occurrence. Outputs distinguish:

- `MATCHED`
- `BASELINE_ONLY`
- `SCENARIO_ONLY`

For matched trades, entry fill, exit fill, quantity, fees, P&L, and return
deltas are reported. Cost changes may alter equity, position sizing, cash
feasibility, later rejections, and therefore the trade population; this is an
expected result to expose rather than suppress.

## Primary robustness gates

Only `C02_S02` is gated. All four conditions must hold:

1. compounded OOS return is strictly positive;
2. average window profit factor is greater than `1.0`;
3. at least `7` of `13` windows have positive returns;
4. stitched maximum drawdown worsening versus baseline is no more than `5.0`
   percentage points.

These gates answer whether the frozen research baseline survives a doubled
execution-cost environment. They do not authorize a model change.

## Required reports

Outputs include:

- window definitions;
- all 208 window/scenario test runs;
- scenario aggregates and baseline comparisons;
- ticker contributions;
- rejection reasons;
- scenario trades and baseline trade pairs;
- stitched equity curves;
- official replay checks;
- primary gates and a quality screen;
- machine-readable JSON and full provenance.

The reports cover compounded return, CAGR, maximum drawdown, return/drawdown,
average profit factor, positive windows, matched-benchmark results, exposure,
trade counts, ticker P&L, rejection reasons, fees, and trade-population drift.

Existing output paths are never overwritten. `--no-save` creates no artifacts.

## Acceptance criteria

Implementation is accepted only when tests prove:

1. the exact 16-scenario grid and identifiers;
2. rejection of unregistered multipliers;
3. cost-only config mutation;
4. exact official baseline replay;
5. deterministic trade-pair statuses and deltas;
6. fixed primary gates and non-authorization flags;
7. complete 208-run coverage with no scenario selection;
8. fail-closed provenance, pre-save mutation detection, and no overwrite;
9. `--no-save` never calls the saver;
10. full official lineage can be loaded and verified.

## Following decision

If the completed stress result passes review, the next roadmap action is a
separate Research Baseline Lock specification. If it fails, this stage reports
the failure and stops; it does not search for cheaper assumptions, tune the
strategy, or alter the baseline.

The separately registered Stop Conditional Forward protocol remains
time-locked and unchanged.
