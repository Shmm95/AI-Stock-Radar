# Research Baseline Lock V1 Specification

## Purpose and authority boundary

Research Baseline Lock V1 freezes the approved AI-Stock-Radar historical
research reference after deterministic portfolio validation, attribution,
holding-path analysis, and execution-cost stress testing. It is a verification
and governance stage, not a strategy experiment.

The lock does not alter or authorize strategy parameters, engine behavior,
event ordering, risk limits, fees, slippage, `src/main.py`, paper trading,
production, broker access, or automation. Passing the lock means only that
future research has an immutable comparator.

## Locked scope

The lock applies only to the controlled nine-asset historical research basket:

```text
AAPL, AMZN, BTC-USD, ETH-USD, GOOGL, META, MSFT, NVDA, TSLA
```

It does not establish broader-universe robustness, capacity, live execution
quality, or deployability.

## Approved lineage

- Snapshot: `20260802_081049_d93145cb1dcf`
- Snapshot fingerprint:
  `d93145cb1dcf3f837415f5b28ac29eeb51c13bc8ce5b8b1187aa34ebfc01ad44`
- Stop Walk-Forward: `20260802_081449`
- Trade Timing Attribution: `20260802_085312`
- Entry Statistics: `20260802_103007`
- Forward Return Statistics: `20260802_134341`
- Holding-Path Attribution: `20260802_155749`
- Execution and Slippage Stress: `20260802_192047`

The machine-readable registry is
`config/research_baseline_lock_v1.json`. Its installed bytes, the authoritative
engine/model files, the baseline runner, the execution-stress implementation,
their relevant tests and specifications, and the baseline document are fixed
by SHA-256.

## Baseline configuration

The registered `PortfolioBacktestConfig` is:

```text
Initial cash:                     10,000 USD
Maximum open positions:          4
Risk per trade:                  1%
Maximum total open risk:         4%
Maximum position allocation:     25%
Maximum crypto allocation:       25%
Stock stop / trailing close:     5% / 7.5%
Crypto stop / trailing close:    5% / 7.5%
Commission / minimum fee:        0.05% / 1 USD
Slippage:                        5 bps
Fractional stocks / crypto:      disabled / enabled
Force close at test end:         enabled
```

The verifier compares every runtime config field against this registration.
Any missing, additional, or changed value fails closed.

## Registered historical reference

The baseline `C01_S01` reference is:

```text
Compounded OOS return:        215.5054%
CAGR:                          19.3457%
Maximum drawdown:              24.4430%
Return / drawdown:              8.8167
Average profit factor:          2.2953
Positive windows:               8 / 13
Matched benchmark return:     532.4012%
Matched benchmark beats:        4 / 13
Completed trades:             256
Average exposure:              51.4822%
```

The sole primary doubled-cost scenario `C02_S02` is also registered:

```text
Compounded OOS return:        172.1317%
Maximum drawdown:              25.5887%
Average profit factor:          2.1898
Positive windows:               8 / 13
Drawdown worsening:             1.1457 percentage points
Matched / baseline-only / scenario-only trades: 248 / 8 / 7
```

These values are historical reference fingerprints, not performance promises.

## Verification protocol

The verifier must:

1. validate the registry schema, scope, lineage and non-authorization flags;
2. hash-check every registered project file;
3. require the exact registered Execution and Slippage Stress provenance;
4. hash-check all 13 stress result files;
5. hash-check every predecessor recorded in stress provenance;
6. require the exact snapshot, Holding-Path and Stop Walk-Forward lineage;
7. compare every runtime baseline-config field with the registry;
8. require all `182` official replay comparisons to pass within `1e-8`;
9. require the complete 16-scenario, 13-window, 208-run grid;
10. require all four primary robustness gates and all six quality checks;
11. require the exact controlled ticker set and registered reference metrics;
12. require no scenario selection and no execution authority.

Any mismatch stops the lock. There is no tolerance-based fallback, artifact
substitution, automatic manifest update, parameter search, or alternate code
path.

## Outputs

A successful official run writes exactly three non-overwriting artifacts:

- verification checks CSV;
- research baseline lock JSON certificate;
- provenance JSON containing every verified source and result hash.

Every verified input is hash-checked again immediately before saving.
`--no-save` must create no artifacts.

## Known limitations retained by the lock

- The controlled basket is small and concentrated.
- Baseline and doubled-cost returns trail their matched benchmarks.
- Cost responses are path-dependent and non-monotonic across the complete
  stress grid.
- NVDA/W09 materially influences the cost-stress attribution.
- `FORCE_CLOSE_END` remains in the official historical population.
- Historical fills do not reproduce broker, liquidity, latency, capacity, or
  production conditions.

The lock records these limitations rather than resolving or concealing them.

## Following stage

After the completed lock certificate is reviewed, the next roadmap action is a
separate **Broader-Universe Robustness** specification. The research baseline
remains unchanged during that work and is the comparator for all candidate
universe results.

Paper-trading architecture remains later in the roadmap and requires separate
approval after broader-universe robustness.
