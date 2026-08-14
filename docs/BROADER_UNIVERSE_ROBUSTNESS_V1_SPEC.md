# Broader-Universe Robustness V1 — Cohort Registration and Data Audit

## Purpose

This protocol is the first, registration-only sub-stage of Broader-Universe
Robustness. It freezes a point-in-time cohort definition and verifies the
prepared market-data snapshot before any broader-universe performance result
is computed.

It does not run a portfolio backtest, select parameters, change the locked
baseline, authorize paper trading, or establish production readiness.

## Locked comparator

Every registration must use research lock
`RBL_V1_20260802_192047_D93145CB1DCF`. The registry file, its hash, the locked
configuration, and all registered project-file hashes are verified before a
cohort can pass.

The following behavior remains unchanged:

- strategy and exit rules;
- risk and allocation limits;
- commission and slippage assumptions;
- next-available-Open execution;
- deterministic tie-breaking and portfolio event ordering;
- `src/main.py`, paper trading, broker access, and automation.

## Point-in-time cohort contract

The input cohort manifest is UTF-8 CSV with one row per security and cohort
date. It must contain exactly these columns:

```text
cohort_date,ticker,asset_type,region,liquidity_rank,eligible_universe_count,
liquidity_measure,liquidity_value,selection_source,source_as_of,
source_snapshot_path,
source_snapshot_sha256,eligible_from,eligible_to,status_at_source,
price_adjustment
```

Dates use ISO `YYYY-MM-DD`. `eligible_to` may be empty for a still-eligible
security. `status_at_source` is one of `active`, `inactive`, or `delisted`.
`source_snapshot_path` identifies the actual immutable point-in-time membership
and liquidity file, and `source_snapshot_sha256` is its verified hash. Relative
paths are resolved against the cohort-manifest directory.

For each cohort date:

- equities are the first 150 eligible US names ranked by trailing 90-day
  median daily dollar volume;
- crypto contains the first `min(20, eligible_universe_count)` eligible global
  names, with at least two names required;
- descending liquidity and ascending ticker provide the deterministic order;
- the referenced source file must exist, match its declared hash, and have
  been available no later than the cohort date;
- current-constituent backfilling and controlled-basket prefiltering are
  prohibited;
- inactive or delisted securities remain present when they were eligible at
  the historical cohort date.

This registration does not claim that a source is survivorship-free merely
because its label says so. The immutable source hash, source date, eligibility
interval, inactive-security status, and cohort ranks are preserved so that
the claim remains auditable.

## Snapshot contract

The market-data input is an immutable snapshot produced with the existing
research snapshot schema. The registration independently verifies:

- snapshot fingerprint and every declared file hash;
- unique, increasing timestamps;
- required prepared-data columns;
- finite and logically consistent OHLC values;
- nonnegative volume;
- at least 200 observations available by each selection date;
- coverage through the complete membership interval;
- maximum missing-date fractions of 10% for US-equity business days and 3%
  for crypto calendar days.

These tolerances detect incomplete files; they do not model exchange holidays,
corporate actions, vendor corrections, or live data quality. Equity rows must
declare split-and-dividend-adjusted prices. Crypto rows must declare raw spot
prices.

## Fail-closed gates

A cohort is registered only when all of the following pass:

1. specification contract;
2. locked registry identity and file hash;
3. locked registry status and authority boundary;
4. locked project-file hashes;
5. cohort schema and value domains;
6. annual schedule and endpoint coverage;
7. deterministic count and ranking policy;
8. point-in-time source evidence;
9. duplicate and tie-break checks;
10. immutable snapshot integrity;
11. prepared-data columns;
12. OHLCV and timestamp quality;
13. indicator warm-up and membership coverage;
14. absence of controlled-basket or outcome prefiltering;
15. no additional research or execution authority.

Failed audits are saved with status `REGISTRATION_REJECTED` when configured,
but they do not authorize a cohort. A passing result has status
`BROADER_UNIVERSE_COHORT_REGISTERED`.

## Outputs

The command writes:

- registration JSON;
- provenance JSON;
- gate checks CSV;
- per-cohort count/rank audit CSV;
- per-ticker data-quality audit CSV;
- one-row screen CSV.

The provenance hashes all inputs and results. Registration does not authorize
the actual broader-universe backtest. That implementation requires a second
review and explicit approval, including deterministic membership enforcement
inside the portfolio research path.
