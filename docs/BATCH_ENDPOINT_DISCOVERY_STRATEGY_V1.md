# Batch Endpoint Discovery — Strategy (V2 Pivot, İş Kalemi 3)

**Status:** design document, RESEARCH_CANDIDATE_NOT_OFFICIAL. Not wired
into any live/control-arm code path. Builds on top of two already-real
pieces of work in this repo, does not restate them:

- `src/research/batch_fetch_pilot/run_batch_feasibility.py` — real,
  already-run pilot: batch endpoint works, 11.9x–20x faster than
  sequential, and characterized two real failure modes (silent-drop on
  an invalid/delisted symbol; the FB→unrelated-security ticker-recycling
  case).
- `src/research/batch_fetch_pilot/identity_safe_batch_fetch.py` (this
  task) — closes the second failure mode via a real SEC CIK
  cross-check, reusing the same SEC EDGAR mapping
  `issuer_identity_preflight.py` already uses.

## 1. Batch size

The pilot's own real full-universe run (80 tickers, one request) took
1.73s with zero errors — no evidence of a hard size ceiling was hit at
that scale. For S&P 500 scale (~500–510 PIT-eligible tickers per
`sp500_pit_membership` data), **do not send one 500-symbol request**:
URL/response-size risk is unverified at that size (the pilot never
tested past 80), and a single oversized request that fails loses the
whole batch (see §2) rather than degrading gracefully.

**Recommendation: fixed chunks of 150 symbols.** Basis:
- Matches the chunk size already used for real in this session's
  cross-sectional-momentum study (`native_proxy.py`'s
  `fetch_point_in_time_bars`, `chunk_size=150`) — real production use
  at this exact size, hundreds of real calls, zero size-related
  failures observed.
- At ~500 tickers this is 4 requests instead of 1 — a request each
  losing independently (§2) is a much smaller blast radius than one
  request covering everything.

## 2. Rate limit / retry behavior — from the SDK's own source, not assumed

Confirmed by direct read of `alpaca/common/rest.py` (see
`run_batch_feasibility.py`'s own docstring for the exact functions):
- One batch call = one HTTP request. Retries automatically ONLY on
  `429`/`504`, up to 3 attempts, 3 seconds apart
  (`DEFAULT_RETRY_ATTEMPTS`/`DEFAULT_RETRY_WAIT_SECONDS`).
- Any other error status raises `APIError` for the WHOLE request — no
  partial success at the transport level. A caller never has to reason
  about "half this batch succeeded over the wire" — that combination
  is structurally impossible with one request/response pair.

**Design implication:** the retry logic already built for the
cross-sectional-momentum study's own fetch
(`native_proxy.fetch_point_in_time_bars` — up to 3 attempts, exponential
backoff, catching `socket.timeout`/`OSError`/`ConnectionError`) is the
right pattern to reuse for an S&P500-scale batch job, not a new
mechanism. The one confirmed real gap that pattern was built to close:
**alpaca-py sets no HTTP timeout anywhere** (confirmed via `grep -n
timeout` over `alpaca/common/rest.py` — zero matches) — a stalled
connection blocks forever without `socket.setdefaulttimeout()` applied
first. Any new batch-fetch code for this initiative must call that
before the first request, exactly as `native_proxy._apply_socket_timeout`
already does.

**Sequencing at 150-ticker chunks, ~500 tickers:** 4 requests. Even
with a worst-case full 429-retry cycle on every chunk (60s timeout × 3
attempts + backoff ≈ 3 minutes each), a full run stays under 15
minutes — well inside a once-daily research-fetch budget (see
`docs/UNIVERSE_ROBUSTNESS_STEPS.md`'s own practicality read, which this
document does not repeat).

## 3. Error tolerance — two independent, complementary layers

Neither layer alone is sufficient; both are needed because they catch
different failure shapes:

| Layer | Catches | Cannot catch |
|---|---|---|
| `verify_requested_vs_returned` (existing) | A requested symbol silently absent from the response (invalid, delisted, genuinely no data) | A symbol present but resolved to the WRONG company (recycled ticker) |
| `verify_ticker_identity_via_cik` (this task) | A returned symbol whose current SEC CIK differs from a previously-recorded expectation (ticker recycling) | A symbol that's simply missing (that's layer 1's job) |

**Required usage pattern for any real adoption:**
1. Batch-fetch a chunk.
2. `verify_requested_vs_returned` — any `missing` entry is either
   accepted as a real, expected gap (e.g. a confirmed PIT removal
   event) or routed to `assess_sequential_fallback_feasibility`
   (existing, evaluates only, per that function's own scope).
3. `verify_ticker_identity_via_cik` against a persisted
   `expected_cik_by_ticker` map (one entry recorded, and never silently
   overwritten, the first time each ticker is successfully checked).
   Any `MISMATCH_POSSIBLE_RECYCLING` result is fail-closed: exclude
   that ticker's data from this run, flag for manual review — never
   silently proceed with data that might belong to a different real
   company than the caller believes.

Both layers are read-only, local-computation checks (no additional
Alpaca calls beyond the batch fetch itself; the SEC CIK lookup reuses
the already-cached `company_tickers.json`, or a weekly-cadence refresh
exactly like `issuer_identity_preflight.py`'s own pattern).

## 4. The ~20-30 candidate, measured-expansion path (Levent's stated preference)

This is explicitly NOT "fetch all ~500 S&P 500 tickers and go live with
them." The batch/identity infrastructure above exists to make the
RESEARCH step of a much smaller, gated expansion cheap and safe, not to
shortcut straight to full scale:

1. **Candidate pool**: S&P 500 PIT-eligible tickers (strict mode,
   `identity_warnings`-excluded — see `membership_query.py`) minus
   tickers already in `live_universe.py`/`control_universe.py`.
2. **Sector/correlation filter**: reuse `live_universe.py`'s own
   already-established methodology (`redundancy_correlation_threshold
   = 0.75`, confirmed in `strategy_config.py`'s
   `EligibilityRules.redundancy_correlation_threshold`) — compute
   pairwise correlation of each candidate against the CURRENT live
   universe, keep only candidates below the threshold, and require
   sector diversity across the surviving set (no existing automated
   sector-diversity check exists in this repo today — this would be
   new work, out of scope for this design pass, flagged here as the
   next concrete gap rather than silently assumed solved).
3. **Batch-fetch the reduced ~20-30 candidate pool** using §1–3 above
   — small enough that even a full sequential fallback (0.44s/ticker)
   costs under 15 seconds if the batch path fails entirely.
4. **Full existing research pipeline** (data quality, correlation,
   provenance — `src/research/performance_report_v1` and this
   session's own PIT/identity tooling) runs on the candidate pool
   BEFORE any code touches `live_universe.py`/`control_universe.py`.
5. Only a human (Levent), reviewing that research output, decides
   whether any candidate is promoted — no code path in this design
   auto-promotes a candidate into the live universe.

## Out of scope for this design pass

- Actually running a real ~500-ticker batch fetch (would need its own
  explicit, scoped API-call approval — this document is a design
  artifact, not an executed one).
- The sector-diversity check named in §4.2 (flagged, not built).
- Wiring `verify_ticker_identity_via_cik`'s `on_mismatch` handling into
  any real notification channel (same "left unwired on purpose"
  discipline `run_batch_feasibility.py`'s own `on_missing` hook
  already established).
