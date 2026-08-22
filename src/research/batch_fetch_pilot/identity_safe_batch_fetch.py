"""Closes the one real, previously-undiscovered gap
`run_batch_feasibility.py`'s own docstring left explicitly open: its
"FB" finding (Facebook's old ticker, recycled by an unrelated real
security after the 2021 META rename) is invisible to
`verify_requested_vs_returned` -- the symbol IS present in the batch
response, just attached to the wrong company's real data, and a
requested-vs-returned symbol diff has no way to see that.

THE FIX: cross-check each batch-fetched ticker's CURRENT real SEC
EDGAR CIK (Central Index Key -- a durable identifier that survives a
ticker rename/recycle, unlike the ticker symbol itself) against an
EXPECTED CIK the caller already knows for that ticker (e.g. from a
prior PIT anchor artifact, or from an earlier successful fetch this
same caller already verified). A mismatch means the ticker symbol now
resolves to a DIFFERENT real company than expected -- exactly the
FB-style recycling case, now catchable.

REAL PROVENANCE FOR THE FB/META CLAIM (2026-08-22, independent audit
asked this to be proven with real expected-CIK provenance, not just
asserted): Meta Platforms, Inc.'s real CIK is 1326801 -- confirmed
directly from the local SEC cache (`sec_map["META"]["cik_str"] ==
1326801`, `title == "Meta Platforms, Inc."`). A CIK is a durable SEC
identifier that does NOT change when a company renames (Facebook,
Inc. -> Meta Platforms, Inc. in 2021 kept the SAME CIK) -- so 1326801
is also the real, correct "expected CIK" for the OLD "FB" ticker a
caller might still remember from before the rename. Querying this
module's own real, local SEC cache for "FB" today returns NO entry at
all (`sec_map.get("FB") is None`, confirmed by direct lookup) -- proving
that even using Facebook's own real, historically-correct CIK as the
expectation, "FB" resolves to `NO_SEC_RECORD` today, not `MATCH`. This
is real evidence the identity check correctly flags something is wrong
with "FB" today, using a real, externally-verifiable identifier
(1326801), not a hand-wavy comparison of two DataFrames' OHLCV values
(the original pilot's own method, still real but weaker evidence than
a durable-identifier cross-check).

SEC CACHE AGE POLICY (2026-08-22, independent audit): reuses
`src.live.issuer_identity_preflight._ensure_sec_data_fresh` --
the SAME weekly-cadence (`SEC_CACHE_MAX_AGE_DAYS` = 8 days) refresh
policy that module already established and tests, rather than
inventing a second, parallel one. Fail-closed if the cache is stale
AND a refresh attempt fails (see that function's own docstring).

Reuses `_load_sec_company_tickers` from
`src.research.pit_universe.membership_query` (via
`_ensure_sec_data_fresh`) -- the SAME real SEC EDGAR ticker->CIK
mapping (with its own already-solved `curl`-not-`requests`
Akamai-bot-blocking workaround, see that module's own docstring)
already used by `src/live/issuer_identity_preflight.py`. Not
reimplemented here.

ISOLATED: does not modify, and is never imported by,
`run_batch_feasibility.py`, `membership_query.py`, or
`issuer_identity_preflight.py` (only reads one function from the last
of these). `fetch_batch_identity_safe` below composes
`run_batch_feasibility.py`'s own `fetch_batch`/
`verify_requested_vs_returned` with this module's own identity check --
the first real, mandatory-fail-closed wrapper doing so; a caller no
longer has to remember to run both checks itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.live.issuer_identity_preflight import _ensure_sec_data_fresh


class BatchFetchIntegrityError(RuntimeError):
    """Raised by `fetch_batch_identity_safe` -- fail-closed, either a
    missing symbol in the batch response or a real CIK identity
    mismatch/absence. Never partially returned; see that function's own
    docstring for exactly which `IdentityCheckResult` statuses trigger
    this."""


@dataclass(frozen=True)
class IdentityCheckResult:
    ticker: str
    current_cik: int | None
    current_title: str | None
    expected_cik: int | None
    status: str  # "MATCH" | "MISMATCH_POSSIBLE_RECYCLING" | "NO_SEC_RECORD" | "NO_EXPECTATION_RECORDED"


def verify_ticker_identity_via_cik(
    tickers: list[str],
    expected_cik_by_ticker: dict[str, int],
    *,
    force_refresh: bool = False,
    sec_cache_path: Path | None = None,
) -> list[IdentityCheckResult]:
    """For each `ticker`, looks up its CURRENT real SEC CIK and compares
    against `expected_cik_by_ticker.get(ticker)`:

    - No SEC record at all for this ticker today -> `NO_SEC_RECORD`
      (genuinely delisted/unlisted, or a data entry issue -- same
      "silently absent" class `verify_requested_vs_returned` already
      flags for the batch-fetch side, now cross-confirmed on the
      identity side too).
    - Caller has no prior expected CIK for this ticker (first time
      seeing it) -> `NO_EXPECTATION_RECORDED`, current CIK still
      returned so the CALLER can persist it as the baseline for next
      time -- this function never guesses an expectation itself.
    - Current CIK matches the caller's expectation -> `MATCH`.
    - Current CIK differs -> `MISMATCH_POSSIBLE_RECYCLING` -- the real,
      previously-uncatchable FB/META case. A caller should treat this
      as fail-closed (exclude the ticker, raise, or route to manual
      review), never silently proceed with that ticker's batch-fetched
      price data.

    `force_refresh=True` bypasses the age check entirely (always
    fetches live); the default `False` defers to
    `_ensure_sec_data_fresh`'s own weekly-cadence policy (see module
    docstring) -- fail-closed (`RuntimeError`) if the cache is stale
    AND a live refresh attempt fails, rather than silently reasoning
    about ticker identity using data that might be arbitrarily old.
    """
    if force_refresh:
        from src.research.pit_universe.membership_query import _load_sec_company_tickers

        sec_map = _load_sec_company_tickers(force_refresh=True)
    else:
        sec_map, age_days, refresh_error = _ensure_sec_data_fresh(sec_cache_path)
        if sec_map is None:
            raise RuntimeError(
                f"SEC ticker->CIK cache is stale (age={age_days} days) and a live "
                f"refresh attempt failed ({refresh_error}) -- refusing to verify "
                f"ticker identity against data that might be arbitrarily old. "
                f"Fail-closed, per _ensure_sec_data_fresh's own contract."
            )

    results: list[IdentityCheckResult] = []
    for ticker in tickers:
        entry = sec_map.get(ticker)
        expected_cik = expected_cik_by_ticker.get(ticker)

        if entry is None:
            results.append(
                IdentityCheckResult(
                    ticker=ticker, current_cik=None, current_title=None,
                    expected_cik=expected_cik, status="NO_SEC_RECORD",
                )
            )
            continue

        current_cik = int(entry["cik_str"])
        current_title = str(entry.get("title"))

        if expected_cik is None:
            status = "NO_EXPECTATION_RECORDED"
        elif current_cik == expected_cik:
            status = "MATCH"
        else:
            status = "MISMATCH_POSSIBLE_RECYCLING"

        results.append(
            IdentityCheckResult(
                ticker=ticker, current_cik=current_cik, current_title=current_title,
                expected_cik=expected_cik, status=status,
            )
        )
    return results


def summarize(results: list[IdentityCheckResult]) -> dict:
    by_status: dict[str, int] = {}
    for result in results:
        by_status[result.status] = by_status.get(result.status, 0) + 1
    return {
        "checked_count": len(results),
        "by_status": by_status,
        "mismatches": [r.ticker for r in results if r.status == "MISMATCH_POSSIBLE_RECYCLING"],
        # Explicit, first-class fields (2026-08-22, independent audit) --
        # previously only reachable by digging into `by_status`, same
        # discipline `mismatches` above already established.
        "no_sec_record": [r.ticker for r in results if r.status == "NO_SEC_RECORD"],
        "no_expectation_recorded": [r.ticker for r in results if r.status == "NO_EXPECTATION_RECORDED"],
    }


def fetch_batch_identity_safe(
    tickers: list[str],
    expected_cik_by_ticker: dict[str, int],
    *,
    sec_cache_path: Path | None = None,
) -> Any:  # pd.DataFrame -- avoided as a hard import, matching run_batch_feasibility.py's own convention
    """Mandatory fail-closed wrapper (2026-08-22, independent audit):
    composes `run_batch_feasibility.fetch_batch` +
    `verify_requested_vs_returned` + `verify_ticker_identity_via_cik`
    into ONE call a caller cannot forget half of. Raises
    `BatchFetchIntegrityError` if EITHER check finds a real anomaly --
    a missing symbol in the batch response, a CIK mismatch
    (`MISMATCH_POSSIBLE_RECYCLING`), or a ticker with no current SEC
    record at all (`NO_SEC_RECORD` -- genuinely delisted tickers should
    already fail the presence check first in most cases, but an
    OTC/pink-sheet-recycled symbol like the real "FB" case can still
    return real price data with no SEC registration under that symbol,
    which this catches independently). `NO_EXPECTATION_RECORDED` is
    NOT treated as an anomaly -- it is the expected, normal outcome the
    first time this caller ever sees a given ticker; the returned
    `IdentityCheckResult`s let the caller persist the observed CIK as
    next time's baseline. Never returns partially-verified data --
    either both checks pass clean (modulo NO_EXPECTATION_RECORDED) or
    this raises before returning anything."""
    from src.research.batch_fetch_pilot.run_batch_feasibility import fetch_batch, verify_requested_vs_returned

    df = fetch_batch(tickers)
    presence_check = verify_requested_vs_returned(tickers, df)
    if presence_check["silent_partial_failure"]:
        raise BatchFetchIntegrityError(
            f"Batch fetch returned fewer symbols than requested: {presence_check['missing']} "
            f"-- refusing to return partially-fetched data. Fail-closed."
        )

    identity_results = verify_ticker_identity_via_cik(
        tickers, expected_cik_by_ticker, sec_cache_path=sec_cache_path
    )
    identity_summary = summarize(identity_results)
    if identity_summary["mismatches"] or identity_summary["no_sec_record"]:
        raise BatchFetchIntegrityError(
            f"Ticker identity check failed: mismatches (possible recycling)="
            f"{identity_summary['mismatches']}, no current SEC record="
            f"{identity_summary['no_sec_record']} -- refusing to return data that "
            f"might belong to the wrong real company. Fail-closed."
        )

    return df
