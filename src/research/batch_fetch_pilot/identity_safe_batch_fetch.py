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

Reuses `_load_sec_company_tickers` from
`src.research.pit_universe.membership_query` -- the SAME real SEC
EDGAR ticker->CIK mapping (with its own already-solved
`curl`-not-`requests` Akamai-bot-blocking workaround, see that
module's own docstring) already used by
`src/live/issuer_identity_preflight.py`. Not reimplemented here.

ISOLATED: does not modify, and is never imported by,
`run_batch_feasibility.py`, `membership_query.py`, or
`issuer_identity_preflight.py`. A caller composes this with
`fetch_batch`/`verify_requested_vs_returned` explicitly; nothing here
performs a batch fetch itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.research.pit_universe.membership_query import _load_sec_company_tickers


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
    """
    sec_map = _load_sec_company_tickers(force_refresh=force_refresh)
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
    }
