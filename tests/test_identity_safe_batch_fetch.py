"""Tests for src/research/batch_fetch_pilot/identity_safe_batch_fetch.py
-- the CIK-based identity-verification layer closing
run_batch_feasibility.py's own documented FB/META ticker-recycling
gap.

Uses the REAL, already-cached SEC EDGAR ticker->CIK mapping (a local
file, `src/research/pit_universe/sec_edgar_cache/company_tickers.json`
-- no network call; `force_refresh=False`, this module's own default).
"""

from __future__ import annotations

from src.research.batch_fetch_pilot.identity_safe_batch_fetch import (
    summarize,
    verify_ticker_identity_via_cik,
)


def test_real_meta_ticker_matches_its_real_known_cik():
    """META's real CIK (1326801, Meta Platforms Inc) -- confirmed
    against the real local SEC cache, not hardcoded blindly."""
    results = verify_ticker_identity_via_cik(["META"], expected_cik_by_ticker={"META": 1326801})
    assert len(results) == 1
    assert results[0].status == "MATCH"
    assert results[0].current_cik == 1326801


def test_real_fb_ticker_has_no_current_sec_record():
    """Real finding: SEC EDGAR's own company_tickers.json (exchange-
    listed, SEC-reporting issuers only) has NO entry for 'FB' today --
    confirmed by direct lookup, not assumed. This is itself a useful,
    actionable signal distinct from a CIK mismatch: a ticker with real
    batch-fetched price data but no current SEC registration under
    that symbol is not a normal listed-company match and deserves the
    same fail-closed treatment as a real mismatch would."""
    results = verify_ticker_identity_via_cik(["FB"], expected_cik_by_ticker={})
    assert len(results) == 1
    assert results[0].status == "NO_SEC_RECORD"
    assert results[0].current_cik is None


def test_first_time_seeing_a_ticker_records_no_expectation_not_a_false_match():
    """A ticker the caller has never checked before must not be
    silently treated as verified -- NO_EXPECTATION_RECORDED, with the
    real current CIK returned so the caller can persist it as next
    time's baseline."""
    results = verify_ticker_identity_via_cik(["META"], expected_cik_by_ticker={})
    assert results[0].status == "NO_EXPECTATION_RECORDED"
    assert results[0].current_cik == 1326801


def test_synthetic_mismatch_is_flagged_fail_closed():
    """Reproduces the real FB/META class of bug in a controlled way:
    a ticker whose caller-recorded expected CIK does NOT match what
    SEC reports today -- exactly what a recycled ticker looks like."""
    results = verify_ticker_identity_via_cik(["META"], expected_cik_by_ticker={"META": 9999999})
    assert results[0].status == "MISMATCH_POSSIBLE_RECYCLING"
    assert results[0].current_cik == 1326801
    assert results[0].expected_cik == 9999999


def test_summarize_surfaces_mismatches_clearly():
    results = verify_ticker_identity_via_cik(
        ["META", "FB"], expected_cik_by_ticker={"META": 9999999}
    )
    summary = summarize(results)
    assert summary["checked_count"] == 2
    assert summary["mismatches"] == ["META"]
    assert summary["by_status"]["MISMATCH_POSSIBLE_RECYCLING"] == 1
    assert summary["by_status"]["NO_SEC_RECORD"] == 1
