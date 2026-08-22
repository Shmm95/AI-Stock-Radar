"""Tests for src/research/batch_fetch_pilot/identity_safe_batch_fetch.py
-- the CIK-based identity-verification layer closing
run_batch_feasibility.py's own documented FB/META ticker-recycling
gap.

Uses the REAL, already-cached SEC EDGAR ticker->CIK mapping (a local
file, `src/research/pit_universe/sec_edgar_cache/company_tickers.json`
-- no network call; `force_refresh=False`, this module's own default).
"""

from __future__ import annotations

import pytest

import src.research.batch_fetch_pilot.identity_safe_batch_fetch as identity_module
from src.research.batch_fetch_pilot.identity_safe_batch_fetch import (
    BatchFetchIntegrityError,
    fetch_batch_identity_safe,
    summarize,
    verify_ticker_identity_via_cik,
)

# Meta Platforms, Inc.'s real CIK -- confirmed directly against the local
# SEC cache (sec_map["META"] == {"cik_str": 1326801, "title": "Meta
# Platforms, Inc.", ...}). A CIK does not change when a company renames
# (Facebook, Inc. -> Meta Platforms, Inc., 2021) -- so this is also the
# real, historically-correct "expected CIK" for the OLD "FB" ticker.
META_REAL_CIK = 1326801


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


# --- Independent audit finding: real expected-CIK provenance for the FB/META claim ---


def test_real_fb_provenance_using_facebooks_own_real_historical_cik():
    """The real provenance proof the audit asked for: querying "FB"
    using Facebook's OWN real, durable CIK (1326801 -- the same CIK
    Meta Platforms, Inc. has today, since a CIK never changes on a
    rename) as the expectation. Real SEC lookup for "FB" today returns
    NO record at all -- proving that even using the historically
    CORRECT identifier, "FB" does not resolve to a MATCH today. This is
    real, externally-verifiable evidence (a durable SEC identifier),
    not a hand-wavy OHLCV DataFrame comparison."""
    results = verify_ticker_identity_via_cik(["FB"], expected_cik_by_ticker={"FB": META_REAL_CIK})
    assert results[0].status == "NO_SEC_RECORD"  # not MATCH, not MISMATCH -- no record to compare against at all
    assert results[0].expected_cik == META_REAL_CIK
    assert results[0].current_cik is None

    # And META, queried with the SAME real CIK as expectation, genuinely matches --
    # confirming 1326801 really is the correct, current identifier for that CIK.
    meta_results = verify_ticker_identity_via_cik(["META"], expected_cik_by_ticker={"META": META_REAL_CIK})
    assert meta_results[0].status == "MATCH"


# --- Independent audit finding: explicit no_sec_record/no_expectation_recorded fields ---


def test_summarize_exposes_no_sec_record_and_no_expectation_recorded_explicitly():
    results = verify_ticker_identity_via_cik(
        ["FB", "META"], expected_cik_by_ticker={}  # no expectation for either
    )
    summary = summarize(results)
    assert summary["no_sec_record"] == ["FB"]
    assert summary["no_expectation_recorded"] == ["META"]


# --- Independent audit finding: SEC cache age policy (reuses issuer_identity_preflight's) ---


def test_verify_ticker_identity_uses_the_shared_weekly_cadence_refresh_policy():
    """Real source-text check: confirms _ensure_sec_data_fresh is
    actually called (the SAME weekly-cadence policy
    issuer_identity_preflight.py already established), not a second,
    parallel age-check reinvented here."""
    import inspect

    source = inspect.getsource(identity_module.verify_ticker_identity_via_cik)
    assert "_ensure_sec_data_fresh" in source


def test_stale_cache_and_failed_refresh_fails_closed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        identity_module, "_ensure_sec_data_fresh", lambda path: (None, 30.0, "ConnectionError: real refresh failed")
    )
    with pytest.raises(RuntimeError, match="stale"):
        verify_ticker_identity_via_cik(["META"], expected_cik_by_ticker={"META": META_REAL_CIK})


# --- Independent audit finding: mandatory fail-closed batch-fetch wrapper ---
# --- Monkeypatched -- no real Alpaca call (the underlying fetch_batch  ---
# --- integration is already proven real in run_batch_feasibility.py's  ---
# --- own docstring; this tests ONLY the wrapper's own composition/     ---
# --- fail-closed logic).                                               ---


class _FakeDataFrame:
    """Minimal stand-in -- fetch_batch_identity_safe never inspects the
    DataFrame itself, only passes it to verify_requested_vs_returned
    (also faked below), so no real pandas object is needed."""


def test_fetch_batch_identity_safe_raises_on_missing_symbol(monkeypatch: pytest.MonkeyPatch):
    import src.research.batch_fetch_pilot.run_batch_feasibility as run_batch_feasibility

    monkeypatch.setattr(run_batch_feasibility, "fetch_batch", lambda tickers: _FakeDataFrame())
    monkeypatch.setattr(
        run_batch_feasibility, "verify_requested_vs_returned",
        lambda requested, df: {"requested_count": 1, "returned_count": 0, "missing": ["ZZZ"], "extra_unexpected": [], "silent_partial_failure": True},
    )
    with pytest.raises(BatchFetchIntegrityError, match="fewer symbols"):
        fetch_batch_identity_safe(["ZZZ"], expected_cik_by_ticker={})


def test_fetch_batch_identity_safe_raises_on_identity_mismatch(monkeypatch: pytest.MonkeyPatch):
    import src.research.batch_fetch_pilot.run_batch_feasibility as run_batch_feasibility

    monkeypatch.setattr(run_batch_feasibility, "fetch_batch", lambda tickers: _FakeDataFrame())
    monkeypatch.setattr(
        run_batch_feasibility, "verify_requested_vs_returned",
        lambda requested, df: {"requested_count": 1, "returned_count": 1, "missing": [], "extra_unexpected": [], "silent_partial_failure": False},
    )
    with pytest.raises(BatchFetchIntegrityError, match="Ticker identity check failed"):
        fetch_batch_identity_safe(["META"], expected_cik_by_ticker={"META": 9999999})


def test_fetch_batch_identity_safe_returns_data_on_a_clean_pass(monkeypatch: pytest.MonkeyPatch):
    import src.research.batch_fetch_pilot.run_batch_feasibility as run_batch_feasibility

    sentinel_df = _FakeDataFrame()
    monkeypatch.setattr(run_batch_feasibility, "fetch_batch", lambda tickers: sentinel_df)
    monkeypatch.setattr(
        run_batch_feasibility, "verify_requested_vs_returned",
        lambda requested, df: {"requested_count": 1, "returned_count": 1, "missing": [], "extra_unexpected": [], "silent_partial_failure": False},
    )
    result = fetch_batch_identity_safe(["META"], expected_cik_by_ticker={"META": META_REAL_CIK})
    assert result is sentinel_df


def test_fetch_batch_identity_safe_does_not_raise_on_no_expectation_recorded(monkeypatch: pytest.MonkeyPatch):
    """First-time-seeing-this-ticker must NOT be treated as an anomaly."""
    import src.research.batch_fetch_pilot.run_batch_feasibility as run_batch_feasibility

    sentinel_df = _FakeDataFrame()
    monkeypatch.setattr(run_batch_feasibility, "fetch_batch", lambda tickers: sentinel_df)
    monkeypatch.setattr(
        run_batch_feasibility, "verify_requested_vs_returned",
        lambda requested, df: {"requested_count": 1, "returned_count": 1, "missing": [], "extra_unexpected": [], "silent_partial_failure": False},
    )
    result = fetch_batch_identity_safe(["META"], expected_cik_by_ticker={})
    assert result is sentinel_df
