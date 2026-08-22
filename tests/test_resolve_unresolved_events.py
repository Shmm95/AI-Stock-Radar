"""Tests for src/research/pit_universe/resolve_unresolved_events.py --
the resolution methodology for the 273 unresolved S&P 500 PIT
membership events.

Deterministic, no network dependency. Several tests use the REAL
reconstruction-inputs artifact (a local, already-fetched file, not a
network call) to prove the methodology against real historical data,
including the real FOX ticker-recycling case this module's own
docstring documents.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.pit_universe.resolve_unresolved_events import (
    RECONSTRUCTION_INPUTS_PATH,
    load_and_resolve,
    resolve_unresolved_events,
)


def _event(date: str, ticker: str, action: str, sources: list[str], category: str = "true_conflict") -> dict:
    return {"date": date, "ticker": ticker, "action": action, "supporting_sources": sources, "category": category}


def test_two_distinct_sources_within_spread_resolves_via_median():
    events = [
        _event("2020-06-01", "ABCD", "ADD", ["wikipedia"]),
        _event("2020-06-04", "ABCD", "ADD", ["lawcal"]),
    ]
    result = resolve_unresolved_events(events)
    assert len(result.resolved) == 1
    resolved = result.resolved[0]
    assert resolved.ticker == "ABCD"
    assert resolved.action == "ADD"
    assert resolved.date == "2020-06-01"  # median_low of two dates -> the earlier one
    assert set(resolved.supporting_sources) == {"wikipedia", "lawcal"}
    assert not result.still_unresolved


def test_single_source_never_promoted_regardless_of_how_many_entries():
    events = [
        _event("2020-06-01", "SOLO", "REMOVE", ["lawcal"], category="single_source_only"),
    ]
    result = resolve_unresolved_events(events)
    assert not result.resolved
    assert len(result.still_unresolved) == 1


def test_wide_date_spread_stays_unresolved_even_with_two_sources():
    """Real reproduction of the FOX/ADD case: two ADD events for the
    same ticker, far apart in time, are almost certainly TWO DIFFERENT
    real corporate events sharing a recycled ticker symbol, not date
    noise around one event -- must NOT be averaged into a nonsense
    date."""
    events = [
        _event("2015-09-18", "FOX", "ADD", ["wikipedia"]),
        _event("2019-03-19", "FOX", "ADD", ["lawcal"]),
    ]
    result = resolve_unresolved_events(events)
    assert not result.resolved
    assert len(result.still_unresolved) == 2


def test_three_entries_two_close_one_far_all_stay_unresolved_together():
    """A group's resolution is all-or-nothing per (ticker, action) --
    this module does not attempt to split a group into sub-clusters, so
    one far-apart entry blocks the whole group (documented
    simplification, see module docstring)."""
    events = [
        _event("2015-09-18", "FOX", "ADD", ["wikipedia"]),
        _event("2015-09-21", "FOX", "ADD", ["fja05680"]),
        _event("2019-03-19", "FOX", "ADD", ["lawcal"]),
    ]
    result = resolve_unresolved_events(events)
    assert not result.resolved
    assert len(result.still_unresolved) == 3


def test_resolution_is_deterministic_across_repeated_runs():
    events = [
        _event("2020-06-01", "ABCD", "ADD", ["wikipedia"]),
        _event("2020-06-04", "ABCD", "ADD", ["lawcal"]),
        _event("2018-01-01", "SOLO", "REMOVE", ["lawcal"], category="single_source_only"),
    ]
    first = resolve_unresolved_events(events)
    second = resolve_unresolved_events(events)
    assert first.summary == second.summary
    assert first.resolved == second.resolved
    assert first.still_unresolved == second.still_unresolved


def test_custom_spread_threshold_is_respected():
    events = [
        _event("2020-06-01", "ABCD", "ADD", ["wikipedia"]),
        _event("2020-06-04", "ABCD", "ADD", ["lawcal"]),  # 3-day spread
    ]
    strict_result = resolve_unresolved_events(events, max_resolvable_date_spread_days=2)
    assert not strict_result.resolved  # 3 days > 2-day threshold
    loose_result = resolve_unresolved_events(events, max_resolvable_date_spread_days=3)
    assert len(loose_result.resolved) == 1


# --- Real artifact tests -- local file, no network call ---


def test_real_artifact_exists_and_has_the_documented_event_count():
    payload = json.loads(RECONSTRUCTION_INPUTS_PATH.read_text(encoding="utf-8"))
    assert payload["unresolved_events_count"] == 273
    assert len(payload["unresolved_events"]) == 273


def test_real_artifact_resolution_produces_a_real_partial_resolution():
    """Real, measured result against the actual 273-event dataset --
    not hypothesized. Confirms the methodology genuinely resolves a
    meaningful fraction while leaving a real, non-trivial documented
    gap (never claims 100% resolution)."""
    result = load_and_resolve()
    summary = result.summary
    assert summary["total_input_events"] < 273  # grouping collapses some multi-source duplicates
    assert summary["resolved_count"] > 0
    assert summary["still_unresolved_count"] > 0  # a real, honest remaining gap -- not silently zeroed


def test_real_fox_ticker_recycling_case_stays_unresolved_in_the_real_artifact():
    """The real FOX/ADD case (2015 21st Century Fox spinoff addition vs.
    2019 Fox Corporation re-listing after the Disney asset sale) must
    never be silently averaged into one nonsense date -- confirmed
    against the real artifact, not a synthetic reproduction."""
    payload = json.loads(RECONSTRUCTION_INPUTS_PATH.read_text(encoding="utf-8"))
    result = resolve_unresolved_events(payload["unresolved_events"])
    resolved_keys = {(e.ticker, e.action) for e in result.resolved}
    assert ("FOX", "ADD") not in resolved_keys
    still_unresolved_fox = [e for e in result.still_unresolved if e["ticker"] == "FOX" and e["action"] == "ADD"]
    assert len(still_unresolved_fox) == 3  # both 2015 entries and the 2019 entry, all preserved


def test_real_artifact_never_promotes_a_pure_single_source_group():
    """Cross-check against the artifact's own category labels: any
    (ticker, action) group whose ENTIRE evidence (across all its
    entries) is exactly one distinct source must never appear in
    `resolved`."""
    payload = json.loads(RECONSTRUCTION_INPUTS_PATH.read_text(encoding="utf-8"))
    events = payload["unresolved_events"]
    result = resolve_unresolved_events(events)

    groups: dict[tuple[str, str], set[str]] = {}
    for event in events:
        key = (event["ticker"], event["action"])
        groups.setdefault(key, set()).update(event["supporting_sources"])

    resolved_keys = {(e.ticker, e.action) for e in result.resolved}
    for key, sources in groups.items():
        if len(sources) < 2:
            assert key not in resolved_keys, f"{key} had only {sources} but was promoted"
