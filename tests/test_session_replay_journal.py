"""Tests for `src/live/session_replay_journal.py` -- "Missing-Equity-
Session Remediation" design (workspace-c), section D. Covers the
PREPARED -> VERIFIED -> COMMITTED -> TERMINAL state machine (and its
direct-to-TERMINAL abort branches from PREPARED/VERIFIED), the
deterministic `session_id`/`intent_id` construction, and crash-recovery
listing/lookup.

`SESSION_REPLAY_DIRECTORY` is monkeypatched to a per-test tmp_path --
same isolation discipline `tests/test_write_ahead_intent_journal.py`
already established for `order_intent.INTENT_DIRECTORY`, never touching
the real out-of-repo guard directory.
"""

from __future__ import annotations

import pytest

import src.live.session_replay_journal as session_journal


@pytest.fixture(autouse=True)
def _isolated_session_replay_directory(tmp_path, monkeypatch):
    directory = tmp_path / "session_replay_intents"
    monkeypatch.setattr(session_journal, "SESSION_REPLAY_DIRECTORY", directory)


def test_build_session_id_is_deterministic_and_includes_bar_hash():
    session_id = session_journal.build_session_id("2026-08-20", "abc123")
    assert session_id == "EQUITY_SESSION:2026-08-20:abc123"
    # A different bar-set hash for the SAME calendar date is a genuinely
    # different session_id -- never conflated (see module docstring).
    assert session_journal.build_session_id("2026-08-20", "def456") != session_id


def test_create_session_intent_starts_prepared_and_persists():
    intent = session_journal.create_session_intent(
        session_id="EQUITY_SESSION:2026-08-20:abc123",
        session_date="2026-08-20",
        bar_set_sha256="abc123",
        replay_sequence_index=1,
        replay_sequence_total=1,
    )
    assert intent.status == session_journal.PREPARED
    reloaded = session_journal.load_session_intent(intent.intent_id)
    assert reloaded == intent


def test_full_happy_path_prepared_to_terminal():
    intent = session_journal.create_session_intent(
        session_id="EQUITY_SESSION:2026-08-20:abc123",
        session_date="2026-08-20",
        bar_set_sha256="abc123",
        replay_sequence_index=1,
        replay_sequence_total=1,
        state_hash_before="hash-before",
    )
    intent = session_journal.transition_session_intent(intent, session_journal.VERIFIED)
    assert intent.status == session_journal.VERIFIED
    intent = session_journal.transition_session_intent(
        intent, session_journal.COMMITTED, state_hash_after="hash-after"
    )
    assert intent.status == session_journal.COMMITTED
    assert intent.state_hash_after == "hash-after"
    intent = session_journal.transition_session_intent(intent, session_journal.TERMINAL, outcome=session_journal.SUCCEEDED)
    assert intent.status == session_journal.TERMINAL
    assert intent.outcome == session_journal.SUCCEEDED
    assert intent.last_reconciled_at is not None


def test_prepared_can_abort_directly_to_terminal_with_error():
    intent = session_journal.create_session_intent(
        session_id="EQUITY_SESSION:2026-08-20:abc123",
        session_date="2026-08-20",
        bar_set_sha256="abc123",
        replay_sequence_index=1,
        replay_sequence_total=1,
    )
    intent = session_journal.transition_session_intent(
        intent, session_journal.TERMINAL, last_error="run_daily_decision() raised", outcome=session_journal.FAILED,
    )
    assert intent.status == session_journal.TERMINAL
    assert intent.last_error == "run_daily_decision() raised"
    assert intent.outcome == session_journal.FAILED


def test_verified_can_abort_directly_to_terminal():
    intent = session_journal.create_session_intent(
        session_id="EQUITY_SESSION:2026-08-20:abc123",
        session_date="2026-08-20",
        bar_set_sha256="abc123",
        replay_sequence_index=1,
        replay_sequence_total=1,
    )
    intent = session_journal.transition_session_intent(intent, session_journal.VERIFIED)
    intent = session_journal.transition_session_intent(
        intent, session_journal.TERMINAL, last_error="mismatch", outcome=session_journal.FAILED,
    )
    assert intent.status == session_journal.TERMINAL


def test_terminal_requires_an_explicit_outcome():
    """independent-audit-round-3 finding #3b: a TERMINAL record must
    always say which of SUCCEEDED/FAILED it is -- omitting `outcome`
    (or passing something else) is rejected, never silently defaulted."""
    intent = session_journal.create_session_intent(
        session_id="EQUITY_SESSION:2026-08-20:abc123",
        session_date="2026-08-20",
        bar_set_sha256="abc123",
        replay_sequence_index=1,
        replay_sequence_total=1,
    )
    with pytest.raises(ValueError):
        session_journal.transition_session_intent(intent, session_journal.TERMINAL)


def test_outcome_rejected_for_a_non_terminal_transition():
    intent = session_journal.create_session_intent(
        session_id="EQUITY_SESSION:2026-08-20:abc123",
        session_date="2026-08-20",
        bar_set_sha256="abc123",
        replay_sequence_index=1,
        replay_sequence_total=1,
    )
    with pytest.raises(ValueError):
        session_journal.transition_session_intent(intent, session_journal.VERIFIED, outcome=session_journal.SUCCEEDED)


def test_prepared_cannot_transition_directly_to_committed():
    intent = session_journal.create_session_intent(
        session_id="EQUITY_SESSION:2026-08-20:abc123",
        session_date="2026-08-20",
        bar_set_sha256="abc123",
        replay_sequence_index=1,
        replay_sequence_total=1,
    )
    with pytest.raises(session_journal.InvalidSessionTransitionError):
        session_journal.transition_session_intent(intent, session_journal.COMMITTED)


def test_terminal_is_a_true_dead_end():
    intent = session_journal.create_session_intent(
        session_id="EQUITY_SESSION:2026-08-20:abc123",
        session_date="2026-08-20",
        bar_set_sha256="abc123",
        replay_sequence_index=1,
        replay_sequence_total=1,
    )
    intent = session_journal.transition_session_intent(intent, session_journal.TERMINAL, outcome=session_journal.SUCCEEDED)
    with pytest.raises(session_journal.InvalidSessionTransitionError):
        session_journal.transition_session_intent(intent, session_journal.VERIFIED)


def test_find_session_intent_by_id_and_list_session_intents():
    assert session_journal.list_session_intents() == []
    assert session_journal.find_session_intent_by_id("EQUITY_SESSION:2026-08-20:abc123") is None
    intent = session_journal.create_session_intent(
        session_id="EQUITY_SESSION:2026-08-20:abc123",
        session_date="2026-08-20",
        bar_set_sha256="abc123",
        replay_sequence_index=1,
        replay_sequence_total=1,
    )
    assert session_journal.find_session_intent_by_id("EQUITY_SESSION:2026-08-20:abc123") == intent
    assert session_journal.list_session_intents() == [intent]
