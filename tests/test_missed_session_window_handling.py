"""Tests for `src/live/missed_session_window_handling.py` -- "Missing-
Equity-Session Remediation" design (workspace-c), section C. Covers the
module's own narrow, deliberately-simplified contract: a stale pending
BUY/EXIT queued FROM `last_processed_session_before` (the cursor value
immediately before the gap began -- the only date any currently-pending
order could genuinely have been queued from) is expired/cleared, and
(fixed independent-audit-round-3 finding #5, with a correction found
while writing these very tests -- see the module's own docstring
"SELECTIVITY" section) a pending entry queued from a DIFFERENT session
is left untouched, not swept up by an overly-broad "clear everything"
pass. `missed_session_dates` is kept as a parameter but is NOT the
membership test -- it never correctly could be, since it lists the
missed sessions themselves, never the (earlier) date a pending order's
own `signal.timestamp` actually carries.
"""

from __future__ import annotations

from types import SimpleNamespace

import src.live.missed_session_window_handling as missed_window


def _pending(queued_from: str) -> SimpleNamespace:
    return SimpleNamespace(signal=SimpleNamespace(timestamp=queued_from))


def _runner_state(pending_buys: dict, pending_exits: dict) -> SimpleNamespace:
    return SimpleNamespace(pending_buys=dict(pending_buys), pending_exits=dict(pending_exits), positions={"HELD": object()})


def test_expires_pending_buys_queued_from_the_last_processed_session():
    state = _runner_state({"AAPL": _pending("2026-08-18"), "MSFT": _pending("2026-08-18")}, {})
    outcome = missed_window.handle_missed_execution_window(
        state, missed_session_dates=("2026-08-19",), last_processed_session_before="2026-08-18",
    )
    assert state.pending_buys == {}
    assert outcome.expired_buy_tickers == ("AAPL", "MSFT")


def test_clears_pending_exits_queued_from_the_last_processed_session():
    state = _runner_state({}, {"NVDA": _pending("2026-08-18")})
    outcome = missed_window.handle_missed_execution_window(
        state, missed_session_dates=("2026-08-19",), last_processed_session_before="2026-08-18",
    )
    assert state.pending_exits == {}
    assert outcome.cleared_for_reconfirmation_exit_tickers == ("NVDA",)


def test_leaves_a_pending_buy_queued_from_an_earlier_session_untouched():
    """independent-audit-round-3 finding #5: the real over-clearing bug
    -- a pending BUY queued from a session OTHER than
    `last_processed_session_before` must never be swept up just because
    some session was missed."""
    state = _runner_state({"AAPL": _pending("2026-08-18"), "MSFT": _pending("2026-08-17")}, {})
    outcome = missed_window.handle_missed_execution_window(
        state, missed_session_dates=("2026-08-19",), last_processed_session_before="2026-08-18",
    )
    assert state.pending_buys == {"MSFT": state.pending_buys["MSFT"]}
    assert outcome.expired_buy_tickers == ("AAPL",)


def test_leaves_a_pending_exit_queued_from_an_earlier_session_untouched():
    state = _runner_state({}, {"AAPL": _pending("2026-08-18"), "NVDA": _pending("2026-08-17")})
    outcome = missed_window.handle_missed_execution_window(
        state, missed_session_dates=("2026-08-19",), last_processed_session_before="2026-08-18",
    )
    assert state.pending_exits == {"NVDA": state.pending_exits["NVDA"]}
    assert outcome.cleared_for_reconfirmation_exit_tickers == ("AAPL",)


def test_missed_session_dates_membership_is_never_the_test_even_if_it_happens_to_match():
    """A regression guard against re-introducing the FIRST (also wrong)
    fix attempt: a pending order whose `signal.timestamp` happens to
    equal one of `missed_session_dates` (rather than
    `last_processed_session_before`) must NOT be cleared -- that date
    combination should not occur in practice (see module docstring), but
    this proves the actual code checks the right field, not this one."""
    state = _runner_state({"AAPL": _pending("2026-08-19")}, {})  # matches missed_session_dates, NOT last_processed_session_before
    outcome = missed_window.handle_missed_execution_window(
        state, missed_session_dates=("2026-08-19",), last_processed_session_before="2026-08-18",
    )
    assert outcome.touched_anything is False
    assert state.pending_buys == {"AAPL": state.pending_buys["AAPL"]}


def test_never_touches_open_positions():
    state = _runner_state({"AAPL": _pending("2026-08-18")}, {"NVDA": _pending("2026-08-18")})
    missed_window.handle_missed_execution_window(
        state, missed_session_dates=("2026-08-19",), last_processed_session_before="2026-08-18",
    )
    assert state.positions == {"HELD": state.positions["HELD"]}


def test_no_op_when_nothing_pending():
    state = _runner_state({}, {})
    outcome = missed_window.handle_missed_execution_window(
        state, missed_session_dates=("2026-08-19",), last_processed_session_before="2026-08-18",
    )
    assert outcome.touched_anything is False
    assert outcome.expired_buy_tickers == ()
    assert outcome.cleared_for_reconfirmation_exit_tickers == ()


def test_touched_anything_true_when_something_was_cleared():
    state = _runner_state({"AAPL": _pending("2026-08-18")}, {})
    outcome = missed_window.handle_missed_execution_window(
        state, missed_session_dates=("2026-08-19",), last_processed_session_before="2026-08-18",
    )
    assert outcome.touched_anything is True
