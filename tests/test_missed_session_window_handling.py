"""Tests for `src/live/missed_session_window_handling.py` -- "Missing-
Equity-Session Remediation" design (workspace-c), section C. Covers the
module's own narrow, deliberately-simplified contract: stale pending
BUYs are expired outright, stale pending EXITs are cleared (handed back
to the next real run's own unmodified engine logic, never
independently recomputed here), and neither `positions` nor any broker
state is ever touched -- see `broker_reconciliation.py`'s own Scenario
C as the sole authority for real fills during the gap.
"""

from __future__ import annotations

from types import SimpleNamespace

import src.live.missed_session_window_handling as missed_window


def _runner_state(pending_buys: dict, pending_exits: dict) -> SimpleNamespace:
    return SimpleNamespace(pending_buys=dict(pending_buys), pending_exits=dict(pending_exits), positions={"HELD": object()})


def test_expires_all_pending_buys():
    state = _runner_state({"AAPL": {"signal": "x"}, "MSFT": {"signal": "y"}}, {})
    outcome = missed_window.handle_missed_execution_window(state, missed_session_dates=("2026-08-20",))
    assert state.pending_buys == {}
    assert outcome.expired_buy_tickers == ("AAPL", "MSFT")


def test_clears_all_pending_exits():
    state = _runner_state({}, {"NVDA": {"reason": "trend"}})
    outcome = missed_window.handle_missed_execution_window(state, missed_session_dates=("2026-08-20",))
    assert state.pending_exits == {}
    assert outcome.cleared_for_reconfirmation_exit_tickers == ("NVDA",)


def test_never_touches_open_positions():
    state = _runner_state({"AAPL": {}}, {"NVDA": {}})
    missed_window.handle_missed_execution_window(state, missed_session_dates=("2026-08-20",))
    assert state.positions == {"HELD": state.positions["HELD"]}


def test_no_op_when_nothing_pending():
    state = _runner_state({}, {})
    outcome = missed_window.handle_missed_execution_window(state, missed_session_dates=("2026-08-20",))
    assert outcome.touched_anything is False
    assert outcome.expired_buy_tickers == ()
    assert outcome.cleared_for_reconfirmation_exit_tickers == ()


def test_touched_anything_true_when_something_was_cleared():
    state = _runner_state({"AAPL": {}}, {})
    outcome = missed_window.handle_missed_execution_window(state, missed_session_dates=("2026-08-20",))
    assert outcome.touched_anything is True
