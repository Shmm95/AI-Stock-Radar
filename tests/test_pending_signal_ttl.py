"""Tests for src/live/pending_signal_ttl.py -- currently covers the
stuck-`reconfirming`-record recovery path only (added after an
independent integrated-chain audit found a real, latent bug: a
`pending_signal_metadata` record could be left in the transient
`reconfirming` status forever if a run failed between
`evaluate_pending_signals` marking it and `finalize_after_run` resolving
it, since its corresponding `pending_exits` entry is already gone by
then and the normal per-ticker loops can never re-encounter it).

Deterministic, no network/broker dependency -- `pending_signal_ttl.py`
itself never calls Alpaca directly (see its own module docstring).
"""

from __future__ import annotations

import pytest

import src.live.pending_signal_ttl as ttl
import src.live.position_state as ps
from src.backtest.portfolio_backtest_engine import _MutablePosition, _PendingOrder
from src.backtest.portfolio_backtest_models import PortfolioSignal


def _stuck_record() -> dict:
    return {
        "signal_id": "XYZ-QUEUED-EXIT-SIGNAL-2026-08-15",
        "source_session_date": "2026-08-15",
        "target_execution_session_date": "2026-08-16",
        "created_at_utc": "2026-08-15T20:00:00Z",
        "status": "reconfirming",
        "expire_reason": None,
    }


def _position(ticker: str) -> _MutablePosition:
    return _MutablePosition(
        ticker=ticker, asset_class="EQUITY", entry_timestamp="2026-08-15T00:00:00Z",
        entry_portfolio_bar_index=1, quantity=5.0, entry_price=20.0, entry_fee=0.0,
        stop_loss_price=18.0, highest_close=20.0, trailing_close_percent=0.0,
        initial_risk_amount=10.0, signal_score=0.0, signal_reason="test",
    )


def test_stuck_reconfirming_record_fails_closed():
    """No corresponding pending_exits entry (an interrupted prior run
    already cleared it before crashing) -- cannot be safely auto-resolved."""
    runner_state = ps.LiveRunnerState()
    runner_state.positions["XYZ"] = _position("XYZ")
    runner_state.pending_signal_metadata["XYZ|EXIT"] = _stuck_record()

    with pytest.raises(ttl.PendingSignalReconciliationRequiredError, match="XYZ"):
        ttl.evaluate_pending_signals(
            runner_state=runner_state,
            expected_session_date="2026-08-17",
            freeze_active=False,
            has_unresolved_journal_trace_for_signal=lambda sid: False,
            has_unresolved_journal_trace_for_ticker=lambda t, k: False,
        )


def test_reconfirming_status_with_active_pending_exit_is_not_stuck():
    """Contrast case: the SAME 'reconfirming' status, but the ticker IS
    currently in pending_exits -- this is a legitimate, in-progress
    reconfirmation from THIS SAME run, not an orphan. Must go through the
    normal stale-exit path instead of the stuck-recovery raise."""
    runner_state = ps.LiveRunnerState()
    runner_state.positions["XYZ"] = _position("XYZ")
    runner_state.pending_exits["XYZ"] = _PendingOrder(
        signal=PortfolioSignal(timestamp="2026-08-16", ticker="XYZ", action="EXIT", reference_price=19.0, reason="test"),
        submitted_portfolio_bar_index=5,
    )
    runner_state.pending_signal_metadata["XYZ|EXIT"] = _stuck_record()

    outcome = ttl.evaluate_pending_signals(
        runner_state=runner_state,
        expected_session_date="2026-08-17",
        freeze_active=False,
        has_unresolved_journal_trace_for_signal=lambda sid: False,
        has_unresolved_journal_trace_for_ticker=lambda t, k: False,
    )
    assert "XYZ" in outcome.reconfirming_exits
    assert "XYZ" not in runner_state.pending_exits  # cleared for this run's own reconfirmation


def test_find_stuck_reconfirming_records_ignores_pending_status():
    """A normal, non-transient 'pending' record must never be mistaken
    for a stuck one."""
    runner_state = ps.LiveRunnerState()
    runner_state.pending_signal_metadata["ABC|EXIT"] = {
        **_stuck_record(),
        "status": ttl.STATUS_PENDING,
    }
    assert ttl._find_stuck_reconfirming_records(runner_state) == []


def test_find_stuck_reconfirming_records_ignores_buy_kind():
    """_STATUS_RECONFIRMING is only ever set for EXIT keys -- a BUY key
    with that status (should never happen in practice) is still not
    treated as a stuck EXIT recovery case."""
    runner_state = ps.LiveRunnerState()
    runner_state.pending_signal_metadata["ABC|BUY"] = _stuck_record()
    assert ttl._find_stuck_reconfirming_records(runner_state) == []
