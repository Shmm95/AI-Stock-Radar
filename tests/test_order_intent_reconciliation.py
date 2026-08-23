"""Tests for src/live/order_intent_reconciliation.py -- Task 3
(2026-08-22), item 6: post-crash recovery for stray order-intent
journal entries left by an interrupted prior run.

Deterministic, no network dependency -- a fake `TradingClient` with
only the two methods this module actually calls
(`get_order_by_client_order_id` via `order_submission`,
`get_order_status` via `order_submission`) is monkeypatched per test.
`order_intent.INTENT_DIRECTORY` is isolated to a per-test `tmp_path`,
same discipline as `test_write_ahead_intent_journal.py`.
"""

from __future__ import annotations

import pytest

import src.live.order_intent as order_intent
import src.live.order_intent_reconciliation as reconciliation
from src.live import order_submission


@pytest.fixture(autouse=True)
def _isolated_intent_directory(tmp_path, monkeypatch):
    intent_dir = tmp_path / "order_intents"
    monkeypatch.setattr(order_intent, "INTENT_DIRECTORY", intent_dir)
    return intent_dir


def _submission_intent(*, status: str, client_order_id: str = "AAPL-ENTRY-MARKET-BUY-2026-08-20") -> order_intent.OrderIntent:
    intent = order_intent.create_intent(
        client_order_id=client_order_id, account_identity="control_arm_v1", ticker="AAPL",
        side="BUY", order_type="market", action_kind="ENTRY_MARKET_BUY",
        source_signal_timestamp="2026-08-20T00:00:00Z",
    )
    if status != order_intent.PREPARED:
        intent = order_intent.transition_intent(intent, order_intent.SUBMITTING)
    return intent


def _cancel_intent(*, status: str, target_broker_order_id: str = "resting-stop-1") -> order_intent.OrderIntent:
    intent = order_intent.create_intent(
        client_order_id="", account_identity="control_arm_v1", ticker="AAPL",
        side="SELL", order_type="stop", action_kind="CANCEL_PROTECTIVE_STOP",
        source_signal_timestamp="2026-08-20T00:00:00Z",
        operation_id="AAPL-CANCEL-PROTECTIVE-STOP-resting-stop-1",
        target_broker_order_id=target_broker_order_id,
    )
    if status != order_intent.PREPARED:
        intent = order_intent.transition_intent(intent, order_intent.SUBMITTING)
    return intent


class _FakeOrder:
    def __init__(self, order_id: str, status: str = "accepted"):
        self.id = order_id
        self.status = status


# --- find_stray_session_intents_from_prior_run ---


def test_finds_only_non_terminal_intents():
    """independent-audit-round-2 (2026-08-23) test-quality note: the
    original version of this test only exercised PREPARED/SUBMITTING/
    TERMINAL -- BROKER_ACKNOWLEDGED, COMMITTED, and UNCERTAIN were never
    covered, even though `_NON_TERMINAL_STATUSES` (finding #2's own fix)
    includes all three. Expanded to prove every non-terminal status is
    actually found, and only TERMINAL is actually excluded."""
    prepared = _submission_intent(status=order_intent.PREPARED)
    submitting = _submission_intent(status=order_intent.SUBMITTING, client_order_id="MSFT-ENTRY-MARKET-BUY-2026-08-20")

    acknowledged = _submission_intent(status=order_intent.SUBMITTING, client_order_id="NVDA-ENTRY-MARKET-BUY-2026-08-20")
    acknowledged = order_intent.transition_intent(
        acknowledged, order_intent.BROKER_ACKNOWLEDGED, broker_order_id="real-1", broker_status="filled",
    )

    committed = _submission_intent(status=order_intent.SUBMITTING, client_order_id="TSLA-ENTRY-MARKET-BUY-2026-08-20")
    committed = order_intent.transition_intent(
        committed, order_intent.BROKER_ACKNOWLEDGED, broker_order_id="real-2", broker_status="filled",
    )
    committed = order_intent.transition_intent(committed, order_intent.COMMITTED)

    uncertain = _submission_intent(status=order_intent.SUBMITTING, client_order_id="AMD-ENTRY-MARKET-BUY-2026-08-20")
    uncertain = order_intent.transition_intent(uncertain, order_intent.UNCERTAIN, last_error="ambiguous")

    terminal_intent = order_intent.create_intent(
        client_order_id="GOOG-ENTRY-MARKET-BUY-2026-08-20", account_identity="control_arm_v1", ticker="GOOG",
        side="BUY", order_type="market", action_kind="ENTRY_MARKET_BUY", source_signal_timestamp="2026-08-20T00:00:00Z",
    )
    terminal_intent = order_intent.transition_intent(terminal_intent, order_intent.TERMINAL)

    stray = reconciliation.find_stray_session_intents_from_prior_run()
    stray_ids = {intent.intent_id for intent in stray}
    assert prepared.intent_id in stray_ids
    assert submitting.intent_id in stray_ids
    assert acknowledged.intent_id in stray_ids
    assert committed.intent_id in stray_ids
    assert uncertain.intent_id in stray_ids
    assert terminal_intent.intent_id not in stray_ids
    assert len(stray) == 5


# --- resolve_stray_order_submission_intent ---


def test_prepared_submission_found_at_broker_becomes_broker_acknowledged(monkeypatch: pytest.MonkeyPatch):
    intent = _submission_intent(status=order_intent.PREPARED)
    monkeypatch.setattr(order_submission, "get_order_by_client_order_id", lambda *a, **k: _FakeOrder("real-order-1", "filled"))

    resolved = reconciliation.resolve_stray_order_submission_intent(client=object(), intent=intent)

    assert resolved.status == order_intent.BROKER_ACKNOWLEDGED
    assert resolved.broker_order_id == "real-order-1"
    assert resolved.broker_status == "filled"
    reloaded = order_intent.load_intent(intent.intent_id)
    assert reloaded.status == order_intent.BROKER_ACKNOWLEDGED


def test_prepared_submission_confirmed_404_becomes_terminal_not_resubmitted(monkeypatch: pytest.MonkeyPatch):
    """A definitive negative from PREPARED -- the broker never received
    this exact submission -- is safe to close TERMINAL. Must NEVER call
    submit() itself; this module only queries, never submits."""
    intent = _submission_intent(status=order_intent.PREPARED)
    monkeypatch.setattr(order_submission, "get_order_by_client_order_id", lambda *a, **k: None)
    monkeypatch.setattr(
        order_submission, "submit_equity_market_order",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must never resubmit")),
    )

    resolved = reconciliation.resolve_stray_order_submission_intent(client=object(), intent=intent)

    assert resolved.status == order_intent.TERMINAL
    assert "never received" in resolved.last_error


def test_submitting_submission_found_at_broker_becomes_broker_acknowledged(monkeypatch: pytest.MonkeyPatch):
    intent = _submission_intent(status=order_intent.SUBMITTING)
    monkeypatch.setattr(order_submission, "get_order_by_client_order_id", lambda *a, **k: _FakeOrder("real-order-2"))

    resolved = reconciliation.resolve_stray_order_submission_intent(client=object(), intent=intent)

    assert resolved.status == order_intent.BROKER_ACKNOWLEDGED
    assert resolved.broker_order_id == "real-order-2"


def test_submitting_submission_not_found_is_uncertain_fail_closed(monkeypatch: pytest.MonkeyPatch):
    """REAL fail-closed proof: unlike the PREPARED case, a 404 from
    SUBMITTING is NOT treated as definitive -- the broker call may have
    been genuinely in flight when the prior run stopped. Must go to
    UNCERTAIN, never TERMINAL (which would silently let a real,
    in-flight order go unresolved forever), and never auto-resubmit."""
    intent = _submission_intent(status=order_intent.SUBMITTING)
    monkeypatch.setattr(order_submission, "get_order_by_client_order_id", lambda *a, **k: None)
    monkeypatch.setattr(
        order_submission, "submit_equity_market_order",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must never resubmit")),
    )

    resolved = reconciliation.resolve_stray_order_submission_intent(client=object(), intent=intent)

    assert resolved.status == order_intent.UNCERTAIN
    assert "ambiguous" in resolved.last_error.lower()


def test_already_broker_acknowledged_submission_returned_unchanged(monkeypatch: pytest.MonkeyPatch):
    """Deliberately NOT resolved further here -- "broker acknowledged,
    local state not caught up" is broker_reconciliation.py's own
    Scenario C job, not reimplemented in this module (see its own
    module docstring)."""
    intent = _submission_intent(status=order_intent.SUBMITTING)
    intent = order_intent.transition_intent(
        intent, order_intent.BROKER_ACKNOWLEDGED, broker_order_id="already-acked-1", broker_status="filled",
    )

    def fail_if_called(*a, **k):
        raise AssertionError("must not query the broker for an already-resolved intent")

    monkeypatch.setattr(order_submission, "get_order_by_client_order_id", fail_if_called)

    resolved = reconciliation.resolve_stray_order_submission_intent(client=object(), intent=intent)
    assert resolved.status == order_intent.BROKER_ACKNOWLEDGED
    assert resolved.broker_order_id == "already-acked-1"


# --- resolve_stray_cancel_intent ---


def test_prepared_cancel_target_still_resting_becomes_terminal(monkeypatch: pytest.MonkeyPatch):
    """A definitive negative from PREPARED -- the cancel was never
    attempted, target order is still genuinely open. Safe to close
    TERMINAL; must never auto-retry the cancel itself."""
    intent = _cancel_intent(status=order_intent.PREPARED)
    monkeypatch.setattr(order_submission, "get_order_status", lambda *a, **k: "new")
    monkeypatch.setattr(
        order_submission, "cancel_order_and_confirm",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must never auto-retry the cancel")),
    )

    resolved = reconciliation.resolve_stray_cancel_intent(client=object(), intent=intent)

    assert resolved.status == order_intent.TERMINAL
    assert "never attempted" in resolved.last_error


def test_prepared_cancel_target_now_canceled_becomes_broker_acknowledged(monkeypatch: pytest.MonkeyPatch):
    intent = _cancel_intent(status=order_intent.PREPARED)
    monkeypatch.setattr(order_submission, "get_order_status", lambda *a, **k: "canceled")

    resolved = reconciliation.resolve_stray_cancel_intent(client=object(), intent=intent)

    assert resolved.status == order_intent.BROKER_ACKNOWLEDGED
    assert resolved.broker_status == "canceled"


def test_submitting_cancel_target_still_resting_is_uncertain_fail_closed(monkeypatch: pytest.MonkeyPatch):
    """REAL fail-closed proof, cancel side: Alpaca's own cancellation is
    asynchronous (a real paper stop can briefly still read `new`
    immediately after a real cancel request, per
    cancel_order_and_confirm's own docstring) -- SUBMITTING + still
    resting is genuinely ambiguous, must be UNCERTAIN, never TERMINAL
    and never an auto-retried cancel."""
    intent = _cancel_intent(status=order_intent.SUBMITTING)
    monkeypatch.setattr(order_submission, "get_order_status", lambda *a, **k: "new")
    monkeypatch.setattr(
        order_submission, "cancel_order_and_confirm",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must never auto-retry the cancel")),
    )

    resolved = reconciliation.resolve_stray_cancel_intent(client=object(), intent=intent)

    assert resolved.status == order_intent.UNCERTAIN
    assert "ambiguous" in resolved.last_error.lower()


def test_target_filled_before_cancel_landed_is_honestly_recorded_not_relabeled(monkeypatch: pytest.MonkeyPatch):
    """The real race _execute_equity_orders's own needs_review path
    already documents (the stop triggered before the cancel could take
    effect) -- a definitive outcome, must be recorded honestly as
    'filled', never silently relabeled as 'canceled'."""
    intent = _cancel_intent(status=order_intent.PREPARED)
    monkeypatch.setattr(order_submission, "get_order_status", lambda *a, **k: "filled")

    resolved = reconciliation.resolve_stray_cancel_intent(client=object(), intent=intent)

    assert resolved.status == order_intent.BROKER_ACKNOWLEDGED
    assert resolved.broker_status == "filled"


# --- resolve_stray_intent dispatch ---


def test_dispatch_routes_submission_type_by_client_order_id(monkeypatch: pytest.MonkeyPatch):
    """`resolve_stray_intent` (the dispatcher) always finalizes through
    to TERMINAL now (independent-audit-round-2 finding #1, 2026-08-23)
    -- unlike `resolve_stray_order_submission_intent` on its own, which
    deliberately still stops at BROKER_ACKNOWLEDGED (see that function's
    own, still-passing `test_already_broker_acknowledged_submission_returned_unchanged`)."""
    intent = _submission_intent(status=order_intent.PREPARED)
    monkeypatch.setattr(order_submission, "get_order_by_client_order_id", lambda *a, **k: _FakeOrder("x"))
    resolved = reconciliation.resolve_stray_intent(client=object(), intent=intent)
    assert resolved.status == order_intent.TERMINAL
    reloaded = order_intent.load_intent(intent.intent_id)
    assert reloaded.status == order_intent.TERMINAL
    assert reloaded.broker_order_id == "x"  # the real broker fact is still preserved through finalization


def test_dispatch_routes_cancel_type_by_target_broker_order_id(monkeypatch: pytest.MonkeyPatch):
    intent = _cancel_intent(status=order_intent.PREPARED)
    monkeypatch.setattr(order_submission, "get_order_status", lambda *a, **k: "canceled")
    resolved = reconciliation.resolve_stray_intent(client=object(), intent=intent)
    assert resolved.status == order_intent.TERMINAL
    reloaded = order_intent.load_intent(intent.intent_id)
    assert reloaded.broker_status == "canceled"


# --- finalize_resolved_intent (independent-audit-round-2 findings #1/#2, 2026-08-23) ---


def test_finalize_closes_broker_acknowledged_through_to_terminal():
    """The exact bug finding #1 describes: a stray intent resolved to
    (or already sitting at) BROKER_ACKNOWLEDGED previously had no path
    forward and stayed there permanently."""
    intent = _submission_intent(status=order_intent.SUBMITTING)
    intent = order_intent.transition_intent(
        intent, order_intent.BROKER_ACKNOWLEDGED, broker_order_id="real-3", broker_status="filled",
    )
    finalized = reconciliation.finalize_resolved_intent(intent)
    assert finalized.status == order_intent.TERMINAL
    reloaded = order_intent.load_intent(intent.intent_id)
    assert reloaded.status == order_intent.TERMINAL
    assert reloaded.broker_order_id == "real-3"


def test_finalize_closes_committed_through_to_terminal():
    """Finding #2's own crash window: a prior run's second save
    (COMMITTED) landed, but the run crashed before the final TERMINAL
    write."""
    intent = _submission_intent(status=order_intent.SUBMITTING)
    intent = order_intent.transition_intent(
        intent, order_intent.BROKER_ACKNOWLEDGED, broker_order_id="real-4", broker_status="filled",
    )
    intent = order_intent.transition_intent(intent, order_intent.COMMITTED)
    finalized = reconciliation.finalize_resolved_intent(intent)
    assert finalized.status == order_intent.TERMINAL


def test_finalize_is_a_no_op_for_prepared_submitting_uncertain_terminal():
    prepared = _submission_intent(status=order_intent.PREPARED)
    assert reconciliation.finalize_resolved_intent(prepared).status == order_intent.PREPARED

    submitting = _submission_intent(status=order_intent.SUBMITTING, client_order_id="MSFT-ENTRY-MARKET-BUY-2026-08-20")
    assert reconciliation.finalize_resolved_intent(submitting).status == order_intent.SUBMITTING

    uncertain = _submission_intent(status=order_intent.SUBMITTING, client_order_id="AMD-ENTRY-MARKET-BUY-2026-08-20")
    uncertain = order_intent.transition_intent(uncertain, order_intent.UNCERTAIN, last_error="ambiguous")
    assert reconciliation.finalize_resolved_intent(uncertain).status == order_intent.UNCERTAIN

    terminal = _submission_intent(status=order_intent.PREPARED, client_order_id="GOOG-ENTRY-MARKET-BUY-2026-08-20")
    terminal = order_intent.transition_intent(terminal, order_intent.TERMINAL)
    assert reconciliation.finalize_resolved_intent(terminal).status == order_intent.TERMINAL


def test_resolve_stray_intent_finalizes_an_already_acknowledged_stray_from_a_prior_run(monkeypatch: pytest.MonkeyPatch):
    """The precise scenario finding #1 names: a stray intent that was
    ALREADY BROKER_ACKNOWLEDGED when found on disk (left over from a
    prior run, never advanced further) -- `resolve_stray_intent` must
    still finalize it through to TERMINAL, not merely return it
    unchanged. Never queries the broker again for an already-resolved
    intent (see `resolve_stray_order_submission_intent`'s own
    docstring) -- only finalization runs here."""
    intent = _submission_intent(status=order_intent.SUBMITTING)
    intent = order_intent.transition_intent(
        intent, order_intent.BROKER_ACKNOWLEDGED, broker_order_id="already-acked-2", broker_status="filled",
    )

    def fail_if_called(*a, **k):
        raise AssertionError("must not query the broker for an already-resolved intent")

    monkeypatch.setattr(order_submission, "get_order_by_client_order_id", fail_if_called)

    resolved = reconciliation.resolve_stray_intent(client=object(), intent=intent)

    assert resolved.status == order_intent.TERMINAL
    assert resolved.broker_order_id == "already-acked-2"


def test_dispatch_raises_for_intent_with_neither_key():
    intent = order_intent.create_intent(
        client_order_id="", account_identity="control_arm_v1", ticker="AAPL",
        side="SELL", order_type="stop", action_kind="CANCEL_PROTECTIVE_STOP",
        source_signal_timestamp="2026-08-20T00:00:00Z",
    )
    with pytest.raises(ValueError, match="neither"):
        reconciliation.resolve_stray_intent(client=object(), intent=intent)
