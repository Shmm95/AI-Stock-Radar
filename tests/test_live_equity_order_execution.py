"""Tests for the equity-only real-order orchestration in run_daily_decision.py.

Deterministic and network-free: `src.live.order_submission`'s functions
are monkeypatched (never the real Alpaca client), so these exercise
the idempotency, protective-stop-placement, and stop-cancellation
logic without depending on market hours, a live account, or an actual
order lifecycle. See the PR-level report for why a small number of
real paper-API smoke calls were used *in addition to* this file rather
than instead of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_daily_decision as runner  # noqa: E402

from alpaca.trading.enums import OrderSide  # noqa: E402
from src.backtest.portfolio_backtest_engine import _MutablePosition  # noqa: E402
from src.backtest.portfolio_backtest_models import PortfolioTrade  # noqa: E402
from src.live import order_submission  # noqa: E402
from src.live.position_state import LiveRunnerState  # noqa: E402


class FakeOrder:
    def __init__(self, order_id: str):
        self.id = order_id


def equity_position(ticker: str = "AAPL", quantity: float = 10.0) -> _MutablePosition:
    return _MutablePosition(
        ticker=ticker,
        asset_class="EQUITY",
        entry_timestamp="2026-08-10",
        entry_portfolio_bar_index=5,
        quantity=quantity,
        entry_price=190.0,
        entry_fee=1.0,
        stop_loss_price=180.5,
        highest_close=190.0,
        trailing_close_percent=7.5,
        initial_risk_amount=95.0,
        signal_score=80.0,
        signal_reason="TEST",
    )


def equity_trade(
    ticker: str = "AAPL", exit_reason: str = "EXIT_SIGNAL_NEXT_OPEN", quantity: float = 10.0
) -> PortfolioTrade:
    return PortfolioTrade(
        ticker=ticker,
        asset_class="EQUITY",
        entry_timestamp="2026-08-10",
        exit_timestamp="2026-08-11",
        entry_portfolio_bar_index=5,
        exit_portfolio_bar_index=6,
        quantity=quantity,
        entry_price=190.0,
        exit_price=195.0,
        entry_fee=1.0,
        exit_fee=1.0,
        total_fees=2.0,
        gross_pnl=50.0,
        net_pnl=48.0,
        return_percent=2.5,
        holding_period_bars=1,
        exit_reason=exit_reason,
        signal_score=80.0,
        signal_reason="TEST",
    )


def test_entry_buy_fills_and_places_stop(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(order_submission, "get_order_by_client_order_id", lambda *a, **k: None)
    monkeypatch.setattr(
        order_submission, "submit_equity_market_order", lambda *a, **k: FakeOrder("entry-1")
    )
    monkeypatch.setattr(order_submission, "wait_for_fill_or_timeout", lambda *a, **k: "filled")
    monkeypatch.setattr(
        order_submission, "submit_equity_stop_sell", lambda *a, **k: FakeOrder("stop-1")
    )

    state = LiveRunnerState()
    actions, review = runner._execute_equity_orders(
        client=None,
        runner_state=state,
        equity_date="2026-08-14",
        newly_opened={"AAPL": equity_position()},
        closed_trades=[],
    )

    assert review == []
    kinds = {(a["ticker"], a["action"]) for a in actions}
    assert ("AAPL", "BUY") in kinds
    assert ("AAPL", "PLACE_STOP") in kinds
    assert state.equity_stop_orders["AAPL"] == "stop-1"
    assert state.submitted_actions["AAPL|ENTRY_MARKET_BUY|2026-08-14"]["status"] == "filled"
    assert state.submitted_actions["AAPL|PROTECTIVE_STOP|2026-08-14"]["order_id"] == "stop-1"


def test_rerun_for_same_bar_never_resubmits(monkeypatch: pytest.MonkeyPatch):
    calls = {"submit": 0}

    def fail_if_called(*a, **k):
        calls["submit"] += 1
        return FakeOrder(f"entry-{calls['submit']}")

    monkeypatch.setattr(order_submission, "get_order_by_client_order_id", lambda *a, **k: None)
    monkeypatch.setattr(order_submission, "submit_equity_market_order", fail_if_called)
    monkeypatch.setattr(order_submission, "wait_for_fill_or_timeout", lambda *a, **k: "filled")
    monkeypatch.setattr(
        order_submission, "submit_equity_stop_sell", lambda *a, **k: FakeOrder("stop-1")
    )
    monkeypatch.setattr(order_submission, "get_order_status", lambda *a, **k: "filled")

    state = LiveRunnerState()
    runner._execute_equity_orders(
        client=None, runner_state=state, equity_date="2026-08-14",
        newly_opened={"AAPL": equity_position()}, closed_trades=[],
    )
    assert calls["submit"] == 1

    # Simulate a re-run for the exact same bar (e.g. after a crash/retry).
    actions, _ = runner._execute_equity_orders(
        client=None, runner_state=state, equity_date="2026-08-14",
        newly_opened={"AAPL": equity_position()}, closed_trades=[],
    )

    assert calls["submit"] == 1, "must not submit a second real order for the same bar"
    assert state.submitted_actions["AAPL|ENTRY_MARKET_BUY|2026-08-14"]["order_id"] == "entry-1"
    buy_actions = [a for a in actions if a["action"] == "BUY"]
    assert buy_actions[0]["order_id"] == "entry-1"


def test_stop_placement_deferred_when_entry_not_yet_filled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(order_submission, "get_order_by_client_order_id", lambda *a, **k: None)
    monkeypatch.setattr(
        order_submission, "submit_equity_market_order", lambda *a, **k: FakeOrder("entry-1")
    )
    monkeypatch.setattr(order_submission, "wait_for_fill_or_timeout", lambda *a, **k: "accepted")

    def fail_if_stop_submitted(*a, **k):
        raise AssertionError("must not place a stop before the entry confirms filled")

    monkeypatch.setattr(order_submission, "submit_equity_stop_sell", fail_if_stop_submitted)

    state = LiveRunnerState()
    actions, review = runner._execute_equity_orders(
        client=None, runner_state=state, equity_date="2026-08-14",
        newly_opened={"AAPL": equity_position()}, closed_trades=[],
    )

    assert "AAPL" not in state.equity_stop_orders
    assert not any(a["action"] == "PLACE_STOP" for a in actions)
    assert review and review[0]["ticker"] == "AAPL"
    assert "deferred" in review[0]["issue"]


def test_reconciliation_places_stop_once_deferred_entry_confirms(monkeypatch: pytest.MonkeyPatch):
    state = LiveRunnerState(
        portfolio_bar_index=3,
        positions={"AAPL": equity_position()},
        submitted_actions={
            "AAPL|ENTRY_MARKET_BUY|3": {
                "order_id": "entry-1", "status": "accepted",
                "kind": "ENTRY_MARKET_BUY", "submitted_at": "x",
            }
        },
    )
    monkeypatch.setattr(order_submission, "get_order_status", lambda *a, **k: "filled")
    monkeypatch.setattr(order_submission, "get_order_by_client_order_id", lambda *a, **k: None)
    monkeypatch.setattr(
        order_submission, "submit_equity_stop_sell", lambda *a, **k: FakeOrder("stop-1")
    )

    review = runner._reconcile_pending_equity_orders(client=None, runner_state=state)

    assert review == []
    assert state.equity_stop_orders["AAPL"] == "stop-1"
    assert state.submitted_actions["AAPL|ENTRY_MARKET_BUY|3"]["status"] == "filled"


def test_reconciliation_flags_rejected_entry(monkeypatch: pytest.MonkeyPatch):
    state = LiveRunnerState(
        portfolio_bar_index=3,
        positions={"AAPL": equity_position()},
        submitted_actions={
            "AAPL|ENTRY_MARKET_BUY|3": {
                "order_id": "entry-1", "status": "accepted",
                "kind": "ENTRY_MARKET_BUY", "submitted_at": "x",
            }
        },
    )
    monkeypatch.setattr(order_submission, "get_order_status", lambda *a, **k: "rejected")

    review = runner._reconcile_pending_equity_orders(client=None, runner_state=state)

    assert len(review) == 1
    assert review[0]["ticker"] == "AAPL"
    assert review[0]["status"] == "rejected"
    assert "AAPL" not in state.equity_stop_orders


def test_signal_exit_cancels_stop_before_selling_and_confirms_cancellation(
    monkeypatch: pytest.MonkeyPatch,
):
    call_order = []
    monkeypatch.setattr(order_submission, "get_order_by_client_order_id", lambda *a, **k: None)
    monkeypatch.setattr(
        order_submission, "submit_equity_market_order",
        lambda *a, **k: (call_order.append("SELL"), FakeOrder("exit-1"))[1],
    )
    monkeypatch.setattr(order_submission, "wait_for_fill_or_timeout", lambda *a, **k: "filled")
    monkeypatch.setattr(
        order_submission, "cancel_order_and_confirm",
        lambda client, order_id: (call_order.append("CANCEL"), "canceled")[1],
    )

    state = LiveRunnerState(equity_stop_orders={"AAPL": "stop-1"})
    actions, review = runner._execute_equity_orders(
        client=None, runner_state=state, equity_date="2026-08-15",
        newly_opened={}, closed_trades=[equity_trade(exit_reason="EXIT_SIGNAL_NEXT_OPEN")],
    )

    assert review == []
    assert call_order == ["CANCEL", "SELL"], "must cancel the stop BEFORE selling, not after"
    assert "AAPL" not in state.equity_stop_orders
    kinds = {a["action"] for a in actions}
    assert "SELL" in kinds
    assert "CANCEL_STOP" in kinds
    cancel_actions = [a for a in actions if a["action"] == "CANCEL_STOP"]
    assert cancel_actions[0]["status"] == "canceled"


def test_signal_exit_never_sells_if_cancel_confirmation_fails(monkeypatch: pytest.MonkeyPatch):
    def fail_if_called(*a, **k):
        raise AssertionError("must not sell while the stop cancellation is unconfirmed")

    monkeypatch.setattr(order_submission, "submit_equity_market_order", fail_if_called)
    monkeypatch.setattr(
        order_submission, "cancel_order_and_confirm", lambda client, order_id: "pending_cancel"
    )

    state = LiveRunnerState(equity_stop_orders={"AAPL": "stop-1"})
    actions, review = runner._execute_equity_orders(
        client=None, runner_state=state, equity_date="2026-08-15",
        newly_opened={}, closed_trades=[equity_trade(exit_reason="EXIT_SIGNAL_NEXT_OPEN")],
    )

    assert not any(a["action"] == "SELL" for a in actions)
    assert "AAPL" in state.equity_stop_orders, "must not drop tracking of an unconfirmed stop"
    assert len(review) == 1
    assert "refusing to sell" in review[0]["issue"]


def test_signal_exit_skips_sell_when_stop_already_filled_first(monkeypatch: pytest.MonkeyPatch):
    """Simulates Alpaca's real `insufficient qty available` scenario in
    reverse: the stop wins the race and fills before our cancel request
    lands, so there is nothing left to sell."""

    def fail_if_called(*a, **k):
        raise AssertionError("must not sell shares the stop already sold")

    monkeypatch.setattr(order_submission, "submit_equity_market_order", fail_if_called)
    monkeypatch.setattr(
        order_submission, "cancel_order_and_confirm", lambda client, order_id: "filled"
    )

    state = LiveRunnerState(equity_stop_orders={"AAPL": "stop-1"})
    actions, review = runner._execute_equity_orders(
        client=None, runner_state=state, equity_date="2026-08-15",
        newly_opened={}, closed_trades=[equity_trade(exit_reason="EXIT_SIGNAL_NEXT_OPEN")],
    )

    assert not any(a["action"] == "SELL" for a in actions)
    assert "AAPL" not in state.equity_stop_orders
    assert len(review) == 1
    assert "already closed at the broker" in review[0]["issue"]


def test_real_held_for_orders_rejection_scenario_is_prevented_by_correct_ordering(
    monkeypatch: pytest.MonkeyPatch,
):
    """Regression test for the exact failure observed against the real
    Alpaca paper API: selling while a protective stop is still active
    is rejected with `insufficient qty available for order` because the
    stop holds the shares as collateral (`held_for_orders`). This test
    simulates that API behavior directly and proves the code's ordering
    (cancel-and-confirm before any sell attempt) never triggers it."""

    class HeldForOrdersError(Exception):
        pass

    stop_state = {"canceled": False}

    def fake_cancel_and_confirm(client, order_id):
        stop_state["canceled"] = True
        return "canceled"

    def fake_submit(client, *, ticker, side, quantity, time_in_force=None, client_order_id=None):
        if side == OrderSide.SELL and not stop_state["canceled"]:
            # Mirrors the real Alpaca APIError observed in production:
            # {"available":"0","existing_qty":"1","held_for_orders":"1",
            #  "message":"insufficient qty available for order
            #  (requested: 1, available: 0)"}
            raise HeldForOrdersError("insufficient qty available for order")
        return FakeOrder("exit-1")

    monkeypatch.setattr(order_submission, "get_order_by_client_order_id", lambda *a, **k: None)
    monkeypatch.setattr(order_submission, "cancel_order_and_confirm", fake_cancel_and_confirm)
    monkeypatch.setattr(order_submission, "submit_equity_market_order", fake_submit)
    monkeypatch.setattr(order_submission, "wait_for_fill_or_timeout", lambda *a, **k: "filled")

    state = LiveRunnerState(equity_stop_orders={"AAPL": "stop-1"})
    actions, review = runner._execute_equity_orders(
        client=None, runner_state=state, equity_date="2026-08-15",
        newly_opened={}, closed_trades=[equity_trade(exit_reason="EXIT_SIGNAL_NEXT_OPEN")],
    )

    assert stop_state["canceled"] is True
    assert review == []
    sell_actions = [a for a in actions if a["action"] == "SELL"]
    assert sell_actions and sell_actions[0]["order_id"] == "exit-1"


def test_stop_loss_exit_never_submits_a_second_sell(monkeypatch: pytest.MonkeyPatch):
    def fail_if_called(*a, **k):
        raise AssertionError("STOP_LOSS exit must never submit a second market sell")

    monkeypatch.setattr(order_submission, "submit_equity_market_order", fail_if_called)
    monkeypatch.setattr(order_submission, "get_order_status", lambda *a, **k: "filled")

    state = LiveRunnerState(equity_stop_orders={"AAPL": "stop-1"})
    actions, review = runner._execute_equity_orders(
        client=None, runner_state=state, equity_date="2026-08-15",
        newly_opened={}, closed_trades=[equity_trade(exit_reason="STOP_LOSS")],
    )

    assert review == []
    assert "AAPL" not in state.equity_stop_orders
    assert any(a["action"] == "RECONCILE_STOP_FILLED" for a in actions)


def test_stop_loss_exit_flags_review_when_broker_stop_not_yet_filled(
    monkeypatch: pytest.MonkeyPatch,
):
    def fail_if_called(*a, **k):
        raise AssertionError("must not guess-submit a sell while the real stop is unresolved")

    monkeypatch.setattr(order_submission, "submit_equity_market_order", fail_if_called)
    monkeypatch.setattr(order_submission, "get_order_status", lambda *a, **k: "new")

    state = LiveRunnerState(equity_stop_orders={"AAPL": "stop-1"})
    actions, review = runner._execute_equity_orders(
        client=None, runner_state=state, equity_date="2026-08-15",
        newly_opened={}, closed_trades=[equity_trade(exit_reason="GAP_STOP_LOSS")],
    )

    assert state.equity_stop_orders["AAPL"] == "stop-1"
    assert not any(a["action"] == "RECONCILE_STOP_FILLED" for a in actions)
    assert len(review) == 1
    assert "does not yet show filled" in review[0]["issue"]


def test_entry_buy_finds_existing_order_by_client_order_id_and_never_resubmits(
    monkeypatch: pytest.MonkeyPatch,
):
    """The actual fix under test, not the pre-existing local-ledger check:
    even with an EMPTY local ledger (simulating a lost/rolled-back
    `submitted_actions`, exactly the scenario the local-ledger-only
    check cannot defend against), a broker that already has an order
    for this client_order_id must stop a real resubmission."""

    def fail_if_called(*a, **k):
        raise AssertionError(
            "must not submit a new order -- the broker already has one for this client_order_id"
        )

    class ExistingOrder:
        id = "already-submitted-entry-1"
        status = "filled"

    monkeypatch.setattr(order_submission, "submit_equity_market_order", fail_if_called)
    monkeypatch.setattr(
        order_submission, "get_order_by_client_order_id", lambda client, client_order_id: ExistingOrder()
    )

    state = LiveRunnerState()  # empty ledger -- nothing locally remembers this order
    actions, review = runner._execute_equity_orders(
        client=None, runner_state=state, equity_date="2026-08-14",
        newly_opened={"AAPL": equity_position()}, closed_trades=[],
    )

    assert review == []
    buy_actions = [a for a in actions if a["action"] == "BUY"]
    assert buy_actions and buy_actions[0]["order_id"] == "already-submitted-entry-1"
    assert state.submitted_actions["AAPL|ENTRY_MARKET_BUY|2026-08-14"]["order_id"] == "already-submitted-entry-1"


def test_signal_exit_finds_existing_order_by_client_order_id_and_never_resubmits(
    monkeypatch: pytest.MonkeyPatch,
):
    def fail_if_called(*a, **k):
        raise AssertionError(
            "must not submit a new sell -- the broker already has one for this client_order_id"
        )

    class ExistingOrder:
        id = "already-submitted-exit-1"
        status = "filled"

    monkeypatch.setattr(order_submission, "submit_equity_market_order", fail_if_called)
    monkeypatch.setattr(
        order_submission, "get_order_by_client_order_id", lambda client, client_order_id: ExistingOrder()
    )

    state = LiveRunnerState()  # empty ledger, no resting stop to cancel first
    actions, review = runner._execute_equity_orders(
        client=None, runner_state=state, equity_date="2026-08-15",
        newly_opened={}, closed_trades=[equity_trade(exit_reason="EXIT_SIGNAL_NEXT_OPEN")],
    )

    assert review == []
    sell_actions = [a for a in actions if a["action"] == "SELL"]
    assert sell_actions and sell_actions[0]["order_id"] == "already-submitted-exit-1"


def test_reconciliation_stop_finds_existing_order_by_client_order_id_and_never_resubmits(
    monkeypatch: pytest.MonkeyPatch,
):
    """Point 2 of the 2026-08-14 review: the reconcile function's own
    PROTECTIVE_STOP placement now goes through the same broker-query
    check, keyed off the entry order's own id rather than a date."""

    def fail_if_called(*a, **k):
        raise AssertionError(
            "must not place a new stop -- the broker already has one for this client_order_id"
        )

    class ExistingStopOrder:
        id = "already-submitted-stop-1"
        status = "new"

    state = LiveRunnerState(
        portfolio_bar_index=3,
        positions={"AAPL": equity_position()},
        submitted_actions={
            "AAPL|ENTRY_MARKET_BUY|3": {
                "order_id": "entry-1", "status": "accepted",
                "kind": "ENTRY_MARKET_BUY", "submitted_at": "x",
            }
        },
    )
    monkeypatch.setattr(order_submission, "get_order_status", lambda *a, **k: "filled")
    monkeypatch.setattr(order_submission, "submit_equity_stop_sell", fail_if_called)
    monkeypatch.setattr(
        order_submission, "get_order_by_client_order_id",
        lambda client, client_order_id: ExistingStopOrder(),
    )

    review = runner._reconcile_pending_equity_orders(client=None, runner_state=state)

    assert review == []
    assert state.equity_stop_orders["AAPL"] == "already-submitted-stop-1"


def test_crypto_positions_never_reach_order_submission(monkeypatch: pytest.MonkeyPatch):
    def fail_if_called(*a, **k):
        raise AssertionError("crypto must never be submitted as a real order")

    monkeypatch.setattr(order_submission, "submit_equity_market_order", fail_if_called)
    monkeypatch.setattr(order_submission, "submit_equity_stop_sell", fail_if_called)

    crypto_position = _MutablePosition(
        ticker="BTC-USD", asset_class="CRYPTO", entry_timestamp="2026-08-10",
        entry_portfolio_bar_index=5, quantity=0.01, entry_price=65000.0, entry_fee=1.0,
        stop_loss_price=61750.0, highest_close=65000.0, trailing_close_percent=7.5,
        initial_risk_amount=32.5, signal_score=80.0, signal_reason="TEST",
    )
    state = LiveRunnerState()
    actions, review = runner._execute_equity_orders(
        client=None, runner_state=state, equity_date="2026-08-14",
        newly_opened={"BTC-USD": crypto_position}, closed_trades=[],
    )

    assert actions == []
    assert review == []
    assert state.submitted_actions == {}
