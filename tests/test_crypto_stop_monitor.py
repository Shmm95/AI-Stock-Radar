"""Tests for src/live/crypto_stop_monitor.py.

Deterministic and network-free: `get_latest_crypto_trade` and
`order_submission`'s functions are monkeypatched. See the PR-level
report for the additional small, self-cleaning real-API check run
alongside these.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.backtest.portfolio_backtest_engine import _MutablePosition
from src.live import crypto_stop_monitor as monitor
from src.live import order_submission
from src.live.position_state import LiveRunnerState


class FakeOrder:
    def __init__(self, order_id: str):
        self.id = order_id


class FakeTrade:
    def __init__(self, price: float):
        self.price = price


def crypto_position(ticker: str = "BTC-USD", quantity: float = 0.01) -> _MutablePosition:
    return _MutablePosition(
        ticker=ticker,
        asset_class="CRYPTO",
        entry_timestamp="2026-08-10",
        entry_portfolio_bar_index=5,
        quantity=quantity,
        entry_price=65000.0,
        entry_fee=1.0,
        stop_loss_price=61750.0,
        highest_close=65000.0,
        trailing_close_percent=7.5,
        initial_risk_amount=32.5,
        signal_score=80.0,
        signal_reason="TEST",
    )


def equity_position() -> _MutablePosition:
    return _MutablePosition(
        ticker="AAPL", asset_class="EQUITY", entry_timestamp="2026-08-10",
        entry_portfolio_bar_index=5, quantity=10.0, entry_price=190.0, entry_fee=1.0,
        stop_loss_price=180.5, highest_close=190.0, trailing_close_percent=7.5,
        initial_risk_amount=95.0, signal_score=80.0, signal_reason="TEST",
    )


def test_is_crypto_stop_triggered_boundary():
    assert monitor.is_crypto_stop_triggered(current_trade_price=61750.0, stop_loss_price=61750.0)
    assert monitor.is_crypto_stop_triggered(current_trade_price=61000.0, stop_loss_price=61750.0)
    assert not monitor.is_crypto_stop_triggered(current_trade_price=61751.0, stop_loss_price=61750.0)


def test_no_action_when_price_above_stop(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(monitor, "get_latest_crypto_trade", lambda symbol: FakeTrade(70000.0))

    def fail_if_called(*a, **k):
        raise AssertionError("must not sell when the stop was not breached")

    monkeypatch.setattr(order_submission, "submit_equity_market_order", fail_if_called)

    state = LiveRunnerState(positions={"BTC-USD": crypto_position()})
    actions = monitor.check_and_execute_crypto_stops(client=None, runner_state=state, check_id="c1")

    assert "BTC-USD" in state.positions
    check_actions = [a for a in actions if a["action"] == "CHECK"]
    assert check_actions[0]["trigger"] == "NOT_TRIGGERED"
    assert not any(a["action"] == "SELL" for a in actions)


def test_breach_submits_ioc_sell_and_closes_position(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(monitor, "get_latest_crypto_trade", lambda symbol: FakeTrade(60000.0))
    monkeypatch.setattr(
        monitor, "_query_available_crypto_quantity", lambda client, symbol: 0.01
    )
    captured_calls = []

    def fake_submit(client, *, ticker, side, quantity, time_in_force):
        captured_calls.append(
            {"ticker": ticker, "side": side, "quantity": quantity, "tif": time_in_force}
        )
        return FakeOrder("sell-1")

    monkeypatch.setattr(order_submission, "submit_equity_market_order", fake_submit)
    monkeypatch.setattr(order_submission, "wait_for_fill_or_timeout", lambda *a, **k: "filled")

    state = LiveRunnerState(positions={"BTC-USD": crypto_position()})
    actions = monitor.check_and_execute_crypto_stops(client=None, runner_state=state, check_id="c1")

    assert "BTC-USD" not in state.positions
    assert captured_calls[0]["ticker"] == "BTC/USD"
    assert captured_calls[0]["quantity"] == 0.01
    assert str(captured_calls[0]["tif"]).lower().endswith("ioc")
    sell_actions = [a for a in actions if a["action"] == "SELL"]
    assert sell_actions[0]["order_id"] == "sell-1"
    assert state.submitted_actions["BTC-USD|CRYPTO_STOP_SELL|c1"]["status"] == "filled"


def test_broker_quantity_below_local_quantity_is_used_and_reconciled(
    monkeypatch: pytest.MonkeyPatch,
):
    """Real paper fills showed in-kind crypto fees can leave less than
    the local `quantity` actually sellable. The sell must use the
    broker-confirmed (smaller) figure, and local state must be
    reconciled to it even though this test never lets the sell fill."""
    monkeypatch.setattr(monitor, "get_latest_crypto_trade", lambda symbol: FakeTrade(60000.0))
    monkeypatch.setattr(
        monitor, "_query_available_crypto_quantity", lambda client, symbol: 0.0095
    )
    captured_calls = []

    def fake_submit(client, *, ticker, side, quantity, time_in_force):
        captured_calls.append(quantity)
        return FakeOrder("sell-1")

    monkeypatch.setattr(order_submission, "submit_equity_market_order", fake_submit)
    # Not yet filled (e.g. still processing) -> position must stay open,
    # but with the reconciled quantity, not the stale local one.
    monkeypatch.setattr(order_submission, "wait_for_fill_or_timeout", lambda *a, **k: "accepted")

    position = crypto_position(quantity=0.01)
    state = LiveRunnerState(positions={"BTC-USD": position})
    actions = monitor.check_and_execute_crypto_stops(client=None, runner_state=state, check_id="c1")

    assert captured_calls == [0.0095], "must sell the broker-confirmed quantity, not local 0.01"
    assert "BTC-USD" in state.positions
    assert state.positions["BTC-USD"].quantity == 0.0095, "local quantity must be reconciled"
    sell_actions = [a for a in actions if a["action"] == "SELL"]
    assert sell_actions[0]["local_quantity"] == 0.01
    assert sell_actions[0]["broker_quantity"] == 0.0095


def test_no_available_quantity_skips_sell_and_flags_review(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(monitor, "get_latest_crypto_trade", lambda symbol: FakeTrade(60000.0))
    monkeypatch.setattr(monitor, "_query_available_crypto_quantity", lambda client, symbol: None)

    def fail_if_called(*a, **k):
        raise AssertionError("must not guess a sell quantity when the broker reports none")

    monkeypatch.setattr(order_submission, "submit_equity_market_order", fail_if_called)

    position = crypto_position(quantity=0.01)
    state = LiveRunnerState(positions={"BTC-USD": position})
    actions = monitor.check_and_execute_crypto_stops(client=None, runner_state=state, check_id="c1")

    assert "BTC-USD" in state.positions
    assert state.positions["BTC-USD"].quantity == 0.01, "must not alter local state without a real figure"
    review = [a for a in actions if a["action"] == "NEEDS_REVIEW"]
    assert len(review) == 1
    assert review[0]["ticker"] == "BTC-USD"
    assert review[0]["local_quantity"] == 0.01
    assert not any(a["action"] == "SELL" for a in actions)


def test_rerun_with_same_check_id_does_not_resubmit(monkeypatch: pytest.MonkeyPatch):
    calls = {"submit": 0}

    def fake_submit(client, *, ticker, side, quantity, time_in_force):
        calls["submit"] += 1
        return FakeOrder(f"sell-{calls['submit']}")

    monkeypatch.setattr(monitor, "get_latest_crypto_trade", lambda symbol: FakeTrade(60000.0))
    monkeypatch.setattr(
        monitor, "_query_available_crypto_quantity", lambda client, symbol: 0.01
    )
    monkeypatch.setattr(order_submission, "submit_equity_market_order", fake_submit)
    monkeypatch.setattr(order_submission, "wait_for_fill_or_timeout", lambda *a, **k: "filled")
    monkeypatch.setattr(order_submission, "get_order_status", lambda *a, **k: "filled")

    state = LiveRunnerState(positions={"BTC-USD": crypto_position()})
    monitor.check_and_execute_crypto_stops(client=None, runner_state=state, check_id="c1")
    assert calls["submit"] == 1

    # Position is already gone, so a second run with a DIFFERENT
    # check_id (simulating the next periodic check) naturally finds
    # nothing to do — this is the primary idempotency mechanism.
    actions_second_run = monitor.check_and_execute_crypto_stops(
        client=None, runner_state=state, check_id="c2"
    )
    assert calls["submit"] == 1
    assert actions_second_run == []


def test_retry_of_same_check_id_after_position_still_present_skips_resubmit(
    monkeypatch: pytest.MonkeyPatch,
):
    # Simulates: submission succeeded and was recorded, but for some
    # reason (e.g. a crash before the position dict was updated) the
    # position is still present when the exact same logical check_id
    # is retried. The ledger entry must still prevent a second sell.
    calls = {"submit": 0}
    monkeypatch.setattr(
        order_submission, "submit_equity_market_order",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not resubmit for a known check_id")),
    )
    monkeypatch.setattr(order_submission, "get_order_status", lambda *a, **k: "filled")

    state = LiveRunnerState(
        positions={"BTC-USD": crypto_position()},
        submitted_actions={
            "BTC-USD|CRYPTO_STOP_SELL|c1": {
                "order_id": "sell-1", "status": "accepted",
                "kind": "CRYPTO_STOP_SELL", "submitted_at": "x",
            }
        },
    )
    actions = monitor.check_and_execute_crypto_stops(client=None, runner_state=state, check_id="c1")

    already_submitted = [a for a in actions if a.get("trigger") == "ALREADY_SUBMITTED"]
    assert already_submitted
    assert already_submitted[0]["status"] == "filled"


def test_equity_positions_never_reach_crypto_monitor(monkeypatch: pytest.MonkeyPatch):
    def fail_if_called(*a, **k):
        raise AssertionError("equity must never be checked/sold by the crypto monitor")

    monkeypatch.setattr(monitor, "get_latest_crypto_trade", fail_if_called)
    monkeypatch.setattr(order_submission, "submit_equity_market_order", fail_if_called)

    state = LiveRunnerState(positions={"AAPL": equity_position()})
    actions = monitor.check_and_execute_crypto_stops(client=None, runner_state=state, check_id="c1")

    assert actions == []
    assert "AAPL" in state.positions


def test_overlapping_invocations_are_refused_by_the_lock(tmp_path: Path):
    lock_path = tmp_path / "crypto_monitor.lock"
    with monitor._monitor_lock(lock_path):
        assert lock_path.exists()
        with pytest.raises(monitor.MonitorAlreadyRunningError):
            with monitor._monitor_lock(lock_path):
                pass
    assert not lock_path.exists()


def test_lock_is_released_even_if_the_body_raises(tmp_path: Path):
    lock_path = tmp_path / "crypto_monitor.lock"
    with pytest.raises(ValueError):
        with monitor._monitor_lock(lock_path):
            raise ValueError("boom")
    assert not lock_path.exists()
