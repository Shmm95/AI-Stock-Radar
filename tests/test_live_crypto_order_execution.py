"""Tests for the crypto real-order orchestration in run_daily_decision.py.

Deterministic and network-free: `src.live.order_submission` and
`src.live.crypto_stop_monitor._query_available_crypto_quantity` are
monkeypatched. See the PR-level report for the additional real (paper)
integration test run alongside these.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_daily_decision as runner  # noqa: E402

from src.backtest.portfolio_backtest_engine import _MutablePosition  # noqa: E402
from src.backtest.portfolio_backtest_models import PortfolioTrade  # noqa: E402
from src.live import order_submission  # noqa: E402
from src.live.position_state import LiveRunnerState  # noqa: E402


class FakeOrder:
    def __init__(self, order_id: str):
        self.id = order_id


def crypto_position(ticker: str = "BTC-USD", quantity: float = 0.01) -> _MutablePosition:
    return _MutablePosition(
        ticker=ticker, asset_class="CRYPTO", entry_timestamp="2026-08-10",
        entry_portfolio_bar_index=5, quantity=quantity, entry_price=65000.0, entry_fee=0.0,
        stop_loss_price=61750.0, highest_close=65000.0, trailing_close_percent=7.5,
        initial_risk_amount=32.5, signal_score=80.0, signal_reason="TEST",
    )


def crypto_trade(
    ticker: str = "BTC-USD", exit_reason: str = "EXIT_SIGNAL_NEXT_OPEN", quantity: float = 0.01
) -> PortfolioTrade:
    return PortfolioTrade(
        ticker=ticker, asset_class="CRYPTO", entry_timestamp="2026-08-10",
        exit_timestamp="2026-08-11", entry_portfolio_bar_index=5, exit_portfolio_bar_index=6,
        quantity=quantity, entry_price=65000.0, exit_price=66000.0, entry_fee=0.0, exit_fee=0.0,
        total_fees=0.0, gross_pnl=10.0, net_pnl=10.0, return_percent=1.5, holding_period_bars=1,
        exit_reason=exit_reason, signal_score=80.0, signal_reason="TEST",
    )


@pytest.fixture
def stub_lock(monkeypatch: pytest.MonkeyPatch):
    """Order-logic tests care about order logic, not the real lock
    file; make acquire/release no-ops. The lock-specific tests at the
    bottom of this file exercise the real functions instead and do
    not request this fixture."""
    monkeypatch.setattr(runner, "_try_acquire_crypto_lock", lambda **_: True)
    monkeypatch.setattr(runner, "_release_crypto_lock", lambda: None)


def test_entry_buy_uses_ioc_and_alpaca_symbol(stub_lock, monkeypatch: pytest.MonkeyPatch):
    captured = []

    def fake_submit(client, *, ticker, side, quantity, time_in_force):
        captured.append({"ticker": ticker, "side": side, "quantity": quantity, "tif": time_in_force})
        return FakeOrder("entry-1")

    monkeypatch.setattr(order_submission, "submit_equity_market_order", fake_submit)
    monkeypatch.setattr(order_submission, "wait_for_fill_or_timeout", lambda *a, **k: "filled")

    state = LiveRunnerState()
    actions, review = runner._execute_crypto_orders(
        client=None, runner_state=state, crypto_date="2026-08-14",
        newly_opened={"BTC-USD": crypto_position()}, closed_trades=[],
    )

    assert review == []
    assert captured[0]["ticker"] == "BTC/USD"
    assert str(captured[0]["tif"]).lower().endswith("ioc")
    buy_actions = [a for a in actions if a["action"] == "BUY"]
    assert buy_actions[0]["order_id"] == "entry-1"
    assert not any(a["action"] == "PLACE_STOP" for a in actions), "crypto has no native stop step"
    assert state.submitted_actions["BTC-USD|ENTRY_MARKET_BUY|2026-08-14"]["status"] == "filled"


def test_rerun_for_same_bar_never_resubmits_entry(stub_lock, monkeypatch: pytest.MonkeyPatch):
    calls = {"submit": 0}

    def fake_submit(client, *, ticker, side, quantity, time_in_force):
        calls["submit"] += 1
        return FakeOrder(f"entry-{calls['submit']}")

    monkeypatch.setattr(order_submission, "submit_equity_market_order", fake_submit)
    monkeypatch.setattr(order_submission, "wait_for_fill_or_timeout", lambda *a, **k: "filled")
    monkeypatch.setattr(order_submission, "get_order_status", lambda *a, **k: "filled")

    state = LiveRunnerState()
    runner._execute_crypto_orders(
        client=None, runner_state=state, crypto_date="2026-08-14",
        newly_opened={"BTC-USD": crypto_position()}, closed_trades=[],
    )
    assert calls["submit"] == 1

    runner._execute_crypto_orders(
        client=None, runner_state=state, crypto_date="2026-08-14",
        newly_opened={"BTC-USD": crypto_position()}, closed_trades=[],
    )
    assert calls["submit"] == 1, "must not submit a second real order for the same bar"


def test_signal_exit_sells_broker_confirmed_quantity_not_local(stub_lock, monkeypatch: pytest.MonkeyPatch):
    captured = []

    def fake_submit(client, *, ticker, side, quantity, time_in_force):
        captured.append(quantity)
        return FakeOrder("exit-1")

    monkeypatch.setattr(
        runner, "_query_available_crypto_quantity", lambda client, symbol: 0.0097
    )
    monkeypatch.setattr(order_submission, "submit_equity_market_order", fake_submit)
    monkeypatch.setattr(order_submission, "wait_for_fill_or_timeout", lambda *a, **k: "filled")

    state = LiveRunnerState()
    actions, review = runner._execute_crypto_orders(
        client=None, runner_state=state, crypto_date="2026-08-15",
        newly_opened={}, closed_trades=[crypto_trade(quantity=0.01)],
    )

    assert review == []
    assert captured == [0.0097], "must sell the broker-confirmed quantity, not local 0.01"
    sell_actions = [a for a in actions if a["action"] == "SELL"]
    assert sell_actions[0]["local_quantity"] == 0.01
    assert sell_actions[0]["broker_quantity"] == 0.0097


def test_signal_exit_with_no_available_quantity_skips_and_flags_review(
    stub_lock, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(runner, "_query_available_crypto_quantity", lambda client, symbol: None)

    def fail_if_called(*a, **k):
        raise AssertionError("must not guess a sell quantity when the broker reports none")

    monkeypatch.setattr(order_submission, "submit_equity_market_order", fail_if_called)

    state = LiveRunnerState()
    actions, review = runner._execute_crypto_orders(
        client=None, runner_state=state, crypto_date="2026-08-15",
        newly_opened={}, closed_trades=[crypto_trade()],
    )

    assert not any(a["action"] == "SELL" for a in actions)
    assert len(review) == 1
    assert review[0]["ticker"] == "BTC-USD"


def test_stop_loss_exit_never_submits_a_sell(stub_lock, monkeypatch: pytest.MonkeyPatch):
    def fail_if_called(*a, **k):
        raise AssertionError("crypto STOP_LOSS exit must never submit a real sell here")

    monkeypatch.setattr(order_submission, "submit_equity_market_order", fail_if_called)
    monkeypatch.setattr(
        runner, "_query_available_crypto_quantity",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not even query")),
    )

    state = LiveRunnerState()
    actions, review = runner._execute_crypto_orders(
        client=None, runner_state=state, crypto_date="2026-08-15",
        newly_opened={}, closed_trades=[crypto_trade(exit_reason="STOP_LOSS")],
    )

    assert actions == []
    assert len(review) == 1
    assert "intraday monitor" in review[0]["issue"]


def test_gap_stop_loss_exit_never_submits_a_sell(stub_lock, monkeypatch: pytest.MonkeyPatch):
    def fail_if_called(*a, **k):
        raise AssertionError("crypto GAP_STOP_LOSS exit must never submit a real sell here")

    monkeypatch.setattr(order_submission, "submit_equity_market_order", fail_if_called)

    state = LiveRunnerState()
    actions, review = runner._execute_crypto_orders(
        client=None, runner_state=state, crypto_date="2026-08-15",
        newly_opened={}, closed_trades=[crypto_trade(exit_reason="GAP_STOP_LOSS")],
    )

    assert actions == []
    assert len(review) == 1


def test_equity_positions_never_reach_crypto_execution(stub_lock, monkeypatch: pytest.MonkeyPatch):
    def fail_if_called(*a, **k):
        raise AssertionError("equity must never be routed through crypto order execution")

    monkeypatch.setattr(order_submission, "submit_equity_market_order", fail_if_called)

    equity_position = _MutablePosition(
        ticker="AAPL", asset_class="EQUITY", entry_timestamp="2026-08-10",
        entry_portfolio_bar_index=5, quantity=10.0, entry_price=190.0, entry_fee=1.0,
        stop_loss_price=180.5, highest_close=190.0, trailing_close_percent=7.5,
        initial_risk_amount=95.0, signal_score=80.0, signal_reason="TEST",
    )
    state = LiveRunnerState()
    actions, review = runner._execute_crypto_orders(
        client=None, runner_state=state, crypto_date="2026-08-14",
        newly_opened={"AAPL": equity_position}, closed_trades=[],
    )

    assert actions == []
    assert review == []


def test_no_op_when_nothing_crypto_pending_does_not_touch_the_lock(
    monkeypatch: pytest.MonkeyPatch,
):
    def fail_if_called(**_):
        raise AssertionError("must not even attempt the lock when there is nothing to do")

    monkeypatch.setattr(runner, "_try_acquire_crypto_lock", fail_if_called)

    state = LiveRunnerState()
    actions, review = runner._execute_crypto_orders(
        client=None, runner_state=state, crypto_date="2026-08-14", newly_opened={}, closed_trades=[],
    )

    assert actions == []
    assert review == []


def test_lock_busy_after_retries_skips_submission_and_flags_both_sides(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(runner, "_try_acquire_crypto_lock", lambda **_: False)

    def fail_if_called(*a, **k):
        raise AssertionError("must not submit any real order while the lock is busy")

    monkeypatch.setattr(order_submission, "submit_equity_market_order", fail_if_called)

    state = LiveRunnerState()
    actions, review = runner._execute_crypto_orders(
        client=None, runner_state=state, crypto_date="2026-08-14",
        newly_opened={"BTC-USD": crypto_position()},
        closed_trades=[crypto_trade(ticker="ETH-USD")],
    )

    assert actions == []
    tickers_flagged = {item["ticker"] for item in review}
    assert tickers_flagged == {"BTC-USD", "ETH-USD"}
    assert all("lock" in item["issue"] for item in review)


def test_lock_acquire_and_release_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    lock_path = tmp_path / "crypto_monitor.lock"
    monkeypatch.setattr(runner, "DEFAULT_LOCK_PATH", lock_path)

    assert runner._try_acquire_crypto_lock(attempts=1, delay_seconds=0.0) is True
    assert lock_path.exists()
    runner._release_crypto_lock()
    assert not lock_path.exists()


def test_lock_retries_then_gives_up_if_held(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    lock_path = tmp_path / "crypto_monitor.lock"
    monkeypatch.setattr(runner, "DEFAULT_LOCK_PATH", lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch()  # simulate the monitor already holding it

    acquired = runner._try_acquire_crypto_lock(attempts=2, delay_seconds=0.01)

    assert acquired is False
    assert lock_path.exists(), "must not remove a lock this call never actually held"
