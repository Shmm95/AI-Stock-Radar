"""Tests for scripts/send_status_update.py.

Deterministic and network-free: Alpaca and Telegram calls are monkeypatched.
Focus is the fail-safe contract this script exists for -- it is itself the
mechanism proving the system is alive, so a failure anywhere inside it
(Alpaca down, position-state file missing/corrupt, no decision log yet)
must degrade that one line to "unavailable"/"unknown", never crash the
whole run and never prevent some message from being sent.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import send_status_update as status  # noqa: E402
from src.backtest.portfolio_backtest_engine import _MutablePosition  # noqa: E402
from src.live.position_state import LiveRunnerState  # noqa: E402


def fake_position(
    *, ticker: str, asset_class: str, quantity: float, entry_price: float,
    entry_timestamp: str = "2026-08-10T00:00:00",
) -> _MutablePosition:
    return _MutablePosition(
        ticker=ticker,
        asset_class=asset_class,
        entry_timestamp=entry_timestamp,
        entry_portfolio_bar_index=1,
        quantity=quantity,
        entry_price=entry_price,
        entry_fee=1.0,
        stop_loss_price=entry_price * 0.95,
        highest_close=entry_price,
        trailing_close_percent=5.0,
        initial_risk_amount=10.0,
        signal_score=1.0,
        signal_reason="test",
    )


def test_build_status_text_reports_no_positions_and_no_decision_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    monkeypatch.setattr(status, "get_live_cash_balance", lambda: 100_000.0)
    monkeypatch.setattr(
        status, "load_position_state", lambda path: LiveRunnerState(positions={})
    )

    text = status.build_status_text(
        state_path=tmp_path / "position_state.json",
        decision_log_directory=tmp_path / "decisions",
    )

    assert "Cash balance: $100,000.00" in text
    assert "Açık pozisyon yok." in text
    assert "no decision log found yet" in text


def test_build_status_text_degrades_gracefully_when_alpaca_query_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    def broken_cash_balance():
        raise RuntimeError("Alpaca API is down")

    monkeypatch.setattr(status, "get_live_cash_balance", broken_cash_balance)
    monkeypatch.setattr(
        status, "load_position_state", lambda path: LiveRunnerState(positions={})
    )

    text = status.build_status_text(
        state_path=tmp_path / "position_state.json",
        decision_log_directory=tmp_path / "decisions",
    )

    assert "Cash balance: unavailable (RuntimeError)" in text
    assert "Alpaca API is down" not in text, "must never leak the raw exception message"


def test_build_status_text_degrades_gracefully_when_position_state_load_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    def broken_load(path):
        raise ValueError("Unsupported position_state schema_version: 99")

    monkeypatch.setattr(status, "get_live_cash_balance", lambda: 1.0)
    monkeypatch.setattr(status, "load_position_state", broken_load)

    text = status.build_status_text(
        state_path=tmp_path / "position_state.json",
        decision_log_directory=tmp_path / "decisions",
    )

    assert "Open positions: unavailable (ValueError)" in text


def test_build_status_text_lists_equity_and_crypto_positions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    positions = {
        "AAPL": fake_position(ticker="AAPL", asset_class="EQUITY", quantity=10, entry_price=200.0),
        "BTC-USD": fake_position(ticker="BTC-USD", asset_class="CRYPTO", quantity=0.5, entry_price=60000.0),
    }
    monkeypatch.setattr(status, "get_live_cash_balance", lambda: 5000.0)
    monkeypatch.setattr(status, "load_position_state", lambda path: LiveRunnerState(positions=positions))

    text = status.build_status_text(
        state_path=tmp_path / "position_state.json",
        decision_log_directory=tmp_path / "decisions",
    )

    assert "AAPL (EQUITY): qty 10" in text
    assert "BTC-USD (CRYPTO): qty 0.5" in text


def _write_decision_log(directory: Path, *, generated_at: str, needs_manual_review: list, name: str = "decision_1.json"):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(
        json.dumps({"generated_at": generated_at, "needs_manual_review": needs_manual_review}),
        encoding="utf-8",
    )


def test_build_status_text_reports_fresh_successful_run_and_review_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    now = datetime(2026, 8, 11, 12, 0, 0, tzinfo=UTC)
    decisions = tmp_path / "decisions"
    _write_decision_log(
        decisions,
        generated_at=(now - timedelta(hours=2)).isoformat(),
        needs_manual_review=[{"ticker": "ETH-USD", "issue": "no available quantity"}],
    )
    monkeypatch.setattr(status, "get_live_cash_balance", lambda: 1.0)
    monkeypatch.setattr(status, "load_position_state", lambda path: LiveRunnerState(positions={}))

    text = status.build_status_text(
        state_path=tmp_path / "position_state.json", decision_log_directory=decisions, now=now,
    )

    assert "(succeeded)" in text
    assert "STALE" not in text
    assert "needs_manual_review pending: 1" in text


def test_build_status_text_flags_a_stale_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    now = datetime(2026, 8, 11, 12, 0, 0, tzinfo=UTC)
    decisions = tmp_path / "decisions"
    _write_decision_log(
        decisions,
        generated_at=(now - timedelta(hours=48)).isoformat(),
        needs_manual_review=[],
    )
    monkeypatch.setattr(status, "get_live_cash_balance", lambda: 1.0)
    monkeypatch.setattr(status, "load_position_state", lambda path: LiveRunnerState(positions={}))

    text = status.build_status_text(
        state_path=tmp_path / "position_state.json", decision_log_directory=decisions, now=now,
    )

    assert "STALE" in text
    assert "48.0h ago" in text


def test_build_status_text_reports_unreadable_decision_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    decisions = tmp_path / "decisions"
    decisions.mkdir()
    (decisions / "decision_bad.json").write_text("{ not valid json", encoding="utf-8")
    monkeypatch.setattr(status, "get_live_cash_balance", lambda: 1.0)
    monkeypatch.setattr(status, "load_position_state", lambda path: LiveRunnerState(positions={}))

    text = status.build_status_text(
        state_path=tmp_path / "position_state.json", decision_log_directory=decisions,
    )

    assert "log unreadable" in text
    assert "decision_bad.json" in text


class _FakeAccount:
    def __init__(self, *, cash: str = "100000.00", equity: str = "100000.00", last_equity: str = "100000.00") -> None:
        self.cash = cash
        self.equity = equity
        self.last_equity = last_equity


class _FakePosition:
    def __init__(
        self, *, symbol: str, side: str = "long", qty: str, avg_entry_price: str, current_price: str,
        market_value: str, unrealized_pl: str, unrealized_plpc: str, asset_class: str = "us_equity",
    ) -> None:
        self.symbol = symbol
        self.side = side
        self.qty = qty
        self.avg_entry_price = avg_entry_price
        self.current_price = current_price
        self.market_value = market_value
        self.unrealized_pl = unrealized_pl
        self.unrealized_plpc = unrealized_plpc
        self.asset_class = asset_class


class _FakePnlClient:
    """No submit/cancel/replace method at all -- see
    test_read_only_portfolio_snapshot.py's identical fake for why that
    absence is itself part of the read-only proof."""

    def __init__(self, *, account: _FakeAccount, positions: list, orders: list | None = None) -> None:
        self._account = account
        self._positions = positions
        self._orders = orders or []

    def get_account(self):
        return self._account

    def get_all_positions(self):
        return list(self._positions)

    def get_orders(self, *, filter=None):
        return list(self._orders)


def test_build_status_text_includes_pnl_snapshot_when_broker_reads_succeed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    monkeypatch.setattr(status, "get_live_cash_balance", lambda: 100_000.0)
    monkeypatch.setattr(status, "load_position_state", lambda path: LiveRunnerState(positions={}))
    fake_client = _FakePnlClient(
        account=_FakeAccount(cash="50000.00", equity="102500.00", last_equity="100000.00"),
        positions=[
            _FakePosition(
                symbol="AAPL", qty="10", avg_entry_price="150.00", current_price="160.00",
                market_value="1600.00", unrealized_pl="100.00", unrealized_plpc="0.0667",
            ),
        ],
    )
    monkeypatch.setattr(status.order_submission, "get_trading_client", lambda: fake_client)

    text = status.build_status_text(
        state_path=tmp_path / "position_state.json", decision_log_directory=tmp_path / "decisions",
    )

    assert "P&L snapshot (broker, live):" in text
    assert "Portfolio value: $102,500.00" in text
    assert "Cash: $50,000.00" in text
    assert "Day P&L: +2,500.00" in text
    assert "AAPL (long): qty 10" in text
    assert "unrealized +100.00" in text
    assert "Realized closed-trade P&L: +0.00" in text
    assert "Broker unrealized P&L: +100.00" in text
    assert "Strategy fill-based P&L (realized + broker unrealized): +100.00" in text
    # The pre-existing, independent cash-balance source is untouched.
    assert "Cash balance: $100,000.00" in text


def test_build_status_text_degrades_gracefully_when_pnl_snapshot_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    monkeypatch.setattr(status, "get_live_cash_balance", lambda: 1.0)
    monkeypatch.setattr(status, "load_position_state", lambda path: LiveRunnerState(positions={}))

    def broken_client():
        raise RuntimeError("ALPACA_API_KEY not set")

    monkeypatch.setattr(status.order_submission, "get_trading_client", broken_client)

    text = status.build_status_text(
        state_path=tmp_path / "position_state.json", decision_log_directory=tmp_path / "decisions",
    )

    assert "P&L snapshot: unavailable (RuntimeError)" in text
    assert "ALPACA_API_KEY not set" not in text, "must never leak the raw exception message"
    # Every other independent source still reports normally.
    assert "Cash balance: $1.00" in text
    assert "Açık pozisyon yok." in text


def test_build_status_text_pnl_snapshot_reports_no_broker_positions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    monkeypatch.setattr(status, "get_live_cash_balance", lambda: 1.0)
    monkeypatch.setattr(status, "load_position_state", lambda path: LiveRunnerState(positions={}))
    fake_client = _FakePnlClient(account=_FakeAccount(), positions=[])
    monkeypatch.setattr(status.order_submission, "get_trading_client", lambda: fake_client)

    text = status.build_status_text(
        state_path=tmp_path / "position_state.json", decision_log_directory=tmp_path / "decisions",
    )

    assert "No broker positions open." in text


def test_notify_safe_reports_success_and_failure(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(status, "send_telegram_message", lambda text: True)
    assert status._notify_safe("hello") is True

    monkeypatch.setattr(status, "send_telegram_message", lambda text: False)
    assert status._notify_safe("hello") is False

    def broken(text):
        raise RuntimeError("notifier bug")

    monkeypatch.setattr(status, "send_telegram_message", broken)
    assert status._notify_safe("hello") is False  # must never raise


def test_main_never_raises_and_sends_something_even_if_build_status_text_breaks(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
):
    sent = []

    def broken_builder(**kwargs):
        raise RuntimeError("unexpected bug inside build_status_text")

    monkeypatch.setattr(status, "build_status_text", broken_builder)
    monkeypatch.setattr(status, "send_telegram_message", lambda text: sent.append(text) or True)
    monkeypatch.setattr(sys, "argv", ["send_status_update.py"])

    status.main()  # must not raise

    assert len(sent) == 1
    assert "FAILED" in sent[0]
    assert "durum sorgulanamadı" in sent[0]


def test_main_warns_on_stderr_when_notification_cannot_be_confirmed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
):
    monkeypatch.setattr(status, "get_live_cash_balance", lambda: 1.0)
    monkeypatch.setattr(status, "load_position_state", lambda path: LiveRunnerState(positions={}))
    monkeypatch.setattr(status, "send_telegram_message", lambda text: False)
    monkeypatch.setattr(
        sys, "argv",
        [
            "send_status_update.py",
            "--state-path", str(tmp_path / "position_state.json"),
            "--decision-log-directory", str(tmp_path / "decisions"),
        ],
    )

    status.main()  # must not raise

    captured = capsys.readouterr()
    assert "WARNING: status-update Telegram notification could not be confirmed sent." in captured.err
    assert "AI-Stock-Radar status check" in captured.out
