"""Tests for scripts/send_performance_report.py.

Deterministic and network-free: Alpaca and Telegram calls are monkeypatched.
Focus is the fail-safe contract this script exists for -- the same class
of Telegram silent-failure bug already hit and fixed elsewhere in this
project (see run_daily_decision.py's own _notify_safe) must not recur here.
generate_performance_report.py's own calculation functions (fetch_filled_orders,
reconstruct_round_trips, compute_metrics) are reused unmodified and are not
retested here -- only this script's own summary-building and notification
wiring is in scope.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import send_performance_report as spr  # noqa: E402
from generate_performance_report import ClosedTrade, OpenLot  # noqa: E402


def test_no_fills_at_all_is_a_clear_normal_message_not_an_error():
    text = spr.build_summary_text([], [])

    assert "Henüz gerçekleşen işlem yok." in text
    assert "FAILED" not in text


def test_closed_trades_report_total_pnl_dollars_and_percent():
    trades = [
        ClosedTrade("AAPL", "us_equity", 100.0, 110.0, 10, "t1", "t2", 100.0),
        ClosedTrade("MSFT", "us_equity", 200.0, 190.0, 5, "t1", "t2", -50.0),
    ]

    text = spr.build_summary_text(trades, [])

    # cumulative pnl = 50.0; deployed capital = 100*10 + 200*5 = 2000; pct = 2.5%
    assert "Total P&L: $50.00 (+2.50%)" in text
    assert "Closed trades: 2, win rate: 50.0%" in text
    assert "us_equity" in text


def test_open_lots_with_no_closed_trades_still_reports_clearly():
    lots = [OpenLot("BTC/USD", "crypto", 0.1, 60000.0, "t1")]

    text = spr.build_summary_text([], lots)

    assert "Henüz kapanmış (P&L hesaplanabilir) işlem yok." in text
    assert "Açık pozisyon (P&L henüz hesaplanmadı): 1" in text


def test_notify_safe_reports_success_and_failure(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(spr, "send_telegram_message", lambda text: True)
    assert spr._notify_safe("hello") is True

    monkeypatch.setattr(spr, "send_telegram_message", lambda text: False)
    assert spr._notify_safe("hello") is False

    def broken(text):
        raise RuntimeError("notifier bug")

    monkeypatch.setattr(spr, "send_telegram_message", broken)
    assert spr._notify_safe("hello") is False  # must never raise


def test_main_sends_a_failed_message_and_reraises_when_report_generation_breaks(
    monkeypatch: pytest.MonkeyPatch,
):
    """The exact failure mode this task warns against: a report-generation
    error must never be silently swallowed -- a FAILED message must still
    go out, and the original exception must still propagate."""
    sent = []

    def broken_get_trading_client():
        raise RuntimeError("Alpaca API is down")

    monkeypatch.setattr(spr.order_submission, "get_trading_client", broken_get_trading_client)
    monkeypatch.setattr(spr, "send_telegram_message", lambda text: sent.append(text) or True)

    with pytest.raises(RuntimeError, match="Alpaca API is down"):
        spr.main()

    assert len(sent) == 1
    assert "FAILED" in sent[0]
    assert "Alpaca API is down" in sent[0]


def test_main_warns_on_stderr_when_notification_cannot_be_confirmed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(spr.order_submission, "get_trading_client", lambda: object())
    monkeypatch.setattr(spr, "fetch_filled_orders", lambda client: [])
    monkeypatch.setattr(spr, "reconstruct_round_trips", lambda orders: ([], []))
    monkeypatch.setattr(spr, "send_telegram_message", lambda text: False)

    spr.main()  # must not raise on the success path

    captured = capsys.readouterr()
    assert "WARNING: performance-summary Telegram notification could not be confirmed sent." in captured.err
    assert "Henüz gerçekleşen işlem yok." in captured.out


def test_main_sends_successfully_with_zero_fills(monkeypatch: pytest.MonkeyPatch):
    """Real current live state (universe just widened today, no organic
    fill yet) -- confirms this normal case is handled without error."""
    sent = []
    monkeypatch.setattr(spr.order_submission, "get_trading_client", lambda: object())
    monkeypatch.setattr(spr, "fetch_filled_orders", lambda client: [])
    monkeypatch.setattr(spr, "reconstruct_round_trips", lambda orders: ([], []))
    monkeypatch.setattr(spr, "send_telegram_message", lambda text: sent.append(text) or True)

    spr.main()

    assert len(sent) == 1
    assert "Henüz gerçekleşen işlem yok." in sent[0]
