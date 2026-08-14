"""Tests for the Telegram notification wiring in run_crypto_stop_monitor.py.

Deterministic and network-free. Focus: routine, nothing-triggered
checks must NEVER notify (spam prevention at a 5-minute cadence), only
a real SELL or a NEEDS_REVIEW anomaly should.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_crypto_stop_monitor.py"
_spec = importlib.util.spec_from_file_location("run_crypto_stop_monitor", MODULE_PATH)
monitor_script = importlib.util.module_from_spec(_spec)
sys.modules["run_crypto_stop_monitor"] = monitor_script
_spec.loader.exec_module(monitor_script)


def test_quiet_check_with_no_trigger_produces_no_notification():
    actions = [
        {
            "ticker": "BTC-USD",
            "action": "CHECK",
            "trigger": "NOT_TRIGGERED",
            "current_trade_price": 65000.0,
            "stop_loss_price": 61750.0,
        }
    ]

    assert monitor_script._build_notification_text(actions) is None


def test_empty_actions_produces_no_notification():
    assert monitor_script._build_notification_text([]) is None


def test_sell_action_produces_a_notification():
    actions = [
        {"ticker": "BTC-USD", "action": "CHECK", "trigger": "STOP_TRIGGERED",
         "current_trade_price": 60000.0, "stop_loss_price": 61750.0},
        {"ticker": "BTC-USD", "action": "SELL", "order_id": "sell-1", "status": "filled"},
    ]

    text = monitor_script._build_notification_text(actions)

    assert text is not None
    assert "STOP TRIGGERED" in text
    assert "BTC-USD" in text
    assert "sell-1" in text


def test_needs_review_action_produces_a_notification():
    actions = [
        {"ticker": "ETH-USD", "action": "NEEDS_REVIEW", "issue": "broker reports no available quantity"}
    ]

    text = monitor_script._build_notification_text(actions)

    assert text is not None
    assert "NEEDS REVIEW" in text
    assert "ETH-USD" in text
    assert "broker reports no available quantity" in text


def test_notify_safe_swallows_a_notifier_exception(monkeypatch: pytest.MonkeyPatch):
    def fail_if_called(text):
        raise RuntimeError("Telegram is on fire")

    monkeypatch.setattr(monitor_script, "send_telegram_message", fail_if_called)

    monitor_script._notify_safe("anything")  # must not raise


def test_main_does_not_notify_on_a_quiet_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from src.live.position_state import LiveRunnerState

    sent = []
    quiet_actions = [
        {"ticker": "BTC-USD", "action": "CHECK", "trigger": "NOT_TRIGGERED",
         "current_trade_price": 65000.0, "stop_loss_price": 61750.0}
    ]

    monkeypatch.setattr(monitor_script, "DEFAULT_LOCK_PATH", tmp_path / "crypto_monitor.lock")
    monkeypatch.setattr(monitor_script, "load_position_state", lambda path: LiveRunnerState())
    monkeypatch.setattr(monitor_script, "save_position_state", lambda state, path: None)
    monkeypatch.setattr(monitor_script.order_submission, "get_trading_client", lambda: None)
    monkeypatch.setattr(
        monitor_script, "check_and_execute_crypto_stops",
        lambda client, state, *, check_id: quiet_actions,
    )
    monkeypatch.setattr(monitor_script, "send_telegram_message", lambda text: sent.append(text) or True)

    exit_code = monitor_script.main()

    assert exit_code == 0
    assert sent == []


def test_main_notifies_when_a_sell_actually_happens(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from src.live.position_state import LiveRunnerState

    sent = []
    trigger_actions = [
        {"ticker": "BTC-USD", "action": "SELL", "order_id": "sell-1", "status": "filled"}
    ]

    monkeypatch.setattr(monitor_script, "DEFAULT_LOCK_PATH", tmp_path / "crypto_monitor.lock")
    monkeypatch.setattr(monitor_script, "load_position_state", lambda path: LiveRunnerState())
    monkeypatch.setattr(monitor_script, "save_position_state", lambda state, path: None)
    monkeypatch.setattr(monitor_script.order_submission, "get_trading_client", lambda: None)
    monkeypatch.setattr(
        monitor_script, "check_and_execute_crypto_stops",
        lambda client, state, *, check_id: trigger_actions,
    )
    monkeypatch.setattr(monitor_script, "send_telegram_message", lambda text: sent.append(text) or True)

    exit_code = monitor_script.main()

    assert exit_code == 0
    assert len(sent) == 1
    assert "STOP TRIGGERED" in sent[0]
