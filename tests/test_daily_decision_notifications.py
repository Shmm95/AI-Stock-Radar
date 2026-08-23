"""Tests for the Telegram notification wiring in run_daily_decision.py.

Deterministic and network-free: `src.notify.telegram_notifier.send_telegram_message`
is monkeypatched. Focus is the fail-safe contract -- a notification
failure (or a bug in the notifier itself) must never change this
script's exit behavior or mask a real trading-flow error.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_daily_decision as runner  # noqa: E402


@contextmanager
def _fake_lock(state_path):
    yield {}


@pytest.fixture(autouse=True)
def _isolate_single_instance_lock(monkeypatch: pytest.MonkeyPatch):
    """`main()` now acquires `single_instance_lock` (independent-audit-
    round-3 finding #4) -- its default `lock_path` is the REAL,
    out-of-repo guard file, never test-isolated by `tmp_path` alone
    (same discipline this codebase's other tests already establish for
    `position_state.py`'s own guard-directory-derived paths). Every test
    in this file calls `runner.main()` with a bare `sys.argv`, so none
    of them pass a real, isolated `--state-path` either -- faking the
    lock itself is simplest and keeps this file's own focus (Telegram
    notification wiring) undiluted by lock mechanics."""
    monkeypatch.setattr(runner, "single_instance_lock", _fake_lock)


def test_no_orders_produces_a_clear_no_action_message():
    decision = {
        "as_of_bar_timestamp": "2026-08-10",
        "equity_order_actions": [],
        "crypto_order_actions": [],
        "needs_manual_review": [],
        "cash_balance_usd": 100000.0,
    }

    text = runner._build_daily_notification_text(decision)

    assert "2026-08-10" in text
    assert "No equity or crypto orders today." in text
    assert "needs_manual_review: 0" in text
    assert "100,000.00" in text


def test_equity_and_crypto_actions_are_both_summarized():
    decision = {
        "as_of_bar_timestamp": "2026-08-10",
        "equity_order_actions": [
            {"ticker": "AAPL", "action": "BUY", "status": "filled", "order_id": "o1"}
        ],
        "crypto_order_actions": [
            {"ticker": "BTC-USD", "action": "SELL", "status": "filled", "order_id": "o2"}
        ],
        "needs_manual_review": [{"ticker": "ETH-USD", "issue": "something odd"}],
        "cash_balance_usd": 42.5,
    }

    text = runner._build_daily_notification_text(decision)

    assert "Equity actions (1):" in text
    assert "AAPL BUY: filled" in text
    assert "Crypto actions (1):" in text
    assert "BTC-USD SELL: filled" in text
    assert "needs_manual_review: 1" in text


def test_review_only_entry_uses_issue_text_not_missing_status():
    decision = {
        "as_of_bar_timestamp": "2026-08-10",
        "equity_order_actions": [
            {"ticker": "AAPL", "action": "BUY", "issue": "not yet confirmed filled"}
        ],
        "crypto_order_actions": [],
        "needs_manual_review": [{"ticker": "AAPL", "issue": "not yet confirmed filled"}],
        "cash_balance_usd": 0.0,
    }

    text = runner._build_daily_notification_text(decision)

    assert "AAPL BUY: not yet confirmed filled" in text


def test_failure_notification_includes_error_type_and_message():
    text = runner._build_failure_notification_text(ValueError("data preparer exploded"))

    assert "FAILED" in text
    assert "ValueError: data preparer exploded" in text


def test_notify_safe_swallows_a_notifier_exception(monkeypatch: pytest.MonkeyPatch):
    def fail_if_called(text):
        raise RuntimeError("Telegram is on fire")

    monkeypatch.setattr(runner, "send_telegram_message", fail_if_called)

    runner._notify_safe("anything")  # must not raise


def test_main_notifies_on_success_and_does_not_affect_normal_exit(
    monkeypatch: pytest.MonkeyPatch,
):
    sent = []
    fake_decision = {
        "as_of_bar_timestamp": "2026-08-10",
        "equity_order_actions": [],
        "crypto_order_actions": [],
        "needs_manual_review": [],
        "cash_balance_usd": 1.0,
    }
    fake_result = {"decision": fake_decision, "log_path": Path("data/live/decisions/fake.json")}

    monkeypatch.setattr(runner, "run_daily_decision", lambda **kwargs: fake_result)
    monkeypatch.setattr(runner, "send_telegram_message", lambda text: sent.append(text) or True)
    monkeypatch.setattr(sys, "argv", ["run_daily_decision.py"])

    runner.main()  # must not raise

    assert len(sent) == 1
    assert "No equity or crypto orders today." in sent[0]


def test_main_still_raises_the_real_error_when_trading_flow_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    """The core fail-safe guarantee: a real failure in the trading flow
    must propagate exactly as before (same exception, same effective
    exit behavior) -- notification is informational only, never a gate."""
    sent = []

    def fake_run_daily_decision(**kwargs):
        raise RuntimeError("Alpaca API is down")

    monkeypatch.setattr(runner, "run_daily_decision", fake_run_daily_decision)
    monkeypatch.setattr(runner, "send_telegram_message", lambda text: sent.append(text) or True)
    monkeypatch.setattr(sys, "argv", ["run_daily_decision.py"])

    with pytest.raises(RuntimeError, match="Alpaca API is down"):
        runner.main()

    assert len(sent) == 1
    assert "FAILED" in sent[0]
    assert "Alpaca API is down" in sent[0]


def test_main_notifies_and_raises_on_the_exact_production_valueerror_shape(
    monkeypatch: pytest.MonkeyPatch,
):
    """Regression for the 2026-08-10 weekend production crash: a
    ValueError raised from inside run_daily_decision() (e.g. the old
    date-mismatch bug) must still both notify and propagate -- proving
    main()'s except clause is not "too narrow" for this exact error type."""
    sent = []

    def fake_run_daily_decision(**kwargs):
        raise ValueError(
            "Prepared tickers do not share a common latest calendar date: "
            "{'AAPL': '2026-08-07', 'BTC-USD': '2026-08-10'}"
        )

    monkeypatch.setattr(runner, "run_daily_decision", fake_run_daily_decision)
    monkeypatch.setattr(runner, "send_telegram_message", lambda text: sent.append(text) or True)
    monkeypatch.setattr(sys, "argv", ["run_daily_decision.py"])

    with pytest.raises(ValueError, match="do not share a common latest calendar date"):
        runner.main()

    assert len(sent) == 1
    assert "FAILED" in sent[0]
    assert "do not share a common latest calendar date" in sent[0]


def test_main_prints_a_loud_warning_when_the_failure_notification_cannot_be_confirmed_sent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """If Telegram delivery itself fails (bad credentials, network issue,
    a bug in the notifier), that failure must ALSO be visible in the
    plain log output -- not just the original exception's traceback --
    so a crash never looks identical to "nothing happened"."""

    def fake_run_daily_decision(**kwargs):
        raise ValueError("Prepared tickers do not share a common latest calendar date: {}")

    monkeypatch.setattr(runner, "run_daily_decision", fake_run_daily_decision)
    monkeypatch.setattr(runner, "send_telegram_message", lambda text: False)
    monkeypatch.setattr(sys, "argv", ["run_daily_decision.py"])

    with pytest.raises(ValueError):
        runner.main()

    captured = capsys.readouterr()
    assert "Telegram failure notification could not be confirmed sent" in captured.err
    assert "do not share a common latest calendar date" in captured.err


def test_main_still_raises_the_real_error_even_if_notification_itself_is_broken(
    monkeypatch: pytest.MonkeyPatch,
):
    """Defense in depth: even if send_telegram_message somehow raises
    (bypassing its own documented never-raises contract), main() must
    still surface the ORIGINAL trading error, not a notifier bug."""

    def fake_run_daily_decision(**kwargs):
        raise RuntimeError("Alpaca API is down")

    def broken_notifier(text):
        raise RuntimeError("notifier bug, should never matter")

    monkeypatch.setattr(runner, "run_daily_decision", fake_run_daily_decision)
    monkeypatch.setattr(runner, "send_telegram_message", broken_notifier)
    monkeypatch.setattr(sys, "argv", ["run_daily_decision.py"])

    with pytest.raises(RuntimeError, match="Alpaca API is down"):
        runner.main()
