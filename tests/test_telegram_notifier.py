"""Tests for src/notify/telegram_notifier.py.

Deterministic and network-free: `requests.post` is monkeypatched.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.notify import telegram_notifier as notifier
from dotenv import load_dotenv


class FakeResponse:
    def __init__(self, *, ok_payload: bool = True, status_code: int = 200):
        self._ok_payload = ok_payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise notifier.requests.HTTPError(
                f"{self.status_code} Client Error: url: https://api.telegram.org/botSECRET/sendMessage"
            )

    def json(self) -> dict:
        return {"ok": self._ok_payload}


def test_missing_credentials_is_a_silent_no_op(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    def fail_if_called(*a, **k):
        raise AssertionError("must not attempt an HTTP call without credentials")

    monkeypatch.setattr(notifier.requests, "post", fail_if_called)

    assert notifier.send_telegram_message("hello") is False


def test_missing_chat_id_only_is_also_a_no_op(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    def fail_if_called(*a, **k):
        raise AssertionError("must not attempt an HTTP call with only half the credentials")

    monkeypatch.setattr(notifier.requests, "post", fail_if_called)

    assert notifier.send_telegram_message("hello") is False


def test_successful_send_posts_correct_payload_and_returns_true(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    captured = {}

    def fake_post(url, *, data, timeout):
        captured["url"] = url
        captured["data"] = data
        captured["timeout"] = timeout
        return FakeResponse(ok_payload=True)

    monkeypatch.setattr(notifier.requests, "post", fake_post)

    result = notifier.send_telegram_message("test message")

    assert result is True
    assert captured["url"] == "https://api.telegram.org/botfake-token/sendMessage"
    assert captured["data"] == {"chat_id": "12345", "text": "test message"}
    assert captured["timeout"] == notifier.REQUEST_TIMEOUT_SECONDS


def test_telegram_reporting_ok_false_returns_false(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    monkeypatch.setattr(notifier.requests, "post", lambda *a, **k: FakeResponse(ok_payload=False))

    assert notifier.send_telegram_message("test message") is False


def test_http_error_is_caught_and_returns_false(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    monkeypatch.setattr(
        notifier.requests, "post", lambda *a, **k: FakeResponse(status_code=404)
    )

    assert notifier.send_telegram_message("test message") is False


def test_network_exception_is_caught_and_returns_false(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")

    def fake_post(*a, **k):
        raise notifier.requests.ConnectionError("no network")

    monkeypatch.setattr(notifier.requests, "post", fake_post)

    assert notifier.send_telegram_message("test message") is False


def test_module_load_dotenv_call_uses_override_true():
    """Pin the actual fix in place: the module's own load_dotenv(...) call
    must pass override=True, not rely on python-dotenv's default (False).
    A regression here (e.g. someone "simplifying" back to load_dotenv(ENV_PATH))
    would silently reopen the exact bug reproduced below."""
    source = Path(notifier.__file__).read_text(encoding="utf-8")
    assert "load_dotenv(ENV_PATH, override=True)" in source


def test_stale_empty_env_var_is_not_overridden_without_override_true(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Reproduces the real production symptom: a manual `python -c` test
    (no pre-existing env var) sent successfully, while send_status_update.py
    returned False with no visible error. Root cause -- confirmed here with
    real python-dotenv, not mocked: python-dotenv's load_dotenv() defaults
    to override=False, which SKIPS a key already present in os.environ even
    when its value is an empty string (e.g. a stale `TELEGRAM_BOT_TOKEN=`
    inherited from a crontab env-var block or systemd EnvironmentFile)."""
    env_file = tmp_path / ".env"
    env_file.write_text("TELEGRAM_BOT_TOKEN=REAL_TOKEN_FROM_ENV_FILE\n")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")  # simulates the stale inherited var

    load_dotenv(env_file)  # the OLD call shape (override defaults to False)

    assert os.environ["TELEGRAM_BOT_TOKEN"] == "", (
        "this demonstrates the bug: without override=True, the real .env "
        "value never overwrites the stale empty variable"
    )


def test_override_true_fixes_the_stale_empty_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """The actual fix, proven directly: override=True makes .env
    authoritative over whatever the process already inherited."""
    env_file = tmp_path / ".env"
    env_file.write_text("TELEGRAM_BOT_TOKEN=REAL_TOKEN_FROM_ENV_FILE\n")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")

    load_dotenv(env_file, override=True)  # the FIXED call shape

    assert os.environ["TELEGRAM_BOT_TOKEN"] == "REAL_TOKEN_FROM_ENV_FILE"


def test_failure_log_never_includes_the_bot_token(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "SUPER-SECRET-TOKEN")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    monkeypatch.setattr(
        notifier.requests, "post", lambda *a, **k: FakeResponse(status_code=500)
    )

    with caplog.at_level("WARNING"):
        notifier.send_telegram_message("test message")

    assert "SUPER-SECRET-TOKEN" not in caplog.text
