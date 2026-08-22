"""Tests for scripts/check_env_health.py -- specifically the live/control
PROFILE distinction added to close a real, independent-audit-found gap
(2026-08-22, finding #4): this script previously treated
LIVE_ACCOUNT_NUMBER_SUFFIX as always-optional and Layer 3 always compared
against the control arm's own hardcoded EXPECTED_ACCOUNT_SUFFIX ("XO4Y")
regardless of which .env file was actually being checked -- so this health
check could report PASS on the LIVE .env (the field merely optional here)
while scripts/run_daily_decision.py's own `_require_live_account_suffix()`
would fail closed on that exact same file (the field is mandatory there).

Deterministic and network-free: Layer 3's real Alpaca/Telegram calls are
monkeypatched, never real (this suite's own tests/conftest.py already
blocks real `requests.post`; Layer 3 also uses `requests.get` and a real
`TradingClient`, both explicitly faked here rather than relying on that
autouse fixture alone).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.check_env_health as ceh

_VALID_ALPACA_API_KEY = "A" * ceh.EXPECTED_ALPACA_API_KEY_LENGTH
_VALID_ALPACA_SECRET_KEY = "B" * ceh.EXPECTED_ALPACA_SECRET_KEY_LENGTH
_VALID_TELEGRAM_BOT_TOKEN = "1" * 10 + ":" + "C" * (ceh.EXPECTED_TELEGRAM_BOT_TOKEN_LENGTH - 11)


def _write_env_file(path: Path, *, live_suffix: str | None) -> Path:
    lines = [
        f"ALPACA_API_KEY={_VALID_ALPACA_API_KEY}",
        f"ALPACA_SECRET_KEY={_VALID_ALPACA_SECRET_KEY}",
        f"TELEGRAM_BOT_TOKEN={_VALID_TELEGRAM_BOT_TOKEN}",
        "TELEGRAM_CHAT_ID=123456789",
    ]
    if live_suffix is not None:
        lines.append(f"LIVE_ACCOUNT_NUMBER_SUFFIX={live_suffix}")
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


# --- Layer 2: LIVE_ACCOUNT_NUMBER_SUFFIX required only for profile=live ---


def test_control_profile_does_not_require_live_account_suffix(tmp_path: Path):
    env_file = _write_env_file(tmp_path / ".env.control", live_suffix=None)
    findings = ceh.check_layer2_field_schema(env_file, profile=ceh.PROFILE_CONTROL)
    assert not any("LIVE_ACCOUNT_NUMBER_SUFFIX is missing or empty" in f.message for f in findings)


def test_live_profile_requires_live_account_suffix(tmp_path: Path):
    """The real gap: run_daily_decision.py's own _require_live_account_suffix()
    treats this field as mandatory -- profile=live must agree, not silently
    allow it to be absent the way profile=control (correctly) does."""
    env_file = _write_env_file(tmp_path / ".env", live_suffix=None)
    findings = ceh.check_layer2_field_schema(env_file, profile=ceh.PROFILE_LIVE)
    assert any("LIVE_ACCOUNT_NUMBER_SUFFIX is missing or empty" in f.message for f in findings)
    assert any(f.severity == "FAIL" for f in findings)


def test_live_profile_passes_layer2_when_suffix_present_and_valid_length(tmp_path: Path):
    env_file = _write_env_file(tmp_path / ".env", live_suffix="ABCD")
    findings = ceh.check_layer2_field_schema(env_file, profile=ceh.PROFILE_LIVE)
    assert not any(f.severity == "FAIL" for f in findings)


def test_live_profile_flags_wrong_length_suffix(tmp_path: Path):
    env_file = _write_env_file(tmp_path / ".env", live_suffix="TOOLONG")
    findings = ceh.check_layer2_field_schema(env_file, profile=ceh.PROFILE_LIVE)
    assert any("LIVE_ACCOUNT_NUMBER_SUFFIX length is 7" in f.message for f in findings)


# --- Layer 3: live profile compares against the FILE'S OWN suffix, never
# the control arm's hardcoded constant ---


class _FakeAccount:
    def __init__(self, account_number: str) -> None:
        self.account_number = account_number


class _FakeTradingClient:
    def __init__(self, *, api_key: str, secret_key: str, paper: bool) -> None:
        self._account_number = "REAL0000LIVEACCT" + _LIVE_ACCOUNT_SUFFIX_FOR_TEST

    def get_account(self) -> _FakeAccount:
        return _FakeAccount(self._account_number)


_LIVE_ACCOUNT_SUFFIX_FOR_TEST = "ZZZZ"  # deliberately NOT the control arm's "XO4Y"


class _FakeTelegramResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload


def _fake_requests_get(url: str, *args, **kwargs) -> _FakeTelegramResponse:
    if "getMe" in url:
        return _FakeTelegramResponse({"ok": True, "result": {"username": "fake_bot"}})
    if "getChat" in url:
        return _FakeTelegramResponse({"ok": True})
    raise AssertionError(f"unexpected URL in test: {url}")


def test_live_profile_layer3_compares_against_the_files_own_suffix_not_the_control_constant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """REAL GAP FOUND AND FIXED (2026-08-22, independent audit finding
    #4): before this fix, Layer 3 always compared against
    EXPECTED_ACCOUNT_SUFFIX ("XO4Y", the CONTROL arm's own constant) no
    matter which file was being checked -- a live .env whose real account
    ends in a DIFFERENT suffix (as it always will; the live account was
    never XO4Y) would always FAIL Layer 3, even though that account is
    exactly the one this file's own LIVE_ACCOUNT_NUMBER_SUFFIX correctly
    declares. profile=live must compare against the file's OWN declared
    value instead."""
    monkeypatch.setattr("alpaca.trading.client.TradingClient", _FakeTradingClient)
    monkeypatch.setattr("requests.get", _fake_requests_get)

    env_file = _write_env_file(tmp_path / ".env", live_suffix=_LIVE_ACCOUNT_SUFFIX_FOR_TEST)
    findings = ceh.check_layer3_cross_identity(env_file, profile=ceh.PROFILE_LIVE)
    assert not any(f.severity == "FAIL" for f in findings), findings

    # Contrast: the OLD (pre-fix) behavior -- comparing against the
    # control constant regardless of profile -- would have failed this
    # exact same real account.
    old_behavior_findings = ceh.check_layer3_cross_identity(
        env_file, expected_account_suffix=ceh.EXPECTED_ACCOUNT_SUFFIX, profile=ceh.PROFILE_CONTROL
    )
    assert any(f.severity == "FAIL" and "get_account()" in f.message for f in old_behavior_findings)


def test_live_profile_layer3_fails_closed_when_suffix_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("alpaca.trading.client.TradingClient", _FakeTradingClient)
    monkeypatch.setattr("requests.get", _fake_requests_get)

    env_file = _write_env_file(tmp_path / ".env", live_suffix=None)
    findings = ceh.check_layer3_cross_identity(env_file, profile=ceh.PROFILE_LIVE)
    assert any(
        f.severity == "FAIL" and "cannot perform the account-identity cross-check" in f.message for f in findings
    )


def test_control_profile_unchanged_default_behavior(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """profile=control (the default) must behave exactly as before this
    task -- a real account ending in the control arm's own XO4Y."""
    class _FakeControlTradingClient(_FakeTradingClient):
        def __init__(self, *, api_key: str, secret_key: str, paper: bool) -> None:
            self._account_number = "REAL0000CONTROL" + ceh.EXPECTED_ACCOUNT_SUFFIX

    monkeypatch.setattr("alpaca.trading.client.TradingClient", _FakeControlTradingClient)
    monkeypatch.setattr("requests.get", _fake_requests_get)

    env_file = _write_env_file(tmp_path / ".env.control", live_suffix=None)
    findings = ceh.check_layer3_cross_identity(env_file, profile=ceh.PROFILE_CONTROL)
    assert not any(f.severity == "FAIL" for f in findings), findings
