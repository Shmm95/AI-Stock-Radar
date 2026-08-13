"""Tests for the network-retry wrapper in src/live/data_preparer.py.

Scoped narrowly to _fetch_bars_with_network_retry, per the task that
introduced it -- not a full data_preparer.py test suite. Mocking is
appropriate here: this tests OUR retry logic's own behavior (attempt
counting, backoff, which exceptions get retried vs propagate
immediately), not real broker behavior.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest
import requests

from src.live import data_preparer


def _fake_bars() -> pd.DataFrame:
    return pd.DataFrame({"open": [1.0]}, index=pd.DatetimeIndex([datetime.now(UTC)]))


def test_succeeds_on_a_later_attempt_after_network_errors(monkeypatch: pytest.MonkeyPatch):
    calls = []
    attempts = iter(
        [
            requests.exceptions.ConnectionError("reset by peer"),
            requests.exceptions.Timeout("read timed out"),
            _fake_bars(),
        ]
    )

    def fake_get_stock_bars(symbol, *, start, end):
        calls.append(symbol)
        outcome = next(attempts)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    sleeps = []
    monkeypatch.setattr(data_preparer.amd, "get_stock_bars", fake_get_stock_bars)
    monkeypatch.setattr(data_preparer.time, "sleep", lambda seconds: sleeps.append(seconds))

    result = data_preparer._fetch_bars_with_network_retry(
        ticker="AAPL", alpaca_symbol="AAPL",
        start=datetime.now(UTC), end=datetime.now(UTC),
    )

    assert len(calls) == 3
    assert not result.empty
    assert sleeps == [1.0, 2.0]  # exponential backoff before attempts 2 and 3


def test_raises_the_last_error_once_all_attempts_are_exhausted(monkeypatch: pytest.MonkeyPatch):
    calls = []
    errors = [
        requests.exceptions.ConnectionError("first reset"),
        requests.exceptions.ConnectionError("second reset"),
        requests.exceptions.ConnectionError("third reset -- this one must be raised"),
    ]

    def fake_get_stock_bars(symbol, *, start, end):
        calls.append(symbol)
        raise errors[len(calls) - 1]

    sleeps = []
    monkeypatch.setattr(data_preparer.amd, "get_stock_bars", fake_get_stock_bars)
    monkeypatch.setattr(data_preparer.time, "sleep", lambda seconds: sleeps.append(seconds))

    with pytest.raises(requests.exceptions.ConnectionError, match="third reset"):
        data_preparer._fetch_bars_with_network_retry(
            ticker="AAPL", alpaca_symbol="AAPL",
            start=datetime.now(UTC), end=datetime.now(UTC),
        )

    assert len(calls) == data_preparer._NETWORK_RETRY_ATTEMPTS == 3
    assert sleeps == [1.0, 2.0]  # no sleep after the final (3rd) attempt


def test_non_network_errors_are_never_retried(monkeypatch: pytest.MonkeyPatch):
    """A bad-symbol/logic error (matching the real BF-B incident) must
    fail on the first attempt, not be masked by three silent retries."""
    calls = []

    def fake_get_stock_bars(symbol, *, start, end):
        calls.append(symbol)
        raise ValueError("invalid symbol: BF-B")

    sleeps = []
    monkeypatch.setattr(data_preparer.amd, "get_stock_bars", fake_get_stock_bars)
    monkeypatch.setattr(data_preparer.time, "sleep", lambda seconds: sleeps.append(seconds))

    with pytest.raises(ValueError, match="invalid symbol"):
        data_preparer._fetch_bars_with_network_retry(
            ticker="BF-B", alpaca_symbol="BF-B",
            start=datetime.now(UTC), end=datetime.now(UTC),
        )

    assert len(calls) == 1  # never retried
    assert sleeps == []


def test_crypto_ticker_routes_to_get_crypto_bars(monkeypatch: pytest.MonkeyPatch):
    calls = []
    monkeypatch.setattr(
        data_preparer.amd, "get_crypto_bars",
        lambda symbol, *, start, end: calls.append(("crypto", symbol)) or _fake_bars(),
    )
    monkeypatch.setattr(
        data_preparer.amd, "get_stock_bars",
        lambda symbol, *, start, end: calls.append(("stock", symbol)) or _fake_bars(),
    )

    data_preparer._fetch_bars_with_network_retry(
        ticker="BTC-USD", alpaca_symbol="BTC/USD",
        start=datetime.now(UTC), end=datetime.now(UTC),
    )

    assert calls == [("crypto", "BTC/USD")]


def test_logs_a_warning_with_ticker_attempt_number_and_error_type(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
):
    def fake_get_stock_bars(symbol, *, start, end):
        raise requests.exceptions.ConnectionError("reset by peer")

    monkeypatch.setattr(data_preparer.amd, "get_stock_bars", fake_get_stock_bars)
    monkeypatch.setattr(data_preparer.time, "sleep", lambda seconds: None)

    with caplog.at_level("WARNING", logger="src.live.data_preparer"):
        with pytest.raises(requests.exceptions.ConnectionError):
            data_preparer._fetch_bars_with_network_retry(
                ticker="MSFT", alpaca_symbol="MSFT",
                start=datetime.now(UTC), end=datetime.now(UTC),
            )

    assert len(caplog.records) == 3
    assert "MSFT" in caplog.text
    assert "1/3" in caplog.text and "2/3" in caplog.text and "3/3" in caplog.text
    assert "ConnectionError" in caplog.text
