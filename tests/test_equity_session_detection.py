"""Tests for `src/live/equity_session_detection.py` -- "Missing-Equity-
Session Remediation" design (workspace-c), section A.

Covers: `fetch_expected_equity_sessions`'s settle-buffer filtering (the
real bug this session self-caught and fixed -- "today" must never
appear as expected before its own close + 15-minute buffer has passed),
`detect_missing_equity_sessions`'s complete/missing/unexpected-future
classification, the crypto-ticker rejection, and the explicit design
decision that this module never raises merely because data is missing
(that decision belongs to the orchestrator, not here).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

import src.live.equity_session_detection as detection


class _FakeCalendarClient:
    def __init__(self, entries: list[SimpleNamespace]) -> None:
        self._entries = entries
        self.calls: list[tuple] = []

    def get_calendar(self, request):
        self.calls.append((request.start, request.end))
        return [e for e in self._entries if request.start <= e.date <= request.end]


def _entry(day: date, close_hour: int = 16, close_minute: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        date=day,
        open=datetime(day.year, day.month, day.day, 9, 30),
        close=datetime(day.year, day.month, day.day, close_hour, close_minute),
    )


# --- fetch_expected_equity_sessions ---------------------------------------


def test_fetch_expected_equity_sessions_excludes_since_date_itself():
    client = _FakeCalendarClient([_entry(date(2026, 8, 19)), _entry(date(2026, 8, 20))])
    result = detection.fetch_expected_equity_sessions(client, since_date="2026-08-19")
    assert "2026-08-19" not in result


def test_fetch_expected_equity_sessions_returns_empty_when_since_date_is_today_or_later():
    client = _FakeCalendarClient([])
    today = datetime.now(timezone.utc).date().isoformat()
    result = detection.fetch_expected_equity_sessions(client, since_date=today)
    assert result == []
    assert client.calls == []


def test_fetch_expected_equity_sessions_excludes_todays_session_before_settle_buffer(monkeypatch):
    """The real bug this session self-caught: without the settle-buffer
    filter, today would spuriously appear as "expected" the instant the
    calendar day rolls over, even before market close."""
    fixed_now = datetime(2026, 8, 21, 19, 0, tzinfo=timezone.utc)  # 15:00 Eastern, before 16:00 close

    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now if tz is not None else fixed_now.replace(tzinfo=None)

    monkeypatch.setattr(detection, "datetime", _FixedDatetime)
    client = _FakeCalendarClient([_entry(date(2026, 8, 20)), _entry(date(2026, 8, 21))])
    result = detection.fetch_expected_equity_sessions(client, since_date="2026-08-19")
    assert result == ["2026-08-20"]


def test_fetch_expected_equity_sessions_includes_todays_session_after_settle_buffer(monkeypatch):
    fixed_now = datetime(2026, 8, 21, 20, 20, tzinfo=timezone.utc)  # 16:20 Eastern, 20 min after close

    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now if tz is not None else fixed_now.replace(tzinfo=None)

    monkeypatch.setattr(detection, "datetime", _FixedDatetime)
    client = _FakeCalendarClient([_entry(date(2026, 8, 20)), _entry(date(2026, 8, 21))])
    result = detection.fetch_expected_equity_sessions(client, since_date="2026-08-19")
    assert result == ["2026-08-20", "2026-08-21"]


def test_fetch_expected_equity_sessions_raises_calendar_fetch_error_on_real_failure():
    class _BrokenClient:
        def get_calendar(self, request):
            raise RuntimeError("network down")

    with pytest.raises(detection.CalendarFetchError):
        detection.fetch_expected_equity_sessions(_BrokenClient(), since_date="2026-08-19")


# --- detect_missing_equity_sessions ---------------------------------------


def _bars_frame(dates: list[str]) -> pd.DataFrame:
    index = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d in dates])
    return pd.DataFrame(
        {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 100}, index=index
    )


def test_detect_missing_equity_sessions_rejects_crypto_tickers():
    client = _FakeCalendarClient([])
    with pytest.raises(ValueError):
        detection.detect_missing_equity_sessions(
            client, tickers=("BTC-USD",), open_position_tickers=frozenset(), since_date="2026-08-19"
        )


def test_detect_missing_equity_sessions_reports_complete_session(monkeypatch):
    client = _FakeCalendarClient([_entry(date(2026, 8, 20), close_hour=20)])
    monkeypatch.setattr(
        detection, "datetime",
        type("_D", (datetime,), {"now": classmethod(lambda cls, tz=None: datetime(2026, 8, 21, 12, tzinfo=timezone.utc))}),
    )
    monkeypatch.setattr(
        detection, "fetch_raw_ticker_bars_for_range",
        lambda ticker, *, start, end: _bars_frame(["2026-08-20"]),
    )
    result = detection.detect_missing_equity_sessions(
        client, tickers=("AAPL", "MSFT"), open_position_tickers=frozenset(), since_date="2026-08-19"
    )
    assert result.expected_sessions == ("2026-08-20",)
    assert result.complete_sessions == ("2026-08-20",)
    assert result.missing_by_session == {}
    assert result.unexpected_future_bars == {}


def test_detect_missing_equity_sessions_reports_missing_ticker_without_raising(monkeypatch):
    client = _FakeCalendarClient([_entry(date(2026, 8, 20), close_hour=20)])
    monkeypatch.setattr(
        detection, "datetime",
        type("_D", (datetime,), {"now": classmethod(lambda cls, tz=None: datetime(2026, 8, 21, 12, tzinfo=timezone.utc))}),
    )

    def _fake_fetch(ticker, *, start, end):
        return _bars_frame(["2026-08-20"]) if ticker == "AAPL" else _bars_frame([])

    monkeypatch.setattr(detection, "fetch_raw_ticker_bars_for_range", _fake_fetch)
    result = detection.detect_missing_equity_sessions(
        client, tickers=("AAPL", "MSFT"), open_position_tickers=frozenset({"MSFT"}), since_date="2026-08-19"
    )
    # Must NOT raise -- reports the fact instead (design decision, see module docstring).
    assert result.missing_by_session == {"2026-08-20": ("MSFT",)}
    assert result.complete_sessions == ()


def test_detect_missing_equity_sessions_flags_unexpected_future_bars(monkeypatch):
    client = _FakeCalendarClient([_entry(date(2026, 8, 20), close_hour=20)])
    monkeypatch.setattr(
        detection, "datetime",
        type("_D", (datetime,), {"now": classmethod(lambda cls, tz=None: datetime(2026, 8, 21, 12, tzinfo=timezone.utc))}),
    )
    monkeypatch.setattr(
        detection, "fetch_raw_ticker_bars_for_range",
        lambda ticker, *, start, end: _bars_frame(["2026-08-20", "2026-08-25"]),
    )
    result = detection.detect_missing_equity_sessions(
        client, tickers=("AAPL",), open_position_tickers=frozenset(), since_date="2026-08-19"
    )
    assert result.unexpected_future_bars == {"AAPL": ("2026-08-25",)}
