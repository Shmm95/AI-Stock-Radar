"""Missing equity-session detection -- "Missing-Equity-Session
Remediation" design (workspace-c), section A.

WHY THIS EXISTS: `portfolio_backtest_engine._validate_market_data`
(frozen, never touched -- see this module's own "ISOLATION" note below)
clips every ticker's prepared data to the narrowest COMMON date range
across the whole batch (`common_start`/`common_end`, that function's own
lines ~234-246). In the live path, if even one ticker's fetch is
stale/incomplete, this silently pulls EVERY other ticker's usable data
back to that ticker's own stale last-date too -- confirmed as the
literal mechanism that let a real position's broker-side stop fill go
undetected in local state for three consecutive days (see
`run_daily_decision.py`'s own docstring, "2026-08-21 governance
finding"). This module detects that condition BEFORE any clipping
happens, by fetching each ticker's own RAW, un-clipped bars directly
and comparing against the REAL Alpaca trading calendar -- not by
inspecting anything `_validate_market_data`/`prepare_live_market_data`
already produced (by the time that dict exists, the clipping has
already happened).

ISOLATION (deliberate, load-bearing -- precise claim, not overstated):
this module never CALLS anything from
`src.backtest.portfolio_backtest_engine` -- in particular, never
`_validate_market_data` (the batch-clipping function) and never
`prepare_live_market_data` (which itself calls that function).
Detection's own logic operates entirely BEFORE/OUTSIDE the frozen
engine's clipping. Note the honest caveat: importing
`src.live.data_preparer` for its raw-fetch primitives below
transitively loads `portfolio_backtest_engine` into the process at the
Python import-system level too (that module's own top-level `from
src.backtest.portfolio_backtest_engine import _validate_market_data`
runs on import) -- this module is never calling any of the engine's
code, only reusing `data_preparer`'s own already-established, read-only
per-ticker fetch primitives (`_fetch_bars_with_network_retry`,
`_alpaca_symbol`, `_is_crypto_ticker`) the same discipline already used
throughout this codebase's live modules (see `CLAUDE.md`'s "Known
private-API coupling" section).

FAIL-CLOSED IS THE ORCHESTRATOR'S DEFAULT, NOT THIS MODULE'S OWN
BEHAVIOR: `detect_missing_equity_sessions` below reports the real,
unfiltered facts (which sessions are missing which tickers' bars) --
it does NOT raise merely because something is missing, and does NOT
distinguish an open-position ticker from a ranking-universe-only one
in its own control flow (both are reported identically in
`missing_by_session`). `equity_session_orchestrator.py` is the single
place that decides what a real `missing_by_session` result means: the
design's own section B (narrow, single-session auto-replay) and
section C (a passed execution window's stale-BUY/EXIT handling) are
both legitimate, sanctioned responses to missing data, not violations
of "fail-closed" -- that phrase describes the orchestrator's own
default posture (never silently proceed with incomplete data, unlike
`_validate_market_data`'s own clipping) once neither of those two
narrow paths applies, not a rule enforced by an exception raised from
this detection module itself. There is no partial-EXCLUSION path
anywhere, though: this module never drops a ticker from its own report
to make a session look complete when it is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest

from src.live.data_preparer import _alpaca_symbol, _fetch_bars_with_network_retry, _is_crypto_ticker

_CALENDAR_FETCH_WINDOW_PAD_DAYS = 3  # small pad past "today" so a same-day settle-buffer edge is never missed
# Same real US equity market timezone + settle buffer as
# run_control_arm_decision.py's own _expected_equity_session_date --
# reused deliberately, not reinvented (stdlib zoneinfo, DST-aware).
_EASTERN = ZoneInfo("America/New_York")
_SESSION_CLOSE_SETTLE_BUFFER = timedelta(minutes=15)


class CalendarFetchError(RuntimeError):
    """A real `get_calendar()` call failed or returned something this
    module cannot interpret. Never silently treated as "no sessions."""


class MissingEquitySessionDataError(RuntimeError):
    """Fail-closed: at least one expected equity session is missing a
    bar for at least one tracked ticker, AND neither of the design's
    two sanctioned exception paths (narrow auto-replay / missed-window
    handling) applies. Raised by `equity_session_orchestrator.py`, NOT
    by this module's own `detect_missing_equity_sessions` (see this
    module's own docstring for why) -- defined here since it is
    conceptually this module's own finding being escalated, not a new,
    unrelated error type."""


def fetch_expected_equity_sessions(client: TradingClient, *, since_date: str) -> list[str]:
    """Every real, SETTLED NYSE session strictly after `since_date`
    (typically `last_processed_equity_session_date`), via one real,
    range-based `get_calendar()` call -- mirrors
    `pending_signal_ttl.next_equity_session_date`'s own already-tested
    range-query pattern for the call shape, and
    `run_control_arm_decision._expected_equity_session_date`'s own
    settle-buffer logic for WHICH sessions count as "expected" -- both
    reused deliberately, not reinvented. Returns an empty list if
    `since_date` is today or later (nothing expected yet).

    SETTLED, not merely "on or before today": today's own session is
    excluded unless it has already closed (+ a 15-minute settle buffer,
    the exact same buffer `_expected_equity_session_date` uses) -- a
    session that merely appears on the calendar but has not actually
    closed yet obviously has no bar to compare against, and including
    it would make `equity_session_orchestrator.py`'s own eligibility
    check (this session being the single, LATEST properly-settled
    expected session) unreliable: without this filter, "today" would
    always show up as a second, spuriously "missing" session the
    moment the calendar day rolls over, even before the market opens.

    Raises `CalendarFetchError` on any real Alpaca call failure --
    never silently treated as "no sessions," which would let a real
    gap masquerade as "everything is fine."""
    start = date.fromisoformat(since_date) + timedelta(days=1)
    today = datetime.now(timezone.utc).date()
    if start > today:
        return []
    end = today + timedelta(days=_CALENDAR_FETCH_WINDOW_PAD_DAYS)
    try:
        calendar = client.get_calendar(GetCalendarRequest(start=start, end=end))
    except Exception as error:  # noqa: BLE001 -- any real failure, never masked as "no sessions"
        raise CalendarFetchError(
            f"get_calendar() failed for the window {start}..{end} (since_date={since_date!r}): "
            f"{type(error).__name__}: {error}. Fail-closed -- refusing to proceed without a real "
            f"calendar answer."
        ) from error

    settled: list[str] = []
    now_utc = datetime.now(timezone.utc)
    for entry in calendar:
        entry_date = entry.date.isoformat()
        if entry_date > today.isoformat():
            continue
        close_utc = entry.close.replace(tzinfo=_EASTERN).astimezone(timezone.utc)
        if now_utc < close_utc + _SESSION_CLOSE_SETTLE_BUFFER:
            continue  # today's own session, not yet closed/settled -- not "expected" yet
        settled.append(entry_date)
    return sorted(set(settled))


def fetch_raw_ticker_bars_for_range(ticker: str, *, start: datetime, end: datetime) -> pd.DataFrame:
    """One ticker's RAW bars over `[start, end]`, BEFORE any indicator
    computation or cross-ticker date-range clipping -- the exact
    `Open/High/Low/Close/Volume` shape
    `src.live.data_preparer._fetch_raw_bars` itself produces, just with
    an explicit date range instead of that function's own
    lookback-from-now-only `fetch_calendar_days` parameter (which
    cannot target an arbitrary past window). Reuses
    `_fetch_bars_with_network_retry`/`_alpaca_symbol` directly -- the
    SAME real network-retry discipline `_fetch_raw_bars` uses, not a
    weaker, reimplemented fetch."""
    alpaca_symbol = _alpaca_symbol(ticker)
    raw = _fetch_bars_with_network_retry(ticker=ticker, alpaca_symbol=alpaca_symbol, start=start, end=end)
    if raw.empty:
        return raw
    frame = raw.xs(alpaca_symbol, level="symbol")
    frame = frame.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    return frame[["Open", "High", "Low", "Close", "Volume"]]


@dataclass(frozen=True)
class DetectionResult:
    """`complete_sessions` -- expected sessions where every tracked
    ticker has a real bar. `missing_by_session` -- expected session
    date -> tuple of tickers missing a bar for that date (empty dict if
    nothing is missing). `unexpected_future_bars` -- ticker -> tuple of
    bar dates found STRICTLY AFTER the latest expected session (a real,
    separate anomaly: data claiming to be from a session the calendar
    does not yet confirm exists -- see `_STOP_LOSS`-adjacent fail-closed
    conditions in the orchestrator for why this also blocks a run)."""

    expected_sessions: tuple[str, ...]
    complete_sessions: tuple[str, ...]
    missing_by_session: dict[str, tuple[str, ...]]
    unexpected_future_bars: dict[str, tuple[str, ...]]
    raw_bars_by_ticker: dict[str, pd.DataFrame]


def detect_missing_equity_sessions(
    client: TradingClient,
    *,
    tickers: tuple[str, ...],
    open_position_tickers: frozenset[str],
    since_date: str,
) -> DetectionResult:
    """The real detection pass -- see this module's own docstring for
    the full "why" and the isolation guarantee. `tickers` is the full
    tracked (ranking) universe; `open_position_tickers` is the subset
    currently held (used only to make a fail-closed error message
    clearer about triage priority -- both cases raise identically, see
    `MissingEquitySessionDataError`'s own docstring).

    Equity tickers only -- crypto trades 24/7 and has no NYSE calendar
    concept; a caller passing crypto tickers here gets a `ValueError`,
    never a silent skip."""
    equity_tickers = [t for t in tickers if not _is_crypto_ticker(t)]
    if len(equity_tickers) != len(tickers):
        raise ValueError(
            "detect_missing_equity_sessions received crypto ticker(s) -- this function is "
            "equity-only (NYSE calendar has no meaning for a 24/7 asset). Filter to equity "
            "tickers before calling."
        )

    expected_sessions = fetch_expected_equity_sessions(client, since_date=since_date)
    if not expected_sessions:
        return DetectionResult(
            expected_sessions=(), complete_sessions=(), missing_by_session={},
            unexpected_future_bars={}, raw_bars_by_ticker={},
        )

    fetch_start = datetime.combine(date.fromisoformat(expected_sessions[0]), datetime.min.time(), tzinfo=timezone.utc)
    fetch_end = datetime.now(timezone.utc)

    raw_bars_by_ticker: dict[str, pd.DataFrame] = {}
    bar_dates_by_ticker: dict[str, set[str]] = {}
    for ticker in equity_tickers:
        frame = fetch_raw_ticker_bars_for_range(ticker, start=fetch_start, end=fetch_end)
        raw_bars_by_ticker[ticker] = frame
        bar_dates_by_ticker[ticker] = {timestamp.date().isoformat() for timestamp in frame.index}

    complete_sessions: list[str] = []
    missing_by_session: dict[str, tuple[str, ...]] = {}
    for session in expected_sessions:
        missing = tuple(sorted(t for t in equity_tickers if session not in bar_dates_by_ticker[t]))
        if missing:
            missing_by_session[session] = missing
        else:
            complete_sessions.append(session)

    latest_expected = expected_sessions[-1]
    unexpected_future_bars: dict[str, tuple[str, ...]] = {}
    for ticker, dates in bar_dates_by_ticker.items():
        future = tuple(sorted(d for d in dates if d > latest_expected))
        if future:
            unexpected_future_bars[ticker] = future

    # Deliberately does NOT raise here even when `missing_by_session` is
    # non-empty -- see this module's own docstring, "FAIL-CLOSED BY
    # DESIGN, NOT BY EXCEPTION HANDLING": that phrase describes the
    # ORCHESTRATOR's own default posture (never silently proceed with
    # incomplete data, exactly like `_validate_market_data`'s clipping
    # must never be allowed to happen), not a rule this function
    # enforces by raising. Section B (the narrow, single-session,
    # execution-window-not-yet-passed auto-replay case) and section C
    # (a passed execution window's own stale-BUY/EXIT handling) are
    # BOTH legitimate, sanctioned responses to a real
    # `missing_by_session` result -- this function's only job is to
    # report the real, unfiltered facts; `equity_session_orchestrator.py`
    # is the single place that decides which of "replay," "handle the
    # missed window," or "fail closed, no automatic path applies" is
    # correct for the specific facts detected. Open-position vs.
    # ranking-universe-only tickers are NOT distinguished in this
    # return value either, for the same reason both are fail-closed by
    # the orchestrator's own default -- see `open_position_tickers`'
    # own docstring note above for where that distinction is still used
    # (an error message's triage priority, never a different code path
    # here).
    return DetectionResult(
        expected_sessions=tuple(expected_sessions),
        complete_sessions=tuple(complete_sessions),
        missing_by_session=missing_by_session,
        unexpected_future_bars=unexpected_future_bars,
        raw_bars_by_ticker=raw_bars_by_ticker,
    )
