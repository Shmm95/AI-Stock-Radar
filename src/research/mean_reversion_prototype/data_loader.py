"""Hourly crypto bars from Alpaca's own historical-bars endpoint --
deliberately NOT yfinance. Today's separate crypto-universe research
found yfinance returning wrong/dead-instrument data for 6 of 33 crypto
tickers (wrong listing dates, price feeds frozen years ago). Alpaca's
own endpoint is the same data source the live system already trades
against, so a signal built on it can never disagree with what the
broker itself would have shown at signal time.

Reuses `src.data.alpaca_market_data.get_crypto_bars` as-is (no engine,
no live-universe files touched).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import NamedTuple

import pandas as pd

from alpaca.data.timeframe import TimeFrame
from src.data.alpaca_market_data import get_crypto_bars

CANDIDATE_SYMBOLS: tuple[str, ...] = ("BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "XRP/USD")

# Requesting from far before any of these listed is deliberate: Alpaca
# simply returns whatever it actually has, so this always gets the
# maximum available history per symbol rather than guessing a cutoff.
_MAX_HISTORY_START = datetime(2015, 1, 1, tzinfo=timezone.utc)


class DataQualityReport(NamedTuple):
    symbol: str
    rows: int
    start: pd.Timestamp
    end: pd.Timestamp
    expected_hourly_bars: int
    missing_bars: int
    missing_percent: float
    invalid_ohlc_rows: int
    zero_volume_bars: int
    duplicate_timestamps_dropped: int


def fetch_hourly_bars(symbol: str, end: datetime | None = None) -> pd.DataFrame:
    """Fetch the maximum available hourly history for one crypto pair.

    Returns a DataFrame indexed by UTC timestamp with columns
    open/high/low/close/volume/trade_count/vwap (vwap is Alpaca's own
    intra-bar volume-weighted price, not this package's rolling VWAP
    signal -- see signals.py).
    """
    end = end or datetime.now(timezone.utc)
    raw = get_crypto_bars(symbol, start=_MAX_HISTORY_START, end=end, timeframe=TimeFrame.Hour)
    frame = raw.reset_index()
    if "symbol" in frame.columns:
        frame = frame.drop(columns=["symbol"])
    frame = frame.set_index("timestamp").sort_index()
    return frame


def assess_quality(symbol: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, DataQualityReport]:
    """Drop duplicate timestamps; report (never silently drop) gaps and
    OHLC sanity violations, mirroring the equity research's hygiene
    checks (drop-worthy vs. merely-flag-worthy is a judgment call left
    to the report, not made silently here)."""
    before = len(frame)
    frame = frame[~frame.index.duplicated(keep="first")]
    duplicate_dropped = before - len(frame)

    full_range = pd.date_range(frame.index.min(), frame.index.max(), freq="h")
    missing = full_range.difference(frame.index)

    max_ocl = frame[["open", "close", "low"]].max(axis=1)
    min_och = frame[["open", "close", "high"]].min(axis=1)
    invalid_ohlc = (frame["high"] < max_ocl) | (frame["low"] > min_och)

    report = DataQualityReport(
        symbol=symbol,
        rows=len(frame),
        start=frame.index.min(),
        end=frame.index.max(),
        expected_hourly_bars=len(full_range),
        missing_bars=len(missing),
        missing_percent=round(100 * len(missing) / len(full_range), 2),
        invalid_ohlc_rows=int(invalid_ohlc.sum()),
        zero_volume_bars=int((frame["volume"] == 0).sum()),
        duplicate_timestamps_dropped=duplicate_dropped,
    )
    return frame, report
