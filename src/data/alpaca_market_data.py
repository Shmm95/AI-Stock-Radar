"""Read-only Alpaca market data client for AI-Stock-Radar.

Scope, deliberately narrow for this research stage:
- Historical and latest-quote data only, for US equities and crypto.
- No order placement, order management, or account/trading endpoints
  are used anywhere in this module. There is nothing here that can
  place a trade.
- Credentials are always read from the ALPACA_API_KEY / ALPACA_SECRET_KEY
  environment variables (via .env). They are expected to be a paper
  trading account's keys for this stage. Note that Alpaca's market
  DATA endpoints (unlike its Trading API) are the same host for paper
  and live accounts — "paper" here describes which account the keys
  belong to, not a distinct data URL. This module never touches
  Alpaca's Trading API host, so it cannot reach live order execution
  regardless of which account the keys belong to.

Free-tier data trade-off (deliberately accepted, not hidden):
- Equity data is fetched via DataFeed.IEX, Alpaca's free-tier feed.
  This is IEX exchange volume only, not the full SIP consolidated
  tape, so it can diverge from the true NBBO and misses trades/quotes
  that only occurred on other venues.
- All data here comes over Alpaca's REST API, which on the free tier
  is subject to Alpaca's ~15 minute real-time delay. Even the
  "latest quote" calls below are not true real-time.
Callers needing full-tape or true real-time data need a paid Alpaca
market data plan; that upgrade is out of scope for this module.

Quote null-handling (SafeQuote):
alpaca-py's own `Quote` model declares `ask_price`/`bid_price` as
required, non-Optional floats. Alpaca's raw API can legitimately
return a one-sided quote (e.g. `ap: 0, as: 0` with a real bid) when a
feed has no resting order on that side — this is common right at the
IEX close or for thinly quoted symbols, since IEX is a fraction of
total market volume, not the full tape. Because the SDK's field is a
required float, it cannot represent "no ask" as `None`; it reports
`0.0`, which is indistinguishable at the type level from an
impossible real price of zero. Returning that raw object directly
would let a caller silently treat "no quote" as "the price is zero".
`get_latest_stock_quote`/`get_latest_crypto_quote` therefore convert
into `SafeQuote` below, which nulls out a side whenever its size is
zero (or missing), so "no quote" and "a real price" can never be
confused.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.models import Quote, Trade
from alpaca.data.requests import (
    CryptoBarsRequest,
    CryptoLatestQuoteRequest,
    CryptoLatestTradeRequest,
    StockBarsRequest,
    StockLatestQuoteRequest,
)
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# override=True: python-dotenv's own default (override=False) skips a key
# already present in os.environ even if it's an empty string -- e.g. a
# stale `ALPACA_API_KEY=` left in a crontab's env-var block or a systemd
# EnvironmentFile. Confirmed as a real bug for the same load_dotenv(ENV_PATH)
# pattern in src/notify/telegram_notifier.py (a manual python -c test
# worked; a cron/service-invoked script silently got an empty credential).
# .env is this project's authoritative credential source (see this
# module's own docstring), so it must always win over a blank inherited
# value.
load_dotenv(ENV_PATH, override=True)


def _require_credentials() -> tuple[str, str]:
    """Read Alpaca credentials from the environment.

    Raises RuntimeError if either is missing. The values are returned
    only for immediate use building a client; callers must not log,
    print, or persist them.
    """
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY were not found. "
            "Copy .env.example to .env and fill in your Alpaca paper "
            "trading account's keys."
        )
    return api_key, secret_key


@dataclass(frozen=True)
class SafeQuote:
    """A quote with `None` (not `0.0`) on any side with no resting order.

    See the module docstring's "Quote null-handling" section for why
    alpaca-py's own `Quote` cannot represent this distinction itself.
    """

    symbol: str
    timestamp: datetime
    bid_price: Optional[float]
    bid_size: float
    bid_exchange: Optional[Any]
    ask_price: Optional[float]
    ask_size: float
    ask_exchange: Optional[Any]
    conditions: Optional[Any]
    tape: Optional[str]


def _to_safe_quote(quote: Quote) -> SafeQuote:
    return SafeQuote(
        symbol=quote.symbol,
        timestamp=quote.timestamp,
        bid_price=quote.bid_price if quote.bid_size else None,
        bid_size=quote.bid_size,
        bid_exchange=quote.bid_exchange,
        ask_price=quote.ask_price if quote.ask_size else None,
        ask_size=quote.ask_size,
        ask_exchange=quote.ask_exchange,
        conditions=quote.conditions,
        tape=quote.tape,
    )


def get_stock_historical_client() -> StockHistoricalDataClient:
    """Build a read-only equities market-data client."""
    api_key, secret_key = _require_credentials()
    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)


def get_crypto_historical_client() -> CryptoHistoricalDataClient:
    """Build a read-only crypto market-data client."""
    api_key, secret_key = _require_credentials()
    return CryptoHistoricalDataClient(api_key=api_key, secret_key=secret_key)


def get_stock_bars(
    symbol: str,
    start: datetime,
    end: Optional[datetime] = None,
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.DataFrame:
    """Fetch historical OHLCV bars for a US equity.

    Uses DataFeed.IEX (free tier) — see module docstring for the
    IEX-vs-SIP trade-off this implies.
    """
    client = get_stock_historical_client()
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )
    bars = client.get_stock_bars(request)
    return bars.df


def get_latest_stock_quote(symbol: str) -> SafeQuote:
    """Fetch the latest available equity quote.

    "Latest" is relative to Alpaca's free-tier REST feed, which runs
    ~15 minutes behind true real-time. Returns a `SafeQuote`, not
    alpaca-py's raw `Quote` — see the module docstring's "Quote
    null-handling" section.
    """
    client = get_stock_historical_client()
    request = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
    quotes = client.get_stock_latest_quote(request)
    return _to_safe_quote(quotes[symbol])


def get_crypto_bars(
    symbol: str,
    start: datetime,
    end: Optional[datetime] = None,
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.DataFrame:
    """Fetch historical OHLCV bars for a crypto pair (e.g. "BTC/USD")."""
    client = get_crypto_historical_client()
    request = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
    )
    bars = client.get_crypto_bars(request)
    return bars.df


def get_latest_crypto_quote(symbol: str) -> SafeQuote:
    """Fetch the latest available crypto quote (e.g. "BTC/USD").

    Returns a `SafeQuote`, not alpaca-py's raw `Quote` — see the
    module docstring's "Quote null-handling" section.
    """
    client = get_crypto_historical_client()
    request = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
    quotes = client.get_crypto_latest_quote(request)
    return _to_safe_quote(quotes[symbol])


def get_latest_crypto_trade(symbol: str) -> Trade:
    """Fetch the latest executed crypto trade (e.g. "BTC/USD").

    Deliberately trade price, not quote: Alpaca's own native stop
    orders trigger off a trade at or through the stop level, not off
    the bid/ask quote. `src/live/crypto_stop_monitor.py` uses this
    function (never `get_latest_crypto_quote`) so its own trigger
    check mirrors that same basis for crypto, which has no native stop
    order type.
    """
    client = get_crypto_historical_client()
    request = CryptoLatestTradeRequest(symbol_or_symbols=symbol)
    trades = client.get_crypto_latest_trade(request)
    return trades[symbol]
