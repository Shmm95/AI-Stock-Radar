"""Prepare live Alpaca market data in the shape the frozen portfolio
backtest engine expects.

Dry-run research tooling only. Nothing here places an order or reads
account state; it only turns Alpaca historical bars into the same
`Open/High/Low/Close/EMA20/EMA50/RSI14/RegimeAllowed` frame the batch
engine consumes, using the frozen strategy's own approved ticker
universe and indicator formulas.

Private-API coupling (documented per CLAUDE.md's "protected areas"
policy — never modified, only imported and called as-is):
- `src.backtest.portfolio_backtest_engine._validate_market_data` — the
  exact OHLC sanity checks, RegimeAllowed defaulting, dropna, and
  common-date-range trimming the batch engine itself relies on. Reused
  here instead of reimplemented so this module cannot silently drift
  from the engine's own validation.
- `src.backtest.run_portfolio_backtest._bull_trend_variant` — selects
  which `RegimeVariant` is the officially active crypto bull-regime
  filter. Imported (not hardcoded) so this module automatically tracks
  whatever variant the backtest pipeline is actually configured with.
- `src.backtest.run_regime_ablation.build_regime_allowed` — the causal
  (forward-fill-only, no backward-fill) regime-gate calculation.

Crypto regime warm-up gotcha: the bull-regime filter needs BTC-USD's
own `EMA200` (`min_periods=200`), a much longer warm-up than the
14-bar RSI14 minimum the rest of the engine needs. Requesting only
~150 calendar days of BTC-USD data would leave `EMA200` — and
therefore `RegimeAllowed` for every crypto ticker — `NaN`/`False` for
the entire window, silently disabling the crypto regime gate. This
module fetches a much longer window (`DEFAULT_FETCH_CALENDAR_DAYS`,
comfortably covering the 200-bar regime warm-up plus the requested
minimum deliverable history) specifically to avoid that failure mode.

Symbol-format note: `LIVE_CONTROLLED_TICKERS` uses the backtest/yfinance
dash convention (`BTC-USD`, `ETH-USD`); Alpaca's crypto API expects a
slash (`BTC/USD`). This module converts only for the Alpaca request
and keeps the dash form as the internal/delivered ticker key, so it
stays consistent with `_is_crypto_ticker`'s suffix rule everywhere
else in the engine.

Universe note: default tickers come from `src.live.live_universe.LIVE_CONTROLLED_TICKERS`,
not `src.backtest.run_portfolio_entry_statistics.CONTROLLED_TICKERS` -- the
live-trading universe is deliberately independent of the frozen 9-ticker
research universe. See that module's docstring for why.

Network-retry note: a real dry-run against a larger candidate universe
(2026-08-13 feasibility check, see docs/BROADER_UNIVERSE research) hit a
`requests.exceptions.ConnectionError` (a raw connection reset) partway
through a sequential fetch -- alpaca-py's own `RESTClient` only retries
HTTP 429 responses (`RetryException`/`retry_exception_codes`, see
`alpaca.common.rest.RESTClient._request`); it does not catch or retry
anything at the socket/connection level, which happens before any HTTP
response even exists. `_fetch_raw_bars` below retries only
`requests.exceptions.ConnectionError`/`Timeout`/`ChunkedEncodingError` --
the raw network-layer exceptions `requests` itself defines -- a fixed
number of times with exponential backoff. Deliberately narrow: an
`alpaca.common.exceptions.APIError` (bad symbol, bad request, any real
HTTP error status) is a different, already-translated exception class
and is never caught here, so a genuine data/logic problem still fails
immediately and loudly instead of being masked by three silent retries.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Iterable

import pandas as pd
import requests

from src.analysis.technical_indicators import add_technical_indicators
from src.backtest.portfolio_backtest_engine import _validate_market_data
from src.backtest.run_portfolio_backtest import _bull_trend_variant
from src.backtest.run_regime_ablation import build_regime_allowed
from src.data import alpaca_market_data as amd
from src.live.live_universe import LIVE_CONTROLLED_TICKERS

logger = logging.getLogger(__name__)

DEFAULT_FETCH_CALENDAR_DAYS = 400
MINIMUM_DELIVERED_BARS = 150
REGIME_SYMBOL = "BTC-USD"

_NETWORK_ERRORS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)
_NETWORK_RETRY_ATTEMPTS = 3
_NETWORK_RETRY_BASE_DELAY_SECONDS = 1.0


def _is_crypto_ticker(ticker: str) -> bool:
    """Mirror portfolio_backtest_engine._is_crypto_ticker's suffix rule."""
    return ticker.strip().upper().endswith(("-USD", "-EUR", "-GBP"))


def _alpaca_symbol(ticker: str) -> str:
    """Convert the engine's dash ticker to Alpaca's crypto slash form."""
    return ticker.replace("-", "/") if _is_crypto_ticker(ticker) else ticker


def _fetch_bars_with_network_retry(
    *, ticker: str, alpaca_symbol: str, start: datetime, end: datetime
) -> pd.DataFrame:
    """Fetch one ticker's bars, retrying only raw network-layer failures.

    See this module's docstring ("Network-retry note") for why this
    exists and why the caught exception set is deliberately narrow.
    """
    last_error: Exception | None = None
    for attempt in range(1, _NETWORK_RETRY_ATTEMPTS + 1):
        try:
            if _is_crypto_ticker(ticker):
                return amd.get_crypto_bars(alpaca_symbol, start=start, end=end)
            return amd.get_stock_bars(alpaca_symbol, start=start, end=end)
        except _NETWORK_ERRORS as error:
            last_error = error
            logger.warning(
                "Network error fetching %s (attempt %d/%d): %s: %s",
                ticker, attempt, _NETWORK_RETRY_ATTEMPTS, type(error).__name__, error,
            )
            if attempt < _NETWORK_RETRY_ATTEMPTS:
                time.sleep(_NETWORK_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)))
    assert last_error is not None
    raise last_error


def _fetch_raw_bars(ticker: str, *, fetch_calendar_days: int) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=fetch_calendar_days)
    alpaca_symbol = _alpaca_symbol(ticker)

    raw = _fetch_bars_with_network_retry(
        ticker=ticker, alpaca_symbol=alpaca_symbol, start=start, end=end
    )

    if raw.empty:
        raise ValueError(f"Alpaca returned no bars for {ticker} ({alpaca_symbol}).")

    frame = raw.xs(alpaca_symbol, level="symbol")
    frame = frame.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    return frame[["Open", "High", "Low", "Close", "Volume"]]


def _regime_data_from(btc_with_indicators: pd.DataFrame) -> pd.DataFrame:
    """Build the MarketClose/MarketEMA50/MarketEMA200 frame build_regime_allowed expects."""
    close = btc_with_indicators["Close"].astype(float)
    return pd.DataFrame(
        {
            "MarketClose": close,
            "MarketEMA50": close.ewm(span=50, adjust=False, min_periods=50).mean(),
            "MarketEMA200": close.ewm(span=200, adjust=False, min_periods=200).mean(),
        },
        index=btc_with_indicators.index,
    )


def prepare_live_market_data(
    tickers: Iterable[str] = LIVE_CONTROLLED_TICKERS,
    *,
    fetch_calendar_days: int = DEFAULT_FETCH_CALENDAR_DAYS,
    minimum_delivered_bars: int = MINIMUM_DELIVERED_BARS,
) -> dict[str, pd.DataFrame]:
    """Fetch, enrich, and validate live Alpaca data for every ticker.

    Returns a dict[ticker, DataFrame] in exactly the shape
    `portfolio_backtest_engine._validate_market_data` and the six
    per-bar step functions expect: Open/High/Low/Close/EMA20/EMA50/
    RSI14/RegimeAllowed, trimmed to a common date range, with no NaNs
    in the required columns.
    """
    tickers = tuple(tickers)
    if REGIME_SYMBOL not in tickers:
        raise ValueError(
            f"{REGIME_SYMBOL} must be included to compute the crypto "
            "bull-regime filter; it cannot be derived from other tickers."
        )

    raw_by_ticker = {
        ticker: _fetch_raw_bars(ticker, fetch_calendar_days=fetch_calendar_days)
        for ticker in tickers
    }

    btc_with_indicators = add_technical_indicators(raw_by_ticker[REGIME_SYMBOL])
    regime_data = _regime_data_from(btc_with_indicators)
    variant = _bull_trend_variant()

    prepared: dict[str, pd.DataFrame] = {}
    for ticker, raw in raw_by_ticker.items():
        enriched = add_technical_indicators(raw)
        if _is_crypto_ticker(ticker):
            enriched["RegimeAllowed"] = build_regime_allowed(
                asset_index=enriched.index,
                regime_data=regime_data,
                variant=variant,
            )
        else:
            enriched["RegimeAllowed"] = True
        prepared[ticker] = enriched

    validated = _validate_market_data(prepared)

    short = {
        ticker: len(frame)
        for ticker, frame in validated.items()
        if len(frame) < minimum_delivered_bars
    }
    if short:
        raise ValueError(
            f"Fewer than {minimum_delivered_bars} valid bars after warm-up "
            f"and validation: {short}. Increase fetch_calendar_days."
        )

    return validated
