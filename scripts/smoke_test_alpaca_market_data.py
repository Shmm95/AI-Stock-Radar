"""Manual smoke test for src/data/alpaca_market_data.py.

Not a pytest test: it makes real network calls to Alpaca using
whatever paper-trading keys are in .env, so it is not meant to run
unattended in CI. Run it by hand after filling in .env:

    .venv/bin/python scripts/smoke_test_alpaca_market_data.py

It only reads data (one equity bar/quote lookup, one crypto
bar/quote lookup) and never places or touches any order.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import alpaca_market_data as amd  # noqa: E402

EQUITY_SYMBOL = "AAPL"
CRYPTO_SYMBOL = "BTC/USD"


def _format_ask_price(ask_price: float | None) -> str:
    if ask_price is None:
        return "N/A (no resting ask quote)"
    return str(ask_price)


def main() -> int:
    try:
        amd._require_credentials()
    except RuntimeError as exc:
        print(f"SKIPPED: {exc}")
        return 0

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=5)

    print(f"Fetching equity bars for {EQUITY_SYMBOL}...")
    equity_bars = amd.get_stock_bars(EQUITY_SYMBOL, start=start, end=end)
    print(f"  got {len(equity_bars)} bar rows")

    print(f"Fetching equity latest quote for {EQUITY_SYMBOL}...")
    equity_quote = amd.get_latest_stock_quote(EQUITY_SYMBOL)
    print(f"  latest ask price: {_format_ask_price(equity_quote.ask_price)}")

    print(f"Fetching crypto bars for {CRYPTO_SYMBOL}...")
    crypto_bars = amd.get_crypto_bars(CRYPTO_SYMBOL, start=start, end=end)
    print(f"  got {len(crypto_bars)} bar rows")

    print(f"Fetching crypto latest quote for {CRYPTO_SYMBOL}...")
    crypto_quote = amd.get_latest_crypto_quote(CRYPTO_SYMBOL)
    print(f"  latest ask price: {_format_ask_price(crypto_quote.ask_price)}")

    print("OK: equity and crypto read-only market data both reachable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
