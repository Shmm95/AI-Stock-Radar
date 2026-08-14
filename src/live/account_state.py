"""Read-only Alpaca paper-account cash balance.

This is the one place in the live dry-run tooling that touches
Alpaca's Trading API host (`src/data/alpaca_market_data.py` is
market-data-only and deliberately never imports it). The only call
made here is `TradingClient.get_account()` — an account-info read.
Nothing in this module can place, modify, or cancel an order; the
trading client's order-submission methods are never called.

Reuses `alpaca_market_data._require_credentials()` for the
ALPACA_API_KEY / ALPACA_SECRET_KEY env-var loading so credential
handling stays in one place; builds its own `TradingClient` because
Phase 1's market-data client classes have no account-balance method.
"""

from __future__ import annotations

from alpaca.trading.client import TradingClient

from src.data.alpaca_market_data import _require_credentials


def get_live_cash_balance() -> float:
    """Return the paper account's current cash balance in USD."""
    api_key, secret_key = _require_credentials()
    client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)
    account = client.get_account()
    return float(account.cash)
