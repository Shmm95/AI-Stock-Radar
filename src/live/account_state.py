"""Read-only Alpaca paper-account cash balance.

This is the one place in the live dry-run tooling that touches
Alpaca's Trading API host (`src/data/alpaca_market_data.py` is
market-data-only and deliberately never imports it). The only call
made here is `TradingClient.get_account()` — an account-info read.
Nothing in this module can place, modify, or cancel an order; the
trading client's order-submission methods are never called.

Reuses `alpaca_market_data._require_credentials()` for the
ALPACA_API_KEY / ALPACA_SECRET_KEY env-var loading so credential
handling stays in one place; builds its own `TradingClient` (via the
optional `client` parameter's own default-None fallback) only when the
caller doesn't already have one, because Phase 1's market-data client
classes have no account-balance method.
"""

from __future__ import annotations

from alpaca.trading.client import TradingClient

from src.data.alpaca_market_data import _require_credentials


def get_live_cash_balance(client: TradingClient | None = None) -> float:
    """Return the paper account's current cash balance in USD.

    `client` -- REAL BUG FOUND AND FIXED (2026-08-22, independent
    audit): `run_daily_decision()`'s broker-reconciliation wiring
    already builds and identity-verifies one real `TradingClient` for
    this run; this function used to unconditionally build a SECOND,
    separate one, meaning cash and the reconciliation snapshot could
    come from two different underlying connections (and, in a test,
    from a real client even when the reconciliation side was fully
    faked). Pass the SAME already-verified client through; only build a
    fresh one (the original behavior, unchanged for every other
    existing caller) when none is given."""
    if client is None:
        api_key, secret_key = _require_credentials()
        client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)
    account = client.get_account()
    return float(account.cash)
