"""Manual smoke test for the crypto stop-monitor's real Alpaca surface.

Not a pytest test: makes real (paper) API calls. Crypto trades 24/7,
so unlike the equity smoke test this does not need to wait for a
market session.

Run by hand:

    .venv/bin/python scripts/smoke_test_crypto_stop_monitor.py

Verifies, against the real API:
1. `get_latest_crypto_trade` (new function) returns a sane real price.
2. `submit_equity_market_order(..., time_in_force=IOC)` — the exact
   call the monitor makes on a triggered stop — actually works for a
   crypto symbol and fills near-instantly.

Self-cleaning: buys a tiny (~$6-7) BTC/USD notional with IOC, confirms
it filled, then immediately sells the same quantity back with IOC, so
the net position returns to ~0 and nothing is left resting.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpaca.trading.enums import OrderSide, TimeInForce

from src.data.alpaca_market_data import get_latest_crypto_trade
from src.live import order_submission

# Alpaca rejects crypto orders below a $10 notional cost basis (learned
# from a real 403: "cost basis must be >= minimal amount of order 10").
# 0.0002 BTC clears that with margin at anything above ~$50k/BTC.
SMOKE_TEST_SYMBOL = "BTC/USD"
SMOKE_TEST_QUANTITY = 0.0002


def main() -> int:
    trade = get_latest_crypto_trade(SMOKE_TEST_SYMBOL)
    print(f"Latest real trade for {SMOKE_TEST_SYMBOL}: {trade.price} at {trade.timestamp}")

    client = order_submission.get_trading_client()

    print(f"\nSubmitting a real PAPER IOC market BUY: {SMOKE_TEST_QUANTITY} {SMOKE_TEST_SYMBOL}...")
    buy_order = order_submission.submit_equity_market_order(
        client,
        ticker=SMOKE_TEST_SYMBOL,
        side=OrderSide.BUY,
        quantity=SMOKE_TEST_QUANTITY,
        time_in_force=TimeInForce.IOC,
    )
    buy_status = order_submission.wait_for_fill_or_timeout(client, str(buy_order.id))
    print(f"  buy order id: {buy_order.id}, status: {buy_status}")

    if buy_status != "filled":
        print(f"Buy did not fill (status={buy_status}); nothing to sell back. Stopping here.")
        return 0

    # Sell back exactly what is actually available, not the requested
    # quantity: a real fill discovered that Alpaca can deduct crypto
    # fees in-kind, leaving the available balance slightly below what
    # was requested (0.0001995 available after requesting 0.0002 here).
    position = client.get_open_position(SMOKE_TEST_SYMBOL.replace("/", ""))
    available_quantity = float(position.qty_available)
    print(f"  actual available balance to sell back: {available_quantity} BTC")

    print(f"\nSubmitting a real PAPER IOC market SELL to flatten the position...")
    sell_order = order_submission.submit_equity_market_order(
        client,
        ticker=SMOKE_TEST_SYMBOL,
        side=OrderSide.SELL,
        quantity=available_quantity,
        time_in_force=TimeInForce.IOC,
    )
    sell_status = order_submission.wait_for_fill_or_timeout(client, str(sell_order.id))
    print(f"  sell order id: {sell_order.id}, status: {sell_status}")

    print(
        "\nOK: real latest-trade fetch + IOC crypto buy/sell round-trip verified; "
        "position flattened back to ~0."
        if sell_status == "filled"
        else "\nWARNING: sell-back did not confirm filled; check the paper account manually."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
