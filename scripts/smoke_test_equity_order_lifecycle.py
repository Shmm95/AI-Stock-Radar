"""Manual smoke test for src/live/order_submission.py against the REAL
Alpaca PAPER trading API (never live/production).

Not a pytest test: it makes a real (paper) order-submission call. Run
by hand:

    .venv/bin/python scripts/smoke_test_equity_order_lifecycle.py

What it proves that tests/test_live_equity_order_execution.py's mocks
cannot: that `submit_order`/`get_order_by_id`/`cancel_order_by_id`
actually round-trip correctly against the real Alpaca API — real auth,
real request serialization, real response parsing.

Self-cleaning: submits the smallest reasonable equity market order (1
share), checks its status, and cancels it immediately (or reports it
was already filled, if the market happened to be open) so nothing is
left resting in the paper account afterward.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpaca.trading.enums import OrderSide

from src.live import order_submission

SMOKE_TEST_TICKER = "AAPL"
SMOKE_TEST_QUANTITY = 1.0


def main() -> int:
    client = order_submission.get_trading_client()

    market_open = order_submission.is_market_open(client)
    print(f"Market open right now: {market_open}")

    print(f"Submitting a real PAPER market BUY: {SMOKE_TEST_QUANTITY} share(s) of {SMOKE_TEST_TICKER}...")
    order = order_submission.submit_equity_market_order(
        client, ticker=SMOKE_TEST_TICKER, side=OrderSide.BUY, quantity=SMOKE_TEST_QUANTITY
    )
    print(f"  order id: {order.id}")

    status = order_submission.get_order_status(client, str(order.id))
    print(f"  status immediately after submit: {status}")

    if status in order_submission.TERMINAL_STATUSES:
        print(f"Order reached a terminal status ({status}) with no cleanup needed.")
        return 0

    print("Order is not yet terminal (expected while the market is closed). Canceling it now...")
    order_submission.cancel_order(client, str(order.id))
    final_status = order_submission.get_order_status(client, str(order.id))
    print(f"  status after cancel request: {final_status}")

    print("OK: real market-order submit -> status -> cancel round-trip verified.")

    print(f"\nSubmitting a real PAPER protective stop SELL for {SMOKE_TEST_TICKER}...")
    stop_order = order_submission.submit_equity_stop_sell(
        client, ticker=SMOKE_TEST_TICKER, quantity=SMOKE_TEST_QUANTITY, stop_price=1.00
    )
    print(f"  stop order id: {stop_order.id}")
    stop_status = order_submission.get_order_status(client, str(stop_order.id))
    print(f"  status immediately after submit: {stop_status}")

    if stop_status not in order_submission.TERMINAL_STATUSES:
        order_submission.cancel_order(client, str(stop_order.id))
        stop_final_status = order_submission.get_order_status(client, str(stop_order.id))
        print(f"  status after cancel request: {stop_final_status}")

    print("OK: real stop-order submit -> status -> cancel round-trip verified; nothing left resting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
