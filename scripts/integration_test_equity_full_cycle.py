"""Real, controlled equity integration test: entry -> protective stop
-> simulated signal-based exit -> stop cancellation, all against the
real Alpaca PAPER API (never live/production).

Calls the ACTUAL production function, `run_daily_decision._execute_equity_orders`,
directly -- not a reimplemented copy of its steps. This validates the
real code path (including the three fixes below) rather than a
parallel version of it that could drift from what actually runs.
`run_daily_decision.run_daily_decision`/`main` (the full daily-cycle
orchestration) is still never invoked, and the real
`data/live/position_state.json` is never opened, read, or written --
this script builds its own throwaway, in-memory `LiveRunnerState()`
and passes that in directly.

MUST be run while the US equity market is open (checked below and
aborts otherwise) -- a market order submitted while closed cannot
reach a real fill, which is the entire point of this test.

This is also the regression check for three real findings from the
first run of this test against the real API:
1. `_execute_equity_orders` used to sell before canceling the
   protective stop. Alpaca rejects that: a resting stop SELL holds its
   shares as collateral (`held_for_orders`), and a second sell against
   them fails with `insufficient qty available for order`. Fixed to
   cancel-and-confirm first, sell only after confirmation.
2. `order_submission.wait_for_fill_or_timeout`'s poll window (12s) was
   shorter than a real fill observed near the market open (~25-30s).
   Raised to 45s (see order_submission.py for the exact reasoning).
3. A status check immediately after requesting cancellation could read
   the pre-cancel status. `cancel_order_and_confirm` now polls the
   same way `wait_for_fill_or_timeout` does rather than trusting a
   single immediate read.

Run by hand, only while the market is open:

    .venv/bin/python scripts/integration_test_equity_full_cycle.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

from src.backtest.portfolio_backtest_engine import _MutablePosition
from src.backtest.portfolio_backtest_models import PortfolioTrade
from src.data.alpaca_market_data import get_latest_stock_quote
from src.live import order_submission
from src.live.position_state import LiveRunnerState

import run_daily_decision as runner

PROGRESS_LOG_PATH = Path("data/live/_integration_test_state.json")
TEST_TICKER = "AAPL"
TEST_QUANTITY = 1.0
STOP_PERCENT_BELOW_ENTRY = 5.0  # matches the frozen baseline's stock_stop_loss_percent


def _write_progress(record: dict) -> None:
    PROGRESS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_LOG_PATH.write_text(
        json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def main() -> int:
    client = order_submission.get_trading_client()
    clock = client.get_clock()
    print(f"Market clock: now={clock.timestamp}, is_open={clock.is_open}")
    if not clock.is_open:
        print(
            f"ABORTING: market is closed (next_open={clock.next_open}). "
            "This test requires a real fill; it cannot proceed while the market is closed."
        )
        return 1

    record: dict = {
        "started_at": datetime.now(UTC).isoformat(),
        "ticker": TEST_TICKER,
        "steps": [],
    }
    _write_progress(record)

    # Estimated entry price, the same role `_attempt_open_position`'s
    # `raw_open_price` plays in the real daily flow -- the stop price
    # is always derived from an estimate taken before the real order
    # exists, never from the order's own eventual fill price.
    quote = get_latest_stock_quote(TEST_TICKER)
    estimated_price = quote.ask_price or quote.bid_price
    if estimated_price is None:
        print("ABORTING: no usable bid or ask price available for a stop-price estimate.")
        return 1
    estimated_price = float(estimated_price)
    stop_loss_price = round(estimated_price * (1 - STOP_PERCENT_BELOW_ENTRY / 100), 2)
    print(f"Estimated entry price: {estimated_price}, planned stop_loss_price: {stop_loss_price}")

    state = LiveRunnerState()
    fake_position = _MutablePosition(
        ticker=TEST_TICKER,
        asset_class="EQUITY",
        entry_timestamp=datetime.now(UTC).isoformat(),
        entry_portfolio_bar_index=0,
        quantity=TEST_QUANTITY,
        entry_price=estimated_price,
        entry_fee=0.0,
        stop_loss_price=stop_loss_price,
        highest_close=estimated_price,
        trailing_close_percent=7.5,
        initial_risk_amount=0.0,
        signal_score=100.0,
        signal_reason="INTEGRATION_TEST_SEEDED_SIGNAL",
    )

    # --- Calling the real production function: entry + protective stop ---
    print(f"\n=== _execute_equity_orders: entry {TEST_QUANTITY} {TEST_TICKER} ===")
    entry_actions, entry_review = runner._execute_equity_orders(
        client,
        runner_state=state,
        today_index=1,
        newly_opened={TEST_TICKER: fake_position},
        closed_trades=[],
    )
    for action in entry_actions:
        print(f"  {action}")
    record["entry_actions"] = entry_actions
    record["entry_review"] = entry_review
    _write_progress(record)

    buy_actions = [a for a in entry_actions if a["action"] == "BUY"]
    stop_actions = [a for a in entry_actions if a["action"] == "PLACE_STOP"]
    if not buy_actions or buy_actions[0]["status"] != "filled":
        print(f"\nABORTING: entry did not confirm filled. needs_review={entry_review}")
        _write_progress(record)
        return 1
    if not stop_actions:
        print(f"\nABORTING: entry filled but no protective stop was placed. needs_review={entry_review}")
        _write_progress(record)
        return 1
    print(f"\nEntry filled: order {buy_actions[0]['order_id']}. "
          f"Stop placed: order {stop_actions[0]['order_id']} at {stop_loss_price}.")

    # --- Calling the real production function: simulated signal-based exit ---
    fake_trade = PortfolioTrade(
        ticker=TEST_TICKER,
        asset_class="EQUITY",
        entry_timestamp=fake_position.entry_timestamp,
        exit_timestamp=datetime.now(UTC).isoformat(),
        entry_portfolio_bar_index=1,
        exit_portfolio_bar_index=2,
        quantity=TEST_QUANTITY,
        entry_price=estimated_price,
        exit_price=estimated_price,
        entry_fee=0.0,
        exit_fee=0.0,
        total_fees=0.0,
        gross_pnl=0.0,
        net_pnl=0.0,
        return_percent=0.0,
        holding_period_bars=1,
        exit_reason="EXIT_SIGNAL_NEXT_OPEN",
        signal_score=100.0,
        signal_reason="INTEGRATION_TEST_SEEDED_EXIT_SIGNAL",
    )
    print(f"\n=== _execute_equity_orders: simulated signal-based exit for {TEST_TICKER} ===")
    exit_actions, exit_review = runner._execute_equity_orders(
        client,
        runner_state=state,
        today_index=2,
        newly_opened={},
        closed_trades=[fake_trade],
    )
    for action in exit_actions:
        print(f"  {action}")
    record["exit_actions"] = exit_actions
    record["exit_review"] = exit_review
    _write_progress(record)

    cancel_actions = [a for a in exit_actions if a["action"] == "CANCEL_STOP"]
    sell_actions = [a for a in exit_actions if a["action"] == "SELL"]
    stop_canceled = bool(cancel_actions) and cancel_actions[0]["status"] == "canceled"
    sell_filled = bool(sell_actions) and sell_actions[0]["status"] == "filled"
    print(f"\nStop canceled: {stop_canceled}. Exit sold: {sell_filled}.")
    if exit_review:
        print(f"needs_review from exit step: {exit_review}")

    # --- Final verification ---
    print("\n=== final verification ===")
    try:
        position = client.get_open_position(TEST_TICKER)
        print(f"  WARNING: an open {TEST_TICKER} position still exists: qty={position.qty}")
        no_open_position = False
    except Exception as error:
        print(f"  confirmed: no open {TEST_TICKER} position ({error})")
        no_open_position = True

    open_orders = client.get_orders(
        filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[TEST_TICKER])
    )
    if open_orders:
        print(f"  WARNING: {len(open_orders)} open order(s) still exist for {TEST_TICKER}: "
              f"{[str(o.id) for o in open_orders]}")
        no_open_orders = False
    else:
        print(f"  confirmed: no open orders remain for {TEST_TICKER}")
        no_open_orders = True

    full_cycle_ok = (
        stop_canceled and sell_filled and no_open_position and no_open_orders
        and not entry_review and not exit_review
    )
    record["outcome"] = "FULL_CYCLE_OK" if full_cycle_ok else "FULL_CYCLE_INCOMPLETE"
    _write_progress(record)

    print(f"\n{'OK' if full_cycle_ok else 'INCOMPLETE'}: full cycle "
          f"{'completed cleanly, no manual intervention needed' if full_cycle_ok else 'did not complete cleanly -- see warnings above'}.")

    PROGRESS_LOG_PATH.unlink(missing_ok=True)
    print(f"Progress log removed ({PROGRESS_LOG_PATH}).")

    return 0 if full_cycle_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
