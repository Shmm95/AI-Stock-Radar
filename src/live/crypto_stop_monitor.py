"""One-shot crypto stop-loss monitor.

Alpaca has no native resting stop order for crypto — only
`market`/`limit`/`stop_limit` (confirmed against Alpaca's own crypto
order-type docs in the Phase 2 order-type investigation). Equities got
a real broker-side stop (see `src/live/order_submission.py`); crypto
needs our own continuous check standing in for that missing primitive.

Data source decision — latest TRADE, not latest quote: Alpaca's own
native equity stop orders trigger off a *trade* on the consolidated
tape at or through the stop level, not off the bid/ask quote (Alpaca's
own forum answer: "stop loss sell orders are triggered by a 'valid'
trade occurring at or below the stop loss. It's not based upon quotes
but rather trades."). This monitor mirrors that same trigger basis so
its notion of "would a stop have fired" matches what Alpaca's own
equity stop orders actually do, rather than reacting to a quote that
might be one-sided or stale (see `SafeQuote` in
`src/data/alpaca_market_data.py` for why a crypto quote can be
`None`-sided). `get_latest_crypto_trade` is used here; the quote
function is never called by this module.

Scope: this file exposes exactly one entry point meant to be invoked
once per call — `check_and_execute_crypto_stops()`. Nothing here
loops, sleeps, or self-schedules. Whatever eventually triggers it
periodically (cron, a scheduler) is a separate deployment task.

Not a batch-engine replay: `_check_intrabar_stops`/`_check_gap_stops`
in `portfolio_backtest_engine.py` look at a bar's already-known
Low/Open — information that does not exist yet for "today" in a live
context. This module is new, conceptually parallel logic (same
protection intent — close a position at or below its stop level —
applied to a live trade-price stream instead of a completed bar). It
does not call, and cannot call, the engine's private step functions
for this; `portfolio_backtest_engine.py` is only referenced here for
the `_MutablePosition` type, exactly as `src/live/position_state.py`
already does.

Sell quantity: queried from the broker, not the local position's
`quantity`. Crypto fees can be deducted in-kind (confirmed with a real
paper fill: requesting 0.0002 BTC left 0.0001995 BTC actually
available), so local bookkeeping can drift above what Alpaca will
actually let us sell — exactly the same "don't trust our own shadow
number, ask the broker" principle already applied to cash in
`src/live/account_state.py`. `_query_available_crypto_quantity` calls
`TradingClient.get_open_position` right before submitting the sell; if
that reports nothing sellable (already closed some other way, a
transient query error), no sell is submitted — the trigger is recorded
in `needs_manual_review` instead of guessed at. Whatever quantity is
confirmed also overwrites `position.quantity` immediately (before the
sell is even attempted), so this drift cannot silently accumulate
across checks.

Idempotency / overlap protection: primarily via state itself — once a
triggered sell is submitted and the position is removed from
`runner_state.positions`, a later check (any later invocation, however
it was triggered) finds nothing to act on for that ticker. A
`submitted_actions` ledger entry additionally guards a retry of the
exact same logical check (same `check_id`) from resubmitting. Neither
of those protects two *simultaneously overlapping* processes reading
state before either has written it back — that is a real gap, closed
here with a plain lock file (`data/live/crypto_monitor.lock`): a
second invocation that finds the lock already held refuses to run
rather than racing. This is a correctness safeguard, not scheduling
infrastructure — nothing here decides when to run, only whether a
second concurrent run is safe to proceed.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce

from src.backtest.portfolio_backtest_engine import _MutablePosition
from src.data.alpaca_market_data import get_latest_crypto_trade
from src.live import order_submission
from src.live.position_state import LiveRunnerState

DEFAULT_LOCK_PATH = Path("data/live/crypto_monitor.lock")


class MonitorAlreadyRunningError(RuntimeError):
    pass


@contextmanager
def _monitor_lock(lock_path: Path = DEFAULT_LOCK_PATH) -> Iterator[None]:
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise MonitorAlreadyRunningError(
            f"{lock_path} already exists — another monitor invocation appears "
            "to be in progress. Refusing to run concurrently."
        ) from error
    os.close(descriptor)
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _alpaca_crypto_symbol(ticker: str) -> str:
    return ticker.replace("-", "/")


def _query_available_crypto_quantity(client: TradingClient, alpaca_symbol: str) -> float | None:
    """Ask the broker how much of `alpaca_symbol` is actually sellable right now.

    Returns None (never 0.0) when nothing reliable was found — a
    closed position, a transient query error, or an explicit zero
    balance are all "do not guess, do not sell" to the caller.
    """
    try:
        position = client.get_open_position(alpaca_symbol.replace("/", ""))
    except APIError:
        return None
    quantity = float(position.qty_available)
    return quantity if quantity > 0 else None


def is_crypto_stop_triggered(*, current_trade_price: float, stop_loss_price: float) -> bool:
    """Pure trigger check: mirrors a native stop's "trade at or through the level" rule.

    Deliberately not a re-implementation of
    `portfolio_backtest_engine._check_intrabar_stops` — that function
    reads a completed bar's Low, which does not exist yet live. This
    is the same protective intent (close at or below the stop) applied
    to one current trade price.
    """
    return current_trade_price <= stop_loss_price


def check_and_execute_crypto_stops(
    client: TradingClient,
    runner_state: LiveRunnerState,
    *,
    check_id: str,
) -> list[dict]:
    """Check every open crypto position once and sell any that breached its stop.

    Not locked internally — callers that want overlap protection should
    wrap this in `_monitor_lock()` (the CLI script does). The return
    value stays a single flat list (unchanged shape) so existing
    callers like `scripts/run_crypto_stop_monitor.py` need no changes;
    a stop that triggered but could not be sold safely is represented
    as an `action: "NEEDS_REVIEW"` entry within that same list rather
    than a second return value.
    """
    actions: list[dict] = []

    for ticker, position in list(runner_state.positions.items()):
        if position.asset_class != "CRYPTO":
            continue

        action_key = f"{ticker}|CRYPTO_STOP_SELL|{check_id}"
        existing = runner_state.submitted_actions.get(action_key)
        if existing is not None:
            existing["status"] = order_submission.get_order_status(client, existing["order_id"])
            actions.append(
                {"ticker": ticker, "action": "SELL", "trigger": "ALREADY_SUBMITTED", **existing}
            )
            continue

        alpaca_symbol = _alpaca_crypto_symbol(ticker)
        trade = get_latest_crypto_trade(alpaca_symbol)
        current_price = float(trade.price)
        triggered = is_crypto_stop_triggered(
            current_trade_price=current_price, stop_loss_price=position.stop_loss_price
        )
        actions.append(
            {
                "ticker": ticker,
                "action": "CHECK",
                "trigger": "STOP_TRIGGERED" if triggered else "NOT_TRIGGERED",
                "current_trade_price": current_price,
                "stop_loss_price": position.stop_loss_price,
            }
        )
        if not triggered:
            continue

        local_quantity = position.quantity
        available_quantity = _query_available_crypto_quantity(client, alpaca_symbol)
        if available_quantity is None:
            actions.append(
                {
                    "ticker": ticker,
                    "action": "NEEDS_REVIEW",
                    "trigger": "STOP_TRIGGERED",
                    "issue": (
                        "stop triggered but the broker reports no available "
                        "quantity for this symbol; refusing to guess a sell "
                        "quantity from local state"
                    ),
                    "local_quantity": local_quantity,
                    "current_trade_price": current_price,
                    "stop_loss_price": position.stop_loss_price,
                }
            )
            continue

        # Reconcile local state to the broker-confirmed figure as soon
        # as it is known, independent of whether the sell below fills —
        # otherwise the same in-kind-fee drift keeps accumulating.
        position.quantity = available_quantity

        order = order_submission.submit_equity_market_order(
            client,
            ticker=alpaca_symbol,
            side=OrderSide.SELL,
            quantity=available_quantity,
            time_in_force=TimeInForce.IOC,
        )
        status = order_submission.wait_for_fill_or_timeout(client, str(order.id))
        record = {
            "order_id": str(order.id),
            "status": status,
            "kind": "CRYPTO_STOP_SELL",
            "submitted_at": order_submission.now_iso(),
            "trigger_price": current_price,
            "stop_loss_price": position.stop_loss_price,
            "local_quantity": local_quantity,
            "broker_quantity": available_quantity,
        }
        runner_state.submitted_actions[action_key] = record
        if status == "filled":
            del runner_state.positions[ticker]
        actions.append({"ticker": ticker, "action": "SELL", "trigger": "STOP_TRIGGERED", **record})

    return actions
