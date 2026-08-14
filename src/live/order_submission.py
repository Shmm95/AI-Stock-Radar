"""Real Alpaca PAPER equity order submission for the live daily runner.

Equity-only, paper-only. Nothing in this module is ever called for a
crypto ticker — `scripts/run_daily_decision.py` keeps crypto entirely
dry-run/log-only, per explicit scope. This module hardcodes
`TradingClient(..., paper=True)`, same as `src/live/account_state.py`;
there is no code path here that can reach Alpaca's live/production
trading host.

Order-type mapping (from the Phase 2 order-type investigation):
- Entry: always a market order (the frozen engine never uses a limit
  price on entry).
- Signal-based exit (`EXIT_SIGNAL_NEXT_OPEN`, from
  `_queue_close_based_exits`): always a market order.
- Protective stop: a native Alpaca `stop` order, placed only once the
  entry market order is confirmed filled. Equities support a plain
  `stop` order (triggers into a market order); that is the closest
  faithful equivalent to the backtest engine's assumption that a
  Low-touch at `stop_loss_price` fills unconditionally.

Why a STOP_LOSS/GAP_STOP_LOSS exit never gets a second sell order from
here: once a protective stop is resting at the broker, the broker
enforces it continuously and independently of whether this script even
runs that day. `scripts/run_daily_decision.py` never calls
`submit_equity_market_order(side=SELL, ...)` for a trade whose
`exit_reason` is `STOP_LOSS` or `GAP_STOP_LOSS` — it only reconciles
local bookkeeping against that stop order's real status. Submitting a
second sell there would risk turning one real exit into two.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.models import Order
from alpaca.trading.requests import MarketOrderRequest, StopOrderRequest

from src.data.alpaca_market_data import _require_credentials

TERMINAL_STATUSES = {"filled", "canceled", "expired", "rejected", "done_for_day"}
# 15 x 3.0s = 45s. A real paper order submitted right after the 09:30
# ET open took ~25-30s to go from `new` to `filled` -- longer than the
# original 6 x 2s = 12s window, which made a real fill look like a
# timeout. 45s gives that observed worst case a comfortable ~1.5x
# margin without making a single slow order eat minutes: most fills
# happen within a couple of seconds, so the loop exits on the first or
# second check in the common case and this ceiling only bites for the
# rare slow-fill/near-open scenario it was raised for.
FILL_POLL_ATTEMPTS = 15
FILL_POLL_INTERVAL_SECONDS = 3.0


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def get_trading_client() -> TradingClient:
    """Build a paper-only trading client. Never pass paper=False here."""
    api_key, secret_key = _require_credentials()
    return TradingClient(api_key=api_key, secret_key=secret_key, paper=True)


def is_market_open(client: TradingClient) -> bool:
    return bool(client.get_clock().is_open)


def submit_equity_market_order(
    client: TradingClient,
    *,
    ticker: str,
    side: OrderSide,
    quantity: float | None = None,
    notional: float | None = None,
    time_in_force: TimeInForce = TimeInForce.DAY,
    client_order_id: str | None = None,
) -> Order:
    """Submit a market order. Despite the name, this is asset-class-agnostic
    at the request level and is reused by `src/live/crypto_stop_monitor.py`
    for crypto stop-triggered sells — with `time_in_force=IOC`, since
    crypto orders reject `DAY` (Alpaca only accepts `gtc`/`ioc` for
    crypto; see the Phase 2 order-type investigation). The default stays
    `DAY` so every existing equity call site is unaffected.

    `notional`/`client_order_id` were added for
    `scripts/run_heartbeat_test.py` (a fixed-dollar-amount pipeline
    exerciser, not a strategy signal) -- every existing call site keeps
    passing `quantity` and omitting the other two, so behavior there is
    unchanged. Exactly one of `quantity`/`notional` must be given,
    exactly like Alpaca's own `MarketOrderRequest` contract.
    """
    if (quantity is None) == (notional is None):
        raise ValueError("Exactly one of quantity or notional must be provided.")
    request = MarketOrderRequest(
        symbol=ticker,
        qty=quantity,
        notional=notional,
        side=side,
        time_in_force=time_in_force,
        client_order_id=client_order_id,
    )
    return client.submit_order(request)


def submit_equity_stop_sell(
    client: TradingClient,
    *,
    ticker: str,
    quantity: float,
    stop_price: float,
    client_order_id: str | None = None,
) -> Order:
    """`client_order_id` added 2026-08-14 alongside the same field on
    `submit_equity_market_order` -- every existing call site omits it
    (defaults to None, Alpaca auto-generates one), so behavior there is
    unchanged."""
    request = StopOrderRequest(
        symbol=ticker,
        qty=quantity,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC,
        stop_price=round(float(stop_price), 2),
        client_order_id=client_order_id,
    )
    return client.submit_order(request)


def cancel_order(client: TradingClient, order_id: str) -> None:
    client.cancel_order_by_id(order_id)


def cancel_order_and_confirm(client: TradingClient, order_id: str) -> str:
    """Request cancellation and poll for it to actually take effect.

    Cancellation is not synchronous: a real paper stop order read back
    as `new` immediately after `cancel_order_by_id` was called, and
    only showed `canceled` moments later. Reuses
    `wait_for_fill_or_timeout`'s poll loop rather than duplicating it —
    `canceled` is one of `TERMINAL_STATUSES` too, so "wait for a
    terminal status" is exactly the right check for a cancellation as
    well as a fill. Can also legitimately return `filled` (the order
    triggered on its own before the cancel request landed) or any
    other terminal status; callers must check the return value rather
    than assume cancellation succeeded.
    """
    cancel_order(client, order_id)
    return wait_for_fill_or_timeout(client, order_id)


def get_order_status(client: TradingClient, order_id: str) -> str:
    order = client.get_order_by_id(order_id)
    status = order.status
    return str(status.value if hasattr(status, "value") else status)


def get_order_by_client_order_id(client: TradingClient, client_order_id: str) -> Order | None:
    """Added 2026-08-14 for `run_daily_decision.py`'s broker-authoritative
    idempotency check: queries Alpaca for a previously-submitted order
    with this exact `client_order_id`. Returns None ONLY on a confirmed
    404 (Alpaca's own "no such order" response) -- any other failure
    (auth, rate limit, a transient network error) is re-raised rather
    than treated as "safe to submit," since misreading a failed lookup
    as "doesn't exist" would silently defeat the entire point of asking
    first. A plain module-level function (not a bare
    `client.get_order_by_client_id(...)` call at each call site) so
    callers can monkeypatch it the same way as every other function
    here, without needing a real or elaborately-faked `client` object.
    """
    try:
        return client.get_order_by_client_id(client_order_id)
    except APIError as error:
        if getattr(error, "status_code", None) == 404:
            return None
        raise


def wait_for_fill_or_timeout(client: TradingClient, order_id: str) -> str:
    """Poll briefly for a terminal status; return whatever status is current on timeout.

    Does not block indefinitely — if the market is closed the order
    will legitimately stay `accepted`/`new`/`pending_new` until the
    next session, which is a normal, expected outcome this function
    reports rather than waits out.
    """
    status = get_order_status(client, order_id)
    attempts = 0
    while status not in TERMINAL_STATUSES and attempts < FILL_POLL_ATTEMPTS:
        time.sleep(FILL_POLL_INTERVAL_SECONDS)
        status = get_order_status(client, order_id)
        attempts += 1
    return status
