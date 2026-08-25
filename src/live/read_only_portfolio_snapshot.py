"""Read-only broker P&L snapshot -- the shared data layer for the P&L
Telegram notification (`scripts/send_status_update.py`) and, later, the
dashboard.

WHY THIS EXISTS: both consumers need the exact same numbers (open
positions with their broker-reported unrealized P&L, realized P&L
reconstructed from real fills, and the day's account-level P&L), and
both must read them off ONE already-identity-verified `TradingClient`
rather than each building their own -- the same "one verified client,
passed through" discipline `src/live/account_state.py`'s own
`get_live_cash_balance` docstring documents fixing a real bug over.
This module never builds a `TradingClient` itself; `fetch_portfolio_pnl`
takes one as a required parameter.

SCOPE: read-only. The only calls made against `client` are
`get_account()`, `get_all_positions()`, and `get_orders()` (a CLOSED-status
query) -- no order is ever placed, modified, or canceled from here, and
this module never imports `order_submission.py` or any other
order-mutating module. Every numeric broker field arrives from the SDK
as a `str` (confirmed against the installed `alpaca-py` models) and is
converted via `Decimal(str(x))`, never `float`, matching this
codebase's own established money-comparison convention (see
`broker_reconciliation.py`'s module docstring).

UNREALIZED P&L: taken directly from the broker `Position` model's own
`unrealized_pl`/`unrealized_plpc`/`current_price`/`market_value` fields
-- no separate market-data call is needed or made. A second, INDEPENDENT
value is also computed locally from `avg_entry_price`/`current_price`/
`quantity`/`side` purely as an integrity cross-check (`integrity_discrepancy`
below); the broker's own `unrealized_pl` is always the authoritative
number shown, never the locally-computed one -- a real broker fee/swap
adjustment this module doesn't model could legitimately make the two
differ by a small amount without that being a bug.

REALIZED P&L: reconstructed from real fills via
`scripts/generate_performance_report.py`'s own existing, pure
`fetch_filled_orders`/`reconstruct_round_trips` functions (reused
as-is, not reimplemented -- that module already does this exact FIFO
matching for `send_performance_report.py`). `fetch_filled_orders`'s own
underlying `get_orders()` call is capped at
`generate_performance_report.DEFAULT_ORDER_LOOKBACK_LIMIT` (500) closed
orders with no pagination -- if the account has more than that many
historical closed orders, older fills silently fall outside the
window. This module makes its own extra raw `get_orders()` call (same
filter, no filtering applied) purely to COUNT how many closed orders
exist within that same limit; if the raw count equals the limit,
`realized_history_complete` is set `False` so callers never present
`realized_closed_trade_pnl` as a definitive, complete number.

COMBINED P&L: `realized_closed_trade_pnl + broker_unrealized_pnl_total`,
always surfaced under the exact label `COMBINED_STRATEGY_PNL_LABEL`
below ("Strategy fill-based P&L (realized + broker unrealized)") --
deliberately never called "lifetime P&L", since it excludes anything
before this module's own order-history window and excludes any
non-fill account activity (deposits/withdrawals/interest).

DAY P&L: `account.equity - account.last_equity`, kept as its own,
separate field -- never folded into the combined strategy P&L above,
since it reflects the WHOLE account's day-over-day change (including
anything the strategy itself didn't do), not fill-based strategy
performance.

Nothing here raises on a "normal" data shape (no positions, no closed
orders); a real network/auth/broker failure is allowed to propagate --
the CALLER (e.g. `send_status_update.py`) is responsible for its own
graceful-degradation wrapping around this whole fetch, the same pattern
its other independent sources already use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from alpaca.common.enums import Sort
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

import scripts.generate_performance_report as generate_performance_report

COMBINED_STRATEGY_PNL_LABEL = "Strategy fill-based P&L (realized + broker unrealized)"


def _enum_value(value: Any) -> str:
    return str(value.value if hasattr(value, "value") else value)


def _decimal_or_zero(value: Any) -> Decimal:
    return Decimal(str(value)) if value is not None else Decimal("0")


@dataclass(slots=True)
class ReadOnlyPositionPnl:
    ticker: str
    asset_class: str
    side: str  # "long" | "short"
    quantity: Decimal
    avg_entry_price: Decimal
    current_price: Decimal
    market_value: Decimal
    broker_unrealized_pnl: Decimal
    broker_unrealized_pnl_pct: Decimal
    # Independent, LOCAL cross-check only -- see module docstring's
    # "UNREALIZED P&L" section for why `broker_unrealized_pnl` above,
    # never this field, is the number callers should display as truth.
    calculated_unrealized_pnl: Decimal
    integrity_discrepancy: Decimal  # calculated_unrealized_pnl - broker_unrealized_pnl


@dataclass(slots=True)
class ReadOnlyPortfolioPnl:
    as_of_utc: str
    portfolio_value: Decimal  # account.equity -- already includes cash + all positions' market value
    cash: Decimal
    day_pnl: Decimal  # account.equity - account.last_equity; a whole-account figure, see module docstring
    positions: list[ReadOnlyPositionPnl]
    broker_unrealized_pnl_total: Decimal
    realized_closed_trade_pnl: Decimal
    realized_history_complete: bool
    combined_strategy_pnl: Decimal
    combined_strategy_pnl_label: str = field(default=COMBINED_STRATEGY_PNL_LABEL)


def _independent_unrealized_pnl(
    *, side: str, quantity: Decimal, avg_entry_price: Decimal, current_price: Decimal
) -> Decimal:
    """The two formulas the audit spec named explicitly:
    LONG: (current_price - entry_price) * quantity
    SHORT: (entry_price - current_price) * abs(quantity)
    `quantity` is taken as-is (Alpaca reports a short position's own
    `qty` as negative), so `abs()` is applied for the SHORT case only,
    matching the spec's own formula literally."""
    if side == "short":
        return (avg_entry_price - current_price) * abs(quantity)
    return (current_price - avg_entry_price) * quantity


def _build_position_pnl(position: Any) -> ReadOnlyPositionPnl:
    side = _enum_value(position.side)
    quantity = Decimal(str(position.qty))
    avg_entry_price = Decimal(str(position.avg_entry_price))
    current_price = _decimal_or_zero(position.current_price)
    market_value = _decimal_or_zero(position.market_value)
    broker_unrealized_pnl = _decimal_or_zero(position.unrealized_pl)
    broker_unrealized_pnl_pct = _decimal_or_zero(position.unrealized_plpc)
    calculated_unrealized_pnl = _independent_unrealized_pnl(
        side=side, quantity=quantity, avg_entry_price=avg_entry_price, current_price=current_price,
    )
    return ReadOnlyPositionPnl(
        ticker=position.symbol,
        asset_class=_enum_value(position.asset_class),
        side=side,
        quantity=quantity,
        avg_entry_price=avg_entry_price,
        current_price=current_price,
        market_value=market_value,
        broker_unrealized_pnl=broker_unrealized_pnl,
        broker_unrealized_pnl_pct=broker_unrealized_pnl_pct,
        calculated_unrealized_pnl=calculated_unrealized_pnl,
        integrity_discrepancy=calculated_unrealized_pnl - broker_unrealized_pnl,
    )


def _count_raw_closed_orders(client: TradingClient) -> int:
    """Same filter `fetch_filled_orders` uses internally, but unfiltered
    -- purely to detect whether the 500-order cap was actually hit (see
    module docstring's "REALIZED P&L" section). A second real API call,
    same duplication this codebase's own `count_heartbeat_fills`
    already accepts for an identical reason."""
    request = GetOrdersRequest(
        status=QueryOrderStatus.CLOSED,
        limit=generate_performance_report.DEFAULT_ORDER_LOOKBACK_LIMIT,
        direction=Sort.ASC,
    )
    return len(client.get_orders(filter=request))


def fetch_portfolio_pnl(client: TradingClient) -> ReadOnlyPortfolioPnl:
    """The one entry point both consumers call. See module docstring
    for the full contract; every broker call here is read-only."""
    account = client.get_account()
    broker_positions = list(client.get_all_positions())

    positions = [_build_position_pnl(position) for position in broker_positions]
    broker_unrealized_pnl_total = sum((p.broker_unrealized_pnl for p in positions), Decimal("0"))

    raw_closed_order_count = _count_raw_closed_orders(client)
    realized_history_complete = raw_closed_order_count < generate_performance_report.DEFAULT_ORDER_LOOKBACK_LIMIT

    filled_orders = generate_performance_report.fetch_filled_orders(client)
    closed_trades, _open_lots = generate_performance_report.reconstruct_round_trips(filled_orders)
    realized_closed_trade_pnl = sum((Decimal(str(trade.pnl)) for trade in closed_trades), Decimal("0"))

    return ReadOnlyPortfolioPnl(
        as_of_utc=datetime.now(UTC).isoformat(),
        portfolio_value=_decimal_or_zero(account.equity),
        cash=_decimal_or_zero(account.cash),
        day_pnl=_decimal_or_zero(account.equity) - _decimal_or_zero(account.last_equity),
        positions=positions,
        broker_unrealized_pnl_total=broker_unrealized_pnl_total,
        realized_closed_trade_pnl=realized_closed_trade_pnl,
        realized_history_complete=realized_history_complete,
        combined_strategy_pnl=realized_closed_trade_pnl + broker_unrealized_pnl_total,
    )
