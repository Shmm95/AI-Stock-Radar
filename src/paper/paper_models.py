"""Data models for the local paper-trading engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    """Return the current UTC timestamp."""

    return datetime.now(UTC).isoformat()


def create_identifier(prefix: str) -> str:
    """Create a short unique identifier."""

    return f"{prefix}-{uuid4().hex[:12]}"


@dataclass(frozen=True, slots=True)
class PaperOrder:
    """One simulated order submitted to the paper broker."""

    order_id: str
    ticker: str
    asset_type: str

    side: str
    order_type: str
    status: str

    quantity: float

    requested_price: float
    fill_price: float | None

    stop_loss: float
    take_profit: float

    estimated_value: float
    filled_value: float | None

    fee: float
    slippage: float

    strategy: str
    reason: str

    created_at: str
    filled_at: str | None

    @classmethod
    def create_buy(
        cls,
        *,
        ticker: str,
        asset_type: str,
        quantity: float,
        requested_price: float,
        stop_loss: float,
        take_profit: float,
        estimated_value: float,
        strategy: str,
        reason: str = "",
        order_type: str = "MARKET",
    ) -> PaperOrder:
        """Create a pending simulated BUY order."""

        return cls(
            order_id=create_identifier("order"),
            ticker=ticker.upper(),
            asset_type=asset_type.lower(),
            side="BUY",
            order_type=order_type.upper(),
            status="PENDING",
            quantity=float(quantity),
            requested_price=float(requested_price),
            fill_price=None,
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            estimated_value=float(estimated_value),
            filled_value=None,
            fee=0.0,
            slippage=0.0,
            strategy=strategy,
            reason=reason,
            created_at=utc_now(),
            filled_at=None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the order to a JSON-compatible dictionary."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class PaperPosition:
    """One currently open simulated position."""

    position_id: str
    order_id: str

    ticker: str
    asset_type: str

    quantity: float

    entry_price: float
    current_price: float

    stop_loss: float
    take_profit: float

    position_value: float
    market_value: float

    open_risk_amount: float

    unrealized_pnl: float
    unrealized_pnl_percent: float

    fee_paid: float

    strategy: str

    opened_at: str
    updated_at: str

    @classmethod
    def from_filled_order(
        cls,
        order: PaperOrder,
    ) -> PaperPosition:
        """Create an open position from a filled order."""

        if order.status != "FILLED":
            raise ValueError(
                "Only filled orders can create a position."
            )

        if order.fill_price is None:
            raise ValueError(
                "Filled order does not contain fill_price."
            )

        if order.filled_value is None:
            raise ValueError(
                "Filled order does not contain filled_value."
            )

        open_risk_amount = (
            abs(order.fill_price - order.stop_loss)
            * order.quantity
        )

        timestamp = utc_now()

        return cls(
            position_id=create_identifier("position"),
            order_id=order.order_id,
            ticker=order.ticker,
            asset_type=order.asset_type,
            quantity=order.quantity,
            entry_price=round(order.fill_price, 6),
            current_price=round(order.fill_price, 6),
            stop_loss=round(order.stop_loss, 6),
            take_profit=round(order.take_profit, 6),
            position_value=round(order.filled_value, 2),
            market_value=round(order.filled_value, 2),
            open_risk_amount=round(open_risk_amount, 2),
            unrealized_pnl=0.0,
            unrealized_pnl_percent=0.0,
            fee_paid=round(order.fee, 2),
            strategy=order.strategy,
            opened_at=timestamp,
            updated_at=timestamp,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the position to a JSON-compatible dictionary."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class PaperTrade:
    """One completed simulated trade."""

    trade_id: str
    position_id: str
    opening_order_id: str
    closing_order_id: str | None

    ticker: str
    asset_type: str

    quantity: float

    entry_price: float
    exit_price: float

    gross_entry_value: float
    gross_exit_value: float

    total_fees: float

    gross_pnl: float
    net_pnl: float
    return_percent: float

    close_reason: str
    strategy: str

    opened_at: str
    closed_at: str

    @classmethod
    def create_from_position(
        cls,
        *,
        position: PaperPosition,
        exit_price: float,
        exit_fee: float,
        close_reason: str,
        closing_order_id: str | None = None,
    ) -> PaperTrade:
        """Create a completed trade from an open position."""

        gross_entry_value = (
            position.entry_price
            * position.quantity
        )

        gross_exit_value = (
            exit_price
            * position.quantity
        )

        gross_pnl = (
            gross_exit_value
            - gross_entry_value
        )

        total_fees = (
            position.fee_paid
            + exit_fee
        )

        net_pnl = gross_pnl - total_fees

        if gross_entry_value <= 0:
            return_percent = 0.0
        else:
            return_percent = (
                net_pnl
                / gross_entry_value
                * 100
            )

        return cls(
            trade_id=create_identifier("trade"),
            position_id=position.position_id,
            opening_order_id=position.order_id,
            closing_order_id=closing_order_id,
            ticker=position.ticker,
            asset_type=position.asset_type,
            quantity=position.quantity,
            entry_price=round(position.entry_price, 6),
            exit_price=round(exit_price, 6),
            gross_entry_value=round(
                gross_entry_value,
                2,
            ),
            gross_exit_value=round(
                gross_exit_value,
                2,
            ),
            total_fees=round(
                total_fees,
                2,
            ),
            gross_pnl=round(
                gross_pnl,
                2,
            ),
            net_pnl=round(
                net_pnl,
                2,
            ),
            return_percent=round(
                return_percent,
                4,
            ),
            close_reason=close_reason,
            strategy=position.strategy,
            opened_at=position.opened_at,
            closed_at=utc_now(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the trade to a JSON-compatible dictionary."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class EquityPoint:
    """One recorded paper-account equity observation."""

    timestamp: str

    cash: float
    positions_market_value: float
    account_value: float

    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float

    open_position_count: int

    @classmethod
    def create(
        cls,
        *,
        cash: float,
        positions_market_value: float,
        realized_pnl: float,
        unrealized_pnl: float,
        open_position_count: int,
    ) -> EquityPoint:
        """Create one portfolio equity observation."""

        account_value = (
            cash
            + positions_market_value
        )

        total_pnl = (
            realized_pnl
            + unrealized_pnl
        )

        return cls(
            timestamp=utc_now(),
            cash=round(cash, 2),
            positions_market_value=round(
                positions_market_value,
                2,
            ),
            account_value=round(
                account_value,
                2,
            ),
            realized_pnl=round(
                realized_pnl,
                2,
            ),
            unrealized_pnl=round(
                unrealized_pnl,
                2,
            ),
            total_pnl=round(
                total_pnl,
                2,
            ),
            open_position_count=int(
                open_position_count
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the equity point to a JSON-compatible dictionary."""

        return asdict(self)