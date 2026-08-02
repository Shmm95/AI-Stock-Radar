"""Local JSON-backed paper broker for AI-Stock-Radar.

This module simulates long-only BUY and SELL transactions.
It never connects to a real broker and cannot send real orders.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.paper.paper_account import (
    EQUITY_CURVE_PATH,
    ORDERS_PATH,
    POSITIONS_PATH,
    TRADES_PATH,
    PaperAccount,
    PaperAccountStore,
)
from src.paper.paper_models import (
    EquityPoint,
    PaperOrder,
    PaperPosition,
    PaperTrade,
    create_identifier,
    utc_now,
)
from src.risk.position_sizer import PositionSizeResult
from src.risk.risk_manager import (
    PortfolioSnapshot,
    RiskDecision,
)
from src.strategy.trade_plan import TradePlan


DEFAULT_COMMISSION_RATE = 0.0005
DEFAULT_MINIMUM_FEE = 1.0
DEFAULT_SLIPPAGE_BPS = 5.0


def _round_money(value: float) -> float:
    """Round a monetary value to two decimal places."""

    return round(float(value), 2)


def _round_price(value: float) -> float:
    """Round a price while retaining crypto precision."""

    return round(float(value), 6)


def _atomic_write_json(
    path: Path,
    payload: Any,
) -> None:
    """Write JSON safely through a temporary file."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        f"{path.suffix}.tmp"
    )

    try:
        with temporary_path.open(
            mode="w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                ensure_ascii=False,
                indent=2,
            )

            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        temporary_path.replace(path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _read_json_list(
    path: Path,
) -> list[dict[str, Any]]:
    """Read and validate a JSON list."""

    if not path.exists():
        return []

    raw_content = path.read_text(
        encoding="utf-8",
    ).strip()

    if not raw_content:
        return []

    try:
        payload = json.loads(raw_content)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON file: {path}"
        ) from error

    if not isinstance(payload, list):
        raise TypeError(
            f"Expected a JSON list in: {path}"
        )

    return payload


def _calculate_fee(
    transaction_value: float,
    commission_rate: float,
    minimum_fee: float,
) -> float:
    """Calculate a simulated transaction commission."""

    if transaction_value < 0:
        raise ValueError(
            "transaction_value cannot be negative."
        )

    if commission_rate < 0:
        raise ValueError(
            "commission_rate cannot be negative."
        )

    if minimum_fee < 0:
        raise ValueError(
            "minimum_fee cannot be negative."
        )

    proportional_fee = (
        transaction_value
        * commission_rate
    )

    return _round_money(
        max(
            proportional_fee,
            minimum_fee,
        )
    )


def _apply_buy_slippage(
    requested_price: float,
    slippage_bps: float,
) -> float:
    """Apply adverse slippage to a BUY transaction."""

    if requested_price <= 0:
        raise ValueError(
            "requested_price must be greater than zero."
        )

    multiplier = (
        1
        + slippage_bps
        / 10_000
    )

    return _round_price(
        requested_price
        * multiplier
    )


def _apply_sell_slippage(
    requested_price: float,
    slippage_bps: float,
) -> float:
    """Apply adverse slippage to a SELL transaction."""

    if requested_price <= 0:
        raise ValueError(
            "requested_price must be greater than zero."
        )

    multiplier = (
        1
        - slippage_bps
        / 10_000
    )

    return _round_price(
        requested_price
        * multiplier
    )


class PaperBroker:
    """Local JSON-backed paper-trading broker."""

    def __init__(
        self,
        *,
        account_store: PaperAccountStore | None = None,
        commission_rate: float = DEFAULT_COMMISSION_RATE,
        minimum_fee: float = DEFAULT_MINIMUM_FEE,
        slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    ) -> None:
        if commission_rate < 0:
            raise ValueError(
                "commission_rate cannot be negative."
            )

        if minimum_fee < 0:
            raise ValueError(
                "minimum_fee cannot be negative."
            )

        if slippage_bps < 0:
            raise ValueError(
                "slippage_bps cannot be negative."
            )

        self.account_store = (
            account_store
            or PaperAccountStore()
        )

        self.commission_rate = float(
            commission_rate
        )

        self.minimum_fee = float(
            minimum_fee
        )

        self.slippage_bps = float(
            slippage_bps
        )

        self.account_store.initialize_storage()

    # =========================================================
    # STORAGE
    # =========================================================

    def list_orders(self) -> list[PaperOrder]:
        """Return all simulated orders."""

        return [
            PaperOrder(**item)
            for item in _read_json_list(
                ORDERS_PATH
            )
        ]

    def list_positions(self) -> list[PaperPosition]:
        """Return all currently open positions."""

        return [
            PaperPosition(**item)
            for item in _read_json_list(
                POSITIONS_PATH
            )
        ]

    def list_trades(self) -> list[PaperTrade]:
        """Return all completed trades."""

        return [
            PaperTrade(**item)
            for item in _read_json_list(
                TRADES_PATH
            )
        ]

    def list_equity_points(self) -> list[EquityPoint]:
        """Return all recorded equity observations."""

        return [
            EquityPoint(**item)
            for item in _read_json_list(
                EQUITY_CURVE_PATH
            )
        ]

    def _save_orders(
        self,
        orders: list[PaperOrder],
    ) -> None:
        _atomic_write_json(
            ORDERS_PATH,
            [
                order.to_dict()
                for order in orders
            ],
        )

    def _save_positions(
        self,
        positions: list[PaperPosition],
    ) -> None:
        _atomic_write_json(
            POSITIONS_PATH,
            [
                position.to_dict()
                for position in positions
            ],
        )

    def _save_trades(
        self,
        trades: list[PaperTrade],
    ) -> None:
        _atomic_write_json(
            TRADES_PATH,
            [
                trade.to_dict()
                for trade in trades
            ],
        )

    def _save_equity_points(
        self,
        points: list[EquityPoint],
    ) -> None:
        _atomic_write_json(
            EQUITY_CURVE_PATH,
            [
                point.to_dict()
                for point in points
            ],
        )

    # =========================================================
    # PORTFOLIO INFORMATION
    # =========================================================

    def find_position(
        self,
        ticker: str,
    ) -> PaperPosition | None:
        """Find an open position by ticker."""

        normalized_ticker = (
            ticker.strip().upper()
        )

        return next(
            (
                position
                for position in self.list_positions()
                if position.ticker.upper()
                == normalized_ticker
            ),
            None,
        )

    def get_positions_market_value(self) -> float:
        """Return the market value of open positions."""

        return _round_money(
            sum(
                position.market_value
                for position in self.list_positions()
            )
        )

    def get_unrealized_pnl(self) -> float:
        """Return total unrealized PnL."""

        return _round_money(
            sum(
                position.unrealized_pnl
                for position in self.list_positions()
            )
        )

    def get_open_risk_amount(self) -> float:
        """Return total stop-loss risk."""

        return _round_money(
            sum(
                position.open_risk_amount
                for position in self.list_positions()
            )
        )

    def get_crypto_market_value(self) -> float:
        """Return total crypto position value."""

        return _round_money(
            sum(
                position.market_value
                for position in self.list_positions()
                if position.asset_type.lower()
                == "crypto"
            )
        )

    def get_daily_realized_pnl(self) -> float:
        """Return PnL from trades closed today in UTC."""

        today = datetime.now(
            UTC
        ).date()

        daily_pnl = 0.0

        for trade in self.list_trades():
            try:
                closed_date = datetime.fromisoformat(
                    trade.closed_at
                ).date()

            except ValueError:
                continue

            if closed_date == today:
                daily_pnl += trade.net_pnl

        return _round_money(
            daily_pnl
        )

    def create_portfolio_snapshot(
        self,
    ) -> PortfolioSnapshot:
        """Create current input for Risk Manager."""

        positions = self.list_positions()

        return self.account_store.create_portfolio_snapshot(
            positions_market_value=sum(
                position.market_value
                for position in positions
            ),
            open_position_count=len(
                positions
            ),
            open_tickers=tuple(
                position.ticker
                for position in positions
            ),
            current_open_risk_amount=sum(
                position.open_risk_amount
                for position in positions
            ),
            current_crypto_value=sum(
                position.market_value
                for position in positions
                if position.asset_type.lower()
                == "crypto"
            ),
            daily_realized_pnl=(
                self.get_daily_realized_pnl()
            ),
        )

    # =========================================================
    # OPEN POSITION
    # =========================================================

    def open_position(
        self,
        *,
        trade_plan: TradePlan,
        position_size: PositionSizeResult,
        risk_decision: RiskDecision,
    ) -> PaperPosition:
        """Open a simulated long position."""

        if not position_size.approved:
            reasons = "; ".join(
                position_size.rejection_reasons
            )

            raise PermissionError(
                "Position sizing rejected the trade: "
                f"{reasons}"
            )

        if not risk_decision.approved:
            reasons = "; ".join(
                risk_decision.rejection_reasons
            )

            raise PermissionError(
                "Risk Manager rejected the trade: "
                f"{reasons}"
            )

        if trade_plan.action.upper() != "BUY":
            raise ValueError(
                "Paper Broker V1 supports BUY entries only."
            )

        if position_size.quantity <= 0:
            raise ValueError(
                "Position quantity must be greater than zero."
            )

        if self.find_position(
            trade_plan.ticker
        ) is not None:
            raise ValueError(
                "An open position already exists for "
                f"{trade_plan.ticker.upper()}."
            )

        account = self.account_store.load()

        pending_order = PaperOrder.create_buy(
            ticker=trade_plan.ticker,
            asset_type=trade_plan.asset_type,
            quantity=position_size.quantity,
            requested_price=trade_plan.entry_price,
            stop_loss=trade_plan.stop_loss,
            take_profit=trade_plan.take_profit,
            estimated_value=position_size.position_value,
            strategy=trade_plan.strategy,
            reason=trade_plan.notes,
            order_type="MARKET",
        )

        fill_price = _apply_buy_slippage(
            requested_price=trade_plan.entry_price,
            slippage_bps=self.slippage_bps,
        )

        filled_value = _round_money(
            fill_price
            * position_size.quantity
        )

        fee = _calculate_fee(
            transaction_value=filled_value,
            commission_rate=self.commission_rate,
            minimum_fee=self.minimum_fee,
        )

        total_cost = _round_money(
            filled_value + fee
        )

        orders = self.list_orders()

        if total_cost > account.cash:
            rejected_order = replace(
                pending_order,
                status="REJECTED",
                reason=(
                    "Insufficient cash after "
                    "commission and slippage."
                ),
            )

            orders.append(
                rejected_order
            )

            self._save_orders(
                orders
            )

            raise ValueError(
                "Insufficient paper cash."
            )

        filled_order = replace(
            pending_order,
            status="FILLED",
            fill_price=fill_price,
            filled_value=filled_value,
            fee=fee,
            slippage=_round_price(
                fill_price
                - trade_plan.entry_price
            ),
            filled_at=utc_now(),
        )

        position = PaperPosition.from_filled_order(
            filled_order
        )

        positions = self.list_positions()

        orders.append(
            filled_order
        )

        positions.append(
            position
        )

        self.account_store.record_buy(
            position_value=filled_value,
            fee=fee,
        )

        self._save_orders(
            orders
        )

        self._save_positions(
            positions
        )

        self.record_equity_point()

        return position

    # =========================================================
    # CLOSE POSITION
    # =========================================================

    def close_position(
        self,
        *,
        ticker: str,
        requested_exit_price: float,
        close_reason: str = "MANUAL",
    ) -> PaperTrade:
        """Close a simulated long position."""

        if requested_exit_price <= 0:
            raise ValueError(
                "requested_exit_price must be greater than zero."
            )

        normalized_ticker = (
            ticker.strip().upper()
        )

        positions = self.list_positions()

        matching_position = next(
            (
                position
                for position in positions
                if position.ticker.upper()
                == normalized_ticker
            ),
            None,
        )

        if matching_position is None:
            raise LookupError(
                "No open position found for "
                f"{normalized_ticker}."
            )

        fill_price = _apply_sell_slippage(
            requested_price=requested_exit_price,
            slippage_bps=self.slippage_bps,
        )

        gross_exit_value = _round_money(
            fill_price
            * matching_position.quantity
        )

        exit_fee = _calculate_fee(
            transaction_value=gross_exit_value,
            commission_rate=self.commission_rate,
            minimum_fee=self.minimum_fee,
        )

        timestamp = utc_now()

        closing_order = PaperOrder(
            order_id=create_identifier(
                "order"
            ),
            ticker=matching_position.ticker,
            asset_type=matching_position.asset_type,
            side="SELL",
            order_type="MARKET",
            status="FILLED",
            quantity=matching_position.quantity,
            requested_price=_round_price(
                requested_exit_price
            ),
            fill_price=fill_price,
            stop_loss=matching_position.stop_loss,
            take_profit=matching_position.take_profit,
            estimated_value=_round_money(
                requested_exit_price
                * matching_position.quantity
            ),
            filled_value=gross_exit_value,
            fee=exit_fee,
            slippage=_round_price(
                requested_exit_price
                - fill_price
            ),
            strategy=matching_position.strategy,
            reason=close_reason,
            created_at=timestamp,
            filled_at=timestamp,
        )

        trade = PaperTrade.create_from_position(
            position=matching_position,
            exit_price=fill_price,
            exit_fee=exit_fee,
            close_reason=close_reason,
            closing_order_id=(
                closing_order.order_id
            ),
        )

        remaining_positions = [
            position
            for position in positions
            if position.position_id
            != matching_position.position_id
        ]

        orders = self.list_orders()
        trades = self.list_trades()

        orders.append(
            closing_order
        )

        trades.append(
            trade
        )

        self.account_store.record_sell(
            sale_value=gross_exit_value,
            realized_pnl=trade.net_pnl,
            fee=exit_fee,
        )

        self._save_orders(
            orders
        )

        self._save_positions(
            remaining_positions
        )

        self._save_trades(
            trades
        )

        self.record_equity_point()

        return trade

    # =========================================================
    # MARKET PRICE UPDATES
    # =========================================================

    def update_market_price(
        self,
        *,
        ticker: str,
        current_price: float,
    ) -> PaperPosition:
        """Update an open position using a supplied price."""

        if current_price <= 0:
            raise ValueError(
                "current_price must be greater than zero."
            )

        normalized_ticker = (
            ticker.strip().upper()
        )

        positions = self.list_positions()

        updated_position: PaperPosition | None = None
        updated_positions: list[PaperPosition] = []

        for position in positions:
            if position.ticker.upper() != normalized_ticker:
                updated_positions.append(
                    position
                )
                continue

            market_value = _round_money(
                current_price
                * position.quantity
            )

            unrealized_pnl = _round_money(
                market_value
                - position.position_value
                - position.fee_paid
            )

            if position.position_value <= 0:
                unrealized_pnl_percent = 0.0
            else:
                unrealized_pnl_percent = round(
                    unrealized_pnl
                    / position.position_value
                    * 100,
                    4,
                )

            updated_position = replace(
                position,
                current_price=_round_price(
                    current_price
                ),
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_percent=(
                    unrealized_pnl_percent
                ),
                updated_at=utc_now(),
            )

            updated_positions.append(
                updated_position
            )

        if updated_position is None:
            raise LookupError(
                "No open position found for "
                f"{normalized_ticker}."
            )

        self._save_positions(
            updated_positions
        )

        self.record_equity_point()

        return updated_position

    def process_exit_rules(
        self,
        prices: dict[str, float],
    ) -> list[PaperTrade]:
        """Close positions that reached stop-loss or take-profit."""

        normalized_prices = {
            ticker.upper(): float(price)
            for ticker, price in prices.items()
        }

        closed_trades: list[PaperTrade] = []

        for position in self.list_positions():
            current_price = normalized_prices.get(
                position.ticker.upper()
            )

            if current_price is None:
                continue

            self.update_market_price(
                ticker=position.ticker,
                current_price=current_price,
            )

            if current_price <= position.stop_loss:
                trade = self.close_position(
                    ticker=position.ticker,
                    requested_exit_price=current_price,
                    close_reason="STOP_LOSS",
                )

                closed_trades.append(
                    trade
                )

            elif current_price >= position.take_profit:
                trade = self.close_position(
                    ticker=position.ticker,
                    requested_exit_price=current_price,
                    close_reason="TAKE_PROFIT",
                )

                closed_trades.append(
                    trade
                )

        return closed_trades

    # =========================================================
    # EQUITY
    # =========================================================

    def record_equity_point(
        self,
    ) -> EquityPoint:
        """Record current paper-account equity."""

        account = self.account_store.load()
        positions = self.list_positions()

        positions_market_value = _round_money(
            sum(
                position.market_value
                for position in positions
            )
        )

        unrealized_pnl = _round_money(
            sum(
                position.unrealized_pnl
                for position in positions
            )
        )

        point = EquityPoint.create(
            cash=account.cash,
            positions_market_value=(
                positions_market_value
            ),
            realized_pnl=account.realized_pnl,
            unrealized_pnl=unrealized_pnl,
            open_position_count=len(
                positions
            ),
        )

        points = self.list_equity_points()

        points.append(
            point
        )

        self._save_equity_points(
            points
        )

        return point

    def reset(
        self,
        *,
        initial_cash: float = 10_000.0,
        currency: str = "EUR",
    ) -> PaperAccount:
        """Reset the complete local paper environment."""

        return self.account_store.reset(
            initial_cash=initial_cash,
            currency=currency,
        )


def print_open_positions(
    positions: list[PaperPosition],
    currency: str = "EUR",
) -> None:
    """Print currently open paper positions."""

    print()
    print("=" * 100)
    print("OPEN PAPER POSITIONS")
    print("=" * 100)

    if not positions:
        print("No open positions.")
        print("=" * 100)
        return

    for position in positions:
        print(
            f"{position.ticker:<12} "
            f"Qty: {position.quantity:<10} "
            f"Entry: {position.entry_price:>10.4f} "
            f"Current: {position.current_price:>10.4f} "
            f"Value: {position.market_value:>10.2f} "
            f"PnL: {position.unrealized_pnl:>+10.2f} "
            f"{currency}"
        )

    print("=" * 100)


def print_completed_trade(
    trade: PaperTrade,
    currency: str = "EUR",
) -> None:
    """Print one completed simulated trade."""

    print()
    print("=" * 76)
    print("COMPLETED PAPER TRADE")
    print("=" * 76)

    print(f"Ticker:           {trade.ticker}")
    print(f"Quantity:         {trade.quantity}")
    print(f"Entry price:      {trade.entry_price:.4f}")
    print(f"Exit price:       {trade.exit_price:.4f}")

    print(
        f"Gross PnL:        "
        f"{trade.gross_pnl:+,.2f} {currency}"
    )

    print(
        f"Total fees:       "
        f"{trade.total_fees:,.2f} {currency}"
    )

    print(
        f"Net PnL:          "
        f"{trade.net_pnl:+,.2f} {currency}"
    )

    print(
        f"Return:           "
        f"{trade.return_percent:+.4f}%"
    )

    print(f"Close reason:     {trade.close_reason}")

    print("=" * 76)