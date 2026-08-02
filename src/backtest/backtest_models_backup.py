"""Core data models for the deterministic backtest engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


def _round_money(value: float) -> float:
    """Round a monetary value to two decimal places."""

    return round(float(value), 2)


def _round_price(value: float) -> float:
    """Round a market price while preserving useful precision."""

    return round(float(value), 6)


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Configuration for one backtest run."""

    initial_cash: float = 10_000.0

    risk_per_trade_percent: float = 1.0
    maximum_position_percent: float = 25.0

    stop_loss_percent: float = 5.0
    take_profit_percent: float = 10.0

    commission_rate: float = 0.0005
    minimum_fee: float = 1.0
    slippage_bps: float = 5.0

    allow_fractional: bool = False
    maximum_open_positions: int = 1

    force_close_at_end: bool = True

    def validate(self) -> None:
        """Validate all configuration values."""

        if self.initial_cash <= 0:
            raise ValueError(
                "initial_cash must be greater than zero."
            )

        percentage_fields = {
            "risk_per_trade_percent": (
                self.risk_per_trade_percent
            ),
            "maximum_position_percent": (
                self.maximum_position_percent
            ),
            "stop_loss_percent": (
                self.stop_loss_percent
            ),
            "take_profit_percent": (
                self.take_profit_percent
            ),
        }

        for field_name, value in percentage_fields.items():
            if value <= 0 or value >= 100:
                raise ValueError(
                    f"{field_name} must be greater than 0 "
                    "and lower than 100."
                )

        if self.commission_rate < 0:
            raise ValueError(
                "commission_rate cannot be negative."
            )

        if self.minimum_fee < 0:
            raise ValueError(
                "minimum_fee cannot be negative."
            )

        if self.slippage_bps < 0:
            raise ValueError(
                "slippage_bps cannot be negative."
            )

        if self.maximum_open_positions < 1:
            raise ValueError(
                "maximum_open_positions must be at least 1."
            )


@dataclass(frozen=True, slots=True)
class BacktestSignal:
    """One strategy signal generated without future information."""

    timestamp: str
    ticker: str

    action: str

    reference_price: float

    overall_score: float = 0.0
    technical_score: float = 0.0
    confidence: float = 0.0

    reason: str = ""

    def validate(self) -> None:
        """Validate a strategy signal."""

        if not self.ticker.strip():
            raise ValueError(
                "Signal ticker cannot be empty."
            )

        if self.action.upper() not in {
            "BUY",
            "EXIT",
            "HOLD",
        }:
            raise ValueError(
                "Signal action must be BUY, EXIT, or HOLD."
            )

        if self.reference_price <= 0:
            raise ValueError(
                "reference_price must be greater than zero."
            )


@dataclass(frozen=True, slots=True)
class BacktestPosition:
    """One currently open backtest position."""

    position_id: str
    ticker: str

    quantity: float

    entry_timestamp: str
    entry_price: float

    stop_loss: float
    take_profit: float

    entry_value: float
    entry_fee: float

    initial_risk_amount: float

    signal_score: float
    signal_confidence: float

    def market_value(
        self,
        current_price: float,
    ) -> float:
        """Return current position market value."""

        return _round_money(
            self.quantity
            * current_price
        )

    def unrealized_pnl(
        self,
        current_price: float,
    ) -> float:
        """Return unrealized PnL before exit commission."""

        return _round_money(
            self.market_value(current_price)
            - self.entry_value
            - self.entry_fee
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the model to a dictionary."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class BacktestTrade:
    """One completed historical trade."""

    trade_id: str
    position_id: str

    ticker: str
    quantity: float

    entry_timestamp: str
    exit_timestamp: str

    entry_price: float
    exit_price: float

    stop_loss: float
    take_profit: float

    entry_value: float
    exit_value: float

    entry_fee: float
    exit_fee: float
    total_fees: float

    gross_pnl: float
    net_pnl: float
    return_percent: float

    holding_period_bars: int
    exit_reason: str

    signal_score: float
    signal_confidence: float

    @classmethod
    def create(
        cls,
        *,
        trade_id: str,
        position: BacktestPosition,
        exit_timestamp: str,
        exit_price: float,
        exit_fee: float,
        holding_period_bars: int,
        exit_reason: str,
    ) -> BacktestTrade:
        """Create a completed trade from an open position."""

        exit_value = (
            position.quantity
            * exit_price
        )

        gross_pnl = (
            exit_value
            - position.entry_value
        )

        total_fees = (
            position.entry_fee
            + exit_fee
        )

        net_pnl = (
            gross_pnl
            - total_fees
        )

        if position.entry_value <= 0:
            return_percent = 0.0
        else:
            return_percent = (
                net_pnl
                / position.entry_value
                * 100
            )

        return cls(
            trade_id=trade_id,
            position_id=position.position_id,
            ticker=position.ticker,
            quantity=position.quantity,
            entry_timestamp=position.entry_timestamp,
            exit_timestamp=exit_timestamp,
            entry_price=_round_price(
                position.entry_price
            ),
            exit_price=_round_price(
                exit_price
            ),
            stop_loss=_round_price(
                position.stop_loss
            ),
            take_profit=_round_price(
                position.take_profit
            ),
            entry_value=_round_money(
                position.entry_value
            ),
            exit_value=_round_money(
                exit_value
            ),
            entry_fee=_round_money(
                position.entry_fee
            ),
            exit_fee=_round_money(
                exit_fee
            ),
            total_fees=_round_money(
                total_fees
            ),
            gross_pnl=_round_money(
                gross_pnl
            ),
            net_pnl=_round_money(
                net_pnl
            ),
            return_percent=round(
                return_percent,
                4,
            ),
            holding_period_bars=int(
                holding_period_bars
            ),
            exit_reason=exit_reason,
            signal_score=round(
                position.signal_score,
                4,
            ),
            signal_confidence=round(
                position.signal_confidence,
                4,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the model to a dictionary."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class BacktestEquityPoint:
    """One point on the historical equity curve."""

    timestamp: str

    cash: float
    positions_market_value: float
    total_equity: float

    realized_pnl: float
    unrealized_pnl: float

    open_position_count: int

    peak_equity: float
    drawdown_amount: float
    drawdown_percent: float

    @classmethod
    def create(
        cls,
        *,
        timestamp: str,
        cash: float,
        positions_market_value: float,
        realized_pnl: float,
        unrealized_pnl: float,
        open_position_count: int,
        previous_peak_equity: float,
    ) -> BacktestEquityPoint:
        """Create an equity observation with drawdown data."""

        total_equity = (
            cash
            + positions_market_value
        )

        peak_equity = max(
            previous_peak_equity,
            total_equity,
        )

        drawdown_amount = max(
            peak_equity
            - total_equity,
            0.0,
        )

        if peak_equity <= 0:
            drawdown_percent = 0.0
        else:
            drawdown_percent = (
                drawdown_amount
                / peak_equity
                * 100
            )

        return cls(
            timestamp=timestamp,
            cash=_round_money(cash),
            positions_market_value=_round_money(
                positions_market_value
            ),
            total_equity=_round_money(
                total_equity
            ),
            realized_pnl=_round_money(
                realized_pnl
            ),
            unrealized_pnl=_round_money(
                unrealized_pnl
            ),
            open_position_count=int(
                open_position_count
            ),
            peak_equity=_round_money(
                peak_equity
            ),
            drawdown_amount=_round_money(
                drawdown_amount
            ),
            drawdown_percent=round(
                drawdown_percent,
                4,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the model to a dictionary."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Complete output of one historical backtest."""

    ticker: str

    started_at: str
    completed_at: str

    start_date: str
    end_date: str

    initial_cash: float
    ending_cash: float
    ending_equity: float

    trades: tuple[BacktestTrade, ...]
    equity_curve: tuple[BacktestEquityPoint, ...]

    rejected_signals: int
    skipped_signals: int

    config: BacktestConfig

    @property
    def total_trades(self) -> int:
        """Return number of completed trades."""

        return len(self.trades)

    @property
    def total_return_amount(self) -> float:
        """Return absolute strategy return."""

        return _round_money(
            self.ending_equity
            - self.initial_cash
        )

    @property
    def total_return_percent(self) -> float:
        """Return strategy return in percent."""

        if self.initial_cash <= 0:
            return 0.0

        return round(
            self.total_return_amount
            / self.initial_cash
            * 100,
            4,
        )

    @property
    def maximum_drawdown_percent(self) -> float:
        """Return maximum equity-curve drawdown."""

        if not self.equity_curve:
            return 0.0

        return max(
            point.drawdown_percent
            for point in self.equity_curve
        )

    @property
    def maximum_drawdown_amount(self) -> float:
        """Return maximum absolute drawdown."""

        if not self.equity_curve:
            return 0.0

        return max(
            point.drawdown_amount
            for point in self.equity_curve
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert complete backtest output to a dictionary."""

        return {
            "ticker": self.ticker,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_cash": self.initial_cash,
            "ending_cash": self.ending_cash,
            "ending_equity": self.ending_equity,
            "total_trades": self.total_trades,
            "total_return_amount": (
                self.total_return_amount
            ),
            "total_return_percent": (
                self.total_return_percent
            ),
            "maximum_drawdown_amount": (
                self.maximum_drawdown_amount
            ),
            "maximum_drawdown_percent": (
                self.maximum_drawdown_percent
            ),
            "rejected_signals": self.rejected_signals,
            "skipped_signals": self.skipped_signals,
            "config": asdict(self.config),
            "trades": [
                trade.to_dict()
                for trade in self.trades
            ],
            "equity_curve": [
                point.to_dict()
                for point in self.equity_curve
            ],
        }


def utc_now() -> str:
    """Return the current UTC timestamp."""

    return datetime.now().astimezone().isoformat()