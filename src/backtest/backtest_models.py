"""Data models for the deterministic AI-Stock-Radar backtest engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


def utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""

    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Configuration for one single-asset historical backtest."""

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
        """Validate configuration values."""

        if self.initial_cash <= 0:
            raise ValueError(
                "initial_cash must be greater than zero."
            )

        if not 0 < self.risk_per_trade_percent <= 100:
            raise ValueError(
                "risk_per_trade_percent must be between 0 and 100."
            )

        if not 0 < self.maximum_position_percent <= 100:
            raise ValueError(
                "maximum_position_percent must be between 0 and 100."
            )

        if self.stop_loss_percent <= 0:
            raise ValueError(
                "stop_loss_percent must be greater than zero."
            )

        if self.take_profit_percent <= 0:
            raise ValueError(
                "take_profit_percent must be greater than zero."
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
                "maximum_open_positions must be at least one."
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class BacktestSignal:
    """A deterministic signal created after a historical bar closes."""

    timestamp: str
    ticker: str
    action: str
    reference_price: float

    overall_score: float = 0.0
    technical_score: float = 0.0
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the signal."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class BacktestPosition:
    """Public representation of an open backtest position."""

    ticker: str
    entry_timestamp: str
    entry_bar_index: int

    quantity: float
    entry_price: float
    entry_fee: float

    stop_loss_price: float
    take_profit_price: float
    signal_reason: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the open position."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class BacktestTrade:
    """One completed historical trade."""

    ticker: str

    entry_timestamp: str
    exit_timestamp: str

    entry_bar_index: int
    exit_bar_index: int

    quantity: float

    entry_price: float
    exit_price: float

    entry_fee: float
    exit_fee: float
    total_fees: float

    gross_pnl: float
    net_pnl: float
    return_percent: float

    holding_period_bars: int
    exit_reason: str
    signal_reason: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the trade."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class BacktestEquityPoint:
    """One marked-to-market account-equity observation."""

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

    @property
    def position_value(self) -> float:
        """Backward-compatible alias."""

        return self.positions_market_value

    def to_dict(self) -> dict[str, Any]:
        """Serialize the equity observation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Complete result of one single-asset historical backtest."""

    ticker: str
    config: BacktestConfig

    start_date: str
    end_date: str
    started_at: str
    finished_at: str

    initial_cash: float
    ending_cash: float
    ending_equity: float

    total_return_amount: float
    total_return_percent: float

    maximum_drawdown_amount: float
    maximum_drawdown_percent: float

    completed_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_percent: float

    gross_profit: float
    gross_loss: float
    profit_factor: float
    total_fees: float

    open_position: BacktestPosition | None

    rejected_signals: int
    skipped_signals: int

    trades: list[BacktestTrade] = field(
        default_factory=list
    )

    equity_curve: list[BacktestEquityPoint] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete result."""

        return {
            "ticker": self.ticker,
            "config": self.config.to_dict(),
            "start_date": self.start_date,
            "end_date": self.end_date,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "initial_cash": self.initial_cash,
            "ending_cash": self.ending_cash,
            "ending_equity": self.ending_equity,
            "total_return_amount": self.total_return_amount,
            "total_return_percent": self.total_return_percent,
            "maximum_drawdown_amount": (
                self.maximum_drawdown_amount
            ),
            "maximum_drawdown_percent": (
                self.maximum_drawdown_percent
            ),
            "completed_trades": self.completed_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate_percent": self.win_rate_percent,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "profit_factor": self.profit_factor,
            "total_fees": self.total_fees,
            "open_position": (
                self.open_position.to_dict()
                if self.open_position is not None
                else None
            ),
            "rejected_signals": self.rejected_signals,
            "skipped_signals": self.skipped_signals,
            "trades": [
                trade.to_dict()
                for trade in self.trades
            ],
            "equity_curve": [
                point.to_dict()
                for point in self.equity_curve
            ],
        }