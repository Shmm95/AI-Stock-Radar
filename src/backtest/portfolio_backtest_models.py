"""Models for deterministic multi-asset portfolio backtesting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


def utc_now() -> str:
    """Return current UTC time in ISO-8601 format."""

    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class PortfolioBacktestConfig:
    """Configuration for one shared-cash multi-asset backtest."""

    initial_cash: float = 10_000.0

    risk_per_trade_percent: float = 1.0
    maximum_position_percent: float = 25.0
    maximum_total_open_risk_percent: float = 4.0
    maximum_crypto_allocation_percent: float = 25.0
    maximum_open_positions: int = 4

    stock_stop_loss_percent: float = 5.0
    crypto_stop_loss_percent: float = 5.0
    stock_trailing_close_percent: float = 7.5
    crypto_trailing_close_percent: float = 7.5

    commission_rate: float = 0.0005
    minimum_fee: float = 1.0
    slippage_bps: float = 5.0

    allow_fractional_stocks: bool = False
    allow_fractional_crypto: bool = True
    force_close_at_end: bool = True

    def validate(self) -> None:
        """Validate all configuration values."""

        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be greater than zero.")

        percentage_fields = {
            "risk_per_trade_percent": self.risk_per_trade_percent,
            "maximum_position_percent": self.maximum_position_percent,
            "maximum_total_open_risk_percent": (
                self.maximum_total_open_risk_percent
            ),
            "maximum_crypto_allocation_percent": (
                self.maximum_crypto_allocation_percent
            ),
            "stock_stop_loss_percent": self.stock_stop_loss_percent,
            "crypto_stop_loss_percent": self.crypto_stop_loss_percent,
            "stock_trailing_close_percent": (
                self.stock_trailing_close_percent
            ),
            "crypto_trailing_close_percent": (
                self.crypto_trailing_close_percent
            ),
        }

        for field_name, value in percentage_fields.items():
            if value <= 0 or value > 100:
                raise ValueError(
                    f"{field_name} must be greater than 0 and at most 100."
                )

        if self.maximum_open_positions < 1:
            raise ValueError(
                "maximum_open_positions must be at least one."
            )

        if self.commission_rate < 0:
            raise ValueError("commission_rate cannot be negative.")

        if self.minimum_fee < 0:
            raise ValueError("minimum_fee cannot be negative.")

        if self.slippage_bps < 0:
            raise ValueError("slippage_bps cannot be negative.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class PortfolioSignal:
    """One causal signal created after a ticker bar closes."""

    timestamp: str
    ticker: str
    action: str
    reference_price: float

    score: float = 0.0
    technical_score: float = 0.0
    confidence: float = 0.0
    reason: str = ""
    priority_rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PortfolioPosition:
    """One open position inside the shared portfolio."""

    ticker: str
    asset_class: str

    entry_timestamp: str
    entry_portfolio_bar_index: int

    quantity: float
    entry_price: float
    entry_fee: float

    stop_loss_price: float
    highest_close: float
    trailing_close_percent: float
    initial_risk_amount: float

    signal_score: float
    signal_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PortfolioTrade:
    """One completed portfolio trade."""

    ticker: str
    asset_class: str

    entry_timestamp: str
    exit_timestamp: str

    entry_portfolio_bar_index: int
    exit_portfolio_bar_index: int

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
    signal_score: float
    signal_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PortfolioRejection:
    """A rejected or skipped portfolio signal."""

    timestamp: str
    ticker: str
    action: str
    reason_code: str
    reason: str

    score: float = 0.0
    requested_value: float = 0.0
    available_cash: float = 0.0
    open_position_count: int = 0
    open_risk_percent: float = 0.0
    crypto_allocation_percent: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PortfolioEquityPoint:
    """One marked-to-market shared-account observation."""

    timestamp: str
    cash: float
    positions_market_value: float
    total_equity: float

    realized_pnl: float
    unrealized_pnl: float
    open_position_count: int

    open_risk_amount: float
    open_risk_percent: float
    crypto_market_value: float
    crypto_allocation_percent: float

    peak_equity: float
    drawdown_amount: float
    drawdown_percent: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PortfolioBenchmarkResult:
    """Equal-weight buy-and-hold benchmark across the tested universe."""

    initial_cash: float
    ending_equity: float
    total_return_amount: float
    total_return_percent: float
    maximum_drawdown_amount: float
    maximum_drawdown_percent: float
    total_fees: float
    invested_tickers: int
    uninvested_cash: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PortfolioBacktestResult:
    """Complete result of a shared-cash multi-asset backtest."""

    config: PortfolioBacktestConfig
    tickers: tuple[str, ...]

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

    average_exposure_percent: float
    maximum_open_positions_observed: int

    rejected_signals: int
    skipped_signals: int

    benchmark: PortfolioBenchmarkResult | None = None

    open_positions: tuple[PortfolioPosition, ...] = ()
    trades: list[PortfolioTrade] = field(default_factory=list)
    rejections: list[PortfolioRejection] = field(default_factory=list)
    equity_curve: list[PortfolioEquityPoint] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete result."""

        return {
            "config": self.config.to_dict(),
            "tickers": list(self.tickers),
            "start_date": self.start_date,
            "end_date": self.end_date,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "initial_cash": self.initial_cash,
            "ending_cash": self.ending_cash,
            "ending_equity": self.ending_equity,
            "total_return_amount": self.total_return_amount,
            "total_return_percent": self.total_return_percent,
            "maximum_drawdown_amount": self.maximum_drawdown_amount,
            "maximum_drawdown_percent": self.maximum_drawdown_percent,
            "completed_trades": self.completed_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate_percent": self.win_rate_percent,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "profit_factor": self.profit_factor,
            "total_fees": self.total_fees,
            "average_exposure_percent": self.average_exposure_percent,
            "maximum_open_positions_observed": (
                self.maximum_open_positions_observed
            ),
            "rejected_signals": self.rejected_signals,
            "skipped_signals": self.skipped_signals,
            "benchmark": (
                self.benchmark.to_dict()
                if self.benchmark is not None
                else None
            ),
            "open_positions": [
                position.to_dict() for position in self.open_positions
            ],
            "trades": [trade.to_dict() for trade in self.trades],
            "rejections": [item.to_dict() for item in self.rejections],
            "equity_curve": [point.to_dict() for point in self.equity_curve],
        }
