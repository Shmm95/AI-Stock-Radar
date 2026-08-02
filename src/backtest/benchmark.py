"""Allocation-matched buy-and-hold benchmark utilities.

The benchmark uses the same initial capital constraints as the strategy:

- Risk-based allocation:
  risk_per_trade_percent / stop_loss_percent

- Maximum allocation:
  maximum_position_percent

The lower of these two values determines benchmark exposure.

Example:
- Risk per trade: 1%
- Stop loss: 5%
- Maximum position: 25%

Risk-based exposure = 1 / 5 = 20%.
The benchmark therefore invests approximately 20% of the account
and keeps the remaining capital in cash.

Commission and adverse slippage are included.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import floor

import pandas as pd

from src.backtest.backtest_models import (
    BacktestConfig,
    BacktestResult,
)


@dataclass(frozen=True, slots=True)
class BuyAndHoldResult:
    """Result of one allocation-matched buy-and-hold simulation."""

    ticker: str

    start_date: str
    end_date: str

    initial_cash: float
    ending_equity: float

    target_allocation_percent: float
    actual_allocation_percent: float
    allocated_capital: float
    remaining_cash_after_entry: float

    quantity: float

    entry_price: float
    exit_price: float

    entry_fee: float
    exit_fee: float
    total_fees: float

    total_return_amount: float
    total_return_percent: float
    invested_capital_return_percent: float

    maximum_drawdown_amount: float
    maximum_drawdown_percent: float

    holding_period_bars: int

    def to_dict(self) -> dict[str, object]:
        """Serialize the benchmark result."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class StrategyBenchmarkComparison:
    """Comparison between active strategy and matched benchmark."""

    ticker: str

    strategy_return_amount: float
    strategy_return_percent: float
    strategy_max_drawdown_percent: float

    benchmark_return_amount: float
    benchmark_return_percent: float
    benchmark_max_drawdown_percent: float

    excess_return_amount: float
    excess_return_percent: float

    strategy_outperformed: bool

    def to_dict(self) -> dict[str, object]:
        """Serialize the comparison."""

        return asdict(self)


def _round_money(
    value: float,
) -> float:
    """Round a monetary value."""

    return round(
        float(value),
        2,
    )


def _round_price(
    value: float,
) -> float:
    """Round a market price."""

    return round(
        float(value),
        8,
    )


def _timestamp_to_string(
    value: object,
) -> str:
    """Convert index values to stable strings."""

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    return str(value)


def _validate_market_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and normalize benchmark data."""

    if data.empty:
        raise ValueError(
            "Benchmark market data cannot be empty."
        )

    normalized = data.copy()

    if isinstance(
        normalized.columns,
        pd.MultiIndex,
    ):
        normalized.columns = [
            (
                column[0]
                if isinstance(
                    column,
                    tuple,
                )
                else column
            )
            for column in normalized.columns
        ]

    required_columns = {
        "Open",
        "Close",
    }

    missing_columns = (
        required_columns
        - set(normalized.columns)
    )

    if missing_columns:
        raise ValueError(
            "Benchmark data is missing columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    if not normalized.index.is_monotonic_increasing:
        normalized = normalized.sort_index()

    normalized = normalized.loc[
        ~normalized.index.duplicated(
            keep="last"
        )
    ]

    for column in required_columns:
        normalized[column] = pd.to_numeric(
            normalized[column],
            errors="coerce",
        )

    normalized = normalized.dropna(
        subset=[
            "Open",
            "Close",
        ]
    )

    normalized = normalized[
        (
            normalized["Open"] > 0
        )
        & (
            normalized["Close"] > 0
        )
    ]

    if normalized.empty:
        raise ValueError(
            "No valid benchmark rows remain after cleaning."
        )

    return normalized


def _calculate_fee(
    *,
    transaction_value: float,
    config: BacktestConfig,
) -> float:
    """Calculate simulated commission."""

    proportional_fee = (
        abs(transaction_value)
        * config.commission_rate
    )

    return _round_money(
        max(
            proportional_fee,
            config.minimum_fee,
        )
    )


def _apply_buy_slippage(
    *,
    raw_price: float,
    config: BacktestConfig,
) -> float:
    """Apply adverse BUY slippage."""

    return _round_price(
        raw_price
        * (
            1
            + config.slippage_bps
            / 10_000
        )
    )


def _apply_sell_slippage(
    *,
    raw_price: float,
    config: BacktestConfig,
) -> float:
    """Apply adverse SELL slippage."""

    return _round_price(
        raw_price
        * (
            1
            - config.slippage_bps
            / 10_000
        )
    )


def calculate_effective_allocation_percent(
    config: BacktestConfig,
) -> float:
    """Calculate benchmark exposure matching strategy constraints.

    Strategy risk sizing:

        position value
        = equity × risk percent / stop percent

    The result is also capped by maximum_position_percent.
    """

    config.validate()

    risk_based_percent = (
        config.risk_per_trade_percent
        / config.stop_loss_percent
        * 100
    )

    effective_percent = min(
        risk_based_percent,
        config.maximum_position_percent,
        100.0,
    )

    return round(
        max(
            effective_percent,
            0.0,
        ),
        6,
    )


def _normalize_quantity(
    quantity: float,
    *,
    allow_fractional: bool,
) -> float:
    """Normalize benchmark quantity."""

    if allow_fractional:
        return round(
            max(
                quantity,
                0.0,
            ),
            6,
        )

    return float(
        max(
            floor(quantity),
            0,
        )
    )


def _calculate_quantity(
    *,
    initial_cash: float,
    target_capital: float,
    entry_price: float,
    config: BacktestConfig,
) -> tuple[float, float]:
    """Calculate allocation-constrained affordable quantity."""

    quantity = _normalize_quantity(
        target_capital
        / entry_price,
        allow_fractional=(
            config.allow_fractional
        ),
    )

    for _ in range(12):
        if quantity <= 0:
            return (
                0.0,
                0.0,
            )

        entry_value = (
            quantity
            * entry_price
        )

        entry_fee = _calculate_fee(
            transaction_value=entry_value,
            config=config,
        )

        total_cost = (
            entry_value
            + entry_fee
        )

        if total_cost <= initial_cash + 1e-9:
            return (
                quantity,
                entry_fee,
            )

        affordable_quantity = max(
            (
                initial_cash
                - entry_fee
            )
            / entry_price,
            0.0,
        )

        new_quantity = _normalize_quantity(
            min(
                quantity,
                affordable_quantity,
            ),
            allow_fractional=(
                config.allow_fractional
            ),
        )

        if new_quantity >= quantity:
            decrement = (
                0.000001
                if config.allow_fractional
                else 1.0
            )

            new_quantity = _normalize_quantity(
                quantity
                - decrement,
                allow_fractional=(
                    config.allow_fractional
                ),
            )

        quantity = new_quantity

    return (
        0.0,
        0.0,
    )


def _calculate_drawdown(
    *,
    initial_cash: float,
    cash_after_entry: float,
    quantity: float,
    close_values: pd.Series,
    final_equity: float,
) -> tuple[float, float]:
    """Calculate total-account benchmark drawdown."""

    peak_equity = float(
        initial_cash
    )

    maximum_drawdown_amount = 0.0
    maximum_drawdown_percent = 0.0

    for close_price in close_values:
        total_equity = (
            cash_after_entry
            + quantity
            * float(close_price)
        )

        peak_equity = max(
            peak_equity,
            total_equity,
        )

        drawdown_amount = max(
            peak_equity
            - total_equity,
            0.0,
        )

        drawdown_percent = (
            drawdown_amount
            / peak_equity
            * 100
            if peak_equity > 0
            else 0.0
        )

        maximum_drawdown_amount = max(
            maximum_drawdown_amount,
            drawdown_amount,
        )

        maximum_drawdown_percent = max(
            maximum_drawdown_percent,
            drawdown_percent,
        )

    peak_equity = max(
        peak_equity,
        final_equity,
    )

    final_drawdown_amount = max(
        peak_equity
        - final_equity,
        0.0,
    )

    final_drawdown_percent = (
        final_drawdown_amount
        / peak_equity
        * 100
        if peak_equity > 0
        else 0.0
    )

    maximum_drawdown_amount = max(
        maximum_drawdown_amount,
        final_drawdown_amount,
    )

    maximum_drawdown_percent = max(
        maximum_drawdown_percent,
        final_drawdown_percent,
    )

    return (
        _round_money(
            maximum_drawdown_amount
        ),
        round(
            maximum_drawdown_percent,
            4,
        ),
    )


def run_buy_and_hold_benchmark(
    *,
    ticker: str,
    data: pd.DataFrame,
    config: BacktestConfig,
) -> BuyAndHoldResult:
    """Run an allocation-matched passive benchmark."""

    normalized_ticker = (
        ticker.strip()
        .upper()
    )

    if not normalized_ticker:
        raise ValueError(
            "ticker cannot be empty."
        )

    config.validate()

    market_data = _validate_market_data(
        data
    )

    target_allocation_percent = (
        calculate_effective_allocation_percent(
            config
        )
    )

    target_capital = (
        config.initial_cash
        * target_allocation_percent
        / 100
    )

    first_open = float(
        market_data.iloc[0][
            "Open"
        ]
    )

    final_close = float(
        market_data.iloc[-1][
            "Close"
        ]
    )

    entry_price = _apply_buy_slippage(
        raw_price=first_open,
        config=config,
    )

    exit_price = _apply_sell_slippage(
        raw_price=final_close,
        config=config,
    )

    quantity, entry_fee = (
        _calculate_quantity(
            initial_cash=(
                config.initial_cash
            ),
            target_capital=(
                target_capital
            ),
            entry_price=entry_price,
            config=config,
        )
    )

    if quantity <= 0:
        raise ValueError(
            "Initial capital is insufficient "
            "for the matched benchmark position."
        )

    entry_value = (
        quantity
        * entry_price
    )

    cash_after_entry = (
        config.initial_cash
        - entry_value
        - entry_fee
    )

    exit_value = (
        quantity
        * exit_price
    )

    exit_fee = _calculate_fee(
        transaction_value=exit_value,
        config=config,
    )

    ending_equity = (
        cash_after_entry
        + exit_value
        - exit_fee
    )

    total_return_amount = (
        ending_equity
        - config.initial_cash
    )

    total_return_percent = (
        total_return_amount
        / config.initial_cash
        * 100
    )

    total_fees = (
        entry_fee
        + exit_fee
    )

    benchmark_trade_pnl = (
        (
            exit_price
            - entry_price
        )
        * quantity
        - total_fees
    )

    invested_capital_return_percent = (
        benchmark_trade_pnl
        / entry_value
        * 100
        if entry_value > 0
        else 0.0
    )

    actual_allocation_percent = (
        entry_value
        / config.initial_cash
        * 100
    )

    (
        maximum_drawdown_amount,
        maximum_drawdown_percent,
    ) = _calculate_drawdown(
        initial_cash=(
            config.initial_cash
        ),
        cash_after_entry=(
            cash_after_entry
        ),
        quantity=quantity,
        close_values=(
            market_data["Close"]
        ),
        final_equity=ending_equity,
    )

    return BuyAndHoldResult(
        ticker=normalized_ticker,
        start_date=_timestamp_to_string(
            market_data.index[0]
        ),
        end_date=_timestamp_to_string(
            market_data.index[-1]
        ),
        initial_cash=_round_money(
            config.initial_cash
        ),
        ending_equity=_round_money(
            ending_equity
        ),
        target_allocation_percent=round(
            target_allocation_percent,
            4,
        ),
        actual_allocation_percent=round(
            actual_allocation_percent,
            4,
        ),
        allocated_capital=_round_money(
            entry_value
        ),
        remaining_cash_after_entry=_round_money(
            cash_after_entry
        ),
        quantity=quantity,
        entry_price=entry_price,
        exit_price=exit_price,
        entry_fee=entry_fee,
        exit_fee=exit_fee,
        total_fees=_round_money(
            total_fees
        ),
        total_return_amount=_round_money(
            total_return_amount
        ),
        total_return_percent=round(
            total_return_percent,
            4,
        ),
        invested_capital_return_percent=round(
            invested_capital_return_percent,
            4,
        ),
        maximum_drawdown_amount=(
            maximum_drawdown_amount
        ),
        maximum_drawdown_percent=(
            maximum_drawdown_percent
        ),
        holding_period_bars=max(
            len(market_data) - 1,
            0,
        ),
    )


def compare_strategy_to_benchmark(
    *,
    strategy_result: BacktestResult,
    benchmark_result: BuyAndHoldResult,
) -> StrategyBenchmarkComparison:
    """Compare active strategy with matched buy-and-hold."""

    if (
        strategy_result.ticker.upper()
        != benchmark_result.ticker.upper()
    ):
        raise ValueError(
            "Strategy and benchmark tickers do not match."
        )

    excess_return_amount = (
        strategy_result.total_return_amount
        - benchmark_result.total_return_amount
    )

    excess_return_percent = (
        strategy_result.total_return_percent
        - benchmark_result.total_return_percent
    )

    return StrategyBenchmarkComparison(
        ticker=strategy_result.ticker,
        strategy_return_amount=_round_money(
            strategy_result.total_return_amount
        ),
        strategy_return_percent=round(
            strategy_result.total_return_percent,
            4,
        ),
        strategy_max_drawdown_percent=round(
            strategy_result.maximum_drawdown_percent,
            4,
        ),
        benchmark_return_amount=_round_money(
            benchmark_result.total_return_amount
        ),
        benchmark_return_percent=round(
            benchmark_result.total_return_percent,
            4,
        ),
        benchmark_max_drawdown_percent=round(
            benchmark_result.maximum_drawdown_percent,
            4,
        ),
        excess_return_amount=_round_money(
            excess_return_amount
        ),
        excess_return_percent=round(
            excess_return_percent,
            4,
        ),
        strategy_outperformed=(
            excess_return_amount > 0
        ),
    )


def print_buy_and_hold_result(
    result: BuyAndHoldResult,
) -> None:
    """Print allocation-matched benchmark performance."""

    print()
    print("=" * 92)
    print("ALLOCATION-MATCHED BUY AND HOLD")
    print("=" * 92)

    print(
        f"Ticker:                    "
        f"{result.ticker}"
    )

    print(
        f"Period start:              "
        f"{result.start_date}"
    )

    print(
        f"Period end:                "
        f"{result.end_date}"
    )

    print()

    print(
        f"Initial capital:           "
        f"{result.initial_cash:,.2f}"
    )

    print(
        f"Target allocation:         "
        f"{result.target_allocation_percent:.4f}%"
    )

    print(
        f"Actual allocation:         "
        f"{result.actual_allocation_percent:.4f}%"
    )

    print(
        f"Allocated capital:         "
        f"{result.allocated_capital:,.2f}"
    )

    print(
        f"Remaining cash:            "
        f"{result.remaining_cash_after_entry:,.2f}"
    )

    print(
        f"Quantity:                  "
        f"{result.quantity}"
    )

    print(
        f"Entry price:               "
        f"{result.entry_price:.6f}"
    )

    print(
        f"Exit price:                "
        f"{result.exit_price:.6f}"
    )

    print(
        f"Total fees:                "
        f"{result.total_fees:,.2f}"
    )

    print()

    print(
        f"Ending equity:             "
        f"{result.ending_equity:,.2f}"
    )

    print(
        f"Account return:            "
        f"{result.total_return_amount:+,.2f}"
    )

    print(
        f"Account return %:          "
        f"{result.total_return_percent:+.4f}%"
    )

    print(
        f"Invested capital return %: "
        f"{result.invested_capital_return_percent:+.4f}%"
    )

    print(
        f"Maximum drawdown:          "
        f"{result.maximum_drawdown_amount:,.2f}"
    )

    print(
        f"Maximum drawdown %:        "
        f"{result.maximum_drawdown_percent:.4f}%"
    )

    print("=" * 92)


def print_strategy_benchmark_comparison(
    comparison: StrategyBenchmarkComparison,
) -> None:
    """Print active strategy versus matched benchmark."""

    print()
    print("=" * 92)
    print("STRATEGY VS MATCHED BUY AND HOLD")
    print("=" * 92)

    print(
        f"Ticker:                    "
        f"{comparison.ticker}"
    )

    print()

    print(
        f"Strategy return:           "
        f"{comparison.strategy_return_amount:+,.2f} "
        f"({comparison.strategy_return_percent:+.4f}%)"
    )

    print(
        f"Matched benchmark return:  "
        f"{comparison.benchmark_return_amount:+,.2f} "
        f"({comparison.benchmark_return_percent:+.4f}%)"
    )

    print(
        f"Excess return:             "
        f"{comparison.excess_return_amount:+,.2f} "
        f"({comparison.excess_return_percent:+.4f}%)"
    )

    print()

    print(
        f"Strategy max drawdown:     "
        f"{comparison.strategy_max_drawdown_percent:.4f}%"
    )

    print(
        f"Benchmark max drawdown:    "
        f"{comparison.benchmark_max_drawdown_percent:.4f}%"
    )

    print()

    print(
        "Outperformed benchmark:   "
        + (
            "YES"
            if comparison.strategy_outperformed
            else "NO"
        )
    )

    print("=" * 92)