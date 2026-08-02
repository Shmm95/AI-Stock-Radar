"""Buy-and-hold benchmark utilities for deterministic backtests.

The benchmark:
- Buys at the first available bar's Open.
- Applies adverse slippage and commission.
- Holds through the complete test period.
- Sells at the final bar's Close.
- Calculates total return and maximum drawdown.

The benchmark intentionally invests almost all available capital.
It provides a passive baseline for evaluating the active strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor

import pandas as pd

from src.backtest.backtest_models import (
    BacktestConfig,
    BacktestResult,
)


@dataclass(frozen=True, slots=True)
class BuyAndHoldResult:
    """Result of one passive buy-and-hold simulation."""

    ticker: str

    start_date: str
    end_date: str

    initial_cash: float
    ending_equity: float

    quantity: float

    entry_price: float
    exit_price: float

    entry_fee: float
    exit_fee: float
    total_fees: float

    total_return_amount: float
    total_return_percent: float

    maximum_drawdown_amount: float
    maximum_drawdown_percent: float

    holding_period_bars: int


@dataclass(frozen=True, slots=True)
class StrategyBenchmarkComparison:
    """Comparison between the strategy and passive benchmark."""

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


def _round_money(
    value: float,
) -> float:
    """Round a monetary value to two decimal places."""

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
        6,
    )


def _timestamp_to_string(
    value: object,
) -> str:
    """Convert an index value to a stable timestamp string."""

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    return str(value)


def _validate_market_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and normalize benchmark market data."""

    if data.empty:
        raise ValueError(
            "Benchmark market data cannot be empty."
        )

    required_columns = {
        "Open",
        "Close",
    }

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Benchmark data is missing columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    normalized = data.copy()

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
        transaction_value
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
    price: float,
    config: BacktestConfig,
) -> float:
    """Apply adverse BUY slippage."""

    return _round_price(
        price
        * (
            1
            + config.slippage_bps
            / 10_000
        )
    )


def _apply_sell_slippage(
    *,
    price: float,
    config: BacktestConfig,
) -> float:
    """Apply adverse SELL slippage."""

    return _round_price(
        price
        * (
            1
            - config.slippage_bps
            / 10_000
        )
    )


def _calculate_quantity(
    *,
    initial_cash: float,
    entry_price: float,
    config: BacktestConfig,
) -> tuple[float, float]:
    """Calculate maximum affordable benchmark quantity and entry fee."""

    available_before_fee = max(
        initial_cash
        - config.minimum_fee,
        0.0,
    )

    if config.allow_fractional:
        quantity = round(
            available_before_fee
            / entry_price,
            6,
        )
    else:
        quantity = float(
            floor(
                available_before_fee
                / entry_price
            )
        )

    while quantity > 0:
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

        if total_cost <= initial_cash:
            return (
                quantity,
                entry_fee,
            )

        if config.allow_fractional:
            difference = (
                total_cost
                - initial_cash
            )

            quantity = round(
                max(
                    quantity
                    - difference
                    / entry_price
                    - 0.000001,
                    0.0,
                ),
                6,
            )

        else:
            quantity -= 1.0

    return (
        0.0,
        0.0,
    )


def _calculate_drawdown(
    *,
    cash_after_entry: float,
    quantity: float,
    close_values: pd.Series,
    initial_cash: float,
) -> tuple[float, float]:
    """Calculate close-to-close benchmark drawdown."""

    peak_equity = initial_cash

    maximum_drawdown_amount = 0.0
    maximum_drawdown_percent = 0.0

    for close_price in close_values:
        equity = (
            cash_after_entry
            + quantity
            * float(close_price)
        )

        if equity > peak_equity:
            peak_equity = equity

        drawdown_amount = max(
            peak_equity
            - equity,
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

        maximum_drawdown_amount = max(
            maximum_drawdown_amount,
            drawdown_amount,
        )

        maximum_drawdown_percent = max(
            maximum_drawdown_percent,
            drawdown_percent,
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
    """Run a cost-adjusted passive buy-and-hold benchmark."""

    normalized_ticker = (
        ticker.strip().upper()
    )

    if not normalized_ticker:
        raise ValueError(
            "ticker cannot be empty."
        )

    config.validate()

    market_data = _validate_market_data(
        data
    )

    first_open = float(
        market_data.iloc[0]["Open"]
    )

    final_close = float(
        market_data.iloc[-1]["Close"]
    )

    entry_price = _apply_buy_slippage(
        price=first_open,
        config=config,
    )

    exit_price = _apply_sell_slippage(
        price=final_close,
        config=config,
    )

    quantity, entry_fee = (
        _calculate_quantity(
            initial_cash=config.initial_cash,
            entry_price=entry_price,
            config=config,
        )
    )

    if quantity <= 0:
        raise ValueError(
            "Initial capital is insufficient "
            "for the benchmark position."
        )

    entry_value = _round_money(
        quantity
        * entry_price
    )

    cash_after_entry = _round_money(
        config.initial_cash
        - entry_value
        - entry_fee
    )

    exit_value = _round_money(
        quantity
        * exit_price
    )

    exit_fee = _calculate_fee(
        transaction_value=exit_value,
        config=config,
    )

    ending_equity = _round_money(
        cash_after_entry
        + exit_value
        - exit_fee
    )

    total_return_amount = _round_money(
        ending_equity
        - config.initial_cash
    )

    total_return_percent = round(
        total_return_amount
        / config.initial_cash
        * 100,
        4,
    )

    (
        maximum_drawdown_amount,
        maximum_drawdown_percent,
    ) = _calculate_drawdown(
        cash_after_entry=cash_after_entry,
        quantity=quantity,
        close_values=market_data["Close"],
        initial_cash=config.initial_cash,
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
        ending_equity=ending_equity,
        quantity=quantity,
        entry_price=entry_price,
        exit_price=exit_price,
        entry_fee=entry_fee,
        exit_fee=exit_fee,
        total_fees=_round_money(
            entry_fee
            + exit_fee
        ),
        total_return_amount=(
            total_return_amount
        ),
        total_return_percent=(
            total_return_percent
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
    """Compare active strategy performance with buy-and-hold."""

    if (
        strategy_result.ticker.upper()
        != benchmark_result.ticker.upper()
    ):
        raise ValueError(
            "Strategy and benchmark tickers do not match."
        )

    excess_return_amount = _round_money(
        strategy_result.total_return_amount
        - benchmark_result.total_return_amount
    )

    excess_return_percent = round(
        strategy_result.total_return_percent
        - benchmark_result.total_return_percent,
        4,
    )

    return StrategyBenchmarkComparison(
        ticker=strategy_result.ticker,
        strategy_return_amount=(
            strategy_result.total_return_amount
        ),
        strategy_return_percent=(
            strategy_result.total_return_percent
        ),
        strategy_max_drawdown_percent=(
            strategy_result.maximum_drawdown_percent
        ),
        benchmark_return_amount=(
            benchmark_result.total_return_amount
        ),
        benchmark_return_percent=(
            benchmark_result.total_return_percent
        ),
        benchmark_max_drawdown_percent=(
            benchmark_result.maximum_drawdown_percent
        ),
        excess_return_amount=(
            excess_return_amount
        ),
        excess_return_percent=(
            excess_return_percent
        ),
        strategy_outperformed=(
            excess_return_amount > 0
        ),
    )


def print_buy_and_hold_result(
    result: BuyAndHoldResult,
) -> None:
    """Print passive benchmark performance."""

    print()
    print("=" * 92)
    print("BUY AND HOLD BENCHMARK")
    print("=" * 92)

    print(f"Ticker:                   {result.ticker}")
    print(f"Period start:             {result.start_date}")
    print(f"Period end:               {result.end_date}")

    print()
    print(
        f"Initial capital:          "
        f"{result.initial_cash:,.2f}"
    )
    print(
        f"Ending equity:            "
        f"{result.ending_equity:,.2f}"
    )
    print(
        f"Quantity:                 "
        f"{result.quantity}"
    )
    print(
        f"Entry price:              "
        f"{result.entry_price:.6f}"
    )
    print(
        f"Exit price:               "
        f"{result.exit_price:.6f}"
    )
    print(
        f"Total fees:               "
        f"{result.total_fees:,.2f}"
    )

    print()
    print(
        f"Total return:             "
        f"{result.total_return_amount:+,.2f}"
    )
    print(
        f"Total return %:           "
        f"{result.total_return_percent:+.4f}%"
    )
    print(
        f"Maximum drawdown:         "
        f"{result.maximum_drawdown_amount:,.2f}"
    )
    print(
        f"Maximum drawdown %:       "
        f"{result.maximum_drawdown_percent:.4f}%"
    )

    print("=" * 92)


def print_strategy_benchmark_comparison(
    comparison: StrategyBenchmarkComparison,
) -> None:
    """Print strategy-versus-benchmark comparison."""

    print()
    print("=" * 92)
    print("STRATEGY VS BUY AND HOLD")
    print("=" * 92)

    print(f"Ticker:                   {comparison.ticker}")

    print()
    print(
        f"Strategy return:          "
        f"{comparison.strategy_return_amount:+,.2f} "
        f"({comparison.strategy_return_percent:+.4f}%)"
    )
    print(
        f"Benchmark return:         "
        f"{comparison.benchmark_return_amount:+,.2f} "
        f"({comparison.benchmark_return_percent:+.4f}%)"
    )
    print(
        f"Excess return:            "
        f"{comparison.excess_return_amount:+,.2f} "
        f"({comparison.excess_return_percent:+.4f}%)"
    )

    print()
    print(
        f"Strategy max drawdown:    "
        f"{comparison.strategy_max_drawdown_percent:.4f}%"
    )
    print(
        f"Benchmark max drawdown:   "
        f"{comparison.benchmark_max_drawdown_percent:.4f}%"
    )

    print()
    print(
        "Outperformed benchmark:  "
        + (
            "YES"
            if comparison.strategy_outperformed
            else "NO"
        )
    )

    print("=" * 92)