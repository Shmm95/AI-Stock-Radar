"""Run deterministic backtests across multiple market symbols.

For every ticker, this module:

1. Downloads historical data.
2. Calculates technical indicators.
3. Runs the deterministic V1 strategy.
4. Runs a cost-adjusted buy-and-hold benchmark.
5. Compares strategy and benchmark performance.
6. Saves a consolidated CSV and JSON report.

This module performs historical simulation only.
It cannot place paper or real-money broker orders.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from math import inf, isfinite
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest.backtest_engine import run_backtest
from src.backtest.backtest_models import (
    BacktestConfig,
    BacktestResult,
)
from src.backtest.benchmark import (
    compare_strategy_to_benchmark,
    run_buy_and_hold_benchmark,
)
from src.backtest.run_backtest import (
    _prepare_market_data,
    deterministic_signal_provider,
)
from src.data.download_stock import download_stock_data


DEFAULT_TICKERS = (
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "BTC-USD",
    "ETH-USD",
)

DEFAULT_PERIOD = "5y"

OUTPUT_DIRECTORY = (
    Path("data")
    / "backtests"
    / "batch"
)


@dataclass(frozen=True, slots=True)
class BatchBacktestRow:
    """Consolidated result for one tested ticker."""

    ticker: str
    success: bool
    error: str

    prepared_bars: int
    completed_trades: int
    winning_trades: int
    losing_trades: int

    win_rate_percent: float
    profit_factor: float
    average_trade: float
    total_fees: float

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
    """Round monetary values."""

    return round(
        float(value),
        2,
    )


def _round_percent(
    value: float,
) -> float:
    """Round percentage values."""

    return round(
        float(value),
        4,
    )


def _normalize_tickers(
    tickers: list[str] | tuple[str, ...],
) -> list[str]:
    """Normalize ticker symbols while preserving order."""

    normalized: list[str] = []
    seen: set[str] = set()

    for ticker in tickers:
        symbol = (
            ticker.strip()
            .upper()
        )

        if not symbol:
            continue

        if symbol in seen:
            continue

        normalized.append(symbol)
        seen.add(symbol)

    if not normalized:
        raise ValueError(
            "At least one ticker must be supplied."
        )

    return normalized


def _is_crypto_ticker(
    ticker: str,
) -> bool:
    """Return whether a Yahoo ticker represents cryptocurrency."""

    normalized = ticker.upper()

    return (
        normalized.endswith("-USD")
        or normalized.endswith("-EUR")
        or normalized.endswith("-GBP")
    )


def _calculate_trade_statistics(
    result: BacktestResult,
) -> dict[str, float | int]:
    """Calculate compact statistics from completed trades."""

    trades = list(
        result.trades
    )

    winning_trades = [
        trade
        for trade in trades
        if trade.net_pnl > 0
    ]

    losing_trades = [
        trade
        for trade in trades
        if trade.net_pnl < 0
    ]

    completed_trades = len(
        trades
    )

    if completed_trades == 0:
        win_rate = 0.0
        average_trade = 0.0

    else:
        win_rate = (
            len(winning_trades)
            / completed_trades
            * 100
        )

        average_trade = (
            sum(
                trade.net_pnl
                for trade in trades
            )
            / completed_trades
        )

    gross_profit = sum(
        trade.net_pnl
        for trade in winning_trades
    )

    gross_loss = abs(
        sum(
            trade.net_pnl
            for trade in losing_trades
        )
    )

    if gross_loss == 0:
        profit_factor = (
            inf
            if gross_profit > 0
            else 0.0
        )

    else:
        profit_factor = (
            gross_profit
            / gross_loss
        )

    total_fees = sum(
        trade.total_fees
        for trade in trades
    )

    return {
        "completed_trades": completed_trades,
        "winning_trades": len(
            winning_trades
        ),
        "losing_trades": len(
            losing_trades
        ),
        "win_rate_percent": (
            _round_percent(
                win_rate
            )
        ),
        "profit_factor": (
            round(
                profit_factor,
                4,
            )
            if isfinite(profit_factor)
            else inf
        ),
        "average_trade": (
            _round_money(
                average_trade
            )
        ),
        "total_fees": (
            _round_money(
                total_fees
            )
        ),
    }


def _create_failed_row(
    *,
    ticker: str,
    error: Exception | str,
) -> BatchBacktestRow:
    """Create a standardized failed-ticker result."""

    return BatchBacktestRow(
        ticker=ticker,
        success=False,
        error=str(error),
        prepared_bars=0,
        completed_trades=0,
        winning_trades=0,
        losing_trades=0,
        win_rate_percent=0.0,
        profit_factor=0.0,
        average_trade=0.0,
        total_fees=0.0,
        strategy_return_amount=0.0,
        strategy_return_percent=0.0,
        strategy_max_drawdown_percent=0.0,
        benchmark_return_amount=0.0,
        benchmark_return_percent=0.0,
        benchmark_max_drawdown_percent=0.0,
        excess_return_amount=0.0,
        excess_return_percent=0.0,
        strategy_outperformed=False,
    )


def run_ticker_backtest(
    *,
    ticker: str,
    period: str,
    base_config: BacktestConfig,
    fractional_crypto: bool = True,
) -> BatchBacktestRow:
    """Run strategy and benchmark simulations for one ticker."""

    normalized_ticker = (
        ticker.strip()
        .upper()
    )

    if not normalized_ticker:
        raise ValueError(
            "Ticker cannot be empty."
        )

    ticker_config = replace(
        base_config,
        allow_fractional=(
            base_config.allow_fractional
            or (
                fractional_crypto
                and _is_crypto_ticker(
                    normalized_ticker
                )
            )
        ),
    )

    downloaded_data = download_stock_data(
        normalized_ticker,
        period=period,
    )

    prepared_data = _prepare_market_data(
        downloaded_data
    )

    strategy_result = run_backtest(
        ticker=normalized_ticker,
        data=prepared_data,
        signal_provider=(
            deterministic_signal_provider
        ),
        config=ticker_config,
    )

    benchmark_result = (
        run_buy_and_hold_benchmark(
            ticker=normalized_ticker,
            data=prepared_data,
            config=ticker_config,
        )
    )

    comparison = (
        compare_strategy_to_benchmark(
            strategy_result=(
                strategy_result
            ),
            benchmark_result=(
                benchmark_result
            ),
        )
    )

    statistics = (
        _calculate_trade_statistics(
            strategy_result
        )
    )

    return BatchBacktestRow(
        ticker=normalized_ticker,
        success=True,
        error="",
        prepared_bars=len(
            prepared_data
        ),
        completed_trades=int(
            statistics[
                "completed_trades"
            ]
        ),
        winning_trades=int(
            statistics[
                "winning_trades"
            ]
        ),
        losing_trades=int(
            statistics[
                "losing_trades"
            ]
        ),
        win_rate_percent=float(
            statistics[
                "win_rate_percent"
            ]
        ),
        profit_factor=float(
            statistics[
                "profit_factor"
            ]
        ),
        average_trade=float(
            statistics[
                "average_trade"
            ]
        ),
        total_fees=float(
            statistics[
                "total_fees"
            ]
        ),
        strategy_return_amount=(
            _round_money(
                comparison
                .strategy_return_amount
            )
        ),
        strategy_return_percent=(
            _round_percent(
                comparison
                .strategy_return_percent
            )
        ),
        strategy_max_drawdown_percent=(
            _round_percent(
                comparison
                .strategy_max_drawdown_percent
            )
        ),
        benchmark_return_amount=(
            _round_money(
                comparison
                .benchmark_return_amount
            )
        ),
        benchmark_return_percent=(
            _round_percent(
                comparison
                .benchmark_return_percent
            )
        ),
        benchmark_max_drawdown_percent=(
            _round_percent(
                comparison
                .benchmark_max_drawdown_percent
            )
        ),
        excess_return_amount=(
            _round_money(
                comparison
                .excess_return_amount
            )
        ),
        excess_return_percent=(
            _round_percent(
                comparison
                .excess_return_percent
            )
        ),
        strategy_outperformed=(
            comparison
            .strategy_outperformed
        ),
    )


def run_batch_backtest(
    *,
    tickers: list[str] | tuple[str, ...],
    period: str,
    config: BacktestConfig,
    fractional_crypto: bool = True,
    stop_on_error: bool = False,
) -> list[BatchBacktestRow]:
    """Run the same deterministic strategy across many tickers."""

    normalized_tickers = (
        _normalize_tickers(
            tickers
        )
    )

    rows: list[
        BatchBacktestRow
    ] = []

    print()
    print("=" * 110)
    print("AI STOCK RADAR — BATCH BACKTEST")
    print("=" * 110)

    print(
        f"Ticker count:        "
        f"{len(normalized_tickers)}"
    )

    print(
        f"Historical period:   "
        f"{period}"
    )

    print(
        f"Initial capital:     "
        f"{config.initial_cash:,.2f}"
    )

    print(
        f"Stop / target:       "
        f"{config.stop_loss_percent:.2f}% / "
        f"{config.take_profit_percent:.2f}%"
    )

    print("=" * 110)

    for index, ticker in enumerate(
        normalized_tickers,
        start=1,
    ):
        print()
        print(
            f"[{index}/{len(normalized_tickers)}] "
            f"Backtesting {ticker}..."
        )

        try:
            row = run_ticker_backtest(
                ticker=ticker,
                period=period,
                base_config=config,
                fractional_crypto=(
                    fractional_crypto
                ),
            )

        except Exception as error:
            row = _create_failed_row(
                ticker=ticker,
                error=error,
            )

            print(
                f"  FAILED: {error}"
            )

            rows.append(row)

            if stop_on_error:
                raise

            continue

        rows.append(row)

        print(
            f"  Trades: "
            f"{row.completed_trades}"
        )

        print(
            f"  Strategy return: "
            f"{row.strategy_return_percent:+.4f}%"
        )

        print(
            f"  Benchmark return: "
            f"{row.benchmark_return_percent:+.4f}%"
        )

        print(
            f"  Excess return: "
            f"{row.excess_return_percent:+.4f}%"
        )

        print(
            "  Outperformed: "
            + (
                "YES"
                if row.strategy_outperformed
                else "NO"
            )
        )

    return rows


def _json_safe_value(
    value: Any,
) -> Any:
    """Convert non-standard floating values for JSON output."""

    if isinstance(
        value,
        float,
    ):
        if not isfinite(value):
            return None

    return value


def _row_to_json_dict(
    row: BatchBacktestRow,
) -> dict[str, Any]:
    """Convert a batch row to JSON-compatible data."""

    return {
        key: _json_safe_value(
            value
        )
        for key, value in asdict(
            row
        ).items()
    }


def save_batch_results(
    rows: list[BatchBacktestRow],
) -> dict[str, Path]:
    """Save consolidated batch results."""

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        UTC
    ).strftime(
        "%Y%m%d_%H%M%S"
    )

    csv_path = (
        OUTPUT_DIRECTORY
        / f"batch_{timestamp}.csv"
    )

    json_path = (
        OUTPUT_DIRECTORY
        / f"batch_{timestamp}.json"
    )

    frame = pd.DataFrame(
        [
            asdict(row)
            for row in rows
        ]
    )

    if not frame.empty:
        frame = frame.sort_values(
            by=[
                "success",
                "excess_return_percent",
            ],
            ascending=[
                False,
                False,
            ],
        )

    frame.to_csv(
        csv_path,
        index=False,
    )

    payload = {
        "created_at": (
            datetime.now(
                UTC
            ).isoformat()
        ),
        "ticker_count": len(
            rows
        ),
        "results": [
            _row_to_json_dict(
                row
            )
            for row in rows
        ],
    }

    with json_path.open(
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

    return {
        "csv": csv_path,
        "json": json_path,
    }


def _format_profit_factor(
    value: float,
) -> str:
    """Format finite and infinite profit factors."""

    if not isfinite(value):
        return "INF"

    return f"{value:.2f}"


def print_batch_summary(
    rows: list[BatchBacktestRow],
) -> None:
    """Print consolidated multi-asset performance."""

    successful = [
        row
        for row in rows
        if row.success
    ]

    failed = [
        row
        for row in rows
        if not row.success
    ]

    outperformed = [
        row
        for row in successful
        if row.strategy_outperformed
    ]

    ranked = sorted(
        successful,
        key=lambda row: (
            row.excess_return_percent
        ),
        reverse=True,
    )

    total_trades = sum(
        row.completed_trades
        for row in successful
    )

    if successful:
        average_strategy_return = (
            sum(
                row.strategy_return_percent
                for row in successful
            )
            / len(successful)
        )

        average_benchmark_return = (
            sum(
                row.benchmark_return_percent
                for row in successful
            )
            / len(successful)
        )

        average_excess_return = (
            sum(
                row.excess_return_percent
                for row in successful
            )
            / len(successful)
        )

    else:
        average_strategy_return = 0.0
        average_benchmark_return = 0.0
        average_excess_return = 0.0

    print()
    print("=" * 132)
    print("BATCH BACKTEST SUMMARY")
    print("=" * 132)

    print(
        f"Successful tickers:       "
        f"{len(successful)}"
    )

    print(
        f"Failed tickers:           "
        f"{len(failed)}"
    )

    print(
        f"Total completed trades:   "
        f"{total_trades}"
    )

    print(
        f"Outperformed benchmark:   "
        f"{len(outperformed)} / "
        f"{len(successful)}"
    )

    print(
        f"Average strategy return:  "
        f"{average_strategy_return:+.4f}%"
    )

    print(
        f"Average benchmark return: "
        f"{average_benchmark_return:+.4f}%"
    )

    print(
        f"Average excess return:    "
        f"{average_excess_return:+.4f}%"
    )

    print()
    print(
        f"{'Ticker':<12}"
        f"{'Trades':>9}"
        f"{'Win %':>10}"
        f"{'PF':>9}"
        f"{'Strategy %':>14}"
        f"{'Benchmark %':>15}"
        f"{'Excess %':>12}"
        f"{'Strat DD %':>13}"
        f"{'Bench DD %':>13}"
        f"{'Beat':>8}"
    )

    print("-" * 132)

    for row in ranked:
        print(
            f"{row.ticker:<12}"
            f"{row.completed_trades:>9}"
            f"{row.win_rate_percent:>10.2f}"
            f"{_format_profit_factor(row.profit_factor):>9}"
            f"{row.strategy_return_percent:>14.2f}"
            f"{row.benchmark_return_percent:>15.2f}"
            f"{row.excess_return_percent:>12.2f}"
            f"{row.strategy_max_drawdown_percent:>13.2f}"
            f"{row.benchmark_max_drawdown_percent:>13.2f}"
            f"{('YES' if row.strategy_outperformed else 'NO'):>8}"
        )

    if failed:
        print()
        print("FAILED TICKERS")

        for row in failed:
            print(
                f"- {row.ticker}: "
                f"{row.error}"
            )

    print("=" * 132)


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the deterministic V1 strategy "
            "across multiple tickers."
        )
    )

    parser.add_argument(
        "tickers",
        nargs="*",
        help=(
            "Ticker list. When omitted, "
            "the default development universe is used."
        ),
    )

    parser.add_argument(
        "--period",
        default=DEFAULT_PERIOD,
    )

    parser.add_argument(
        "--initial-cash",
        type=float,
        default=10_000.0,
    )

    parser.add_argument(
        "--risk",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--max-position",
        type=float,
        default=25.0,
    )

    parser.add_argument(
        "--stop",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--target",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--commission-rate",
        type=float,
        default=0.0005,
    )

    parser.add_argument(
        "--minimum-fee",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--fractional-all",
        action="store_true",
        help=(
            "Allow fractional quantities "
            "for every ticker."
        ),
    )

    parser.add_argument(
        "--no-fractional-crypto",
        action="store_true",
        help=(
            "Disable automatic fractional quantities "
            "for crypto tickers."
        ),
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    """Run one consolidated batch backtest."""

    arguments = (
        _parse_arguments()
    )

    tickers = (
        arguments.tickers
        if arguments.tickers
        else list(
            DEFAULT_TICKERS
        )
    )

    config = BacktestConfig(
        initial_cash=(
            arguments.initial_cash
        ),
        risk_per_trade_percent=(
            arguments.risk
        ),
        maximum_position_percent=(
            arguments.max_position
        ),
        stop_loss_percent=(
            arguments.stop
        ),
        take_profit_percent=(
            arguments.target
        ),
        commission_rate=(
            arguments.commission_rate
        ),
        minimum_fee=(
            arguments.minimum_fee
        ),
        slippage_bps=(
            arguments.slippage_bps
        ),
        allow_fractional=(
            arguments.fractional_all
        ),
        maximum_open_positions=1,
        force_close_at_end=True,
    )

    config.validate()

    rows = run_batch_backtest(
        tickers=tickers,
        period=arguments.period,
        config=config,
        fractional_crypto=(
            not arguments
            .no_fractional_crypto
        ),
        stop_on_error=(
            arguments.stop_on_error
        ),
    )

    print_batch_summary(
        rows
    )

    if not arguments.no_save:
        paths = save_batch_results(
            rows
        )

        print()
        print("=" * 110)
        print("BATCH BACKTEST FILES")
        print("=" * 110)

        print(
            f"CSV:  "
            f"{paths['csv'].resolve()}"
        )

        print(
            f"JSON: "
            f"{paths['json'].resolve()}"
        )

        print("=" * 110)

    print()
    print(
        "Batch backtest completed successfully."
    )


if __name__ == "__main__":
    main()