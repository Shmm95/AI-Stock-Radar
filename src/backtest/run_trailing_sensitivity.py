"""Trailing-close sensitivity analysis for AI-Stock-Radar.

This module compares multiple Close-based trailing distances while
keeping every other strategy component constant.

Shared entry:
- EMA20 > EMA50
- Close > EMA20
- RSI14 between 45 and 70
- Entry only when the setup becomes newly valid

Shared protection:
- 5% initial stop by default
- No practical fixed take-profit
- EMA20 below EMA50 exit
- Close-based trailing exit
- Signals execute at the next bar Open through the shared engine

Default trailing distances:
- 7.5%
- 10.0%
- 12.5%
- 15.0%

Results are summarized separately for:
- ALL assets
- EQUITY assets
- CRYPTO assets
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from math import inf, isfinite
from pathlib import Path
from statistics import mean, median
from typing import Any

import pandas as pd

from src.backtest.backtest_engine import run_backtest
from src.backtest.backtest_models import (
    BacktestConfig,
    BacktestResult,
)
from src.backtest.benchmark import (
    BuyAndHoldResult,
    run_buy_and_hold_benchmark,
)
from src.backtest.run_backtest import _prepare_market_data
from src.backtest.run_exit_ablation import (
    DISABLED_TARGET_PERCENT,
    ExitSignalProvider,
    ExitVariant,
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

DEFAULT_TRAILING_VALUES = (
    7.5,
    10.0,
    12.5,
    15.0,
)

DEFAULT_PERIOD = "5y"

OUTPUT_DIRECTORY = (
    Path("data")
    / "backtests"
    / "trailing_sensitivity"
)


@dataclass(frozen=True, slots=True)
class TrailingSensitivityRow:
    """One ticker and trailing-distance result."""

    ticker: str
    asset_class: str
    trailing_close_percent: float

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

    profitable: bool
    strategy_outperformed: bool


@dataclass(frozen=True, slots=True)
class TrailingSensitivitySummary:
    """Aggregate result for one scope and distance."""

    scope: str
    rank: int
    trailing_close_percent: float

    successful_tickers: int
    failed_tickers: int

    total_trades: int
    total_winning_trades: int
    total_losing_trades: int

    profitable_tickers: int
    profitable_tickers_percent: float

    outperformed_count: int
    outperformed_percent: float

    average_strategy_return_percent: float
    median_strategy_return_percent: float

    best_ticker_return_percent: float
    worst_ticker_return_percent: float

    average_benchmark_return_percent: float

    average_excess_return_percent: float
    median_excess_return_percent: float

    average_max_drawdown_percent: float
    average_benchmark_drawdown_percent: float

    average_profit_factor: float
    average_win_rate_percent: float


def _round_money(
    value: float,
) -> float:
    """Round monetary values."""

    return round(
        float(value),
        2,
    )


def _round_metric(
    value: float,
) -> float:
    """Round percentages and metrics."""

    return round(
        float(value),
        4,
    )


def _is_crypto_ticker(
    ticker: str,
) -> bool:
    """Return whether a ticker represents cryptocurrency."""

    normalized = ticker.strip().upper()

    return normalized.endswith(
        (
            "-USD",
            "-EUR",
            "-GBP",
        )
    )


def _asset_class(
    ticker: str,
) -> str:
    """Return the asset class for a ticker."""

    if _is_crypto_ticker(ticker):
        return "CRYPTO"

    return "EQUITY"


def _normalize_tickers(
    tickers: list[str] | tuple[str, ...],
) -> list[str]:
    """Normalize tickers and remove duplicates."""

    normalized: list[str] = []
    seen: set[str] = set()

    for ticker in tickers:
        symbol = ticker.strip().upper()

        if (
            symbol
            and symbol not in seen
        ):
            normalized.append(symbol)
            seen.add(symbol)

    if not normalized:
        raise ValueError(
            "At least one ticker must be supplied."
        )

    return normalized


def normalize_trailing_values(
    values: list[float] | tuple[float, ...],
) -> list[float]:
    """Validate, sort, and deduplicate trailing distances."""

    normalized: list[float] = []

    for raw_value in values:
        value = float(raw_value)

        if not isfinite(value):
            raise ValueError(
                "Trailing values must be finite."
            )

        if value <= 0:
            raise ValueError(
                "Trailing values must be greater than zero."
            )

        if value >= 100:
            raise ValueError(
                "Trailing values must be below 100 percent."
            )

        rounded = round(
            value,
            6,
        )

        if rounded not in normalized:
            normalized.append(rounded)

    normalized.sort()

    if not normalized:
        raise ValueError(
            "At least one trailing value must be supplied."
        )

    return normalized


def _create_trailing_variant(
    trailing_close_percent: float,
) -> ExitVariant:
    """Create one trailing-only exit variant."""

    formatted_distance = (
        f"{trailing_close_percent:g}"
        .replace(
            ".",
            "_",
        )
    )

    return ExitVariant(
        name=(
            "TRAILING_CLOSE_"
            f"{formatted_distance}"
        ),
        description=(
            "5% initial stop, no fixed target, "
            f"{trailing_close_percent:g}% "
            "highest-Close trailing exit, "
            "plus EMA20 below EMA50 exit."
        ),
        take_profit_percent=(
            DISABLED_TARGET_PERCENT
        ),
        use_trend_rsi_exit=False,
        trailing_close_percent=(
            trailing_close_percent
        ),
    )


def _average_trade(
    result: BacktestResult,
) -> float:
    """Calculate average completed-trade PnL."""

    if not result.trades:
        return 0.0

    return _round_money(
        sum(
            trade.net_pnl
            for trade in result.trades
        )
        / len(result.trades)
    )


def _failed_row(
    *,
    ticker: str,
    trailing_close_percent: float,
    error: Exception | str,
) -> TrailingSensitivityRow:
    """Create a standardized failed result."""

    return TrailingSensitivityRow(
        ticker=ticker,
        asset_class=_asset_class(
            ticker
        ),
        trailing_close_percent=(
            trailing_close_percent
        ),
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
        profitable=False,
        strategy_outperformed=False,
    )


def _successful_row(
    *,
    ticker: str,
    trailing_close_percent: float,
    prepared_bars: int,
    result: BacktestResult,
    benchmark: BuyAndHoldResult,
) -> TrailingSensitivityRow:
    """Create a successful sensitivity result."""

    excess_return_amount = (
        result.total_return_amount
        - benchmark.total_return_amount
    )

    excess_return_percent = (
        result.total_return_percent
        - benchmark.total_return_percent
    )

    profit_factor = (
        round(
            result.profit_factor,
            4,
        )
        if isfinite(
            result.profit_factor
        )
        else inf
    )

    return TrailingSensitivityRow(
        ticker=ticker,
        asset_class=_asset_class(
            ticker
        ),
        trailing_close_percent=(
            trailing_close_percent
        ),
        success=True,
        error="",
        prepared_bars=prepared_bars,
        completed_trades=(
            result.completed_trades
        ),
        winning_trades=(
            result.winning_trades
        ),
        losing_trades=(
            result.losing_trades
        ),
        win_rate_percent=(
            _round_metric(
                result.win_rate_percent
            )
        ),
        profit_factor=profit_factor,
        average_trade=(
            _average_trade(
                result
            )
        ),
        total_fees=(
            _round_money(
                result.total_fees
            )
        ),
        strategy_return_amount=(
            _round_money(
                result.total_return_amount
            )
        ),
        strategy_return_percent=(
            _round_metric(
                result.total_return_percent
            )
        ),
        strategy_max_drawdown_percent=(
            _round_metric(
                result.maximum_drawdown_percent
            )
        ),
        benchmark_return_amount=(
            _round_money(
                benchmark.total_return_amount
            )
        ),
        benchmark_return_percent=(
            _round_metric(
                benchmark.total_return_percent
            )
        ),
        benchmark_max_drawdown_percent=(
            _round_metric(
                benchmark.maximum_drawdown_percent
            )
        ),
        excess_return_amount=(
            _round_money(
                excess_return_amount
            )
        ),
        excess_return_percent=(
            _round_metric(
                excess_return_percent
            )
        ),
        profitable=(
            result.total_return_amount > 0
        ),
        strategy_outperformed=(
            excess_return_amount > 0
        ),
    )


def run_trailing_sensitivity(
    *,
    tickers: list[str] | tuple[str, ...],
    trailing_values: list[float] | tuple[float, ...],
    period: str,
    base_config: BacktestConfig,
    fractional_crypto: bool = True,
    stop_on_error: bool = False,
) -> list[TrailingSensitivityRow]:
    """Run every trailing distance across every ticker."""

    symbols = _normalize_tickers(
        tickers
    )

    distances = normalize_trailing_values(
        trailing_values
    )

    rows: list[
        TrailingSensitivityRow
    ] = []

    print()
    print("=" * 120)
    print("AI STOCK RADAR — TRAILING SENSITIVITY")
    print("=" * 120)

    print(
        f"Tickers:                   "
        f"{len(symbols)}"
    )

    print(
        f"Trailing distances:        "
        f"{', '.join(f'{value:g}%' for value in distances)}"
    )

    print(
        f"Historical period:         "
        f"{period}"
    )

    print(
        f"Initial stop:              "
        f"{base_config.stop_loss_percent:.2f}%"
    )

    print(
        f"Initial capital:           "
        f"{base_config.initial_cash:,.2f}"
    )

    print("=" * 120)

    for ticker_index, ticker in enumerate(
        symbols,
        start=1,
    ):
        print()
        print(
            f"[{ticker_index}/{len(symbols)}] "
            f"Preparing {ticker}..."
        )

        ticker_config = replace(
            base_config,
            allow_fractional=(
                base_config.allow_fractional
                or (
                    fractional_crypto
                    and _is_crypto_ticker(
                        ticker
                    )
                )
            ),
            take_profit_percent=(
                DISABLED_TARGET_PERCENT
            ),
        )

        try:
            downloaded_data = (
                download_stock_data(
                    ticker,
                    period=period,
                )
            )

            prepared_data = (
                _prepare_market_data(
                    downloaded_data
                )
            )

            benchmark = (
                run_buy_and_hold_benchmark(
                    ticker=ticker,
                    data=prepared_data,
                    config=ticker_config,
                )
            )

        except Exception as error:
            print(
                f"  PREPARATION FAILED: "
                f"{error}"
            )

            for distance in distances:
                rows.append(
                    _failed_row(
                        ticker=ticker,
                        trailing_close_percent=(
                            distance
                        ),
                        error=error,
                    )
                )

            if stop_on_error:
                raise

            continue

        print(
            f"  Prepared bars: "
            f"{len(prepared_data)}"
        )

        for distance in distances:
            variant = (
                _create_trailing_variant(
                    distance
                )
            )

            signal_provider = (
                ExitSignalProvider(
                    variant=variant,
                    config=ticker_config,
                )
            )

            print(
                f"  Running trailing "
                f"{distance:>5.2f}%  ",
                end="",
            )

            try:
                result = run_backtest(
                    ticker=ticker,
                    data=prepared_data,
                    signal_provider=(
                        signal_provider
                    ),
                    config=ticker_config,
                )

                row = _successful_row(
                    ticker=ticker,
                    trailing_close_percent=(
                        distance
                    ),
                    prepared_bars=len(
                        prepared_data
                    ),
                    result=result,
                    benchmark=benchmark,
                )

            except Exception as error:
                row = _failed_row(
                    ticker=ticker,
                    trailing_close_percent=(
                        distance
                    ),
                    error=error,
                )

                rows.append(row)

                print(
                    f"FAILED: {error}"
                )

                if stop_on_error:
                    raise

                continue

            rows.append(row)

            print(
                f"Trades={row.completed_trades:<3} "
                f"Return={row.strategy_return_percent:>+8.2f}% "
                f"PF={_format_profit_factor(row.profit_factor):>6} "
                f"Excess={row.excess_return_percent:>+8.2f}% "
                f"DD={row.strategy_max_drawdown_percent:>6.2f}%"
            )

    return rows


def _rows_for_scope(
    rows: list[TrailingSensitivityRow],
    scope: str,
) -> list[TrailingSensitivityRow]:
    """Filter rows for one summary scope."""

    if scope == "ALL":
        return rows

    return [
        row
        for row in rows
        if row.asset_class == scope
    ]


def _safe_mean(
    values: list[float],
) -> float:
    """Return a safe arithmetic mean."""

    if not values:
        return 0.0

    return float(
        mean(values)
    )


def _safe_median(
    values: list[float],
) -> float:
    """Return a safe median."""

    if not values:
        return 0.0

    return float(
        median(values)
    )


def _create_summary(
    *,
    scope: str,
    trailing_close_percent: float,
    rows: list[TrailingSensitivityRow],
) -> TrailingSensitivitySummary:
    """Create one aggregate summary."""

    matching_rows = [
        row
        for row in _rows_for_scope(
            rows,
            scope,
        )
        if row.trailing_close_percent
        == trailing_close_percent
    ]

    successful_rows = [
        row
        for row in matching_rows
        if row.success
    ]

    failed_rows = [
        row
        for row in matching_rows
        if not row.success
    ]

    strategy_returns = [
        row.strategy_return_percent
        for row in successful_rows
    ]

    benchmark_returns = [
        row.benchmark_return_percent
        for row in successful_rows
    ]

    excess_returns = [
        row.excess_return_percent
        for row in successful_rows
    ]

    strategy_drawdowns = [
        row.strategy_max_drawdown_percent
        for row in successful_rows
    ]

    benchmark_drawdowns = [
        row.benchmark_max_drawdown_percent
        for row in successful_rows
    ]

    finite_profit_factors = [
        row.profit_factor
        for row in successful_rows
        if isfinite(
            row.profit_factor
        )
    ]

    win_rates = [
        row.win_rate_percent
        for row in successful_rows
    ]

    profitable_count = sum(
        1
        for row in successful_rows
        if row.profitable
    )

    outperformed_count = sum(
        1
        for row in successful_rows
        if row.strategy_outperformed
    )

    successful_count = len(
        successful_rows
    )

    return TrailingSensitivitySummary(
        scope=scope,
        rank=0,
        trailing_close_percent=(
            trailing_close_percent
        ),
        successful_tickers=(
            successful_count
        ),
        failed_tickers=len(
            failed_rows
        ),
        total_trades=sum(
            row.completed_trades
            for row in successful_rows
        ),
        total_winning_trades=sum(
            row.winning_trades
            for row in successful_rows
        ),
        total_losing_trades=sum(
            row.losing_trades
            for row in successful_rows
        ),
        profitable_tickers=(
            profitable_count
        ),
        profitable_tickers_percent=(
            _round_metric(
                profitable_count
                / successful_count
                * 100
                if successful_count
                else 0.0
            )
        ),
        outperformed_count=(
            outperformed_count
        ),
        outperformed_percent=(
            _round_metric(
                outperformed_count
                / successful_count
                * 100
                if successful_count
                else 0.0
            )
        ),
        average_strategy_return_percent=(
            _round_metric(
                _safe_mean(
                    strategy_returns
                )
            )
        ),
        median_strategy_return_percent=(
            _round_metric(
                _safe_median(
                    strategy_returns
                )
            )
        ),
        best_ticker_return_percent=(
            _round_metric(
                max(
                    strategy_returns,
                    default=0.0,
                )
            )
        ),
        worst_ticker_return_percent=(
            _round_metric(
                min(
                    strategy_returns,
                    default=0.0,
                )
            )
        ),
        average_benchmark_return_percent=(
            _round_metric(
                _safe_mean(
                    benchmark_returns
                )
            )
        ),
        average_excess_return_percent=(
            _round_metric(
                _safe_mean(
                    excess_returns
                )
            )
        ),
        median_excess_return_percent=(
            _round_metric(
                _safe_median(
                    excess_returns
                )
            )
        ),
        average_max_drawdown_percent=(
            _round_metric(
                _safe_mean(
                    strategy_drawdowns
                )
            )
        ),
        average_benchmark_drawdown_percent=(
            _round_metric(
                _safe_mean(
                    benchmark_drawdowns
                )
            )
        ),
        average_profit_factor=(
            round(
                _safe_mean(
                    finite_profit_factors
                ),
                4,
            )
        ),
        average_win_rate_percent=(
            _round_metric(
                _safe_mean(
                    win_rates
                )
            )
        ),
    )


def summarize_sensitivity(
    rows: list[TrailingSensitivityRow],
) -> list[TrailingSensitivitySummary]:
    """Create ranked ALL, EQUITY, and CRYPTO summaries."""

    trailing_values = sorted(
        {
            row.trailing_close_percent
            for row in rows
        }
    )

    summaries: list[
        TrailingSensitivitySummary
    ] = []

    for scope in (
        "ALL",
        "EQUITY",
        "CRYPTO",
    ):
        scope_summaries = [
            _create_summary(
                scope=scope,
                trailing_close_percent=(
                    trailing_value
                ),
                rows=rows,
            )
            for trailing_value
            in trailing_values
        ]

        scope_summaries.sort(
            key=lambda item: (
                item.median_strategy_return_percent,
                item.average_strategy_return_percent,
                item.average_profit_factor,
                -item.average_max_drawdown_percent,
            ),
            reverse=True,
        )

        ranked_scope_summaries = [
            replace(
                item,
                rank=index,
            )
            for index, item in enumerate(
                scope_summaries,
                start=1,
            )
        ]

        summaries.extend(
            ranked_scope_summaries
        )

    return summaries


def _format_profit_factor(
    value: float,
) -> str:
    """Format profit factor output."""

    if not isfinite(value):
        return "INF"

    return f"{value:.2f}"


def print_sensitivity_summary(
    summaries: list[TrailingSensitivitySummary],
) -> None:
    """Print ranked sensitivity summaries."""

    for scope in (
        "ALL",
        "EQUITY",
        "CRYPTO",
    ):
        scope_summaries = [
            item
            for item in summaries
            if item.scope == scope
        ]

        if not scope_summaries:
            continue

        print()
        print("=" * 154)
        print(
            f"TRAILING SENSITIVITY SUMMARY — {scope}"
        )
        print("=" * 154)

        print(
            f"{'Rank':>5}"
            f"{'Trail %':>10}"
            f"{'Tickers':>10}"
            f"{'Trades':>10}"
            f"{'Win %':>9}"
            f"{'Avg PF':>9}"
            f"{'Avg Ret':>11}"
            f"{'Med Ret':>11}"
            f"{'Worst':>10}"
            f"{'Avg Excess':>13}"
            f"{'Med Excess':>13}"
            f"{'Avg DD':>10}"
            f"{'Positive':>11}"
            f"{'Beat':>9}"
        )

        print("-" * 154)

        for item in scope_summaries:
            profitable_text = (
                f"{item.profitable_tickers}/"
                f"{item.successful_tickers}"
            )

            outperformed_text = (
                f"{item.outperformed_count}/"
                f"{item.successful_tickers}"
            )

            print(
                f"{item.rank:>5}"
                f"{item.trailing_close_percent:>10.2f}"
                f"{item.successful_tickers:>10}"
                f"{item.total_trades:>10}"
                f"{item.average_win_rate_percent:>9.2f}"
                f"{_format_profit_factor(item.average_profit_factor):>9}"
                f"{item.average_strategy_return_percent:>11.2f}"
                f"{item.median_strategy_return_percent:>11.2f}"
                f"{item.worst_ticker_return_percent:>10.2f}"
                f"{item.average_excess_return_percent:>13.2f}"
                f"{item.median_excess_return_percent:>13.2f}"
                f"{item.average_max_drawdown_percent:>10.2f}"
                f"{profitable_text:>11}"
                f"{outperformed_text:>9}"
            )

        print("=" * 154)

        winner = scope_summaries[0]

        print(
            f"Provisional {scope} winner: "
            f"{winner.trailing_close_percent:.2f}%"
        )

        print(
            f"Median return:             "
            f"{winner.median_strategy_return_percent:+.4f}%"
        )

        print(
            f"Average return:            "
            f"{winner.average_strategy_return_percent:+.4f}%"
        )

        print(
            f"Average profit factor:     "
            f"{_format_profit_factor(winner.average_profit_factor)}"
        )

        print(
            f"Average drawdown:          "
            f"{winner.average_max_drawdown_percent:.4f}%"
        )


def print_sensitivity_details(
    rows: list[TrailingSensitivityRow],
) -> None:
    """Print detailed ticker results."""

    successful_rows = sorted(
        [
            row
            for row in rows
            if row.success
        ],
        key=lambda row: (
            row.ticker,
            row.trailing_close_percent,
        ),
    )

    print()
    print("=" * 151)
    print("TRAILING SENSITIVITY DETAILS")
    print("=" * 151)

    print(
        f"{'Ticker':<12}"
        f"{'Class':<9}"
        f"{'Trail %':>9}"
        f"{'Trades':>9}"
        f"{'Win %':>9}"
        f"{'PF':>8}"
        f"{'Return %':>11}"
        f"{'Benchmark':>12}"
        f"{'Excess %':>11}"
        f"{'Strat DD':>10}"
        f"{'Bench DD':>10}"
        f"{'Positive':>10}"
        f"{'Beat':>7}"
    )

    print("-" * 151)

    for row in successful_rows:
        print(
            f"{row.ticker:<12}"
            f"{row.asset_class:<9}"
            f"{row.trailing_close_percent:>9.2f}"
            f"{row.completed_trades:>9}"
            f"{row.win_rate_percent:>9.2f}"
            f"{_format_profit_factor(row.profit_factor):>8}"
            f"{row.strategy_return_percent:>11.2f}"
            f"{row.benchmark_return_percent:>12.2f}"
            f"{row.excess_return_percent:>11.2f}"
            f"{row.strategy_max_drawdown_percent:>10.2f}"
            f"{row.benchmark_max_drawdown_percent:>10.2f}"
            f"{('YES' if row.profitable else 'NO'):>10}"
            f"{('YES' if row.strategy_outperformed else 'NO'):>7}"
        )

    failed_rows = [
        row
        for row in rows
        if not row.success
    ]

    if failed_rows:
        print()
        print("FAILED TESTS")

        for row in failed_rows:
            print(
                f"- {row.ticker} / "
                f"{row.trailing_close_percent:.2f}%: "
                f"{row.error}"
            )

    print("=" * 151)


def _json_safe_value(
    value: Any,
) -> Any:
    """Convert non-finite floats to JSON-safe values."""

    if (
        isinstance(value, float)
        and not isfinite(value)
    ):
        return None

    return value


def _json_safe_dictionary(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Convert a dictionary to JSON-safe values."""

    return {
        key: _json_safe_value(value)
        for key, value in payload.items()
    }


def save_sensitivity_results(
    *,
    rows: list[TrailingSensitivityRow],
    summaries: list[TrailingSensitivitySummary],
    trailing_values: list[float],
    period: str,
) -> dict[str, Path]:
    """Save detailed CSV, summary CSV, and JSON reports."""

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = (
        datetime.now(UTC)
        .strftime("%Y%m%d_%H%M%S")
    )

    details_path = (
        OUTPUT_DIRECTORY
        / (
            "trailing_sensitivity_details_"
            f"{timestamp}.csv"
        )
    )

    summary_path = (
        OUTPUT_DIRECTORY
        / (
            "trailing_sensitivity_summary_"
            f"{timestamp}.csv"
        )
    )

    json_path = (
        OUTPUT_DIRECTORY
        / (
            "trailing_sensitivity_"
            f"{timestamp}.json"
        )
    )

    pd.DataFrame(
        [
            asdict(row)
            for row in rows
        ]
    ).to_csv(
        details_path,
        index=False,
    )

    pd.DataFrame(
        [
            asdict(summary)
            for summary in summaries
        ]
    ).to_csv(
        summary_path,
        index=False,
    )

    payload = {
        "created_at": (
            datetime.now(UTC)
            .isoformat()
        ),
        "period": period,
        "entry_rule": (
            "EMA20>EMA50, Close>EMA20, "
            "RSI14 between 45 and 70, "
            "new transition only."
        ),
        "exit_rule": (
            "5% initial stop by default, "
            "EMA20 below EMA50 or "
            "Close-based trailing exit, "
            "filled at next Open."
        ),
        "trailing_values": (
            trailing_values
        ),
        "summaries": [
            _json_safe_dictionary(
                asdict(summary)
            )
            for summary in summaries
        ],
        "details": [
            _json_safe_dictionary(
                asdict(row)
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
        "details_csv": details_path,
        "summary_csv": summary_path,
        "json": json_path,
    }


def _parse_arguments() -> argparse.Namespace:
    """Parse terminal arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare multiple Close-based "
            "trailing distances."
        )
    )

    parser.add_argument(
        "tickers",
        nargs="*",
    )

    parser.add_argument(
        "--period",
        default=DEFAULT_PERIOD,
    )

    parser.add_argument(
        "--trailing-values",
        nargs="+",
        type=float,
        default=list(
            DEFAULT_TRAILING_VALUES
        ),
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
    )

    parser.add_argument(
        "--no-fractional-crypto",
        action="store_true",
    )

    parser.add_argument(
        "--show-details",
        action="store_true",
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
    """Run trailing sensitivity analysis."""

    arguments = _parse_arguments()

    tickers = (
        arguments.tickers
        if arguments.tickers
        else list(
            DEFAULT_TICKERS
        )
    )

    trailing_values = (
        normalize_trailing_values(
            arguments.trailing_values
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
            DISABLED_TARGET_PERCENT
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

    rows = run_trailing_sensitivity(
        tickers=tickers,
        trailing_values=trailing_values,
        period=arguments.period,
        base_config=config,
        fractional_crypto=(
            not arguments
            .no_fractional_crypto
        ),
        stop_on_error=(
            arguments.stop_on_error
        ),
    )

    summaries = summarize_sensitivity(
        rows
    )

    print_sensitivity_summary(
        summaries
    )

    if arguments.show_details:
        print_sensitivity_details(
            rows
        )

    if not arguments.no_save:
        paths = save_sensitivity_results(
            rows=rows,
            summaries=summaries,
            trailing_values=(
                trailing_values
            ),
            period=arguments.period,
        )

        print()
        print("=" * 120)
        print("TRAILING SENSITIVITY FILES")
        print("=" * 120)

        print(
            f"Details CSV: "
            f"{paths['details_csv'].resolve()}"
        )

        print(
            f"Summary CSV: "
            f"{paths['summary_csv'].resolve()}"
        )

        print(
            f"JSON:        "
            f"{paths['json'].resolve()}"
        )

        print("=" * 120)

    print()
    print(
        "Trailing sensitivity completed successfully."
    )


if __name__ == "__main__":
    main()