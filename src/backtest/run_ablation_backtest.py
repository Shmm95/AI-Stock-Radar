"""Technical strategy ablation tests for AI-Stock-Radar.

This module compares multiple deterministic technical strategies:

1. TREND_ONLY
2. TREND_MACD
3. TREND_RSI
4. FULL_TECHNICAL

For each ticker:
- Historical data is downloaded once.
- Indicators are calculated once.
- Every strategy variant uses the same price history.
- Every strategy is compared with the same buy-and-hold benchmark.
- Commission, slippage, stop-loss, and take-profit are included.

Fundamental and news ablations are intentionally excluded because
point-in-time historical datasets are not yet available. Using today's
fundamental or news values in historical bars would create look-ahead bias.

This module performs historical simulation only.
It cannot place paper or real-money orders.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from math import inf, isfinite
from pathlib import Path
from typing import Callable

import pandas as pd

from src.backtest.backtest_engine import run_backtest
from src.backtest.backtest_models import (
    BacktestConfig,
    BacktestResult,
    BacktestSignal,
)
from src.backtest.benchmark import (
    BuyAndHoldResult,
    run_buy_and_hold_benchmark,
)
from src.backtest.run_backtest import _prepare_market_data
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
    / "ablation"
)


EntryRule = Callable[[pd.Series], bool]
ExitRule = Callable[[pd.Series], bool]


@dataclass(frozen=True, slots=True)
class StrategyVariant:
    """Definition of one deterministic strategy variant."""

    name: str
    description: str

    entry_rule: EntryRule
    exit_rule: ExitRule


@dataclass(frozen=True, slots=True)
class AblationRow:
    """One strategy result for one ticker."""

    ticker: str
    variant: str

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


@dataclass(frozen=True, slots=True)
class VariantSummary:
    """Aggregate statistics for one strategy variant."""

    variant: str

    successful_tickers: int
    failed_tickers: int

    total_trades: int
    total_winning_trades: int
    total_losing_trades: int

    outperformed_count: int
    outperformed_percent: float

    average_strategy_return_percent: float
    average_benchmark_return_percent: float
    average_excess_return_percent: float

    median_strategy_return_percent: float
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


def _round_percent(
    value: float,
) -> float:
    """Round percentage values."""

    return round(
        float(value),
        4,
    )


def _timestamp_to_string(
    value: object,
) -> str:
    """Convert a DataFrame index value to a stable string."""

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    return str(value)


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

        normalized.append(
            symbol
        )

        seen.add(
            symbol
        )

    if not normalized:
        raise ValueError(
            "At least one ticker must be supplied."
        )

    return normalized


def _is_crypto_ticker(
    ticker: str,
) -> bool:
    """Return whether the ticker is treated as cryptocurrency."""

    normalized = ticker.upper()

    return (
        normalized.endswith("-USD")
        or normalized.endswith("-EUR")
        or normalized.endswith("-GBP")
    )


def _trend_entry(
    row: pd.Series,
) -> bool:
    """Trend-only long entry rule."""

    ema20 = float(
        row["EMA20"]
    )

    ema50 = float(
        row["EMA50"]
    )

    close = float(
        row["Close"]
    )

    return (
        ema20 > ema50
        and close > ema20
    )


def _trend_exit(
    row: pd.Series,
) -> bool:
    """Trend-only exit rule."""

    ema20 = float(
        row["EMA20"]
    )

    ema50 = float(
        row["EMA50"]
    )

    return ema20 < ema50


def _trend_macd_entry(
    row: pd.Series,
) -> bool:
    """Trend plus MACD entry rule."""

    return (
        _trend_entry(row)
        and float(
            row["MACD"]
        ) > 0
    )


def _trend_macd_exit(
    row: pd.Series,
) -> bool:
    """Trend plus MACD exit rule."""

    return (
        _trend_exit(row)
        or float(
            row["MACD"]
        ) < 0
    )


def _trend_rsi_entry(
    row: pd.Series,
) -> bool:
    """Trend plus RSI entry rule."""

    rsi14 = float(
        row["RSI14"]
    )

    return (
        _trend_entry(row)
        and 45 <= rsi14 <= 70
    )


def _trend_rsi_exit(
    row: pd.Series,
) -> bool:
    """Trend plus RSI exit rule."""

    return (
        _trend_exit(row)
        or float(
            row["RSI14"]
        ) >= 75
    )


def _full_technical_entry(
    row: pd.Series,
) -> bool:
    """Full V1 technical entry rule."""

    return (
        _trend_entry(row)
        and float(
            row["MACD"]
        ) > 0
        and 45
        <= float(
            row["RSI14"]
        )
        <= 70
    )


def _full_technical_exit(
    row: pd.Series,
) -> bool:
    """Full V1 technical exit rule."""

    return (
        _trend_exit(row)
        or float(
            row["MACD"]
        ) < 0
        or float(
            row["RSI14"]
        ) >= 75
    )


STRATEGY_VARIANTS = (
    StrategyVariant(
        name="TREND_ONLY",
        description=(
            "EMA20 above EMA50 and price above EMA20."
        ),
        entry_rule=_trend_entry,
        exit_rule=_trend_exit,
    ),
    StrategyVariant(
        name="TREND_MACD",
        description=(
            "Trend filters plus positive MACD."
        ),
        entry_rule=_trend_macd_entry,
        exit_rule=_trend_macd_exit,
    ),
    StrategyVariant(
        name="TREND_RSI",
        description=(
            "Trend filters plus RSI between 45 and 70."
        ),
        entry_rule=_trend_rsi_entry,
        exit_rule=_trend_rsi_exit,
    ),
    StrategyVariant(
        name="FULL_TECHNICAL",
        description=(
            "Trend, MACD, and RSI combined."
        ),
        entry_rule=_full_technical_entry,
        exit_rule=_full_technical_exit,
    ),
)


def _technical_score(
    *,
    row: pd.Series,
    variant: StrategyVariant,
) -> int:
    """Create a compact score for trade explainability."""

    score = 0

    ema20 = float(
        row["EMA20"]
    )

    ema50 = float(
        row["EMA50"]
    )

    close = float(
        row["Close"]
    )

    macd = float(
        row["MACD"]
    )

    rsi14 = float(
        row["RSI14"]
    )

    if ema20 > ema50:
        score += 30

    if close > ema20:
        score += 20

    if (
        variant.name
        in {
            "TREND_MACD",
            "FULL_TECHNICAL",
        }
        and macd > 0
    ):
        score += 25

    if (
        variant.name
        in {
            "TREND_RSI",
            "FULL_TECHNICAL",
        }
        and 45 <= rsi14 <= 70
    ):
        score += 25

    if variant.name == "TREND_ONLY":
        score *= 2

    return min(
        int(score),
        100,
    )


def create_signal_provider(
    variant: StrategyVariant,
) -> Callable[
    [pd.DataFrame, int, str],
    BacktestSignal | None,
]:
    """Create a causal signal provider for one variant."""

    def provider(
        visible_data: pd.DataFrame,
        bar_index: int,
        ticker: str,
    ) -> BacktestSignal | None:
        del bar_index

        if len(visible_data) < 2:
            return None

        current_row = (
            visible_data.iloc[-1]
        )

        previous_row = (
            visible_data.iloc[-2]
        )

        current_entry = (
            variant.entry_rule(
                current_row
            )
        )

        previous_entry = (
            variant.entry_rule(
                previous_row
            )
        )

        current_exit = (
            variant.exit_rule(
                current_row
            )
        )

        previous_exit = (
            variant.exit_rule(
                previous_row
            )
        )

        timestamp = (
            _timestamp_to_string(
                visible_data.index[-1]
            )
        )

        close_price = float(
            current_row["Close"]
        )

        score = _technical_score(
            row=current_row,
            variant=variant,
        )

        if (
            current_entry
            and not previous_entry
        ):
            return BacktestSignal(
                timestamp=timestamp,
                ticker=ticker,
                action="BUY",
                reference_price=close_price,
                overall_score=float(
                    score
                ),
                technical_score=float(
                    score
                ),
                confidence=float(
                    score
                ),
                reason=(
                    f"{variant.name} entry: "
                    f"{variant.description}"
                ),
            )

        if (
            current_exit
            and not previous_exit
        ):
            return BacktestSignal(
                timestamp=timestamp,
                ticker=ticker,
                action="EXIT",
                reference_price=close_price,
                overall_score=float(
                    score
                ),
                technical_score=float(
                    score
                ),
                confidence=float(
                    score
                ),
                reason=(
                    f"{variant.name} exit condition."
                ),
            )

        return None

    return provider


def _calculate_trade_statistics(
    result: BacktestResult,
) -> dict[str, float | int]:
    """Calculate trade-level strategy statistics."""

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
        "completed_trades": (
            completed_trades
        ),
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
            if isfinite(
                profit_factor
            )
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
    variant: str,
    error: Exception | str,
) -> AblationRow:
    """Create a standardized failed test row."""

    return AblationRow(
        ticker=ticker,
        variant=variant,
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


def _create_result_row(
    *,
    ticker: str,
    variant: StrategyVariant,
    prepared_bars: int,
    result: BacktestResult,
    benchmark: BuyAndHoldResult,
) -> AblationRow:
    """Convert one completed simulation into an ablation row."""

    statistics = (
        _calculate_trade_statistics(
            result
        )
    )

    excess_return_amount = (
        result.total_return_amount
        - benchmark.total_return_amount
    )

    excess_return_percent = (
        result.total_return_percent
        - benchmark.total_return_percent
    )

    return AblationRow(
        ticker=ticker,
        variant=variant.name,
        success=True,
        error="",
        prepared_bars=prepared_bars,
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
                result.total_return_amount
            )
        ),
        strategy_return_percent=(
            _round_percent(
                result.total_return_percent
            )
        ),
        strategy_max_drawdown_percent=(
            _round_percent(
                result.maximum_drawdown_percent
            )
        ),
        benchmark_return_amount=(
            _round_money(
                benchmark.total_return_amount
            )
        ),
        benchmark_return_percent=(
            _round_percent(
                benchmark.total_return_percent
            )
        ),
        benchmark_max_drawdown_percent=(
            _round_percent(
                benchmark.maximum_drawdown_percent
            )
        ),
        excess_return_amount=(
            _round_money(
                excess_return_amount
            )
        ),
        excess_return_percent=(
            _round_percent(
                excess_return_percent
            )
        ),
        strategy_outperformed=(
            excess_return_amount > 0
        ),
    )


def run_ablation_tests(
    *,
    tickers: list[str] | tuple[str, ...],
    period: str,
    base_config: BacktestConfig,
    fractional_crypto: bool = True,
    stop_on_error: bool = False,
) -> list[AblationRow]:
    """Run every technical variant across every ticker."""

    normalized_tickers = (
        _normalize_tickers(
            tickers
        )
    )

    rows: list[AblationRow] = []

    print()
    print("=" * 116)
    print("AI STOCK RADAR — TECHNICAL ABLATION TEST")
    print("=" * 116)

    print(
        f"Ticker count:       "
        f"{len(normalized_tickers)}"
    )

    print(
        f"Variant count:      "
        f"{len(STRATEGY_VARIANTS)}"
    )

    print(
        f"Historical period:  "
        f"{period}"
    )

    print(
        f"Stop / target:      "
        f"{base_config.stop_loss_percent:.2f}% / "
        f"{base_config.take_profit_percent:.2f}%"
    )

    print("=" * 116)

    for ticker_index, ticker in enumerate(
        normalized_tickers,
        start=1,
    ):
        print()
        print(
            f"[{ticker_index}/{len(normalized_tickers)}] "
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
                f"  PREPARATION FAILED: {error}"
            )

            for variant in STRATEGY_VARIANTS:
                rows.append(
                    _create_failed_row(
                        ticker=ticker,
                        variant=variant.name,
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

        for variant in STRATEGY_VARIANTS:
            print(
                f"  Running "
                f"{variant.name:<16}",
                end="",
            )

            try:
                result = run_backtest(
                    ticker=ticker,
                    data=prepared_data,
                    signal_provider=(
                        create_signal_provider(
                            variant
                        )
                    ),
                    config=ticker_config,
                )

                row = _create_result_row(
                    ticker=ticker,
                    variant=variant,
                    prepared_bars=len(
                        prepared_data
                    ),
                    result=result,
                    benchmark=benchmark,
                )

            except Exception as error:
                row = _create_failed_row(
                    ticker=ticker,
                    variant=variant.name,
                    error=error,
                )

                print(
                    f" FAILED: {error}"
                )

                rows.append(row)

                if stop_on_error:
                    raise

                continue

            rows.append(row)

            print(
                f" Trades={row.completed_trades:<3} "
                f"Return={row.strategy_return_percent:>+8.2f}% "
                f"Excess={row.excess_return_percent:>+8.2f}% "
                f"DD={row.strategy_max_drawdown_percent:>6.2f}%"
            )

    return rows


def _safe_average(
    values: list[float],
) -> float:
    """Return a safe arithmetic average."""

    if not values:
        return 0.0

    return sum(
        values
    ) / len(
        values
    )


def _safe_median(
    values: list[float],
) -> float:
    """Return a safe median."""

    if not values:
        return 0.0

    ordered = sorted(
        values
    )

    size = len(
        ordered
    )

    middle = (
        size // 2
    )

    if size % 2 == 1:
        return float(
            ordered[middle]
        )

    return (
        ordered[
            middle - 1
        ]
        + ordered[
            middle
        ]
    ) / 2


def summarize_variants(
    rows: list[AblationRow],
) -> list[VariantSummary]:
    """Aggregate performance by strategy variant."""

    summaries: list[
        VariantSummary
    ] = []

    for variant in STRATEGY_VARIANTS:
        variant_rows = [
            row
            for row in rows
            if row.variant
            == variant.name
        ]

        successful = [
            row
            for row in variant_rows
            if row.success
        ]

        failed = [
            row
            for row in variant_rows
            if not row.success
        ]

        outperformers = [
            row
            for row in successful
            if row.strategy_outperformed
        ]

        finite_profit_factors = [
            row.profit_factor
            for row in successful
            if isfinite(
                row.profit_factor
            )
        ]

        if successful:
            outperformed_percent = (
                len(outperformers)
                / len(successful)
                * 100
            )

        else:
            outperformed_percent = 0.0

        summary = VariantSummary(
            variant=variant.name,
            successful_tickers=len(
                successful
            ),
            failed_tickers=len(
                failed
            ),
            total_trades=sum(
                row.completed_trades
                for row in successful
            ),
            total_winning_trades=sum(
                row.winning_trades
                for row in successful
            ),
            total_losing_trades=sum(
                row.losing_trades
                for row in successful
            ),
            outperformed_count=len(
                outperformers
            ),
            outperformed_percent=(
                _round_percent(
                    outperformed_percent
                )
            ),
            average_strategy_return_percent=(
                _round_percent(
                    _safe_average(
                        [
                            row.strategy_return_percent
                            for row in successful
                        ]
                    )
                )
            ),
            average_benchmark_return_percent=(
                _round_percent(
                    _safe_average(
                        [
                            row.benchmark_return_percent
                            for row in successful
                        ]
                    )
                )
            ),
            average_excess_return_percent=(
                _round_percent(
                    _safe_average(
                        [
                            row.excess_return_percent
                            for row in successful
                        ]
                    )
                )
            ),
            median_strategy_return_percent=(
                _round_percent(
                    _safe_median(
                        [
                            row.strategy_return_percent
                            for row in successful
                        ]
                    )
                )
            ),
            median_excess_return_percent=(
                _round_percent(
                    _safe_median(
                        [
                            row.excess_return_percent
                            for row in successful
                        ]
                    )
                )
            ),
            average_max_drawdown_percent=(
                _round_percent(
                    _safe_average(
                        [
                            row.strategy_max_drawdown_percent
                            for row in successful
                        ]
                    )
                )
            ),
            average_benchmark_drawdown_percent=(
                _round_percent(
                    _safe_average(
                        [
                            row.benchmark_max_drawdown_percent
                            for row in successful
                        ]
                    )
                )
            ),
            average_profit_factor=(
                round(
                    _safe_average(
                        finite_profit_factors
                    ),
                    4,
                )
            ),
            average_win_rate_percent=(
                _round_percent(
                    _safe_average(
                        [
                            row.win_rate_percent
                            for row in successful
                        ]
                    )
                )
            ),
        )

        summaries.append(
            summary
        )

    summaries.sort(
        key=lambda item: (
            item.average_excess_return_percent,
            -item.average_max_drawdown_percent,
        ),
        reverse=True,
    )

    return summaries


def _format_profit_factor(
    value: float,
) -> str:
    """Format finite and infinite ratios."""

    if not isfinite(
        value
    ):
        return "INF"

    return f"{value:.2f}"


def print_variant_summary(
    summaries: list[VariantSummary],
) -> None:
    """Print aggregate strategy comparison."""

    print()
    print("=" * 136)
    print("TECHNICAL ABLATION SUMMARY")
    print("=" * 136)

    print(
        f"{'Variant':<18}"
        f"{'Tickers':>9}"
        f"{'Trades':>9}"
        f"{'Win %':>10}"
        f"{'Avg PF':>10}"
        f"{'Avg Ret %':>12}"
        f"{'Med Ret %':>12}"
        f"{'Avg Excess':>13}"
        f"{'Med Excess':>13}"
        f"{'Avg DD %':>11}"
        f"{'Beat':>12}"
    )

    print("-" * 136)

    for summary in summaries:
        beat_text = (
            f"{summary.outperformed_count}/"
            f"{summary.successful_tickers}"
        )

        print(
            f"{summary.variant:<18}"
            f"{summary.successful_tickers:>9}"
            f"{summary.total_trades:>9}"
            f"{summary.average_win_rate_percent:>10.2f}"
            f"{_format_profit_factor(summary.average_profit_factor):>10}"
            f"{summary.average_strategy_return_percent:>12.2f}"
            f"{summary.median_strategy_return_percent:>12.2f}"
            f"{summary.average_excess_return_percent:>13.2f}"
            f"{summary.median_excess_return_percent:>13.2f}"
            f"{summary.average_max_drawdown_percent:>11.2f}"
            f"{beat_text:>12}"
        )

    print("=" * 136)

    if summaries:
        best = summaries[0]

        print()
        print(
            f"Best average excess-return variant: "
            f"{best.variant}"
        )

        print(
            f"Average excess return: "
            f"{best.average_excess_return_percent:+.4f}%"
        )

        print(
            f"Average drawdown: "
            f"{best.average_max_drawdown_percent:.4f}%"
        )


def print_ticker_details(
    rows: list[AblationRow],
) -> None:
    """Print detailed ticker/variant results."""

    successful = [
        row
        for row in rows
        if row.success
    ]

    ranked = sorted(
        successful,
        key=lambda row: (
            row.ticker,
            -row.excess_return_percent,
        ),
    )

    print()
    print("=" * 142)
    print("ABLATION DETAILS")
    print("=" * 142)

    print(
        f"{'Ticker':<12}"
        f"{'Variant':<18}"
        f"{'Trades':>8}"
        f"{'Win %':>9}"
        f"{'PF':>8}"
        f"{'Return %':>11}"
        f"{'Benchmark':>12}"
        f"{'Excess %':>11}"
        f"{'Strat DD':>10}"
        f"{'Bench DD':>10}"
        f"{'Beat':>7}"
    )

    print("-" * 142)

    for row in ranked:
        print(
            f"{row.ticker:<12}"
            f"{row.variant:<18}"
            f"{row.completed_trades:>8}"
            f"{row.win_rate_percent:>9.2f}"
            f"{_format_profit_factor(row.profit_factor):>8}"
            f"{row.strategy_return_percent:>11.2f}"
            f"{row.benchmark_return_percent:>12.2f}"
            f"{row.excess_return_percent:>11.2f}"
            f"{row.strategy_max_drawdown_percent:>10.2f}"
            f"{row.benchmark_max_drawdown_percent:>10.2f}"
            f"{('YES' if row.strategy_outperformed else 'NO'):>7}"
        )

    failed = [
        row
        for row in rows
        if not row.success
    ]

    if failed:
        print()
        print("FAILED TESTS")

        for row in failed:
            print(
                f"- {row.ticker} / "
                f"{row.variant}: "
                f"{row.error}"
            )

    print("=" * 142)


def _json_safe_value(
    value: object,
) -> object:
    """Convert non-finite floats for JSON output."""

    if isinstance(
        value,
        float,
    ):
        if not isfinite(
            value
        ):
            return None

    return value


def _json_safe_dict(
    payload: dict[str, object],
) -> dict[str, object]:
    """Create a JSON-safe dictionary."""

    return {
        key: _json_safe_value(
            value
        )
        for key, value in payload.items()
    }


def save_ablation_results(
    *,
    rows: list[AblationRow],
    summaries: list[VariantSummary],
) -> dict[str, Path]:
    """Save detailed and aggregate ablation results."""

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        UTC
    ).strftime(
        "%Y%m%d_%H%M%S"
    )

    details_csv_path = (
        OUTPUT_DIRECTORY
        / f"ablation_details_{timestamp}.csv"
    )

    summary_csv_path = (
        OUTPUT_DIRECTORY
        / f"ablation_summary_{timestamp}.csv"
    )

    json_path = (
        OUTPUT_DIRECTORY
        / f"ablation_{timestamp}.json"
    )

    details_frame = pd.DataFrame(
        [
            asdict(
                row
            )
            for row in rows
        ]
    )

    summary_frame = pd.DataFrame(
        [
            asdict(
                summary
            )
            for summary in summaries
        ]
    )

    details_frame.to_csv(
        details_csv_path,
        index=False,
    )

    summary_frame.to_csv(
        summary_csv_path,
        index=False,
    )

    payload = {
        "created_at": (
            datetime.now(
                UTC
            ).isoformat()
        ),
        "variants": [
            {
                "name": variant.name,
                "description": (
                    variant.description
                ),
            }
            for variant in STRATEGY_VARIANTS
        ],
        "summaries": [
            _json_safe_dict(
                asdict(
                    summary
                )
            )
            for summary in summaries
        ],
        "details": [
            _json_safe_dict(
                asdict(
                    row
                )
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
        "details_csv": (
            details_csv_path
        ),
        "summary_csv": (
            summary_csv_path
        ),
        "json": json_path,
    }


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare deterministic technical "
            "strategy components."
        )
    )

    parser.add_argument(
        "tickers",
        nargs="*",
        help=(
            "Ticker list. When omitted, the "
            "default development universe is used."
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
    """Run the complete technical ablation study."""

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

    rows = run_ablation_tests(
        tickers=tickers,
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

    summaries = summarize_variants(
        rows
    )

    print_variant_summary(
        summaries
    )

    if arguments.show_details:
        print_ticker_details(
            rows
        )

    if not arguments.no_save:
        paths = save_ablation_results(
            rows=rows,
            summaries=summaries,
        )

        print()
        print("=" * 116)
        print("ABLATION FILES")
        print("=" * 116)

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

        print("=" * 116)

    print()
    print(
        "Technical ablation test "
        "completed successfully."
    )


if __name__ == "__main__":
    main()