"""Exit-strategy ablation for AI-Stock-Radar.

All variants use the same TREND_RSI entry:
- EMA20 > EMA50
- Close > EMA20
- RSI14 between 45 and 70
- BUY only when the setup newly becomes valid

Exit variants:
- FIXED_TARGET_10: 5% stop, 10% target, EMA/RSI exit
- TREND_EXIT_ONLY: 5% stop, no practical target, EMA/RSI exit
- WIDE_TARGET_20: 5% stop, 20% target, EMA/RSI exit
- TRAILING_CLOSE_10: 5% initial stop, no practical target,
  exit when Close is 10% below the highest Close since entry or
  EMA20 falls below EMA50. The signal fills at the next Open.

TRAILING_CLOSE_10 is deliberately close-based. The current shared
engine has no native intrabar trailing-stop order, so this module does
not pretend that an intrabar trailing fill occurred.
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
DISABLED_TARGET_PERCENT = 1_000_000.0

OUTPUT_DIRECTORY = (
    Path("data")
    / "backtests"
    / "exit_ablation"
)


@dataclass(frozen=True, slots=True)
class ExitVariant:
    """Definition of one deterministic exit method."""

    name: str
    description: str
    take_profit_percent: float
    use_trend_rsi_exit: bool
    trailing_close_percent: float | None = None


@dataclass(frozen=True, slots=True)
class ExitAblationRow:
    """One exit-variant result for one ticker."""

    ticker: str
    asset_class: str
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
class ExitVariantSummary:
    """Aggregate result for one exit method."""

    variant: str

    successful_tickers: int
    failed_tickers: int

    total_trades: int
    total_winning_trades: int
    total_losing_trades: int

    outperformed_count: int
    outperformed_percent: float

    profitable_tickers: int
    profitable_tickers_percent: float

    average_strategy_return_percent: float
    median_strategy_return_percent: float

    average_benchmark_return_percent: float
    average_excess_return_percent: float
    median_excess_return_percent: float

    average_max_drawdown_percent: float
    average_benchmark_drawdown_percent: float

    average_profit_factor: float
    average_win_rate_percent: float


EXIT_VARIANTS = (
    ExitVariant(
        name="FIXED_TARGET_10",
        description=(
            "5% stop, 10% target, "
            "EMA20/EMA50 or RSI>=75 exit."
        ),
        take_profit_percent=10.0,
        use_trend_rsi_exit=True,
    ),
    ExitVariant(
        name="TREND_EXIT_ONLY",
        description=(
            "5% stop, no practical target, "
            "EMA20/EMA50 or RSI>=75 exit."
        ),
        take_profit_percent=(
            DISABLED_TARGET_PERCENT
        ),
        use_trend_rsi_exit=True,
    ),
    ExitVariant(
        name="WIDE_TARGET_20",
        description=(
            "5% stop, 20% target, "
            "EMA20/EMA50 or RSI>=75 exit."
        ),
        take_profit_percent=20.0,
        use_trend_rsi_exit=True,
    ),
    ExitVariant(
        name="TRAILING_CLOSE_10",
        description=(
            "5% initial stop, no fixed target, "
            "highest-Close trailing exit."
        ),
        take_profit_percent=(
            DISABLED_TARGET_PERCENT
        ),
        use_trend_rsi_exit=False,
        trailing_close_percent=10.0,
    ),
)


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
    """Convert an index value to a stable timestamp."""

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    return str(value)


def _normalize_tickers(
    tickers: list[str] | tuple[str, ...],
) -> list[str]:
    """Normalize symbols while preserving order."""

    normalized: list[str] = []
    seen: set[str] = set()

    for ticker in tickers:
        symbol = (
            ticker.strip()
            .upper()
        )

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


def _is_crypto_ticker(
    ticker: str,
) -> bool:
    """Return whether the symbol is cryptocurrency."""

    symbol = ticker.upper()

    return symbol.endswith(
        (
            "-USD",
            "-EUR",
            "-GBP",
        )
    )


def _asset_class(
    ticker: str,
) -> str:
    """Return a compact asset-class name."""

    if _is_crypto_ticker(ticker):
        return "CRYPTO"

    return "EQUITY"


def _trend_rsi_entry(
    row: pd.Series,
) -> bool:
    """Evaluate the shared TREND_RSI entry."""

    return (
        float(row["EMA20"])
        > float(row["EMA50"])
        and float(row["Close"])
        > float(row["EMA20"])
        and 45.0
        <= float(row["RSI14"])
        <= 70.0
    )


def _trend_rsi_exit(
    row: pd.Series,
) -> bool:
    """Evaluate the baseline EMA/RSI exit."""

    return (
        float(row["EMA20"])
        < float(row["EMA50"])
        or float(row["RSI14"])
        >= 75.0
    )


def _apply_buy_slippage(
    raw_price: float,
    config: BacktestConfig,
) -> float:
    """Replicate the engine's BUY slippage."""

    return round(
        raw_price
        * (
            1.0
            + config.slippage_bps
            / 10_000.0
        ),
        8,
    )


class ExitSignalProvider:
    """Stateful causal provider for one exit-ablation run.

    The provider mirrors only enough position state to know whether
    it should produce BUY or EXIT. The shared backtest engine remains
    responsible for fills, fees, sizing, PnL, and account statistics.
    """

    def __init__(
        self,
        *,
        variant: ExitVariant,
        config: BacktestConfig,
    ) -> None:
        self.variant = variant
        self.config = config

        self.in_position = False
        self.pending_buy = False
        self.pending_exit = False

        self.stop_price: float | None = None
        self.target_price: float | None = None
        self.highest_close: float | None = None

        self.last_bar_index = -1

    def _reset_position(
        self,
    ) -> None:
        """Clear mirrored position state."""

        self.in_position = False
        self.pending_exit = False

        self.stop_price = None
        self.target_price = None
        self.highest_close = None

    def _open_at_current_bar(
        self,
        row: pd.Series,
    ) -> None:
        """Mirror a pending BUY filled at Open."""

        entry_price = (
            _apply_buy_slippage(
                float(row["Open"]),
                self.config,
            )
        )

        self.in_position = True
        self.pending_buy = False

        self.stop_price = (
            entry_price
            * (
                1.0
                - self.config.stop_loss_percent
                / 100.0
            )
        )

        self.target_price = (
            entry_price
            * (
                1.0
                + self.config.take_profit_percent
                / 100.0
            )
        )

        self.highest_close = float(
            row["Close"]
        )

    def _protective_exit_was_triggered(
        self,
        row: pd.Series,
    ) -> bool:
        """Mirror engine gap and intrabar protection."""

        if (
            not self.in_position
            or self.stop_price is None
            or self.target_price is None
        ):
            return False

        open_price = float(
            row["Open"]
        )

        high_price = float(
            row["High"]
        )

        low_price = float(
            row["Low"]
        )

        if (
            open_price
            <= self.stop_price
            or open_price
            >= self.target_price
        ):
            self._reset_position()
            return True

        if (
            low_price
            <= self.stop_price
            or high_price
            >= self.target_price
        ):
            self._reset_position()
            return True

        return False

    def _synchronize_bar(
        self,
        row: pd.Series,
    ) -> None:
        """Mirror the engine lifecycle through the current bar."""

        # Engine order:
        # 1. Overnight protection
        # 2. Pending EXIT
        # 3. Pending BUY
        # 4. Current-bar High/Low protection

        if self.in_position:
            self._protective_exit_was_triggered(
                row
            )

        if self.pending_exit:
            if self.in_position:
                self._reset_position()
            else:
                self.pending_exit = False

        if self.pending_buy:
            if self.in_position:
                self.pending_buy = False
            else:
                self._open_at_current_bar(
                    row
                )

        if self.in_position:
            self._protective_exit_was_triggered(
                row
            )

        if self.in_position:
            close_price = float(
                row["Close"]
            )

            if self.highest_close is None:
                self.highest_close = close_price
            else:
                self.highest_close = max(
                    self.highest_close,
                    close_price,
                )

    def _trailing_exit_is_valid(
        self,
        row: pd.Series,
    ) -> bool:
        """Evaluate the highest-Close trailing exit."""

        distance = (
            self.variant
            .trailing_close_percent
        )

        if (
            distance is None
            or self.highest_close is None
        ):
            return False

        trailing_level = (
            self.highest_close
            * (
                1.0
                - distance
                / 100.0
            )
        )

        return (
            float(row["Close"])
            <= trailing_level
        )

    def __call__(
        self,
        visible_data: pd.DataFrame,
        bar_index: int,
        ticker: str,
    ) -> BacktestSignal | None:
        """Generate a causal BUY or EXIT signal."""

        if bar_index <= self.last_bar_index:
            raise RuntimeError(
                "ExitSignalProvider requires "
                "sequential bars."
            )

        self.last_bar_index = bar_index

        current_row = (
            visible_data.iloc[-1]
        )

        self._synchronize_bar(
            current_row
        )

        timestamp = (
            _timestamp_to_string(
                visible_data.index[-1]
            )
        )

        close_price = float(
            current_row["Close"]
        )

        if (
            self.in_position
            and not self.pending_exit
        ):
            exit_valid = False
            exit_reason = ""

            if (
                self.variant
                .use_trend_rsi_exit
                and _trend_rsi_exit(
                    current_row
                )
            ):
                exit_valid = True
                exit_reason = (
                    "EMA20 below EMA50 or "
                    "RSI14 at/above 75."
                )

            elif not (
                self.variant
                .use_trend_rsi_exit
            ):
                if (
                    float(
                        current_row["EMA20"]
                    )
                    < float(
                        current_row["EMA50"]
                    )
                ):
                    exit_valid = True
                    exit_reason = (
                        "EMA20 below EMA50."
                    )

                elif self._trailing_exit_is_valid(
                    current_row
                ):
                    exit_valid = True

                    exit_reason = (
                        f"Close fell "
                        f"{self.variant.trailing_close_percent:.2f}% "
                        f"below the highest Close since entry."
                    )

            if exit_valid:
                self.pending_exit = True

                return BacktestSignal(
                    timestamp=timestamp,
                    ticker=ticker,
                    action="EXIT",
                    reference_price=close_price,
                    confidence=100.0,
                    reason=(
                        f"{self.variant.name} exit: "
                        f"{exit_reason}"
                    ),
                )

        if (
            not self.in_position
            and not self.pending_buy
            and len(visible_data) >= 2
        ):
            previous_row = (
                visible_data.iloc[-2]
            )

            current_entry = (
                _trend_rsi_entry(
                    current_row
                )
            )

            previous_entry = (
                _trend_rsi_entry(
                    previous_row
                )
            )

            if (
                current_entry
                and not previous_entry
            ):
                self.pending_buy = True

                return BacktestSignal(
                    timestamp=timestamp,
                    ticker=ticker,
                    action="BUY",
                    reference_price=close_price,
                    overall_score=80.0,
                    technical_score=80.0,
                    confidence=80.0,
                    reason=(
                        "TREND_RSI entry: "
                        "EMA20 above EMA50, "
                        "Close above EMA20, "
                        "RSI14 between 45 and 70."
                    ),
                )

        return None


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
    variant: str,
    error: Exception | str,
) -> ExitAblationRow:
    """Create a standardized failed row."""

    return ExitAblationRow(
        ticker=ticker,
        asset_class=_asset_class(
            ticker
        ),
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


def _success_row(
    *,
    ticker: str,
    variant: ExitVariant,
    prepared_bars: int,
    result: BacktestResult,
    benchmark: BuyAndHoldResult,
) -> ExitAblationRow:
    """Create a successful result row."""

    excess_amount = (
        result.total_return_amount
        - benchmark.total_return_amount
    )

    excess_percent = (
        result.total_return_percent
        - benchmark.total_return_percent
    )

    return ExitAblationRow(
        ticker=ticker,
        asset_class=_asset_class(
            ticker
        ),
        variant=variant.name,
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
            _round_percent(
                result.win_rate_percent
            )
        ),
        profit_factor=(
            round(
                result.profit_factor,
                4,
            )
            if isfinite(
                result.profit_factor
            )
            else inf
        ),
        average_trade=(
            _average_trade(result)
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
                excess_amount
            )
        ),
        excess_return_percent=(
            _round_percent(
                excess_percent
            )
        ),
        strategy_outperformed=(
            excess_amount > 0
        ),
    )


def _active_variants(
    trailing_close_percent: float,
) -> tuple[ExitVariant, ...]:
    """Apply the requested trailing distance."""

    return tuple(
        replace(
            variant,
            trailing_close_percent=(
                trailing_close_percent
                if variant.name
                == "TRAILING_CLOSE_10"
                else variant
                .trailing_close_percent
            ),
        )
        for variant in EXIT_VARIANTS
    )


def run_exit_ablation(
    *,
    tickers: list[str] | tuple[str, ...],
    period: str,
    base_config: BacktestConfig,
    trailing_close_percent: float,
    fractional_crypto: bool = True,
    stop_on_error: bool = False,
) -> list[ExitAblationRow]:
    """Run every exit method across every ticker."""

    symbols = _normalize_tickers(
        tickers
    )

    variants = _active_variants(
        trailing_close_percent
    )

    rows: list[
        ExitAblationRow
    ] = []

    print()
    print("=" * 120)
    print("AI STOCK RADAR — EXIT ABLATION")
    print("=" * 120)

    print(
        f"Ticker count:              "
        f"{len(symbols)}"
    )

    print(
        f"Variant count:             "
        f"{len(variants)}"
    )

    print(
        f"Historical period:         "
        f"{period}"
    )

    print(
        f"Stop loss:                 "
        f"{base_config.stop_loss_percent:.2f}%"
    )

    print(
        f"Trailing close distance:   "
        f"{trailing_close_percent:.2f}%"
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
        )

        try:
            downloaded = (
                download_stock_data(
                    ticker,
                    period=period,
                )
            )

            prepared = (
                _prepare_market_data(
                    downloaded
                )
            )

            benchmark = (
                run_buy_and_hold_benchmark(
                    ticker=ticker,
                    data=prepared,
                    config=ticker_config,
                )
            )

        except Exception as error:
            print(
                f"  PREPARATION FAILED: "
                f"{error}"
            )

            rows.extend(
                _failed_row(
                    ticker=ticker,
                    variant=variant.name,
                    error=error,
                )
                for variant in variants
            )

            if stop_on_error:
                raise

            continue

        print(
            f"  Prepared bars: "
            f"{len(prepared)}"
        )

        for variant in variants:
            variant_config = replace(
                ticker_config,
                take_profit_percent=(
                    variant.take_profit_percent
                ),
            )

            provider = ExitSignalProvider(
                variant=variant,
                config=variant_config,
            )

            print(
                f"  Running "
                f"{variant.name:<20}",
                end="",
            )

            try:
                result = run_backtest(
                    ticker=ticker,
                    data=prepared,
                    signal_provider=provider,
                    config=variant_config,
                )

                row = _success_row(
                    ticker=ticker,
                    variant=variant,
                    prepared_bars=len(
                        prepared
                    ),
                    result=result,
                    benchmark=benchmark,
                )

            except Exception as error:
                row = _failed_row(
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
                f" Trades="
                f"{row.completed_trades:<3}"
                f" Return="
                f"{row.strategy_return_percent:>+8.2f}%"
                f" PF="
                f"{_format_pf(row.profit_factor):>6}"
                f" Excess="
                f"{row.excess_return_percent:>+8.2f}%"
                f" DD="
                f"{row.strategy_max_drawdown_percent:>6.2f}%"
            )

    return rows


def _safe_average(
    values: list[float],
) -> float:
    """Return a safe average."""

    if not values:
        return 0.0

    return (
        sum(values)
        / len(values)
    )


def _safe_median(
    values: list[float],
) -> float:
    """Return a safe median."""

    if not values:
        return 0.0

    ordered = sorted(values)
    middle = len(ordered) // 2

    if len(ordered) % 2:
        return float(
            ordered[middle]
        )

    return (
        ordered[middle - 1]
        + ordered[middle]
    ) / 2.0


def summarize_exit_variants(
    rows: list[ExitAblationRow],
) -> list[ExitVariantSummary]:
    """Aggregate results by exit method."""

    summaries: list[
        ExitVariantSummary
    ] = []

    for variant in EXIT_VARIANTS:
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

        profitable = [
            row
            for row in successful
            if row.strategy_return_percent > 0
        ]

        finite_profit_factors = [
            row.profit_factor
            for row in successful
            if isfinite(
                row.profit_factor
            )
        ]

        count = len(
            successful
        )

        summaries.append(
            ExitVariantSummary(
                variant=variant.name,
                successful_tickers=count,
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
                        len(outperformers)
                        / count
                        * 100
                        if count
                        else 0.0
                    )
                ),
                profitable_tickers=len(
                    profitable
                ),
                profitable_tickers_percent=(
                    _round_percent(
                        len(profitable)
                        / count
                        * 100
                        if count
                        else 0.0
                    )
                ),
                average_strategy_return_percent=(
                    _round_percent(
                        _safe_average(
                            [
                                row
                                .strategy_return_percent
                                for row
                                in successful
                            ]
                        )
                    )
                ),
                median_strategy_return_percent=(
                    _round_percent(
                        _safe_median(
                            [
                                row
                                .strategy_return_percent
                                for row
                                in successful
                            ]
                        )
                    )
                ),
                average_benchmark_return_percent=(
                    _round_percent(
                        _safe_average(
                            [
                                row
                                .benchmark_return_percent
                                for row
                                in successful
                            ]
                        )
                    )
                ),
                average_excess_return_percent=(
                    _round_percent(
                        _safe_average(
                            [
                                row
                                .excess_return_percent
                                for row
                                in successful
                            ]
                        )
                    )
                ),
                median_excess_return_percent=(
                    _round_percent(
                        _safe_median(
                            [
                                row
                                .excess_return_percent
                                for row
                                in successful
                            ]
                        )
                    )
                ),
                average_max_drawdown_percent=(
                    _round_percent(
                        _safe_average(
                            [
                                row
                                .strategy_max_drawdown_percent
                                for row
                                in successful
                            ]
                        )
                    )
                ),
                average_benchmark_drawdown_percent=(
                    _round_percent(
                        _safe_average(
                            [
                                row
                                .benchmark_max_drawdown_percent
                                for row
                                in successful
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
                                row
                                .win_rate_percent
                                for row
                                in successful
                            ]
                        )
                    )
                ),
            )
        )

    summaries.sort(
        key=lambda item: (
            item.average_excess_return_percent,
            item.average_strategy_return_percent,
            -item.average_max_drawdown_percent,
        ),
        reverse=True,
    )

    return summaries


def _format_pf(
    value: float,
) -> str:
    """Format a profit factor."""

    if not isfinite(value):
        return "INF"

    return f"{value:.2f}"


def print_exit_summary(
    summaries: list[ExitVariantSummary],
) -> None:
    """Print aggregate results."""

    print()
    print("=" * 150)
    print("EXIT ABLATION SUMMARY")
    print("=" * 150)

    print(
        f"{'Variant':<22}"
        f"{'Tickers':>9}"
        f"{'Trades':>9}"
        f"{'Win %':>9}"
        f"{'Avg PF':>9}"
        f"{'Avg Ret %':>12}"
        f"{'Med Ret %':>12}"
        f"{'Avg Excess':>13}"
        f"{'Med Excess':>13}"
        f"{'Avg DD %':>11}"
        f"{'Positive':>11}"
        f"{'Beat':>10}"
    )

    print("-" * 150)

    for item in summaries:
        positive_text = (
            f"{item.profitable_tickers}/"
            f"{item.successful_tickers}"
        )

        beat_text = (
            f"{item.outperformed_count}/"
            f"{item.successful_tickers}"
        )

        print(
            f"{item.variant:<22}"
            f"{item.successful_tickers:>9}"
            f"{item.total_trades:>9}"
            f"{item.average_win_rate_percent:>9.2f}"
            f"{_format_pf(item.average_profit_factor):>9}"
            f"{item.average_strategy_return_percent:>12.2f}"
            f"{item.median_strategy_return_percent:>12.2f}"
            f"{item.average_excess_return_percent:>13.2f}"
            f"{item.median_excess_return_percent:>13.2f}"
            f"{item.average_max_drawdown_percent:>11.2f}"
            f"{positive_text:>11}"
            f"{beat_text:>10}"
        )

    print("=" * 150)

    if summaries:
        best = summaries[0]

        print()
        print(
            "Best average excess-return exit: "
            f"{best.variant}"
        )

        print(
            "Average strategy return: "
            f"{best.average_strategy_return_percent:+.4f}%"
        )

        print(
            "Average excess return: "
            f"{best.average_excess_return_percent:+.4f}%"
        )

        print(
            "Average maximum drawdown: "
            f"{best.average_max_drawdown_percent:.4f}%"
        )


def print_exit_details(
    rows: list[ExitAblationRow],
) -> None:
    """Print per-ticker details."""

    successful = sorted(
        [
            row
            for row in rows
            if row.success
        ],
        key=lambda row: (
            row.ticker,
            -row.excess_return_percent,
        ),
    )

    print()
    print("=" * 153)
    print("EXIT ABLATION DETAILS")
    print("=" * 153)

    print(
        f"{'Ticker':<12}"
        f"{'Class':<9}"
        f"{'Variant':<22}"
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

    print("-" * 153)

    for row in successful:
        print(
            f"{row.ticker:<12}"
            f"{row.asset_class:<9}"
            f"{row.variant:<22}"
            f"{row.completed_trades:>8}"
            f"{row.win_rate_percent:>9.2f}"
            f"{_format_pf(row.profit_factor):>8}"
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

    print("=" * 153)


def _json_safe(
    value: Any,
) -> Any:
    """Convert non-finite floats to JSON-safe values."""

    if (
        isinstance(value, float)
        and not isfinite(value)
    ):
        return None

    return value


def _json_safe_dict(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Create a JSON-safe dictionary."""

    return {
        key: _json_safe(value)
        for key, value
        in payload.items()
    }


def save_exit_ablation_results(
    *,
    rows: list[ExitAblationRow],
    summaries: list[ExitVariantSummary],
    trailing_close_percent: float,
) -> dict[str, Path]:
    """Save CSV and JSON reports."""

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
        / f"exit_ablation_details_{timestamp}.csv"
    )

    summary_path = (
        OUTPUT_DIRECTORY
        / f"exit_ablation_summary_{timestamp}.csv"
    )

    json_path = (
        OUTPUT_DIRECTORY
        / f"exit_ablation_{timestamp}.json"
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
            asdict(item)
            for item in summaries
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
        "entry_rule": (
            "EMA20>EMA50, Close>EMA20, "
            "RSI14 45-70; new transition only."
        ),
        "trailing_close_percent": (
            trailing_close_percent
        ),
        "variants": [
            {
                "name": variant.name,
                "description": (
                    variant.description
                ),
                "take_profit_percent": (
                    None
                    if variant.take_profit_percent
                    == DISABLED_TARGET_PERCENT
                    else variant.take_profit_percent
                ),
                "trailing_close_percent": (
                    trailing_close_percent
                    if variant.name
                    == "TRAILING_CLOSE_10"
                    else variant
                    .trailing_close_percent
                ),
            }
            for variant in EXIT_VARIANTS
        ],
        "summaries": [
            _json_safe_dict(
                asdict(item)
            )
            for item in summaries
        ],
        "details": [
            _json_safe_dict(
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
            "Compare deterministic exits "
            "using the TREND_RSI entry rule."
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
        "--trailing-close",
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
    """Run the exit-ablation study."""

    arguments = _parse_arguments()

    if arguments.trailing_close <= 0:
        raise ValueError(
            "--trailing-close must be "
            "greater than zero."
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
        take_profit_percent=10.0,
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

    rows = run_exit_ablation(
        tickers=tickers,
        period=arguments.period,
        base_config=config,
        trailing_close_percent=(
            arguments.trailing_close
        ),
        fractional_crypto=(
            not arguments
            .no_fractional_crypto
        ),
        stop_on_error=(
            arguments.stop_on_error
        ),
    )

    summaries = (
        summarize_exit_variants(
            rows
        )
    )

    print_exit_summary(
        summaries
    )

    if arguments.show_details:
        print_exit_details(
            rows
        )

    if not arguments.no_save:
        paths = (
            save_exit_ablation_results(
                rows=rows,
                summaries=summaries,
                trailing_close_percent=(
                    arguments
                    .trailing_close
                ),
            )
        )

        print()
        print("=" * 120)
        print("EXIT ABLATION FILES")
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
        "Exit ablation completed successfully."
    )


if __name__ == "__main__":
    main()