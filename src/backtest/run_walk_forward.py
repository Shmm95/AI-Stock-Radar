"""Rolling walk-forward validation for AI-Stock-Radar.

This module compares two predetermined Close-based trailing exits:

- 7.5% baseline
- 15.0% challenger

Method:
1. Use a fixed rolling training window.
2. Test both trailing values across the complete ticker universe.
3. Select one global trailing value using training results only.
4. Apply the selected value to the immediately following unseen period.
5. Also evaluate both fixed candidates in every test period.
6. Repeat by moving forward through time.

Default windows:
- Training: 24 calendar months
- Testing: 6 calendar months
- Step: 6 calendar months

Important:
- The selected trailing value is global for the entire universe.
- There is no ticker-specific parameter optimization.
- Indicators are prepared causally from historical market data.
- Orders execute through the shared backtest engine.
- BUY and EXIT signals fill at the next bar Open.
- Every train/test run starts with a fresh account.
- This is strategy validation, not yet a portfolio backtest.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from math import inf, isfinite
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

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
    15.0,
)

DEFAULT_PERIOD = "5y"

DEFAULT_TRAIN_MONTHS = 24
DEFAULT_TEST_MONTHS = 6
DEFAULT_STEP_MONTHS = 6

DEFAULT_MINIMUM_TRAIN_BARS = 200
DEFAULT_MINIMUM_TEST_BARS = 40

OUTPUT_DIRECTORY = (
    Path("data")
    / "backtests"
    / "walk_forward"
)


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    """One rolling train and test period."""

    window_id: str

    train_start: pd.Timestamp
    train_end: pd.Timestamp

    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def to_dict(
        self,
    ) -> dict[str, str]:
        """Serialize the window using half-open date ranges."""

        return {
            "window_id": self.window_id,
            "train_start": (
                self.train_start.isoformat()
            ),
            "train_end_exclusive": (
                self.train_end.isoformat()
            ),
            "test_start": (
                self.test_start.isoformat()
            ),
            "test_end_exclusive": (
                self.test_end.isoformat()
            ),
        }


@dataclass(frozen=True, slots=True)
class WalkForwardRunRow:
    """One candidate, ticker, phase, and window result."""

    window_id: str
    phase: str

    ticker: str
    asset_class: str

    trailing_close_percent: float
    selected_for_window: bool

    success: bool
    error: str

    period_start: str
    period_end: str
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
class CandidateWindowSummary:
    """Aggregate metrics for one candidate inside one window."""

    window_id: str
    phase: str

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


@dataclass(frozen=True, slots=True)
class WalkForwardWindowSummary:
    """Selected candidate and out-of-sample performance."""

    window_id: str

    train_start: str
    train_end_exclusive: str

    test_start: str
    test_end_exclusive: str

    selected_trailing_close_percent: float

    train_selected_average_return_percent: float
    train_selected_median_return_percent: float
    train_selected_average_profit_factor: float
    train_selected_average_drawdown_percent: float
    train_selected_profitable_tickers: int

    test_selected_average_return_percent: float
    test_selected_median_return_percent: float
    test_selected_average_excess_return_percent: float
    test_selected_median_excess_return_percent: float
    test_selected_average_profit_factor: float
    test_selected_average_drawdown_percent: float

    test_selected_profitable_tickers: int
    test_selected_outperformed_count: int
    test_selected_total_trades: int


@dataclass(frozen=True, slots=True)
class WalkForwardAggregateSummary:
    """Aggregate out-of-sample result across all windows."""

    label: str

    successful_runs: int
    failed_runs: int

    windows: int
    tickers: int

    total_trades: int
    total_winning_trades: int
    total_losing_trades: int

    profitable_runs: int
    profitable_runs_percent: float

    outperformed_runs: int
    outperformed_runs_percent: float

    average_strategy_return_percent: float
    median_strategy_return_percent: float

    best_run_return_percent: float
    worst_run_return_percent: float

    average_excess_return_percent: float
    median_excess_return_percent: float

    average_max_drawdown_percent: float
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
    """Round percentages and ratios."""

    return round(
        float(value),
        4,
    )


def _safe_mean(
    values: Iterable[float],
) -> float:
    """Return a safe arithmetic mean."""

    items = list(values)

    if not items:
        return 0.0

    return float(
        mean(items)
    )


def _safe_median(
    values: Iterable[float],
) -> float:
    """Return a safe median."""

    items = list(values)

    if not items:
        return 0.0

    return float(
        median(items)
    )


def _normalize_timestamp(
    value: object,
) -> pd.Timestamp:
    """Convert a value to a timezone-naive Timestamp."""

    timestamp = pd.Timestamp(
        value
    )

    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(
            None
        )

    return timestamp


def _timestamp_to_string(
    value: object,
) -> str:
    """Convert an index value to a stable string."""

    return (
        _normalize_timestamp(value)
        .isoformat()
    )


def _is_crypto_ticker(
    ticker: str,
) -> bool:
    """Return whether a ticker represents cryptocurrency."""

    normalized = (
        ticker.strip()
        .upper()
    )

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
    """Return a compact asset-class name."""

    if _is_crypto_ticker(ticker):
        return "CRYPTO"

    return "EQUITY"


def _normalize_tickers(
    tickers: list[str] | tuple[str, ...],
) -> list[str]:
    """Normalize ticker symbols and remove duplicates."""

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


def normalize_candidate_values(
    values: list[float] | tuple[float, ...],
) -> list[float]:
    """Validate and normalize trailing candidates."""

    normalized: list[float] = []

    for raw_value in values:
        value = float(
            raw_value
        )

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
            normalized.append(
                rounded
            )

    normalized.sort()

    if len(normalized) < 2:
        raise ValueError(
            "Walk-forward validation requires "
            "at least two trailing candidates."
        )

    return normalized


def generate_walk_forward_windows(
    *,
    common_start: object,
    common_end: object,
    train_months: int,
    test_months: int,
    step_months: int,
) -> list[WalkForwardWindow]:
    """Generate fixed-length rolling walk-forward windows.

    The date intervals are half-open:

        train_start <= date < train_end
        test_start <= date < test_end
    """

    if train_months <= 0:
        raise ValueError(
            "train_months must be greater than zero."
        )

    if test_months <= 0:
        raise ValueError(
            "test_months must be greater than zero."
        )

    if step_months <= 0:
        raise ValueError(
            "step_months must be greater than zero."
        )

    start = (
        _normalize_timestamp(
            common_start
        )
        .normalize()
    )

    final_inclusive = (
        _normalize_timestamp(
            common_end
        )
        .normalize()
    )

    final_exclusive = (
        final_inclusive
        + pd.Timedelta(
            days=1
        )
    )

    if start >= final_exclusive:
        raise ValueError(
            "common_start must be before common_end."
        )

    windows: list[
        WalkForwardWindow
    ] = []

    cursor = start
    window_number = 1

    while True:
        train_start = cursor

        train_end = (
            train_start
            + pd.DateOffset(
                months=train_months
            )
        )

        test_start = train_end

        test_end = (
            test_start
            + pd.DateOffset(
                months=test_months
            )
        )

        if test_end > final_exclusive:
            break

        windows.append(
            WalkForwardWindow(
                window_id=(
                    f"W{window_number:02d}"
                ),
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

        cursor = (
            cursor
            + pd.DateOffset(
                months=step_months
            )
        )

        window_number += 1

    if not windows:
        raise ValueError(
            "The available date range is too short "
            "for the requested walk-forward windows."
        )

    return windows


def _normalize_prepared_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize a prepared DataFrame index."""

    normalized = data.copy()

    normalized.index = pd.to_datetime(
        normalized.index
    )

    if normalized.index.tz is not None:
        normalized.index = (
            normalized.index
            .tz_convert(None)
        )

    normalized = normalized.sort_index()

    normalized = normalized.loc[
        ~normalized.index.duplicated(
            keep="last"
        )
    ]

    return normalized


def _slice_data(
    data: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Return a half-open date slice."""

    return data.loc[
        (
            data.index >= start
        )
        & (
            data.index < end
        )
    ].copy()


def _create_trailing_variant(
    trailing_close_percent: float,
) -> ExitVariant:
    """Create one Close-based trailing variant."""

    formatted = (
        f"{trailing_close_percent:g}"
        .replace(
            ".",
            "_",
        )
    )

    return ExitVariant(
        name=(
            f"TRAILING_CLOSE_{formatted}"
        ),
        description=(
            "5% initial stop by default, "
            "no fixed target, "
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


def _failed_run_row(
    *,
    window_id: str,
    phase: str,
    ticker: str,
    trailing_close_percent: float,
    selected_for_window: bool,
    period_start: object,
    period_end: object,
    prepared_bars: int,
    error: Exception | str,
) -> WalkForwardRunRow:
    """Create a standardized failed row."""

    return WalkForwardRunRow(
        window_id=window_id,
        phase=phase,
        ticker=ticker,
        asset_class=_asset_class(
            ticker
        ),
        trailing_close_percent=(
            trailing_close_percent
        ),
        selected_for_window=(
            selected_for_window
        ),
        success=False,
        error=str(error),
        period_start=(
            _timestamp_to_string(
                period_start
            )
        ),
        period_end=(
            _timestamp_to_string(
                period_end
            )
        ),
        prepared_bars=prepared_bars,
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


def _successful_run_row(
    *,
    window_id: str,
    phase: str,
    ticker: str,
    trailing_close_percent: float,
    selected_for_window: bool,
    prepared_bars: int,
    result: BacktestResult,
    benchmark: BuyAndHoldResult,
) -> WalkForwardRunRow:
    """Create one successful train or test result."""

    excess_amount = (
        result.total_return_amount
        - benchmark.total_return_amount
    )

    excess_percent = (
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

    return WalkForwardRunRow(
        window_id=window_id,
        phase=phase,
        ticker=ticker,
        asset_class=_asset_class(
            ticker
        ),
        trailing_close_percent=(
            trailing_close_percent
        ),
        selected_for_window=(
            selected_for_window
        ),
        success=True,
        error="",
        period_start=(
            result.start_date
        ),
        period_end=(
            result.end_date
        ),
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
                excess_amount
            )
        ),
        excess_return_percent=(
            _round_metric(
                excess_percent
            )
        ),
        profitable=(
            result.total_return_amount > 0
        ),
        strategy_outperformed=(
            excess_amount > 0
        ),
    )


def _run_candidate(
    *,
    window_id: str,
    phase: str,
    ticker: str,
    data: pd.DataFrame,
    trailing_close_percent: float,
    config: BacktestConfig,
    benchmark: BuyAndHoldResult,
    selected_for_window: bool,
) -> WalkForwardRunRow:
    """Run one ticker and one candidate."""

    variant = _create_trailing_variant(
        trailing_close_percent
    )

    provider = ExitSignalProvider(
        variant=variant,
        config=config,
    )

    try:
        result = run_backtest(
            ticker=ticker,
            data=data,
            signal_provider=provider,
            config=config,
        )

    except Exception as error:
        return _failed_run_row(
            window_id=window_id,
            phase=phase,
            ticker=ticker,
            trailing_close_percent=(
                trailing_close_percent
            ),
            selected_for_window=(
                selected_for_window
            ),
            period_start=data.index[0],
            period_end=data.index[-1],
            prepared_bars=len(data),
            error=error,
        )

    return _successful_run_row(
        window_id=window_id,
        phase=phase,
        ticker=ticker,
        trailing_close_percent=(
            trailing_close_percent
        ),
        selected_for_window=(
            selected_for_window
        ),
        prepared_bars=len(data),
        result=result,
        benchmark=benchmark,
    )


def _summary_for_candidate(
    *,
    window_id: str,
    phase: str,
    trailing_close_percent: float,
    rows: list[WalkForwardRunRow],
) -> CandidateWindowSummary:
    """Aggregate one candidate inside one window and phase."""

    matching_rows = [
        row
        for row in rows
        if (
            row.window_id
            == window_id
            and row.phase
            == phase
            and row.trailing_close_percent
            == trailing_close_percent
        )
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

    drawdowns = [
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

    return CandidateWindowSummary(
        window_id=window_id,
        phase=phase,
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
                    drawdowns
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
                    row.win_rate_percent
                    for row in successful_rows
                )
            )
        ),
    )


def rank_candidate_summaries(
    summaries: list[CandidateWindowSummary],
) -> list[CandidateWindowSummary]:
    """Rank candidates using training robustness metrics.

    Ranking priority:
    1. Number of successful ticker tests
    2. Median strategy return
    3. Average strategy return
    4. Number of profitable tickers
    5. Average profit factor
    6. Lower average drawdown
    """

    ordered = sorted(
        summaries,
        key=lambda item: (
            item.successful_tickers,
            item.median_strategy_return_percent,
            item.average_strategy_return_percent,
            item.profitable_tickers,
            item.average_profit_factor,
            -item.average_max_drawdown_percent,
        ),
        reverse=True,
    )

    return [
        replace(
            item,
            rank=index,
        )
        for index, item in enumerate(
            ordered,
            start=1,
        )
    ]


def _summarize_window_candidates(
    *,
    window_id: str,
    phase: str,
    candidates: list[float],
    rows: list[WalkForwardRunRow],
) -> list[CandidateWindowSummary]:
    """Create ranked summaries for one phase and window."""

    raw_summaries = [
        _summary_for_candidate(
            window_id=window_id,
            phase=phase,
            trailing_close_percent=(
                trailing
            ),
            rows=rows,
        )
        for trailing in candidates
    ]

    return rank_candidate_summaries(
        raw_summaries
    )


def _selected_summary(
    summaries: list[CandidateWindowSummary],
    *,
    trailing_close_percent: float,
) -> CandidateWindowSummary:
    """Find one candidate summary."""

    for summary in summaries:
        if (
            summary.trailing_close_percent
            == trailing_close_percent
        ):
            return summary

    raise ValueError(
        "Selected candidate summary was not found."
    )


def _window_summary(
    *,
    window: WalkForwardWindow,
    selected_trailing: float,
    train_summaries: list[CandidateWindowSummary],
    test_summaries: list[CandidateWindowSummary],
) -> WalkForwardWindowSummary:
    """Create one selected out-of-sample window summary."""

    train_selected = _selected_summary(
        train_summaries,
        trailing_close_percent=(
            selected_trailing
        ),
    )

    test_selected = _selected_summary(
        test_summaries,
        trailing_close_percent=(
            selected_trailing
        ),
    )

    return WalkForwardWindowSummary(
        window_id=window.window_id,
        train_start=(
            window.train_start.isoformat()
        ),
        train_end_exclusive=(
            window.train_end.isoformat()
        ),
        test_start=(
            window.test_start.isoformat()
        ),
        test_end_exclusive=(
            window.test_end.isoformat()
        ),
        selected_trailing_close_percent=(
            selected_trailing
        ),
        train_selected_average_return_percent=(
            train_selected
            .average_strategy_return_percent
        ),
        train_selected_median_return_percent=(
            train_selected
            .median_strategy_return_percent
        ),
        train_selected_average_profit_factor=(
            train_selected
            .average_profit_factor
        ),
        train_selected_average_drawdown_percent=(
            train_selected
            .average_max_drawdown_percent
        ),
        train_selected_profitable_tickers=(
            train_selected
            .profitable_tickers
        ),
        test_selected_average_return_percent=(
            test_selected
            .average_strategy_return_percent
        ),
        test_selected_median_return_percent=(
            test_selected
            .median_strategy_return_percent
        ),
        test_selected_average_excess_return_percent=(
            test_selected
            .average_excess_return_percent
        ),
        test_selected_median_excess_return_percent=(
            test_selected
            .median_excess_return_percent
        ),
        test_selected_average_profit_factor=(
            test_selected
            .average_profit_factor
        ),
        test_selected_average_drawdown_percent=(
            test_selected
            .average_max_drawdown_percent
        ),
        test_selected_profitable_tickers=(
            test_selected
            .profitable_tickers
        ),
        test_selected_outperformed_count=(
            test_selected
            .outperformed_count
        ),
        test_selected_total_trades=(
            test_selected
            .total_trades
        ),
    )


def build_oos_aggregate(
    *,
    rows: list[WalkForwardRunRow],
    label: str,
    trailing_close_percent: float | None = None,
    selected_only: bool = False,
) -> WalkForwardAggregateSummary:
    """Aggregate out-of-sample rows.

    Use either:
    - trailing_close_percent for one fixed candidate
    - selected_only=True for the dynamically selected candidate
    """

    if (
        trailing_close_percent is None
        and not selected_only
    ):
        raise ValueError(
            "Provide a trailing value or selected_only=True."
        )

    matching_rows = [
        row
        for row in rows
        if (
            row.phase == "TEST"
            and (
                row.selected_for_window
                if selected_only
                else (
                    row.trailing_close_percent
                    == trailing_close_percent
                )
            )
        )
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

    returns = [
        row.strategy_return_percent
        for row in successful_rows
    ]

    excess_returns = [
        row.excess_return_percent
        for row in successful_rows
    ]

    drawdowns = [
        row.strategy_max_drawdown_percent
        for row in successful_rows
    ]

    finite_profit_factors = [
        row.profit_factor
        for row in successful_rows
        if isfinite(
            row.profit_factor
        )
    ]

    profitable_runs = sum(
        1
        for row in successful_rows
        if row.profitable
    )

    outperformed_runs = sum(
        1
        for row in successful_rows
        if row.strategy_outperformed
    )

    successful_count = len(
        successful_rows
    )

    return WalkForwardAggregateSummary(
        label=label,
        successful_runs=successful_count,
        failed_runs=len(
            failed_rows
        ),
        windows=len(
            {
                row.window_id
                for row in successful_rows
            }
        ),
        tickers=len(
            {
                row.ticker
                for row in successful_rows
            }
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
        profitable_runs=(
            profitable_runs
        ),
        profitable_runs_percent=(
            _round_metric(
                profitable_runs
                / successful_count
                * 100
                if successful_count
                else 0.0
            )
        ),
        outperformed_runs=(
            outperformed_runs
        ),
        outperformed_runs_percent=(
            _round_metric(
                outperformed_runs
                / successful_count
                * 100
                if successful_count
                else 0.0
            )
        ),
        average_strategy_return_percent=(
            _round_metric(
                _safe_mean(
                    returns
                )
            )
        ),
        median_strategy_return_percent=(
            _round_metric(
                _safe_median(
                    returns
                )
            )
        ),
        best_run_return_percent=(
            _round_metric(
                max(
                    returns,
                    default=0.0,
                )
            )
        ),
        worst_run_return_percent=(
            _round_metric(
                min(
                    returns,
                    default=0.0,
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
                    drawdowns
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
                    row.win_rate_percent
                    for row in successful_rows
                )
            )
        ),
    )


def run_walk_forward(
    *,
    tickers: list[str] | tuple[str, ...],
    trailing_values: list[float] | tuple[float, ...],
    period: str,
    base_config: BacktestConfig,
    train_months: int,
    test_months: int,
    step_months: int,
    minimum_train_bars: int,
    minimum_test_bars: int,
    fractional_crypto: bool = True,
    stop_on_error: bool = False,
) -> tuple[
    list[WalkForwardWindow],
    list[WalkForwardRunRow],
    list[CandidateWindowSummary],
    list[WalkForwardWindowSummary],
    list[WalkForwardAggregateSummary],
]:
    """Run global rolling walk-forward validation."""

    symbols = _normalize_tickers(
        tickers
    )

    candidates = normalize_candidate_values(
        trailing_values
    )

    if minimum_train_bars < 2:
        raise ValueError(
            "minimum_train_bars must be at least 2."
        )

    if minimum_test_bars < 2:
        raise ValueError(
            "minimum_test_bars must be at least 2."
        )

    print()
    print("=" * 124)
    print("AI STOCK RADAR — WALK-FORWARD VALIDATION")
    print("=" * 124)

    print(
        f"Tickers:                   "
        f"{len(symbols)}"
    )

    print(
        f"Candidates:                "
        f"{', '.join(f'{value:g}%' for value in candidates)}"
    )

    print(
        f"Historical period:         "
        f"{period}"
    )

    print(
        f"Training window:           "
        f"{train_months} months"
    )

    print(
        f"Testing window:            "
        f"{test_months} months"
    )

    print(
        f"Step:                      "
        f"{step_months} months"
    )

    print("=" * 124)

    prepared_by_ticker: dict[
        str,
        pd.DataFrame
    ] = {}

    print()
    print("Preparing market data...")

    for index, ticker in enumerate(
        symbols,
        start=1,
    ):
        print(
            f"[{index}/{len(symbols)}] "
            f"{ticker}"
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

            prepared = (
                _normalize_prepared_data(
                    prepared
                )
            )

            if prepared.empty:
                raise ValueError(
                    f"No prepared data for {ticker}."
                )

            prepared_by_ticker[
                ticker
            ] = prepared

        except Exception:
            if stop_on_error:
                raise

            raise RuntimeError(
                f"Market-data preparation failed "
                f"for {ticker}. "
                f"Walk-forward requires a consistent universe."
            )

    common_start = max(
        data.index.min()
        for data in prepared_by_ticker.values()
    )

    common_end = min(
        data.index.max()
        for data in prepared_by_ticker.values()
    )

    windows = generate_walk_forward_windows(
        common_start=common_start,
        common_end=common_end,
        train_months=train_months,
        test_months=test_months,
        step_months=step_months,
    )

    print()
    print(
        f"Common data start:         "
        f"{_timestamp_to_string(common_start)}"
    )

    print(
        f"Common data end:           "
        f"{_timestamp_to_string(common_end)}"
    )

    print(
        f"Walk-forward windows:      "
        f"{len(windows)}"
    )

    all_rows: list[
        WalkForwardRunRow
    ] = []

    all_candidate_summaries: list[
        CandidateWindowSummary
    ] = []

    window_summaries: list[
        WalkForwardWindowSummary
    ] = []

    for window_index, window in enumerate(
        windows,
        start=1,
    ):
        print()
        print("=" * 124)

        print(
            f"[{window_index}/{len(windows)}] "
            f"{window.window_id}"
        )

        print(
            f"Train: "
            f"{window.train_start.date()} "
            f"to "
            f"{window.train_end.date()} "
            f"(exclusive)"
        )

        print(
            f"Test:  "
            f"{window.test_start.date()} "
            f"to "
            f"{window.test_end.date()} "
            f"(exclusive)"
        )

        print("=" * 124)

        train_rows: list[
            WalkForwardRunRow
        ] = []

        for ticker in symbols:
            ticker_data = (
                prepared_by_ticker[
                    ticker
                ]
            )

            train_data = _slice_data(
                ticker_data,
                start=window.train_start,
                end=window.train_end,
            )

            if (
                len(train_data)
                < minimum_train_bars
            ):
                raise ValueError(
                    f"{window.window_id} / {ticker}: "
                    f"training data contains only "
                    f"{len(train_data)} bars; "
                    f"minimum is {minimum_train_bars}."
                )

            ticker_config = replace(
                base_config,
                take_profit_percent=(
                    DISABLED_TARGET_PERCENT
                ),
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

            benchmark = (
                run_buy_and_hold_benchmark(
                    ticker=ticker,
                    data=train_data,
                    config=ticker_config,
                )
            )

            for candidate in candidates:
                row = _run_candidate(
                    window_id=(
                        window.window_id
                    ),
                    phase="TRAIN",
                    ticker=ticker,
                    data=train_data,
                    trailing_close_percent=(
                        candidate
                    ),
                    config=ticker_config,
                    benchmark=benchmark,
                    selected_for_window=False,
                )

                train_rows.append(
                    row
                )

                if (
                    not row.success
                    and stop_on_error
                ):
                    raise RuntimeError(
                        row.error
                    )

        train_summaries = (
            _summarize_window_candidates(
                window_id=(
                    window.window_id
                ),
                phase="TRAIN",
                candidates=candidates,
                rows=train_rows,
            )
        )

        if not train_summaries:
            raise RuntimeError(
                f"No training summaries for "
                f"{window.window_id}."
            )

        selected_trailing = (
            train_summaries[0]
            .trailing_close_percent
        )

        train_rows = [
            replace(
                row,
                selected_for_window=(
                    row.trailing_close_percent
                    == selected_trailing
                ),
            )
            for row in train_rows
        ]

        print()
        print("Training ranking:")

        for summary in train_summaries:
            print(
                f"  Rank {summary.rank}: "
                f"Trailing "
                f"{summary.trailing_close_percent:>5.2f}% | "
                f"Median "
                f"{summary.median_strategy_return_percent:>+8.3f}% | "
                f"Average "
                f"{summary.average_strategy_return_percent:>+8.3f}% | "
                f"PF "
                f"{_format_profit_factor(summary.average_profit_factor):>6} | "
                f"DD "
                f"{summary.average_max_drawdown_percent:>6.3f}% | "
                f"Positive "
                f"{summary.profitable_tickers}/"
                f"{summary.successful_tickers}"
            )

        print(
            f"Selected globally:         "
            f"{selected_trailing:.2f}%"
        )

        test_rows: list[
            WalkForwardRunRow
        ] = []

        for ticker in symbols:
            ticker_data = (
                prepared_by_ticker[
                    ticker
                ]
            )

            test_data = _slice_data(
                ticker_data,
                start=window.test_start,
                end=window.test_end,
            )

            if (
                len(test_data)
                < minimum_test_bars
            ):
                raise ValueError(
                    f"{window.window_id} / {ticker}: "
                    f"test data contains only "
                    f"{len(test_data)} bars; "
                    f"minimum is {minimum_test_bars}."
                )

            ticker_config = replace(
                base_config,
                take_profit_percent=(
                    DISABLED_TARGET_PERCENT
                ),
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

            benchmark = (
                run_buy_and_hold_benchmark(
                    ticker=ticker,
                    data=test_data,
                    config=ticker_config,
                )
            )

            for candidate in candidates:
                row = _run_candidate(
                    window_id=(
                        window.window_id
                    ),
                    phase="TEST",
                    ticker=ticker,
                    data=test_data,
                    trailing_close_percent=(
                        candidate
                    ),
                    config=ticker_config,
                    benchmark=benchmark,
                    selected_for_window=(
                        candidate
                        == selected_trailing
                    ),
                )

                test_rows.append(
                    row
                )

                if (
                    not row.success
                    and stop_on_error
                ):
                    raise RuntimeError(
                        row.error
                    )

        test_summaries = (
            _summarize_window_candidates(
                window_id=(
                    window.window_id
                ),
                phase="TEST",
                candidates=candidates,
                rows=test_rows,
            )
        )

        selected_test_summary = (
            _selected_summary(
                test_summaries,
                trailing_close_percent=(
                    selected_trailing
                ),
            )
        )

        print()
        print("Out-of-sample selected result:")

        print(
            f"  Average return:          "
            f"{selected_test_summary.average_strategy_return_percent:+.4f}%"
        )

        print(
            f"  Median return:           "
            f"{selected_test_summary.median_strategy_return_percent:+.4f}%"
        )

        print(
            f"  Average excess return:   "
            f"{selected_test_summary.average_excess_return_percent:+.4f}%"
        )

        print(
            f"  Average profit factor:   "
            f"{_format_profit_factor(selected_test_summary.average_profit_factor)}"
        )

        print(
            f"  Average drawdown:        "
            f"{selected_test_summary.average_max_drawdown_percent:.4f}%"
        )

        print(
            f"  Profitable tickers:      "
            f"{selected_test_summary.profitable_tickers}/"
            f"{selected_test_summary.successful_tickers}"
        )

        print(
            f"  Benchmark outperformers: "
            f"{selected_test_summary.outperformed_count}/"
            f"{selected_test_summary.successful_tickers}"
        )

        all_rows.extend(
            train_rows
        )

        all_rows.extend(
            test_rows
        )

        all_candidate_summaries.extend(
            train_summaries
        )

        all_candidate_summaries.extend(
            test_summaries
        )

        window_summaries.append(
            _window_summary(
                window=window,
                selected_trailing=(
                    selected_trailing
                ),
                train_summaries=(
                    train_summaries
                ),
                test_summaries=(
                    test_summaries
                ),
            )
        )

    aggregate_summaries = [
        build_oos_aggregate(
            rows=all_rows,
            label=(
                f"FIXED_TRAILING_{candidate:g}"
            ),
            trailing_close_percent=(
                candidate
            ),
        )
        for candidate in candidates
    ]

    aggregate_summaries.append(
        build_oos_aggregate(
            rows=all_rows,
            label="DYNAMIC_TRAIN_SELECTED",
            selected_only=True,
        )
    )

    aggregate_summaries.sort(
        key=lambda item: (
            item.median_strategy_return_percent,
            item.average_strategy_return_percent,
            item.average_profit_factor,
            -item.average_max_drawdown_percent,
        ),
        reverse=True,
    )

    return (
        windows,
        all_rows,
        all_candidate_summaries,
        window_summaries,
        aggregate_summaries,
    )


def _format_profit_factor(
    value: float,
) -> str:
    """Format a profit-factor value."""

    if not isfinite(value):
        return "INF"

    return f"{value:.2f}"


def print_window_summaries(
    summaries: list[WalkForwardWindowSummary],
) -> None:
    """Print selected out-of-sample window results."""

    print()
    print("=" * 157)
    print("WALK-FORWARD SELECTED WINDOW RESULTS")
    print("=" * 157)

    print(
        f"{'Window':<9}"
        f"{'Selected':>10}"
        f"{'Train Med':>12}"
        f"{'Test Avg':>11}"
        f"{'Test Med':>11}"
        f"{'Avg Excess':>12}"
        f"{'Avg PF':>9}"
        f"{'Avg DD':>9}"
        f"{'Positive':>11}"
        f"{'Beat':>9}"
        f"{'Trades':>9}"
        f"{'Test Start':>13}"
        f"{'Test End':>13}"
    )

    print("-" * 157)

    for summary in summaries:
        positive_text = (
            f"{summary.test_selected_profitable_tickers}"
            f"/{len(DEFAULT_TICKERS)}"
        )

        beat_text = (
            f"{summary.test_selected_outperformed_count}"
            f"/{len(DEFAULT_TICKERS)}"
        )

        print(
            f"{summary.window_id:<9}"
            f"{summary.selected_trailing_close_percent:>10.2f}"
            f"{summary.train_selected_median_return_percent:>12.2f}"
            f"{summary.test_selected_average_return_percent:>11.2f}"
            f"{summary.test_selected_median_return_percent:>11.2f}"
            f"{summary.test_selected_average_excess_return_percent:>12.2f}"
            f"{_format_profit_factor(summary.test_selected_average_profit_factor):>9}"
            f"{summary.test_selected_average_drawdown_percent:>9.2f}"
            f"{positive_text:>11}"
            f"{beat_text:>9}"
            f"{summary.test_selected_total_trades:>9}"
            f"{summary.test_start[:10]:>13}"
            f"{summary.test_end_exclusive[:10]:>13}"
        )

    print("=" * 157)


def print_aggregate_summaries(
    summaries: list[WalkForwardAggregateSummary],
) -> None:
    """Print aggregate out-of-sample comparisons."""

    print()
    print("=" * 154)
    print("WALK-FORWARD OUT-OF-SAMPLE AGGREGATE")
    print("=" * 154)

    print(
        f"{'Label':<30}"
        f"{'Runs':>8}"
        f"{'Trades':>9}"
        f"{'Win %':>9}"
        f"{'Avg PF':>9}"
        f"{'Avg Ret':>11}"
        f"{'Med Ret':>11}"
        f"{'Worst':>10}"
        f"{'Avg Excess':>13}"
        f"{'Med Excess':>13}"
        f"{'Avg DD':>10}"
        f"{'Positive':>11}"
        f"{'Beat':>10}"
    )

    print("-" * 154)

    for summary in summaries:
        positive_text = (
            f"{summary.profitable_runs}/"
            f"{summary.successful_runs}"
        )

        beat_text = (
            f"{summary.outperformed_runs}/"
            f"{summary.successful_runs}"
        )

        print(
            f"{summary.label:<30}"
            f"{summary.successful_runs:>8}"
            f"{summary.total_trades:>9}"
            f"{summary.average_win_rate_percent:>9.2f}"
            f"{_format_profit_factor(summary.average_profit_factor):>9}"
            f"{summary.average_strategy_return_percent:>11.2f}"
            f"{summary.median_strategy_return_percent:>11.2f}"
            f"{summary.worst_run_return_percent:>10.2f}"
            f"{summary.average_excess_return_percent:>13.2f}"
            f"{summary.median_excess_return_percent:>13.2f}"
            f"{summary.average_max_drawdown_percent:>10.2f}"
            f"{positive_text:>11}"
            f"{beat_text:>10}"
        )

    print("=" * 154)

    if summaries:
        winner = summaries[0]

        print()
        print(
            f"Provisional OOS leader:    "
            f"{winner.label}"
        )

        print(
            f"Median OOS return:         "
            f"{winner.median_strategy_return_percent:+.4f}%"
        )

        print(
            f"Average OOS return:        "
            f"{winner.average_strategy_return_percent:+.4f}%"
        )

        print(
            f"Average OOS drawdown:      "
            f"{winner.average_max_drawdown_percent:.4f}%"
        )


def _json_safe_value(
    value: Any,
) -> Any:
    """Convert non-finite floats to valid JSON values."""

    if (
        isinstance(value, float)
        and not isfinite(value)
    ):
        return None

    return value


def _json_safe_dictionary(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Convert dictionary values to JSON-safe values."""

    return {
        key: _json_safe_value(value)
        for key, value in payload.items()
    }


def save_walk_forward_results(
    *,
    windows: list[WalkForwardWindow],
    rows: list[WalkForwardRunRow],
    candidate_summaries: list[CandidateWindowSummary],
    window_summaries: list[WalkForwardWindowSummary],
    aggregate_summaries: list[WalkForwardAggregateSummary],
    period: str,
    trailing_values: list[float],
    train_months: int,
    test_months: int,
    step_months: int,
) -> dict[str, Path]:
    """Save all walk-forward reports."""

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = (
        datetime.now(UTC)
        .strftime("%Y%m%d_%H%M%S")
    )

    runs_path = (
        OUTPUT_DIRECTORY
        / f"walk_forward_runs_{timestamp}.csv"
    )

    candidates_path = (
        OUTPUT_DIRECTORY
        / (
            "walk_forward_candidates_"
            f"{timestamp}.csv"
        )
    )

    windows_path = (
        OUTPUT_DIRECTORY
        / (
            "walk_forward_windows_"
            f"{timestamp}.csv"
        )
    )

    aggregate_path = (
        OUTPUT_DIRECTORY
        / (
            "walk_forward_aggregate_"
            f"{timestamp}.csv"
        )
    )

    json_path = (
        OUTPUT_DIRECTORY
        / f"walk_forward_{timestamp}.json"
    )

    pd.DataFrame(
        [
            asdict(row)
            for row in rows
        ]
    ).to_csv(
        runs_path,
        index=False,
    )

    pd.DataFrame(
        [
            asdict(summary)
            for summary
            in candidate_summaries
        ]
    ).to_csv(
        candidates_path,
        index=False,
    )

    pd.DataFrame(
        [
            asdict(summary)
            for summary
            in window_summaries
        ]
    ).to_csv(
        windows_path,
        index=False,
    )

    pd.DataFrame(
        [
            asdict(summary)
            for summary
            in aggregate_summaries
        ]
    ).to_csv(
        aggregate_path,
        index=False,
    )

    payload = {
        "created_at": (
            datetime.now(UTC)
            .isoformat()
        ),
        "period": period,
        "method": (
            "Global rolling walk-forward. "
            "Each training window selects one "
            "trailing value for the complete universe. "
            "The selection is evaluated on the "
            "immediately following unseen test window."
        ),
        "entry_rule": (
            "EMA20>EMA50, Close>EMA20, "
            "RSI14 between 45 and 70, "
            "new transition only."
        ),
        "exit_rule": (
            "5% initial stop by default, "
            "EMA20 below EMA50 or "
            "highest-Close trailing exit, "
            "filled at next Open."
        ),
        "trailing_values": (
            trailing_values
        ),
        "train_months": train_months,
        "test_months": test_months,
        "step_months": step_months,
        "windows": [
            window.to_dict()
            for window in windows
        ],
        "window_summaries": [
            _json_safe_dictionary(
                asdict(summary)
            )
            for summary
            in window_summaries
        ],
        "aggregate_summaries": [
            _json_safe_dictionary(
                asdict(summary)
            )
            for summary
            in aggregate_summaries
        ],
        "candidate_summaries": [
            _json_safe_dictionary(
                asdict(summary)
            )
            for summary
            in candidate_summaries
        ],
        "runs": [
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
        "runs_csv": runs_path,
        "candidates_csv": candidates_path,
        "windows_csv": windows_path,
        "aggregate_csv": aggregate_path,
        "json": json_path,
    }


def _parse_arguments() -> argparse.Namespace:
    """Parse terminal arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run global rolling walk-forward "
            "validation for trailing exits."
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
        "--train-months",
        type=int,
        default=(
            DEFAULT_TRAIN_MONTHS
        ),
    )

    parser.add_argument(
        "--test-months",
        type=int,
        default=(
            DEFAULT_TEST_MONTHS
        ),
    )

    parser.add_argument(
        "--step-months",
        type=int,
        default=(
            DEFAULT_STEP_MONTHS
        ),
    )

    parser.add_argument(
        "--minimum-train-bars",
        type=int,
        default=(
            DEFAULT_MINIMUM_TRAIN_BARS
        ),
    )

    parser.add_argument(
        "--minimum-test-bars",
        type=int,
        default=(
            DEFAULT_MINIMUM_TEST_BARS
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
        "--stop-on-error",
        action="store_true",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    """Run walk-forward validation."""

    arguments = _parse_arguments()

    tickers = (
        arguments.tickers
        if arguments.tickers
        else list(
            DEFAULT_TICKERS
        )
    )

    trailing_values = (
        normalize_candidate_values(
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

    (
        windows,
        rows,
        candidate_summaries,
        window_summaries,
        aggregate_summaries,
    ) = run_walk_forward(
        tickers=tickers,
        trailing_values=(
            trailing_values
        ),
        period=arguments.period,
        base_config=config,
        train_months=(
            arguments.train_months
        ),
        test_months=(
            arguments.test_months
        ),
        step_months=(
            arguments.step_months
        ),
        minimum_train_bars=(
            arguments.minimum_train_bars
        ),
        minimum_test_bars=(
            arguments.minimum_test_bars
        ),
        fractional_crypto=(
            not arguments
            .no_fractional_crypto
        ),
        stop_on_error=(
            arguments.stop_on_error
        ),
    )

    print_window_summaries(
        window_summaries
    )

    print_aggregate_summaries(
        aggregate_summaries
    )

    if not arguments.no_save:
        paths = save_walk_forward_results(
            windows=windows,
            rows=rows,
            candidate_summaries=(
                candidate_summaries
            ),
            window_summaries=(
                window_summaries
            ),
            aggregate_summaries=(
                aggregate_summaries
            ),
            period=arguments.period,
            trailing_values=(
                trailing_values
            ),
            train_months=(
                arguments.train_months
            ),
            test_months=(
                arguments.test_months
            ),
            step_months=(
                arguments.step_months
            ),
        )

        print()
        print("=" * 124)
        print("WALK-FORWARD FILES")
        print("=" * 124)

        print(
            f"Runs CSV:       "
            f"{paths['runs_csv'].resolve()}"
        )

        print(
            f"Candidates CSV: "
            f"{paths['candidates_csv'].resolve()}"
        )

        print(
            f"Windows CSV:    "
            f"{paths['windows_csv'].resolve()}"
        )

        print(
            f"Aggregate CSV:  "
            f"{paths['aggregate_csv'].resolve()}"
        )

        print(
            f"JSON:           "
            f"{paths['json'].resolve()}"
        )

        print("=" * 124)

    print()
    print(
        "Walk-forward validation completed successfully."
    )


if __name__ == "__main__":
    main()