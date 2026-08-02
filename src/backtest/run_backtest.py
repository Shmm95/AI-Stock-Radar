"""Command-line runner for deterministic AI-Stock-Radar backtests.

Strategy V1:
- Long-only.
- Uses EMA20, EMA50, RSI14, and MACD.
- Uses only information available at the current historical bar.
- BUY signals execute at the following bar's Open.
- EXIT signals execute at the current bar's Close.
- Commission, slippage, stop-loss, and take-profit are included.
- Strategy performance is compared with buy-and-hold.

This module performs historical simulation only.
It cannot send broker orders.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.technical_indicators import add_technical_indicators
from src.backtest.backtest_engine import (
    print_backtest_result,
    print_backtest_trades,
    run_backtest,
)
from src.backtest.backtest_models import (
    BacktestConfig,
    BacktestResult,
    BacktestSignal,
)
from src.backtest.benchmark import (
    compare_strategy_to_benchmark,
    print_buy_and_hold_result,
    print_strategy_benchmark_comparison,
    run_buy_and_hold_benchmark,
)
from src.data.download_stock import download_stock_data


DEFAULT_TICKER = "NVDA"
DEFAULT_PERIOD = "5y"

OUTPUT_DIRECTORY = (
    Path("data")
    / "backtests"
)

REQUIRED_COLUMNS = (
    "Open",
    "High",
    "Low",
    "Close",
    "EMA20",
    "EMA50",
    "RSI14",
    "MACD",
)


def _timestamp_to_string(
    value: object,
) -> str:
    """Convert an index value to the format expected by the engine."""

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    return str(value)


def _flatten_columns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Flatten Yahoo Finance MultiIndex columns when necessary."""

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

    return normalized


def _prepare_market_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize downloaded data and add technical indicators."""

    if data.empty:
        raise ValueError(
            "Downloaded market data is empty."
        )

    prepared = _flatten_columns(
        data
    )

    indicator_result = (
        add_technical_indicators(
            prepared
        )
    )

    if isinstance(
        indicator_result,
        pd.DataFrame,
    ):
        prepared = indicator_result

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in prepared.columns
    ]

    if missing_columns:
        raise ValueError(
            "Prepared market data is missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    prepared = prepared.copy()

    for column in REQUIRED_COLUMNS:
        prepared[column] = pd.to_numeric(
            prepared[column],
            errors="coerce",
        )

    prepared = prepared.dropna(
        subset=list(
            REQUIRED_COLUMNS
        )
    )

    prepared = prepared[
        (
            prepared["Open"] > 0
        )
        & (
            prepared["High"] > 0
        )
        & (
            prepared["Low"] > 0
        )
        & (
            prepared["Close"] > 0
        )
    ]

    if prepared.empty:
        raise ValueError(
            "No valid rows remain after indicator preparation."
        )

    if not prepared.index.is_monotonic_increasing:
        prepared = prepared.sort_index()

    prepared = prepared.loc[
        ~prepared.index.duplicated(
            keep="last"
        )
    ]

    return prepared


def _is_buy_setup(
    row: pd.Series,
) -> bool:
    """Return whether one bar satisfies the V1 entry rules."""

    ema20 = float(
        row["EMA20"]
    )

    ema50 = float(
        row["EMA50"]
    )

    rsi14 = float(
        row["RSI14"]
    )

    macd = float(
        row["MACD"]
    )

    close = float(
        row["Close"]
    )

    return (
        ema20 > ema50
        and close > ema20
        and macd > 0
        and 45 <= rsi14 <= 70
    )


def _is_exit_setup(
    row: pd.Series,
) -> bool:
    """Return whether one bar satisfies the V1 exit rules."""

    ema20 = float(
        row["EMA20"]
    )

    ema50 = float(
        row["EMA50"]
    )

    rsi14 = float(
        row["RSI14"]
    )

    macd = float(
        row["MACD"]
    )

    return (
        ema20 < ema50
        or macd < 0
        or rsi14 >= 75
    )


def _calculate_technical_score(
    row: pd.Series,
) -> int:
    """Calculate a deterministic technical score."""

    score = 0

    ema20 = float(
        row["EMA20"]
    )

    ema50 = float(
        row["EMA50"]
    )

    rsi14 = float(
        row["RSI14"]
    )

    macd = float(
        row["MACD"]
    )

    close = float(
        row["Close"]
    )

    if ema20 > ema50:
        score += 30

    if close > ema20:
        score += 20

    if macd > 0:
        score += 25

    if 45 <= rsi14 <= 70:
        score += 25

    return min(
        score,
        100,
    )


def deterministic_signal_provider(
    visible_data: pd.DataFrame,
    bar_index: int,
    ticker: str,
) -> BacktestSignal | None:
    """Generate causal deterministic BUY and EXIT signals.

    Only visible historical rows are inspected.
    Future data is unavailable to this function.
    """

    del bar_index

    if len(visible_data) < 2:
        return None

    current_row = (
        visible_data.iloc[-1]
    )

    previous_row = (
        visible_data.iloc[-2]
    )

    current_timestamp = (
        _timestamp_to_string(
            visible_data.index[-1]
        )
    )

    current_close = float(
        current_row["Close"]
    )

    current_buy = _is_buy_setup(
        current_row
    )

    previous_buy = _is_buy_setup(
        previous_row
    )

    current_exit = _is_exit_setup(
        current_row
    )

    previous_exit = _is_exit_setup(
        previous_row
    )

    technical_score = (
        _calculate_technical_score(
            current_row
        )
    )

    # Create a BUY only when the setup newly becomes valid.
    if (
        current_buy
        and not previous_buy
    ):
        return BacktestSignal(
            timestamp=current_timestamp,
            ticker=ticker,
            action="BUY",
            reference_price=current_close,
            overall_score=float(
                technical_score
            ),
            technical_score=float(
                technical_score
            ),
            confidence=float(
                technical_score
            ),
            reason=(
                "EMA20 above EMA50; "
                "price above EMA20; "
                "MACD positive; "
                "RSI within entry range."
            ),
        )

    # Create an EXIT only when the exit condition newly appears.
    if (
        current_exit
        and not previous_exit
    ):
        return BacktestSignal(
            timestamp=current_timestamp,
            ticker=ticker,
            action="EXIT",
            reference_price=current_close,
            overall_score=float(
                technical_score
            ),
            technical_score=float(
                technical_score
            ),
            confidence=float(
                technical_score
            ),
            reason=(
                "EMA trend, MACD, or RSI "
                "exit condition triggered."
            ),
        )

    return None


def _safe_ticker_name(
    ticker: str,
) -> str:
    """Convert a ticker into a safe filename component."""

    return (
        ticker.strip()
        .upper()
        .replace(
            "/",
            "_",
        )
        .replace(
            "\\",
            "_",
        )
        .replace(
            "^",
            "INDEX_",
        )
        .replace(
            "=",
            "_",
        )
    )


def _write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write formatted JSON output."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
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


def save_backtest_result(
    result: BacktestResult,
) -> dict[str, Path]:
    """Save result summary, trades, and equity curve."""

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        UTC
    ).strftime(
        "%Y%m%d_%H%M%S"
    )

    ticker_name = (
        _safe_ticker_name(
            result.ticker
        )
    )

    base_name = (
        f"{ticker_name}_{timestamp}"
    )

    json_path = (
        OUTPUT_DIRECTORY
        / f"{base_name}_result.json"
    )

    trades_path = (
        OUTPUT_DIRECTORY
        / f"{base_name}_trades.csv"
    )

    equity_path = (
        OUTPUT_DIRECTORY
        / f"{base_name}_equity.csv"
    )

    _write_json(
        json_path,
        result.to_dict(),
    )

    trades_frame = pd.DataFrame(
        [
            trade.to_dict()
            for trade in result.trades
        ]
    )

    equity_frame = pd.DataFrame(
        [
            point.to_dict()
            for point in result.equity_curve
        ]
    )

    trades_frame.to_csv(
        trades_path,
        index=False,
    )

    equity_frame.to_csv(
        equity_path,
        index=False,
    )

    return {
        "json": json_path,
        "trades": trades_path,
        "equity": equity_path,
    }


def _print_saved_paths(
    paths: dict[str, Path],
) -> None:
    """Print saved backtest file locations."""

    print()
    print("=" * 92)
    print("BACKTEST FILES")
    print("=" * 92)

    print(
        f"JSON result:  "
        f"{paths['json'].resolve()}"
    )

    print(
        f"Trades CSV:   "
        f"{paths['trades'].resolve()}"
    )

    print(
        f"Equity CSV:   "
        f"{paths['equity'].resolve()}"
    )

    print("=" * 92)


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run a deterministic historical backtest "
            "and compare it with buy-and-hold."
        )
    )

    parser.add_argument(
        "ticker",
        nargs="?",
        default=DEFAULT_TICKER,
        help=(
            "Yahoo Finance ticker, such as "
            "NVDA, AAPL, MSFT, or ETH-USD."
        ),
    )

    parser.add_argument(
        "--period",
        default=DEFAULT_PERIOD,
        help=(
            "Historical period, such as "
            "1y, 2y, 3y, or 5y."
        ),
    )

    parser.add_argument(
        "--initial-cash",
        type=float,
        default=10_000.0,
        help="Initial simulated account value.",
    )

    parser.add_argument(
        "--risk",
        type=float,
        default=1.0,
        help=(
            "Maximum account risk per trade "
            "in percent."
        ),
    )

    parser.add_argument(
        "--max-position",
        type=float,
        default=25.0,
        help=(
            "Maximum position allocation "
            "in percent."
        ),
    )

    parser.add_argument(
        "--stop",
        type=float,
        default=5.0,
        help="Stop-loss distance in percent.",
    )

    parser.add_argument(
        "--target",
        type=float,
        default=10.0,
        help="Take-profit distance in percent.",
    )

    parser.add_argument(
        "--commission-rate",
        type=float,
        default=0.0005,
        help="Simulated proportional commission rate.",
    )

    parser.add_argument(
        "--minimum-fee",
        type=float,
        default=1.0,
        help="Minimum simulated fee per transaction.",
    )

    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=5.0,
        help="Adverse simulated slippage in basis points.",
    )

    parser.add_argument(
        "--fractional",
        action="store_true",
        help="Allow fractional position quantities.",
    )

    parser.add_argument(
        "--keep-open",
        action="store_true",
        help=(
            "Do not force-close an open position "
            "at the end of the test."
        ),
    )

    parser.add_argument(
        "--show-trades",
        action="store_true",
        help="Print every completed trade.",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save JSON and CSV outputs.",
    )

    return parser.parse_args()


def main() -> None:
    """Download data and run one real-market backtest."""

    arguments = _parse_arguments()

    ticker = (
        arguments.ticker
        .strip()
        .upper()
    )

    if not ticker:
        raise ValueError(
            "Ticker cannot be empty."
        )

    print()
    print("=" * 92)
    print("AI STOCK RADAR — DETERMINISTIC BACKTEST")
    print("=" * 92)

    print(f"Ticker:             {ticker}")
    print(f"Period:             {arguments.period}")

    print(
        f"Initial cash:       "
        f"{arguments.initial_cash:,.2f}"
    )

    print(
        f"Risk per trade:     "
        f"{arguments.risk:.2f}%"
    )

    print(
        f"Maximum position:   "
        f"{arguments.max_position:.2f}%"
    )

    print(
        f"Stop / target:      "
        f"{arguments.stop:.2f}% / "
        f"{arguments.target:.2f}%"
    )

    print(
        f"Commission rate:    "
        f"{arguments.commission_rate:.6f}"
    )

    print(
        f"Minimum fee:        "
        f"{arguments.minimum_fee:.2f}"
    )

    print(
        f"Slippage:           "
        f"{arguments.slippage_bps:.2f} bps"
    )

    print("=" * 92)

    print()
    print("Downloading historical market data...")

    market_data = download_stock_data(
        ticker,
        period=arguments.period,
    )

    prepared_data = _prepare_market_data(
        market_data
    )

    print(
        f"Prepared rows: "
        f"{len(prepared_data)}"
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
            arguments.fractional
        ),
        maximum_open_positions=1,
        force_close_at_end=(
            not arguments.keep_open
        ),
    )

    # =========================================================
    # ACTIVE STRATEGY BACKTEST
    # =========================================================

    result = run_backtest(
        ticker=ticker,
        data=prepared_data,
        signal_provider=(
            deterministic_signal_provider
        ),
        config=config,
    )

    print_backtest_result(
        result
    )

    # =========================================================
    # PASSIVE BUY-AND-HOLD BENCHMARK
    # =========================================================

    benchmark_result = (
        run_buy_and_hold_benchmark(
            ticker=ticker,
            data=prepared_data,
            config=config,
        )
    )

    comparison = (
        compare_strategy_to_benchmark(
            strategy_result=result,
            benchmark_result=benchmark_result,
        )
    )

    print_buy_and_hold_result(
        benchmark_result
    )

    print_strategy_benchmark_comparison(
        comparison
    )

    # =========================================================
    # OPTIONAL TRADE DETAILS
    # =========================================================

    if arguments.show_trades:
        print_backtest_trades(
            result
        )

    # =========================================================
    # SAVE RESULTS
    # =========================================================

    if not arguments.no_save:
        saved_paths = (
            save_backtest_result(
                result
            )
        )

        _print_saved_paths(
            saved_paths
        )

    print()
    print("Backtest completed successfully.")


if __name__ == "__main__":
    main()