"""Market-regime ablation for AI-Stock-Radar.

Baseline strategy:
- TREND_RSI entry
- 5% initial stop
- 7.5% highest-Close trailing exit
- EMA20 below EMA50 exit
- Signals fill at the next Open

Regime symbols:
- Equities: SPY
- Cryptocurrencies: BTC-USD

Variants:
- NO_FILTER
- MARKET_ABOVE_EMA200
- MARKET_BULL_TREND:
  market Close > EMA200 and EMA50 > EMA200

The regime filter controls new entries only. It does not force an
existing position to close when the regime becomes unfavorable.
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
from src.backtest.backtest_models import BacktestConfig, BacktestResult
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

DEFAULT_PERIOD = "5y"
DEFAULT_REGIME_PERIOD = "10y"
DEFAULT_TRAILING_CLOSE_PERCENT = 7.5

OUTPUT_DIRECTORY = (
    Path("data")
    / "backtests"
    / "regime_ablation"
)


@dataclass(frozen=True, slots=True)
class RegimeVariant:
    """Definition of one market-regime rule."""

    name: str
    description: str
    require_price_above_ema200: bool
    require_ema50_above_ema200: bool


@dataclass(frozen=True, slots=True)
class RegimeResultRow:
    """One ticker and one regime-filter result."""

    ticker: str
    asset_class: str
    regime_symbol: str
    variant: str

    success: bool
    error: str

    prepared_bars: int
    regime_allowed_bars: int
    regime_allowed_percent: float

    completed_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_percent: float
    profit_factor: float
    total_fees: float

    strategy_return_percent: float
    benchmark_return_percent: float
    excess_return_percent: float

    strategy_max_drawdown_percent: float
    benchmark_max_drawdown_percent: float

    profitable: bool
    strategy_outperformed: bool


@dataclass(frozen=True, slots=True)
class RegimeSummary:
    """Aggregate result for a scope and regime variant."""

    scope: str
    rank: int
    variant: str

    successful_tickers: int
    failed_tickers: int

    total_trades: int
    winning_trades: int
    losing_trades: int

    profitable_tickers: int
    profitable_tickers_percent: float

    outperformed_tickers: int
    outperformed_tickers_percent: float

    average_regime_allowed_percent: float

    average_strategy_return_percent: float
    median_strategy_return_percent: float
    worst_ticker_return_percent: float

    average_benchmark_return_percent: float
    average_excess_return_percent: float
    median_excess_return_percent: float

    average_max_drawdown_percent: float
    average_profit_factor: float
    average_win_rate_percent: float


REGIME_VARIANTS = (
    RegimeVariant(
        name="NO_FILTER",
        description="No market-regime entry filter.",
        require_price_above_ema200=False,
        require_ema50_above_ema200=False,
    ),
    RegimeVariant(
        name="MARKET_ABOVE_EMA200",
        description="New entries require market Close above EMA200.",
        require_price_above_ema200=True,
        require_ema50_above_ema200=False,
    ),
    RegimeVariant(
        name="MARKET_BULL_TREND",
        description=(
            "New entries require market Close above EMA200 "
            "and market EMA50 above EMA200."
        ),
        require_price_above_ema200=True,
        require_ema50_above_ema200=True,
    ),
)


def _round_metric(value: float) -> float:
    """Round percentages and ratios."""

    return round(float(value), 4)


def _is_crypto_ticker(ticker: str) -> bool:
    """Return whether a ticker represents cryptocurrency."""

    return ticker.strip().upper().endswith(
        (
            "-USD",
            "-EUR",
            "-GBP",
        )
    )


def _asset_class(ticker: str) -> str:
    """Return a compact asset-class name."""

    return "CRYPTO" if _is_crypto_ticker(ticker) else "EQUITY"


def _regime_symbol(ticker: str) -> str:
    """Return the regime instrument for one asset."""

    return "BTC-USD" if _is_crypto_ticker(ticker) else "SPY"


def _normalize_index(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize, sort, and deduplicate a DataFrame index."""

    normalized = data.copy()
    normalized.index = pd.to_datetime(normalized.index)

    if normalized.index.tz is not None:
        normalized.index = normalized.index.tz_convert(None)

    normalized = normalized.sort_index()

    return normalized.loc[
        ~normalized.index.duplicated(keep="last")
    ]


def _prepare_regime_data(
    symbol: str,
    *,
    period: str,
) -> pd.DataFrame:
    """Download and prepare causal regime indicators."""

    downloaded = download_stock_data(
        symbol,
        period=period,
    )

    prepared = _prepare_market_data(downloaded)
    prepared = _normalize_index(prepared)

    close = prepared["Close"].astype(float)

    regime = pd.DataFrame(
        {
            "MarketClose": close,
            "MarketEMA50": close.ewm(
                span=50,
                adjust=False,
                min_periods=50,
            ).mean(),
            "MarketEMA200": close.ewm(
                span=200,
                adjust=False,
                min_periods=200,
            ).mean(),
        },
        index=prepared.index,
    )

    return regime


def build_regime_allowed(
    *,
    asset_index: pd.Index,
    regime_data: pd.DataFrame,
    variant: RegimeVariant,
) -> pd.Series:
    """Align the regime causally and return an entry-permission mask.

    Reindexing uses forward-fill only. Values before the first available
    regime observation remain unavailable and therefore become False.
    No backward-fill is used.
    """

    normalized_regime = _normalize_index(regime_data)

    target_index = pd.to_datetime(asset_index)

    if target_index.tz is not None:
        target_index = target_index.tz_convert(None)

    if variant.name == "NO_FILTER":
        return pd.Series(
            True,
            index=target_index,
            dtype=bool,
            name="RegimeAllowed",
        )

    required_columns = {
        "MarketClose",
        "MarketEMA50",
        "MarketEMA200",
    }

    missing_columns = required_columns.difference(
        normalized_regime.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing regime columns: "
            + ", ".join(sorted(missing_columns))
        )

    aligned = normalized_regime.reindex(
        target_index,
        method="ffill",
    )

    allowed = pd.Series(
        True,
        index=target_index,
        dtype=bool,
    )

    if variant.require_price_above_ema200:
        allowed &= (
            aligned["MarketClose"]
            > aligned["MarketEMA200"]
        )

    if variant.require_ema50_above_ema200:
        allowed &= (
            aligned["MarketEMA50"]
            > aligned["MarketEMA200"]
        )

    allowed = allowed.fillna(False).astype(bool)
    allowed.name = "RegimeAllowed"

    return allowed


class RegimeFilteredProvider:
    """Apply a causal regime gate to an ExitSignalProvider.

    ExitSignalProvider uses only the current and previous visible rows
    for entry decisions. On regime-disabled rows, RSI14 is replaced by
    zero in a temporary two-row view. This makes TREND_RSI entry false
    without changing market prices, exits, trailing state, or the
    original DataFrame.
    """

    def __init__(
        self,
        *,
        base_provider: ExitSignalProvider,
        enabled: bool,
    ) -> None:
        self.base_provider = base_provider
        self.enabled = enabled

    def __call__(
        self,
        visible_data: pd.DataFrame,
        bar_index: int,
        ticker: str,
    ):
        """Return the base strategy signal after applying the gate."""

        provider_data = visible_data.tail(2).copy()

        if self.enabled:
            if "RegimeAllowed" not in provider_data.columns:
                raise ValueError(
                    "RegimeAllowed column is required."
                )

            disabled = ~provider_data[
                "RegimeAllowed"
            ].fillna(False).astype(bool)

            provider_data.loc[
                disabled,
                "RSI14",
            ] = 0.0

        return self.base_provider(
            provider_data,
            bar_index,
            ticker,
        )


def _trailing_variant(
    trailing_close_percent: float,
) -> ExitVariant:
    """Create the fixed trailing-exit strategy."""

    return ExitVariant(
        name=f"TRAILING_CLOSE_{trailing_close_percent:g}",
        description=(
            f"{trailing_close_percent:g}% highest-Close trailing exit."
        ),
        take_profit_percent=DISABLED_TARGET_PERCENT,
        use_trend_rsi_exit=False,
        trailing_close_percent=trailing_close_percent,
    )


def _profit_factor(result: BacktestResult) -> float:
    """Return a rounded or infinite profit factor."""

    if not isfinite(result.profit_factor):
        return inf

    return round(result.profit_factor, 4)


def _success_row(
    *,
    ticker: str,
    regime_symbol: str,
    variant: RegimeVariant,
    data: pd.DataFrame,
    result: BacktestResult,
    benchmark: BuyAndHoldResult,
) -> RegimeResultRow:
    """Create one successful result row."""

    allowed_bars = int(
        data["RegimeAllowed"]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    allowed_percent = (
        allowed_bars
        / len(data)
        * 100
        if len(data)
        else 0.0
    )

    excess_return = (
        result.total_return_percent
        - benchmark.total_return_percent
    )

    return RegimeResultRow(
        ticker=ticker,
        asset_class=_asset_class(ticker),
        regime_symbol=regime_symbol,
        variant=variant.name,
        success=True,
        error="",
        prepared_bars=len(data),
        regime_allowed_bars=allowed_bars,
        regime_allowed_percent=_round_metric(
            allowed_percent
        ),
        completed_trades=result.completed_trades,
        winning_trades=result.winning_trades,
        losing_trades=result.losing_trades,
        win_rate_percent=_round_metric(
            result.win_rate_percent
        ),
        profit_factor=_profit_factor(result),
        total_fees=round(result.total_fees, 2),
        strategy_return_percent=_round_metric(
            result.total_return_percent
        ),
        benchmark_return_percent=_round_metric(
            benchmark.total_return_percent
        ),
        excess_return_percent=_round_metric(
            excess_return
        ),
        strategy_max_drawdown_percent=_round_metric(
            result.maximum_drawdown_percent
        ),
        benchmark_max_drawdown_percent=_round_metric(
            benchmark.maximum_drawdown_percent
        ),
        profitable=result.total_return_amount > 0,
        strategy_outperformed=excess_return > 0,
    )


def _failed_row(
    *,
    ticker: str,
    regime_symbol: str,
    variant: RegimeVariant,
    error: Exception | str,
) -> RegimeResultRow:
    """Create a standardized failed row."""

    return RegimeResultRow(
        ticker=ticker,
        asset_class=_asset_class(ticker),
        regime_symbol=regime_symbol,
        variant=variant.name,
        success=False,
        error=str(error),
        prepared_bars=0,
        regime_allowed_bars=0,
        regime_allowed_percent=0.0,
        completed_trades=0,
        winning_trades=0,
        losing_trades=0,
        win_rate_percent=0.0,
        profit_factor=0.0,
        total_fees=0.0,
        strategy_return_percent=0.0,
        benchmark_return_percent=0.0,
        excess_return_percent=0.0,
        strategy_max_drawdown_percent=0.0,
        benchmark_max_drawdown_percent=0.0,
        profitable=False,
        strategy_outperformed=False,
    )


def run_regime_ablation(
    *,
    tickers: list[str] | tuple[str, ...],
    period: str,
    regime_period: str,
    trailing_close_percent: float,
    base_config: BacktestConfig,
    fractional_crypto: bool = True,
    stop_on_error: bool = False,
) -> list[RegimeResultRow]:
    """Run all regime variants across the universe."""

    symbols = list(
        dict.fromkeys(
            ticker.strip().upper()
            for ticker in tickers
            if ticker.strip()
        )
    )

    if not symbols:
        raise ValueError(
            "At least one ticker must be supplied."
        )

    print()
    print("=" * 120)
    print("AI STOCK RADAR — MARKET REGIME ABLATION")
    print("=" * 120)
    print(f"Tickers:                   {len(symbols)}")
    print(f"Historical period:         {period}")
    print(f"Regime history:            {regime_period}")
    print(
        f"Trailing close:            "
        f"{trailing_close_percent:.2f}%"
    )
    print("=" * 120)

    regime_data_by_symbol = {
        "SPY": _prepare_regime_data(
            "SPY",
            period=regime_period,
        ),
        "BTC-USD": _prepare_regime_data(
            "BTC-USD",
            period=regime_period,
        ),
    }

    rows: list[RegimeResultRow] = []

    for ticker_index, ticker in enumerate(
        symbols,
        start=1,
    ):
        print()
        print(
            f"[{ticker_index}/{len(symbols)}] "
            f"Preparing {ticker}..."
        )

        regime_symbol = _regime_symbol(ticker)

        ticker_config = replace(
            base_config,
            take_profit_percent=DISABLED_TARGET_PERCENT,
            allow_fractional=(
                base_config.allow_fractional
                or (
                    fractional_crypto
                    and _is_crypto_ticker(ticker)
                )
            ),
        )

        try:
            downloaded = download_stock_data(
                ticker,
                period=period,
            )

            prepared = _prepare_market_data(downloaded)
            prepared = _normalize_index(prepared)

            benchmark = run_buy_and_hold_benchmark(
                ticker=ticker,
                data=prepared,
                config=ticker_config,
            )

        except Exception as error:
            for variant in REGIME_VARIANTS:
                rows.append(
                    _failed_row(
                        ticker=ticker,
                        regime_symbol=regime_symbol,
                        variant=variant,
                        error=error,
                    )
                )

            if stop_on_error:
                raise

            continue

        for variant in REGIME_VARIANTS:
            test_data = prepared.copy()

            test_data["RegimeAllowed"] = (
                build_regime_allowed(
                    asset_index=test_data.index,
                    regime_data=(
                        regime_data_by_symbol[
                            regime_symbol
                        ]
                    ),
                    variant=variant,
                )
            )

            base_provider = ExitSignalProvider(
                variant=_trailing_variant(
                    trailing_close_percent
                ),
                config=ticker_config,
            )

            provider = RegimeFilteredProvider(
                base_provider=base_provider,
                enabled=variant.name != "NO_FILTER",
            )

            print(
                f"  {variant.name:<24}",
                end="",
            )

            try:
                result = run_backtest(
                    ticker=ticker,
                    data=test_data,
                    signal_provider=provider,
                    config=ticker_config,
                )

                row = _success_row(
                    ticker=ticker,
                    regime_symbol=regime_symbol,
                    variant=variant,
                    data=test_data,
                    result=result,
                    benchmark=benchmark,
                )

            except Exception as error:
                row = _failed_row(
                    ticker=ticker,
                    regime_symbol=regime_symbol,
                    variant=variant,
                    error=error,
                )

                rows.append(row)
                print(f"FAILED: {error}")

                if stop_on_error:
                    raise

                continue

            rows.append(row)

            print(
                f"Allowed={row.regime_allowed_percent:>6.2f}% "
                f"Trades={row.completed_trades:<3} "
                f"Return={row.strategy_return_percent:>+8.2f}% "
                f"PF={_format_pf(row.profit_factor):>6} "
                f"Excess={row.excess_return_percent:>+8.2f}% "
                f"DD={row.strategy_max_drawdown_percent:>6.2f}%"
            )

    return rows


def _safe_mean(values: list[float]) -> float:
    """Return a safe arithmetic mean."""

    return float(mean(values)) if values else 0.0


def _safe_median(values: list[float]) -> float:
    """Return a safe median."""

    return float(median(values)) if values else 0.0


def _scope_rows(
    rows: list[RegimeResultRow],
    scope: str,
) -> list[RegimeResultRow]:
    """Filter rows by summary scope."""

    if scope == "ALL":
        return rows

    return [
        row
        for row in rows
        if row.asset_class == scope
    ]


def summarize_regime_results(
    rows: list[RegimeResultRow],
) -> list[RegimeSummary]:
    """Create ranked ALL, EQUITY, and CRYPTO summaries."""

    summaries: list[RegimeSummary] = []

    for scope in (
        "ALL",
        "EQUITY",
        "CRYPTO",
    ):
        raw_scope_summaries: list[
            RegimeSummary
        ] = []

        scoped_rows = _scope_rows(
            rows,
            scope,
        )

        for variant in REGIME_VARIANTS:
            matching = [
                row
                for row in scoped_rows
                if row.variant == variant.name
            ]

            successful = [
                row
                for row in matching
                if row.success
            ]

            failed = [
                row
                for row in matching
                if not row.success
            ]

            returns = [
                row.strategy_return_percent
                for row in successful
            ]

            excess_returns = [
                row.excess_return_percent
                for row in successful
            ]

            finite_profit_factors = [
                row.profit_factor
                for row in successful
                if isfinite(row.profit_factor)
            ]

            profitable_count = sum(
                row.profitable
                for row in successful
            )

            outperformed_count = sum(
                row.strategy_outperformed
                for row in successful
            )

            count = len(successful)

            raw_scope_summaries.append(
                RegimeSummary(
                    scope=scope,
                    rank=0,
                    variant=variant.name,
                    successful_tickers=count,
                    failed_tickers=len(failed),
                    total_trades=sum(
                        row.completed_trades
                        for row in successful
                    ),
                    winning_trades=sum(
                        row.winning_trades
                        for row in successful
                    ),
                    losing_trades=sum(
                        row.losing_trades
                        for row in successful
                    ),
                    profitable_tickers=(
                        profitable_count
                    ),
                    profitable_tickers_percent=(
                        _round_metric(
                            profitable_count
                            / count
                            * 100
                            if count
                            else 0.0
                        )
                    ),
                    outperformed_tickers=(
                        outperformed_count
                    ),
                    outperformed_tickers_percent=(
                        _round_metric(
                            outperformed_count
                            / count
                            * 100
                            if count
                            else 0.0
                        )
                    ),
                    average_regime_allowed_percent=(
                        _round_metric(
                            _safe_mean(
                                [
                                    row.regime_allowed_percent
                                    for row in successful
                                ]
                            )
                        )
                    ),
                    average_strategy_return_percent=(
                        _round_metric(
                            _safe_mean(returns)
                        )
                    ),
                    median_strategy_return_percent=(
                        _round_metric(
                            _safe_median(returns)
                        )
                    ),
                    worst_ticker_return_percent=(
                        _round_metric(
                            min(
                                returns,
                                default=0.0,
                            )
                        )
                    ),
                    average_benchmark_return_percent=(
                        _round_metric(
                            _safe_mean(
                                [
                                    row.benchmark_return_percent
                                    for row in successful
                                ]
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
                                [
                                    row
                                    .strategy_max_drawdown_percent
                                    for row in successful
                                ]
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
                                [
                                    row.win_rate_percent
                                    for row in successful
                                ]
                            )
                        )
                    ),
                )
            )

        raw_scope_summaries.sort(
            key=lambda summary: (
                summary.median_strategy_return_percent,
                summary.average_strategy_return_percent,
                summary.profitable_tickers,
                summary.average_profit_factor,
                -summary.average_max_drawdown_percent,
            ),
            reverse=True,
        )

        summaries.extend(
            replace(
                summary,
                rank=index,
            )
            for index, summary in enumerate(
                raw_scope_summaries,
                start=1,
            )
        )

    return summaries


def _format_pf(value: float) -> str:
    """Format a profit-factor value."""

    if not isfinite(value):
        return "INF"

    return f"{value:.2f}"


def print_regime_summaries(
    summaries: list[RegimeSummary],
) -> None:
    """Print ranked summaries."""

    for scope in (
        "ALL",
        "EQUITY",
        "CRYPTO",
    ):
        selected = [
            summary
            for summary in summaries
            if summary.scope == scope
        ]

        if not selected:
            continue

        print()
        print("=" * 148)
        print(
            f"MARKET REGIME SUMMARY — {scope}"
        )
        print("=" * 148)

        print(
            f"{'Rank':>5}"
            f"{'Variant':<26}"
            f"{'Allowed':>10}"
            f"{'Trades':>9}"
            f"{'Win %':>9}"
            f"{'Avg PF':>9}"
            f"{'Avg Ret':>11}"
            f"{'Med Ret':>11}"
            f"{'Worst':>10}"
            f"{'Avg Excess':>13}"
            f"{'Avg DD':>10}"
            f"{'Positive':>11}"
            f"{'Beat':>9}"
        )

        print("-" * 148)

        for summary in selected:
            print(
                f"{summary.rank:>5}"
                f"{summary.variant:<26}"
                f"{summary.average_regime_allowed_percent:>9.2f}%"
                f"{summary.total_trades:>9}"
                f"{summary.average_win_rate_percent:>9.2f}"
                f"{_format_pf(summary.average_profit_factor):>9}"
                f"{summary.average_strategy_return_percent:>11.2f}"
                f"{summary.median_strategy_return_percent:>11.2f}"
                f"{summary.worst_ticker_return_percent:>10.2f}"
                f"{summary.average_excess_return_percent:>13.2f}"
                f"{summary.average_max_drawdown_percent:>10.2f}"
                f"{summary.profitable_tickers:>5}/"
                f"{summary.successful_tickers:<5}"
                f"{summary.outperformed_tickers:>4}/"
                f"{summary.successful_tickers:<4}"
            )

        print("=" * 148)


def _json_safe(value: Any) -> Any:
    """Convert non-finite floats to JSON-safe values."""

    if isinstance(value, float) and not isfinite(value):
        return None

    return value


def save_regime_results(
    *,
    rows: list[RegimeResultRow],
    summaries: list[RegimeSummary],
    period: str,
    regime_period: str,
    trailing_close_percent: float,
) -> dict[str, Path]:
    """Save detailed, summary, and JSON reports."""

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(UTC).strftime(
        "%Y%m%d_%H%M%S"
    )

    details_path = (
        OUTPUT_DIRECTORY
        / f"regime_ablation_details_{timestamp}.csv"
    )

    summary_path = (
        OUTPUT_DIRECTORY
        / f"regime_ablation_summary_{timestamp}.csv"
    )

    json_path = (
        OUTPUT_DIRECTORY
        / f"regime_ablation_{timestamp}.json"
    )

    pd.DataFrame(
        [asdict(row) for row in rows]
    ).to_csv(
        details_path,
        index=False,
    )

    pd.DataFrame(
        [asdict(summary) for summary in summaries]
    ).to_csv(
        summary_path,
        index=False,
    )

    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "period": period,
        "regime_period": regime_period,
        "trailing_close_percent": (
            trailing_close_percent
        ),
        "entry_rule": (
            "TREND_RSI entry, gated by the selected "
            "market-regime rule."
        ),
        "exit_rule": (
            "5% initial stop, EMA20 below EMA50, "
            f"{trailing_close_percent:g}% "
            "highest-Close trailing exit."
        ),
        "regime_symbols": {
            "EQUITY": "SPY",
            "CRYPTO": "BTC-USD",
        },
        "variants": [
            asdict(variant)
            for variant in REGIME_VARIANTS
        ],
        "summaries": [
            {
                key: _json_safe(value)
                for key, value in asdict(summary).items()
            }
            for summary in summaries
        ],
        "details": [
            {
                key: _json_safe(value)
                for key, value in asdict(row).items()
            }
            for row in rows
        ],
    }

    with json_path.open(
        "w",
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
            "Compare market-regime entry filters."
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
        "--regime-period",
        default=DEFAULT_REGIME_PERIOD,
    )

    parser.add_argument(
        "--trailing-close",
        type=float,
        default=DEFAULT_TRAILING_CLOSE_PERCENT,
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
    """Run the market-regime ablation."""

    arguments = _parse_arguments()

    if arguments.trailing_close <= 0:
        raise ValueError(
            "--trailing-close must be greater than zero."
        )

    tickers = (
        arguments.tickers
        if arguments.tickers
        else list(DEFAULT_TICKERS)
    )

    config = BacktestConfig(
        initial_cash=arguments.initial_cash,
        risk_per_trade_percent=arguments.risk,
        maximum_position_percent=(
            arguments.max_position
        ),
        stop_loss_percent=arguments.stop,
        take_profit_percent=(
            DISABLED_TARGET_PERCENT
        ),
        commission_rate=(
            arguments.commission_rate
        ),
        minimum_fee=arguments.minimum_fee,
        slippage_bps=arguments.slippage_bps,
        allow_fractional=(
            arguments.fractional_all
        ),
        maximum_open_positions=1,
        force_close_at_end=True,
    )

    config.validate()

    rows = run_regime_ablation(
        tickers=tickers,
        period=arguments.period,
        regime_period=arguments.regime_period,
        trailing_close_percent=(
            arguments.trailing_close
        ),
        base_config=config,
        fractional_crypto=(
            not arguments.no_fractional_crypto
        ),
        stop_on_error=arguments.stop_on_error,
    )

    summaries = summarize_regime_results(
        rows
    )

    print_regime_summaries(
        summaries
    )

    if not arguments.no_save:
        paths = save_regime_results(
            rows=rows,
            summaries=summaries,
            period=arguments.period,
            regime_period=(
                arguments.regime_period
            ),
            trailing_close_percent=(
                arguments.trailing_close
            ),
        )

        print()
        print("=" * 120)
        print("REGIME ABLATION FILES")
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
        "Market-regime ablation completed successfully."
    )


if __name__ == "__main__":
    main()