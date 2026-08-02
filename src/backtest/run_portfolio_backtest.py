"""Command-line runner for shared-cash portfolio backtesting.

Current provisional profiles:

EQUITY
- TREND_RSI entry
- 5% initial stop
- 7.5% highest-Close trailing exit
- EMA20 below EMA50 exit
- no market-regime entry filter

CRYPTO
- TREND_RSI entry
- 5% initial stop
- 7.5% highest-Close trailing exit
- EMA20 below EMA50 exit
- BTC Close above EMA200 and BTC EMA50 above EMA200 entry filter

All candidates compete for one shared account. Signals created after a Close
execute at the ticker's next available Open.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest.portfolio_backtest_engine import (
    print_portfolio_backtest_result,
    run_portfolio_backtest,
)
from src.backtest.portfolio_backtest_models import (
    PortfolioBacktestConfig,
    PortfolioBacktestResult,
)
from src.backtest.run_backtest import _prepare_market_data
from src.backtest.run_regime_ablation import (
    REGIME_VARIANTS,
    _prepare_regime_data,
    build_regime_allowed,
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

OUTPUT_DIRECTORY = Path("data") / "backtests" / "portfolio"


def _is_crypto_ticker(ticker: str) -> bool:
    return ticker.strip().upper().endswith(("-USD", "-EUR", "-GBP"))


def _normalize_tickers(tickers: list[str] | tuple[str, ...]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()

    for raw_ticker in tickers:
        ticker = raw_ticker.strip().upper()
        if ticker and ticker not in seen:
            normalized.append(ticker)
            seen.add(ticker)

    if not normalized:
        raise ValueError("At least one ticker must be supplied.")

    return normalized


def _normalize_index(data: pd.DataFrame) -> pd.DataFrame:
    normalized = data.copy()
    normalized.index = pd.to_datetime(normalized.index)
    if normalized.index.tz is not None:
        normalized.index = normalized.index.tz_convert(None)
    normalized = normalized.sort_index()
    return normalized.loc[~normalized.index.duplicated(keep="last")]


def _bull_trend_variant():
    return next(
        variant
        for variant in REGIME_VARIANTS
        if variant.name == "MARKET_BULL_TREND"
    )


def prepare_portfolio_data(
    *,
    tickers: list[str] | tuple[str, ...],
    period: str,
    regime_period: str,
    use_crypto_regime: bool = True,
) -> dict[str, pd.DataFrame]:
    """Download and prepare all ticker data with causal regime masks."""

    symbols = _normalize_tickers(tickers)
    prepared_by_ticker: dict[str, pd.DataFrame] = {}

    crypto_regime_data: pd.DataFrame | None = None
    if use_crypto_regime and any(_is_crypto_ticker(ticker) for ticker in symbols):
        print("Preparing BTC bull-trend regime data...")
        crypto_regime_data = _prepare_regime_data(
            "BTC-USD",
            period=regime_period,
        )

    print()
    print("=" * 100)
    print("PORTFOLIO MARKET-DATA PREPARATION")
    print("=" * 100)

    for index, ticker in enumerate(symbols, start=1):
        print(f"[{index}/{len(symbols)}] {ticker}")

        downloaded = download_stock_data(ticker, period=period)
        prepared = _prepare_market_data(downloaded)
        prepared = _normalize_index(prepared)

        if _is_crypto_ticker(ticker) and use_crypto_regime:
            if crypto_regime_data is None:
                raise RuntimeError("Crypto regime data was not prepared.")

            prepared["RegimeAllowed"] = build_regime_allowed(
                asset_index=prepared.index,
                regime_data=crypto_regime_data,
                variant=_bull_trend_variant(),
            )
        else:
            prepared["RegimeAllowed"] = True

        prepared_by_ticker[ticker] = prepared

        allowed_percent = (
            prepared["RegimeAllowed"].astype(bool).mean() * 100
            if len(prepared)
            else 0.0
        )
        print(
            f"  Bars={len(prepared):<5} "
            f"Start={prepared.index.min().date()} "
            f"End={prepared.index.max().date()} "
            f"Regime allowed={allowed_percent:.2f}%"
        )

    print("=" * 100)
    return prepared_by_ticker


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def save_portfolio_backtest_result(
    result: PortfolioBacktestResult,
) -> dict[str, Path]:
    """Save summary JSON and detailed CSV reports."""

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    json_path = OUTPUT_DIRECTORY / f"portfolio_backtest_{timestamp}.json"
    trades_path = OUTPUT_DIRECTORY / f"portfolio_trades_{timestamp}.csv"
    equity_path = OUTPUT_DIRECTORY / f"portfolio_equity_{timestamp}.csv"
    rejections_path = (
        OUTPUT_DIRECTORY / f"portfolio_rejections_{timestamp}.csv"
    )
    summary_path = OUTPUT_DIRECTORY / f"portfolio_summary_{timestamp}.csv"

    payload = _json_safe(result.to_dict())
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")

    pd.DataFrame([trade.to_dict() for trade in result.trades]).to_csv(
        trades_path,
        index=False,
    )
    pd.DataFrame([point.to_dict() for point in result.equity_curve]).to_csv(
        equity_path,
        index=False,
    )
    pd.DataFrame([item.to_dict() for item in result.rejections]).to_csv(
        rejections_path,
        index=False,
    )

    benchmark_return = (
        result.benchmark.total_return_percent
        if result.benchmark is not None
        else 0.0
    )

    summary = {
        "start_date": result.start_date,
        "end_date": result.end_date,
        "ticker_count": len(result.tickers),
        "initial_cash": result.initial_cash,
        "ending_cash": result.ending_cash,
        "ending_equity": result.ending_equity,
        "total_return_amount": result.total_return_amount,
        "total_return_percent": result.total_return_percent,
        "maximum_drawdown_amount": result.maximum_drawdown_amount,
        "maximum_drawdown_percent": result.maximum_drawdown_percent,
        "completed_trades": result.completed_trades,
        "winning_trades": result.winning_trades,
        "losing_trades": result.losing_trades,
        "win_rate_percent": result.win_rate_percent,
        "profit_factor": (
            result.profit_factor if isfinite(result.profit_factor) else None
        ),
        "total_fees": result.total_fees,
        "average_exposure_percent": result.average_exposure_percent,
        "maximum_open_positions_observed": (
            result.maximum_open_positions_observed
        ),
        "rejected_signals": result.rejected_signals,
        "skipped_signals": result.skipped_signals,
        "benchmark_return_percent": benchmark_return,
        "excess_return_percent": result.total_return_percent - benchmark_return,
    }

    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    return {
        "json": json_path,
        "summary_csv": summary_path,
        "trades_csv": trades_path,
        "equity_csv": equity_path,
        "rejections_csv": rejections_path,
    }


def print_recent_activity(
    result: PortfolioBacktestResult,
    *,
    maximum_rows: int = 10,
) -> None:
    """Print recent trades and the most common rejection reasons."""

    print()
    print("=" * 100)
    print("RECENT PORTFOLIO TRADES")
    print("=" * 100)

    if not result.trades:
        print("No completed trades.")
    else:
        for trade in result.trades[-maximum_rows:]:
            print(
                f"{trade.exit_timestamp[:10]}  "
                f"{trade.ticker:<10} "
                f"Qty={trade.quantity:<10g} "
                f"Entry={trade.entry_price:>10.4f} "
                f"Exit={trade.exit_price:>10.4f} "
                f"PnL={trade.net_pnl:>+9.2f} "
                f"Reason={trade.exit_reason}"
            )

    print()
    print("REJECTION COUNTS")

    if not result.rejections:
        print("No rejected signals.")
    else:
        counts: dict[str, int] = {}
        for item in result.rejections:
            counts[item.reason_code] = counts.get(item.reason_code, 0) + 1

        for reason_code, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            print(f"{reason_code:<28} {count:>5}")

    print("=" * 100)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a deterministic shared-cash portfolio backtest."
    )

    parser.add_argument("tickers", nargs="*")
    parser.add_argument("--period", default=DEFAULT_PERIOD)
    parser.add_argument("--regime-period", default=DEFAULT_REGIME_PERIOD)

    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--risk", type=float, default=1.0)
    parser.add_argument("--max-position", type=float, default=25.0)
    parser.add_argument("--max-total-risk", type=float, default=4.0)
    parser.add_argument("--max-crypto", type=float, default=25.0)
    parser.add_argument("--max-open-positions", type=int, default=4)

    parser.add_argument("--stock-stop", type=float, default=5.0)
    parser.add_argument("--crypto-stop", type=float, default=5.0)
    parser.add_argument("--stock-trailing", type=float, default=7.5)
    parser.add_argument("--crypto-trailing", type=float, default=7.5)

    parser.add_argument("--commission-rate", type=float, default=0.0005)
    parser.add_argument("--minimum-fee", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)

    parser.add_argument("--fractional-stocks", action="store_true")
    parser.add_argument("--no-fractional-crypto", action="store_true")
    parser.add_argument("--no-crypto-regime", action="store_true")
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--no-force-close", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--show-activity", action="store_true")

    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()

    tickers = (
        arguments.tickers
        if arguments.tickers
        else list(DEFAULT_TICKERS)
    )

    config = PortfolioBacktestConfig(
        initial_cash=arguments.initial_cash,
        risk_per_trade_percent=arguments.risk,
        maximum_position_percent=arguments.max_position,
        maximum_total_open_risk_percent=arguments.max_total_risk,
        maximum_crypto_allocation_percent=arguments.max_crypto,
        maximum_open_positions=arguments.max_open_positions,
        stock_stop_loss_percent=arguments.stock_stop,
        crypto_stop_loss_percent=arguments.crypto_stop,
        stock_trailing_close_percent=arguments.stock_trailing,
        crypto_trailing_close_percent=arguments.crypto_trailing,
        commission_rate=arguments.commission_rate,
        minimum_fee=arguments.minimum_fee,
        slippage_bps=arguments.slippage_bps,
        allow_fractional_stocks=arguments.fractional_stocks,
        allow_fractional_crypto=(not arguments.no_fractional_crypto),
        force_close_at_end=(not arguments.no_force_close),
    )
    config.validate()

    prepared_data = prepare_portfolio_data(
        tickers=tickers,
        period=arguments.period,
        regime_period=arguments.regime_period,
        use_crypto_regime=(not arguments.no_crypto_regime),
    )

    result = run_portfolio_backtest(
        data_by_ticker=prepared_data,
        config=config,
        include_benchmark=(not arguments.no_benchmark),
    )

    print_portfolio_backtest_result(result)

    if arguments.show_activity:
        print_recent_activity(result)

    if not arguments.no_save:
        paths = save_portfolio_backtest_result(result)

        print()
        print("=" * 100)
        print("PORTFOLIO BACKTEST FILES")
        print("=" * 100)
        print(f"JSON:           {paths['json'].resolve()}")
        print(f"Summary CSV:    {paths['summary_csv'].resolve()}")
        print(f"Trades CSV:     {paths['trades_csv'].resolve()}")
        print(f"Equity CSV:     {paths['equity_csv'].resolve()}")
        print(f"Rejections CSV: {paths['rejections_csv'].resolve()}")
        print("=" * 100)

    print()
    print("Portfolio backtest completed successfully.")


if __name__ == "__main__":
    main()
