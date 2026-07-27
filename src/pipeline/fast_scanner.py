"""Fast technical scanner for the AI-Stock-Radar pipeline."""

from collections.abc import Iterable

import pandas as pd

from src.analysis.technical_indicators import (
    add_technical_indicators,
)
from src.data.download_stock import (
    download_stock_data,
    save_stock_data,
)
from src.pipeline.pipeline_models import FastScanResult
from src.scoring.radar_score import calculate_radar_score
from src.signals.signal_engine import generate_signal


REQUIRED_COLUMNS = [
    "Close",
    "EMA20",
    "EMA50",
    "RSI14",
    "MACD",
]


def _get_latest_valid_row(
    data: pd.DataFrame,
) -> pd.Series:
    """Return the newest row containing all technical values."""

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        missing_text = ", ".join(missing_columns)

        raise ValueError(
            "Technical dataframe is missing columns: "
            f"{missing_text}"
        )

    valid = data.dropna(
        subset=REQUIRED_COLUMNS,
    )

    if valid.empty:
        raise ValueError(
            "No row contains all required technical indicators."
        )

    return valid.iloc[-1]


def _to_float(value: object) -> float:
    """Safely convert a pandas or numeric value to float."""

    if isinstance(value, pd.Series):
        if value.empty:
            raise ValueError(
                "Cannot convert an empty Series to float."
            )

        return float(value.iloc[0])

    return float(value)


def scan_symbol(
    ticker: str,
    asset_type: str,
    period: str = "1y",
    save_raw_data: bool = False,
) -> FastScanResult:
    """Run the lightweight technical scan for one symbol."""

    if asset_type not in {"stock", "crypto"}:
        raise ValueError(
            "asset_type must be either 'stock' or 'crypto'."
        )

    market_data = download_stock_data(
        ticker,
        period=period,
    )

    if save_raw_data:
        save_stock_data(
            market_data,
            ticker,
        )

    technical_data = add_technical_indicators(
        market_data
    )

    latest = _get_latest_valid_row(
        technical_data
    )

    radar_score = calculate_radar_score(
        technical_data
    )

    signal = generate_signal(
        technical_data
    )

    reasons = list(radar_score.reasons)

    for reason in signal.reasons:
        signal_reason = f"Signal: {reason}"

        if signal_reason not in reasons:
            reasons.append(signal_reason)

    return FastScanResult(
        ticker=ticker,
        asset_type=asset_type,
        technical_score=int(radar_score.score),
        technical_stars=int(radar_score.stars),
        technical_signal=str(signal.action),
        technical_confidence=int(signal.confidence),
        close=_to_float(latest["Close"]),
        ema20=_to_float(latest["EMA20"]),
        ema50=_to_float(latest["EMA50"]),
        rsi14=_to_float(latest["RSI14"]),
        macd=_to_float(latest["MACD"]),
        reasons=reasons,
    )


def _scan_symbol_group(
    symbols: Iterable[str],
    asset_type: str,
    period: str,
    save_raw_data: bool,
) -> tuple[list[FastScanResult], list[str]]:
    """Scan one group and return successful and failed symbols."""

    symbol_list = list(symbols)
    total = len(symbol_list)

    successful: list[FastScanResult] = []
    failed: list[str] = []

    for index, ticker in enumerate(
        symbol_list,
        start=1,
    ):
        print(
            f"[{index}/{total}] Fast scanning "
            f"{asset_type}: {ticker}"
        )

        try:
            result = scan_symbol(
                ticker=ticker,
                asset_type=asset_type,
                period=period,
                save_raw_data=save_raw_data,
            )

        except Exception as error:
            failed.append(ticker)

            print(
                f"  Failed: {ticker} — {error}"
            )

            continue

        successful.append(result)

        print(
            f"  Score: {result.technical_score}/100 | "
            f"Signal: {result.technical_signal} | "
            f"Confidence: {result.technical_confidence}%"
        )

    successful.sort(
        key=lambda item: (
            item.technical_score,
            item.technical_confidence,
            item.macd,
        ),
        reverse=True,
    )

    return successful, failed


def scan_market_universe(
    stocks: Iterable[str],
    crypto: Iterable[str],
    period: str = "1y",
    save_raw_data: bool = False,
) -> tuple[
    list[FastScanResult],
    list[FastScanResult],
    list[str],
]:
    """Fast-scan all configured stocks and crypto assets."""

    print()
    print("=" * 80)
    print("STAGE 1 — FAST TECHNICAL SCAN")
    print("=" * 80)

    stock_results, failed_stocks = _scan_symbol_group(
        symbols=stocks,
        asset_type="stock",
        period=period,
        save_raw_data=save_raw_data,
    )

    print()
    print("-" * 80)
    print()

    crypto_results, failed_crypto = _scan_symbol_group(
        symbols=crypto,
        asset_type="crypto",
        period=period,
        save_raw_data=save_raw_data,
    )

    failed_symbols = (
        failed_stocks
        + failed_crypto
    )

    print()
    print("=" * 80)
    print("FAST SCAN COMPLETE")
    print("=" * 80)
    print(
        f"Successful stocks: {len(stock_results)}"
    )
    print(
        f"Successful crypto: {len(crypto_results)}"
    )
    print(
        f"Failed symbols: {len(failed_symbols)}"
    )
    print("=" * 80)

    return (
        stock_results,
        crypto_results,
        failed_symbols,
    )


def select_fast_scan_finalists(
    stock_results: list[FastScanResult],
    crypto_results: list[FastScanResult],
    stock_limit: int = 30,
    crypto_limit: int = 5,
) -> tuple[
    list[FastScanResult],
    list[FastScanResult],
]:
    """Select technical finalists for deep analysis."""

    if stock_limit < 1:
        raise ValueError(
            "stock_limit must be at least 1."
        )

    if crypto_limit < 1:
        raise ValueError(
            "crypto_limit must be at least 1."
        )

    selected_stocks = stock_results[:stock_limit]
    selected_crypto = crypto_results[:crypto_limit]

    return (
        selected_stocks,
        selected_crypto,
    )


def print_fast_scan_ranking(
    results: list[FastScanResult],
    title: str,
    limit: int = 10,
) -> None:
    """Print a compact technical ranking."""

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    if not results:
        print("No successful results.")
        return

    for rank, result in enumerate(
        results[:limit],
        start=1,
    ):
        stars = "⭐" * result.technical_stars

        print(
            f"{rank:>2}. "
            f"{result.ticker:<14} "
            f"Score: {result.technical_score:>3}/100  "
            f"Signal: {result.technical_signal:<5}  "
            f"Confidence: "
            f"{result.technical_confidence:>3}%  "
            f"{stars}"
        )