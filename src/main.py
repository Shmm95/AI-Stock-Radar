"""Entry point for AI-Stock-Radar."""

import pandas as pd

from config.watchlist import WATCHLIST
from src.analysis.technical_indicators import add_technical_indicators
from src.data.download_stock import download_stock_data, save_stock_data
from src.signals.signal_engine import generate_signal


def print_latest_summary(ticker: str, data: pd.DataFrame) -> None:
    """Print the latest available indicator values."""

    valid = data.dropna(
        subset=["Close", "EMA20", "EMA50", "RSI14", "MACD"]
    )

    if valid.empty:
        print(f"No valid data for {ticker}")
        return

    latest = valid.iloc[-1]

    print(f"Ticker: {ticker}")
    print(f"Close:  {float(latest['Close']):.2f}")
    print(f"EMA20:  {float(latest['EMA20']):.2f}")
    print(f"EMA50:  {float(latest['EMA50']):.2f}")
    print(f"RSI14:  {float(latest['RSI14']):.2f}")
    print(f"MACD:   {float(latest['MACD']):.4f}")


def main() -> None:
    """Download watchlist data, calculate indicators, and print signals."""

    period = "1y"
    total = len(WATCHLIST)

    print(f"Downloading {total} symbols ({period} daily data)...\n")

    for index, ticker in enumerate(WATCHLIST, start=1):
        print(f"[{index}/{total}] Downloading {ticker}...")

        data = download_stock_data(ticker, period=period)
        output_path = save_stock_data(data, ticker)
        print(f"Saved to {output_path}")

        data = add_technical_indicators(data)
        print_latest_summary(ticker, data)

        signal = generate_signal(data)

        print(f"Signal: {signal.action}")
        print(f"Confidence: {signal.confidence}%")

        for reason in signal.reasons:
            print(f" - {reason}")

        print("-" * 40)

    print("Done.")


if __name__ == "__main__":
    main()