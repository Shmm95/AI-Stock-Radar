"""Entry point for AI-Stock-Radar."""

from config.watchlist import WATCHLIST
from src.data.download_stock import download_and_save


def main() -> None:
    """Download one year of daily data for every symbol in the watchlist."""
    period = "1y"
    total = len(WATCHLIST)

    print(f"Downloading {total} symbols ({period} daily data)...\n")

    for index, ticker in enumerate(WATCHLIST, start=1):
        print(f"[{index}/{total}] Downloading {ticker}...")
        output_path = download_and_save(ticker, period=period)
        print(f"  Saved to {output_path}\n")

    print("Done.")


if __name__ == "__main__":
    main()
