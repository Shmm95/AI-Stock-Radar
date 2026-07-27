"""Entry point for AI-Stock-Radar."""

from src.data.download_stock import download_stock_data, save_stock_data


def main() -> None:
    """Download one year of AAPL data, save it, and preview the result."""
    ticker = "AAPL"
    period = "1y"

    data = download_stock_data(ticker, period=period)
    output_path = save_stock_data(data, ticker)
    print(f"Saved {ticker} data to {output_path}")

    print(f"\nFirst 5 rows of {ticker} ({period}):")
    print(data.head())


if __name__ == "__main__":
    main()
