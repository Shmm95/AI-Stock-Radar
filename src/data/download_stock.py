"""Download historical stock data and persist it to CSV."""

from pathlib import Path

import pandas as pd
import yfinance as yf


def get_project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent.parent


def ensure_raw_data_dir() -> Path:
    """Create data/raw if missing and return its path."""
    raw_dir = get_project_root() / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


def download_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch daily OHLCV history for a ticker over the given period."""
    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        progress=False,
    )

    if data.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


def save_stock_data(data: pd.DataFrame, ticker: str) -> Path:
    """Save stock data to data/raw/{ticker}.csv and return the file path."""
    raw_dir = ensure_raw_data_dir()
    output_path = raw_dir / f"{ticker}.csv"
    data.to_csv(output_path)
    return output_path


def download_and_save(ticker: str, period: str = "1y") -> Path:
    """Download stock history and save it as a CSV file."""
    data = download_stock_data(ticker, period=period)
    return save_stock_data(data, ticker)
