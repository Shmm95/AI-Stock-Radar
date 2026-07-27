"""Technical indicator calculations using pandas."""

import pandas as pd


def _get_close(data: pd.DataFrame) -> pd.Series:
    """Return the Close price series from raw or multi-index OHLCV data."""
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            return close.iloc[:, 0]
        return close

    if "Close" in data.columns:
        return data["Close"]

    raise ValueError("DataFrame must contain a Close column.")


def add_ema20(data: pd.DataFrame) -> pd.DataFrame:
    """Add a 20-period exponential moving average column."""
    result = data.copy()
    result["EMA20"] = _get_close(result).ewm(span=20, adjust=False).mean()
    return result


def add_ema50(data: pd.DataFrame) -> pd.DataFrame:
    """Add a 50-period exponential moving average column."""
    result = data.copy()
    result["EMA50"] = _get_close(result).ewm(span=50, adjust=False).mean()
    return result


def add_rsi14(data: pd.DataFrame) -> pd.DataFrame:
    """Add a 14-period Relative Strength Index column."""
    result = data.copy()
    close = _get_close(result)
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    rs = avg_gain / avg_loss
    result["RSI14"] = 100 - (100 / (1 + rs))
    return result


def add_macd(data: pd.DataFrame) -> pd.DataFrame:
    """Add a MACD line column (12/26 EMA spread)."""
    result = data.copy()
    close = _get_close(result)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    result["MACD"] = ema12 - ema26
    return result


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Apply all technical indicators and return the enriched dataframe."""
    with_indicators = add_ema20(data)
    with_indicators = add_ema50(with_indicators)
    with_indicators = add_rsi14(with_indicators)
    return add_macd(with_indicators)
