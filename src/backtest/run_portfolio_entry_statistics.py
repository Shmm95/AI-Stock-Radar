"""Observational entry statistics for completed portfolio baseline trades.

This module enriches a provenanced Trade Timing Attribution trade artifact
with causal signal-close and next-available-Open information from an immutable
research snapshot.  It does not replay the portfolio engine, alter trades, or
place orders.  MACD and all derived feature values are diagnostics only.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import numpy as np

from src.backtest.run_portfolio_entry_score_feature_audit import (
    causal_atr14_percent,
)
from src.backtest.run_research_data_snapshot import load_snapshot, sha256_file


SCHEMA_VERSION = 1
MODEL_BASELINE = "FIXED_BASELINE"
APPROVED_TIMING_STAMP = "20260802_085312"
APPROVED_SNAPSHOT_ID = "20260802_081049_d93145cb1dcf"
APPROVED_SOURCE_STAMP = "20260802_081449"
EXPECTED_TRADE_COUNT = 256
EXPECTED_WINDOW_COUNT = 13
APPROVED_STOP_PERCENT = 5.0
CONTROLLED_TICKERS = (
    "AAPL",
    "AMZN",
    "GOOGL",
    "META",
    "MSFT",
    "NVDA",
    "TSLA",
    "BTC-USD",
    "ETH-USD",
)

DEFAULT_SOURCE_DIRECTORY = Path("data/backtests/portfolio/stop_walk_forward")
DEFAULT_OUTPUT_DIRECTORY = Path("data/backtests/portfolio/entry_statistics")

GAP_BUCKETS = (
    "LT_NEG_2",
    "NEG_2_TO_NEG_1",
    "NEG_1_TO_NEG_0P5",
    "NEG_0P5_TO_0",
    "ZERO_TO_POS_0P5",
    "POS_0P5_TO_POS_1",
    "POS_1_TO_POS_2",
    "GE_POS_2",
)
OUTCOME_ORDER = ("WINNER", "LOSER", "FLAT")
ASSET_CLASS_ORDER = ("EQUITY", "CRYPTO")

REQUIRED_SOURCE_TRADE_COLUMNS = {
    "window_id",
    "model",
    "stock_stop_loss_percent",
    "crypto_stop_loss_percent",
    "ticker",
    "asset_class",
    "entry_timestamp",
    "exit_timestamp",
    "entry_portfolio_bar_index",
    "exit_portfolio_bar_index",
    "quantity",
    "entry_price",
    "exit_price",
    "entry_fee",
    "exit_fee",
    "total_fees",
    "gross_pnl",
    "net_pnl",
    "return_percent",
    "holding_period_bars",
    "exit_reason",
    "signal_score",
    "signal_reason",
    "exit_category",
}

TRADE_COLUMNS = (
    "entry_statistics_stamp",
    "timing_stamp",
    "source_stop_walk_forward_stamp",
    "snapshot_id",
    "snapshot_fingerprint",
    "source_trade_row",
    "trade_id",
    "window_id",
    "model",
    "ticker",
    "asset_class",
    "stock_stop_loss_percent",
    "crypto_stop_loss_percent",
    "signal_timestamp",
    "signal_year",
    "signal_close",
    "entry_timestamp",
    "entry_year",
    "entry_open",
    "entry_fill_price",
    "raw_entry_gap_percent",
    "raw_entry_gap_amount",
    "entry_gap_direction",
    "entry_gap_bucket",
    "entry_slippage_per_unit",
    "entry_slippage_percent",
    "entry_slippage_amount",
    "entry_fee",
    "entry_commission_amount",
    "entry_commission_percent",
    "entry_transaction_cost_amount",
    "entry_transaction_cost_percent",
    "signed_entry_timing_effect_amount",
    "signed_entry_timing_effect_percent",
    "adverse_entry_timing_cost_amount",
    "favorable_entry_timing_benefit_amount",
    "signal_score",
    "signal_reason",
    "signal_ema20",
    "signal_ema50",
    "signal_rsi14",
    "signal_macd",
    "signal_volume",
    "signal_regime_allowed",
    "trend_spread_percent",
    "price_extension_percent",
    "rsi_quality",
    "momentum_20_percent",
    "macd_percent",
    "atr14_percent",
    "low_atr14_quality",
    "quantity",
    "entry_notional",
    "initial_stop_percent",
    "initial_stop_price",
    "initial_stop_distance",
    "initial_risk_amount",
    "entry_portfolio_bar_index",
    "exit_timestamp",
    "exit_year",
    "exit_price",
    "exit_portfolio_bar_index",
    "exit_fee",
    "total_fees",
    "gross_pnl",
    "net_pnl",
    "return_percent",
    "holding_period_portfolio_bars",
    "holding_period_ticker_bars",
    "holding_period_calendar_days",
    "exit_reason",
    "exit_category",
    "outcome_class",
)

STATISTIC_METRICS = (
    "signal_close",
    "entry_open",
    "entry_fill_price",
    "raw_entry_gap_percent",
    "raw_entry_gap_amount",
    "entry_slippage_per_unit",
    "entry_slippage_percent",
    "entry_slippage_amount",
    "entry_commission_amount",
    "entry_commission_percent",
    "entry_transaction_cost_amount",
    "entry_transaction_cost_percent",
    "signed_entry_timing_effect_amount",
    "signed_entry_timing_effect_percent",
    "adverse_entry_timing_cost_amount",
    "favorable_entry_timing_benefit_amount",
    "signal_score",
    "signal_ema20",
    "signal_ema50",
    "signal_rsi14",
    "signal_macd",
    "signal_volume",
    "trend_spread_percent",
    "price_extension_percent",
    "rsi_quality",
    "momentum_20_percent",
    "macd_percent",
    "atr14_percent",
    "low_atr14_quality",
    "quantity",
    "entry_notional",
    "initial_stop_price",
    "initial_stop_distance",
    "initial_risk_amount",
    "exit_price",
    "exit_fee",
    "total_fees",
    "gross_pnl",
    "net_pnl",
    "return_percent",
    "holding_period_portfolio_bars",
    "holding_period_ticker_bars",
    "holding_period_calendar_days",
)

STATISTICS_COLUMNS = (
    "entry_statistics_stamp",
    "population",
    "group_type",
    "group_value",
    "metric",
    "population_trade_count",
    "count",
    "missing_count",
    "mean",
    "median",
    "population_standard_deviation",
    "minimum",
    "p01",
    "p05",
    "p10",
    "p25",
    "p75",
    "p90",
    "p95",
    "p99",
    "maximum",
    "negative_count",
    "negative_frequency",
    "flat_count",
    "flat_frequency",
    "positive_count",
    "positive_frequency",
)

FORMULAS = {
    "raw_entry_gap_percent": "(entry_open / signal_close - 1) * 100",
    "raw_entry_gap_amount": "(entry_open - signal_close) * quantity",
    "entry_slippage_per_unit": "entry_fill_price - entry_open",
    "entry_slippage_percent": "(entry_fill_price / entry_open - 1) * 100",
    "entry_slippage_amount": "(entry_fill_price - entry_open) * quantity",
    "entry_commission_percent": "entry_fee / (entry_fill_price * quantity) * 100",
    "entry_transaction_cost_amount": "entry_slippage_amount + entry_fee",
    "entry_transaction_cost_percent": (
        "entry_transaction_cost_amount / (entry_open * quantity) * 100"
    ),
    "signed_entry_timing_effect_amount": (
        "(entry_fill_price - signal_close) * quantity + entry_fee"
    ),
    "signed_entry_timing_effect_percent": (
        "signed_entry_timing_effect_amount / (signal_close * quantity) * 100"
    ),
    "adverse_entry_timing_cost_amount": (
        "max(signed_entry_timing_effect_amount, 0)"
    ),
    "favorable_entry_timing_benefit_amount": (
        "max(-signed_entry_timing_effect_amount, 0)"
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    if not Path(path).exists():
        raise FileNotFoundError(path)
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if value is None or value is pd.NA:
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value) if isfinite(float(value)) else None
    return value


def _expected_result_name(prefix: str, key: str, stamp: str) -> str:
    if key == "json":
        return f"{prefix}{stamp}.json"
    return f"{prefix}{key}_{stamp}.csv"


def verify_provenance_result_files(
    *,
    directory: Path,
    provenance: Mapping[str, Any],
    prefix: str,
    stamp: str,
) -> dict[str, Path]:
    """Verify every result declared by a source provenance document."""

    records = provenance.get("result_files")
    if not isinstance(records, dict) or not records:
        raise ValueError("Provenance has no result_files mapping.")
    verified: dict[str, Path] = {}
    for key, metadata in records.items():
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid provenance metadata for {key}.")
        expected_name = _expected_result_name(prefix, str(key), stamp)
        declared = Path(str(metadata.get("path", "")))
        if declared.name != expected_name:
            raise ValueError(
                f"Unexpected provenanced filename for {key}: {declared.name}"
            )
        path = Path(directory) / expected_name
        if not path.exists():
            raise FileNotFoundError(path)
        expected_hash = str(metadata.get("sha256", ""))
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Provenance hash mismatch for {path}.")
        verified[str(key)] = path
    return verified


def _asset_class(ticker: str) -> str:
    return "CRYPTO" if ticker.upper().endswith(("-USD", "-EUR", "-GBP")) else "EQUITY"


def _normalized_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp


def _normalize_market_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    output = frame.copy(deep=True)
    output.index = pd.to_datetime(output.index)
    if output.index.tz is not None:
        output.index = output.index.tz_convert(None)
    if output.index.duplicated().any():
        raise ValueError(f"Snapshot has duplicate timestamps for {ticker}.")
    return output.sort_index()


def _validated_regime_allowed(
    row: pd.Series, *, ticker: str, timestamp: pd.Timestamp
) -> bool:
    if "RegimeAllowed" not in row.index:
        raise ValueError(
            f"Invalid RegimeAllowed for ticker {ticker} at {timestamp}: field is missing."
        )
    value = row["RegimeAllowed"]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(
        f"Invalid RegimeAllowed for ticker {ticker} at {timestamp}: "
        f"invalid value {value!r}; expected a real boolean."
    )


def _is_entry_setup(
    row: pd.Series,
    *,
    ticker: str,
    regime_allowed: bool,
) -> bool:
    return (
        float(row["EMA20"]) > float(row["EMA50"])
        and float(row["Close"]) > float(row["EMA20"])
        and 45 <= float(row["RSI14"]) <= 70
        and (_asset_class(ticker) != "CRYPTO" or regime_allowed)
    )


def _entry_score(row: pd.Series) -> float:
    ema20 = float(row["EMA20"])
    ema50 = float(row["EMA50"])
    close = float(row["Close"])
    rsi = float(row["RSI14"])
    trend_spread = max((ema20 / ema50 - 1) * 100, 0.0)
    price_extension = max((close / ema20 - 1) * 100, 0.0)
    rsi_quality = max(0.0, 15.0 - abs(rsi - 57.5) * 1.2)
    score = (
        50.0
        + min(trend_spread * 5.0, 20.0)
        + min(price_extension * 3.0, 15.0)
        + rsi_quality
    )
    return round(min(score, 100.0), 4)


def classify_outcome(net_pnl: float) -> str:
    if float(net_pnl) > 0:
        return "WINNER"
    if float(net_pnl) < 0:
        return "LOSER"
    return "FLAT"


def classify_gap_direction(raw_entry_gap_percent: float) -> str:
    if float(raw_entry_gap_percent) > 0:
        return "ADVERSE"
    if float(raw_entry_gap_percent) < 0:
        return "FAVORABLE"
    return "FLAT"


def entry_gap_bucket(raw_entry_gap_percent: float) -> str:
    value = float(raw_entry_gap_percent)
    if value < -2:
        return "LT_NEG_2"
    if value < -1:
        return "NEG_2_TO_NEG_1"
    if value < -0.5:
        return "NEG_1_TO_NEG_0P5"
    if value < 0:
        return "NEG_0P5_TO_0"
    if value < 0.5:
        return "ZERO_TO_POS_0P5"
    if value < 1:
        return "POS_0P5_TO_POS_1"
    if value < 2:
        return "POS_1_TO_POS_2"
    return "GE_POS_2"


def calculate_entry_timing_fields(
    *,
    signal_close: float,
    entry_open: float,
    entry_fill_price: float,
    quantity: float,
    entry_fee: float,
) -> dict[str, float | str]:
    values = (signal_close, entry_open, entry_fill_price, quantity)
    if any(not isfinite(float(value)) or float(value) <= 0 for value in values):
        raise ValueError("Entry timing inputs must be finite and positive.")
    if not isfinite(float(entry_fee)) or float(entry_fee) < 0:
        raise ValueError("entry_fee must be finite and non-negative.")

    raw_gap_percent = (entry_open / signal_close - 1) * 100
    raw_gap_amount = (entry_open - signal_close) * quantity
    slippage_per_unit = entry_fill_price - entry_open
    slippage_percent = (entry_fill_price / entry_open - 1) * 100
    slippage_amount = slippage_per_unit * quantity
    commission_percent = entry_fee / (entry_fill_price * quantity) * 100
    transaction_cost_amount = slippage_amount + entry_fee
    transaction_cost_percent = transaction_cost_amount / (entry_open * quantity) * 100
    signed_effect_amount = (entry_fill_price - signal_close) * quantity + entry_fee
    signed_effect_percent = signed_effect_amount / (signal_close * quantity) * 100
    return {
        "raw_entry_gap_percent": raw_gap_percent,
        "raw_entry_gap_amount": raw_gap_amount,
        "entry_gap_direction": classify_gap_direction(raw_gap_percent),
        "entry_gap_bucket": entry_gap_bucket(raw_gap_percent),
        "entry_slippage_per_unit": slippage_per_unit,
        "entry_slippage_percent": slippage_percent,
        "entry_slippage_amount": slippage_amount,
        "entry_commission_percent": commission_percent,
        "entry_transaction_cost_amount": transaction_cost_amount,
        "entry_transaction_cost_percent": transaction_cost_percent,
        "signed_entry_timing_effect_amount": signed_effect_amount,
        "signed_entry_timing_effect_percent": signed_effect_percent,
        "adverse_entry_timing_cost_amount": max(signed_effect_amount, 0.0),
        "favorable_entry_timing_benefit_amount": max(-signed_effect_amount, 0.0),
    }


def reconstruct_initial_stop(
    *, entry_fill_price: float, quantity: float, stop_percent: float
) -> dict[str, float]:
    if float(stop_percent) <= 0 or float(stop_percent) >= 100:
        raise ValueError("stop_percent must be between zero and 100.")
    stop_price = round(entry_fill_price * (1 - stop_percent / 100), 8)
    distance = entry_fill_price - stop_price
    return {
        "initial_stop_percent": float(stop_percent),
        "initial_stop_price": stop_price,
        "initial_stop_distance": distance,
        "initial_risk_amount": round(distance * quantity, 2),
    }


def _optional_float(row: pd.Series, name: str) -> float | None:
    value = row.get(name)
    if value is None or pd.isna(value):
        return None
    result = float(value)
    return result if isfinite(result) else None


def causal_signal_features(
    data: pd.DataFrame, signal_position: int
) -> dict[str, float | None]:
    """Return existing diagnostic definitions using no post-signal rows."""

    row = data.iloc[signal_position]
    close = float(row["Close"])
    ema20 = float(row["EMA20"])
    ema50 = float(row["EMA50"])
    rsi = float(row["RSI14"])
    momentum: float | None = None
    if signal_position >= 20:
        prior_close = float(data.iloc[signal_position - 20]["Close"])
        if prior_close > 0:
            momentum = (close / prior_close - 1) * 100
    atr_percent = causal_atr14_percent(data, signal_position)
    macd = _optional_float(row, "MACD")
    return {
        "trend_spread_percent": (ema20 / ema50 - 1) * 100,
        "price_extension_percent": (close / ema20 - 1) * 100,
        "rsi_quality": max(0.0, 15.0 - abs(rsi - 57.5) * 1.2),
        "momentum_20_percent": momentum,
        "macd_percent": (macd / close * 100) if macd is not None and close > 0 else None,
        "atr14_percent": atr_percent,
        "low_atr14_quality": -atr_percent if atr_percent is not None else None,
    }


def _window_lookup(windows: pd.DataFrame) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    required = {"window_id", "test_start", "test_end_exclusive"}
    missing = required.difference(windows.columns)
    if missing:
        raise ValueError(f"Snapshot windows missing columns: {sorted(missing)}")
    if windows["window_id"].astype(str).duplicated().any():
        raise ValueError("Snapshot has duplicate window_id values.")
    result: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for row in windows.to_dict("records"):
        start = _normalized_timestamp(row["test_start"])
        end = _normalized_timestamp(row["test_end_exclusive"])
        if end <= start:
            raise ValueError(f"Invalid test window: {row['window_id']}")
        result[str(row["window_id"])] = (start, end)
    return result


def _validate_baseline_trades(
    trades: pd.DataFrame,
    *,
    expected_population: int | None,
    expected_windows: int | None,
) -> pd.DataFrame:
    missing = REQUIRED_SOURCE_TRADE_COLUMNS.difference(trades.columns)
    if missing:
        raise ValueError(f"Timing trades missing columns: {sorted(missing)}")
    source = trades.copy(deep=True)
    if "source_trade_row" not in source:
        source.insert(0, "source_trade_row", range(len(source)))
    baseline = source.loc[source["model"].astype(str) == MODEL_BASELINE].copy()
    if baseline.empty:
        raise ValueError("Timing source has no FIXED_BASELINE trades.")
    if expected_population is not None and len(baseline) != expected_population:
        raise ValueError(
            f"Expected {expected_population} FIXED_BASELINE trades; found {len(baseline)}."
        )
    window_count = baseline["window_id"].astype(str).nunique()
    if expected_windows is not None and window_count != expected_windows:
        raise ValueError(
            f"Expected {expected_windows} baseline windows; found {window_count}."
        )
    for column in ("stock_stop_loss_percent", "crypto_stop_loss_percent"):
        values = pd.to_numeric(baseline[column], errors="raise")
        if not values.eq(APPROVED_STOP_PERCENT).all():
            raise ValueError(
                f"FIXED_BASELINE must use 5% stock and crypto stops; {column} contradicts the approved baseline."
            )
    return baseline


def enrich_completed_trades(
    *,
    trades: pd.DataFrame,
    data_by_ticker: Mapping[str, pd.DataFrame],
    windows: pd.DataFrame,
    timing_stamp: str,
    source_stamp: str,
    snapshot_id: str,
    snapshot_fingerprint: str,
    entry_statistics_stamp: str,
    expected_population: int | None = None,
    expected_windows: int | None = None,
) -> pd.DataFrame:
    """Enrich completed baseline trades without replaying any decisions."""

    baseline = _validate_baseline_trades(
        trades,
        expected_population=expected_population,
        expected_windows=expected_windows,
    )
    window_by_id = _window_lookup(windows)
    normalized_data = {
        str(ticker).strip().upper(): _normalize_market_frame(frame, str(ticker))
        for ticker, frame in data_by_ticker.items()
    }
    records: list[dict[str, Any]] = []

    for source in baseline.to_dict("records"):
        ticker = str(source["ticker"]).strip().upper()
        window_id = str(source["window_id"])
        if ticker not in normalized_data:
            raise ValueError(f"Snapshot missing trade ticker: {ticker}")
        if window_id not in window_by_id:
            raise ValueError(f"Snapshot missing trade window: {window_id}")
        start, end = window_by_id[window_id]
        full_data = normalized_data[ticker]
        window_data = full_data.loc[(full_data.index >= start) & (full_data.index < end)]
        if len(window_data) < 2:
            raise ValueError(f"Insufficient window data for {window_id}/{ticker}.")

        entry_timestamp = _normalized_timestamp(source["entry_timestamp"])
        matches = int((window_data.index == entry_timestamp).sum())
        if matches != 1:
            raise ValueError(
                f"Expected exactly one entry row for {window_id}/{ticker}/{entry_timestamp}; found {matches}."
            )
        entry_position = int(window_data.index.get_loc(entry_timestamp))
        if entry_position < 2:
            raise ValueError(
                f"Cannot validate newly-valid signal for {window_id}/{ticker}/{entry_timestamp}."
            )
        signal_position_window = entry_position - 1
        signal_timestamp = pd.Timestamp(window_data.index[signal_position_window])
        signal_row = window_data.iloc[signal_position_window]
        previous_row = window_data.iloc[signal_position_window - 1]
        previous_timestamp = pd.Timestamp(window_data.index[signal_position_window - 1])
        signal_regime_allowed = _validated_regime_allowed(
            signal_row,
            ticker=ticker,
            timestamp=signal_timestamp,
        )
        previous_regime_allowed = _validated_regime_allowed(
            previous_row,
            ticker=ticker,
            timestamp=previous_timestamp,
        )
        if not _is_entry_setup(
            signal_row,
            ticker=ticker,
            regime_allowed=signal_regime_allowed,
        ) or _is_entry_setup(
            previous_row,
            ticker=ticker,
            regime_allowed=previous_regime_allowed,
        ):
            raise ValueError(
                f"Reconstructed signal is not newly valid for {window_id}/{ticker}/{entry_timestamp}."
            )
        if pd.Timestamp(window_data.index[signal_position_window + 1]) != entry_timestamp:
            raise ValueError(
                f"Entry is not the next available ticker Open for {window_id}/{ticker}."
            )

        full_signal_matches = full_data.index == signal_timestamp
        if int(full_signal_matches.sum()) != 1:
            raise ValueError(f"Signal row is not unique for {window_id}/{ticker}.")
        signal_position_full = int(full_data.index.get_loc(signal_timestamp))
        signal_close = float(signal_row["Close"])
        entry_open = float(window_data.iloc[entry_position]["Open"])
        entry_fill_price = float(source["entry_price"])
        quantity = float(source["quantity"])
        entry_fee = float(source["entry_fee"])
        if signal_close <= 0 or entry_open <= 0 or entry_fill_price <= 0 or quantity <= 0:
            raise ValueError(f"Non-positive entry input for {window_id}/{ticker}.")

        expected_asset_class = _asset_class(ticker)
        if str(source["asset_class"]) != expected_asset_class:
            raise ValueError(f"Asset-class mismatch for {window_id}/{ticker}.")
        expected_fill = round(entry_open * 1.0005, 8)
        if abs(entry_fill_price - expected_fill) > 0.00000001:
            raise ValueError(f"Entry fill/slippage mismatch for {window_id}/{ticker}.")
        expected_fee = round(max(entry_fill_price * quantity * 0.0005, 1.0), 2)
        if abs(entry_fee - expected_fee) > 0.00000001:
            raise ValueError(f"Entry fee mismatch for {window_id}/{ticker}.")
        expected_score = _entry_score(signal_row)
        if abs(float(source["signal_score"]) - expected_score) > 0.00005:
            raise ValueError(f"Signal-score mismatch for {window_id}/{ticker}.")

        stop_percent = float(
            source[
                "crypto_stop_loss_percent"
                if expected_asset_class == "CRYPTO"
                else "stock_stop_loss_percent"
            ]
        )
        if stop_percent != APPROVED_STOP_PERCENT:
            raise ValueError(f"Non-baseline stop for {window_id}/{ticker}.")
        stop_fields = reconstruct_initial_stop(
            entry_fill_price=entry_fill_price,
            quantity=quantity,
            stop_percent=stop_percent,
        )
        timing_fields = calculate_entry_timing_fields(
            signal_close=signal_close,
            entry_open=entry_open,
            entry_fill_price=entry_fill_price,
            quantity=quantity,
            entry_fee=entry_fee,
        )
        feature_fields = causal_signal_features(full_data, signal_position_full)

        exit_timestamp = _normalized_timestamp(source["exit_timestamp"])
        if exit_timestamp < entry_timestamp or exit_timestamp >= end:
            raise ValueError(f"Exit timestamp outside its test window for {window_id}/{ticker}.")
        last_exit_position = int(window_data.index.searchsorted(exit_timestamp, side="right")) - 1
        if last_exit_position < entry_position:
            raise ValueError(f"No causal ticker exit bar for {window_id}/{ticker}.")
        holding_ticker_bars = last_exit_position - entry_position
        holding_calendar_days = int((exit_timestamp - entry_timestamp).days)
        holding_portfolio_bars = int(source["holding_period_bars"])
        expected_portfolio_bars = max(
            int(source["exit_portfolio_bar_index"])
            - int(source["entry_portfolio_bar_index"]),
            0,
        )
        if holding_portfolio_bars != expected_portfolio_bars:
            raise ValueError(f"Portfolio holding-period mismatch for {window_id}/{ticker}.")

        exit_price = float(source["exit_price"])
        exit_fee = float(source["exit_fee"])
        expected_gross = round((exit_price - entry_fill_price) * quantity, 2)
        expected_net = round(
            (exit_price - entry_fill_price) * quantity - entry_fee - exit_fee,
            2,
        )
        if abs(float(source["gross_pnl"]) - expected_gross) > 0.00000001:
            raise ValueError(f"Gross-P&L mismatch for {window_id}/{ticker}.")
        if abs(float(source["net_pnl"]) - expected_net) > 0.00000001:
            raise ValueError(f"Net-P&L mismatch for {window_id}/{ticker}.")

        source_trade_row = int(source["source_trade_row"])
        record = {
            "entry_statistics_stamp": entry_statistics_stamp,
            "timing_stamp": timing_stamp,
            "source_stop_walk_forward_stamp": source_stamp,
            "snapshot_id": snapshot_id,
            "snapshot_fingerprint": snapshot_fingerprint,
            "source_trade_row": source_trade_row,
            "trade_id": f"{timing_stamp}:{source_trade_row:06d}",
            "window_id": window_id,
            "model": MODEL_BASELINE,
            "ticker": ticker,
            "asset_class": expected_asset_class,
            "stock_stop_loss_percent": float(source["stock_stop_loss_percent"]),
            "crypto_stop_loss_percent": float(source["crypto_stop_loss_percent"]),
            "signal_timestamp": signal_timestamp,
            "signal_year": int(signal_timestamp.year),
            "signal_close": signal_close,
            "entry_timestamp": entry_timestamp,
            "entry_year": int(entry_timestamp.year),
            "entry_open": entry_open,
            "entry_fill_price": entry_fill_price,
            **timing_fields,
            "entry_fee": entry_fee,
            "entry_commission_amount": entry_fee,
            "signal_score": float(source["signal_score"]),
            "signal_reason": str(source["signal_reason"]),
            "signal_ema20": float(signal_row["EMA20"]),
            "signal_ema50": float(signal_row["EMA50"]),
            "signal_rsi14": float(signal_row["RSI14"]),
            "signal_macd": _optional_float(signal_row, "MACD"),
            "signal_volume": _optional_float(signal_row, "Volume"),
            "signal_regime_allowed": signal_regime_allowed,
            **feature_fields,
            "quantity": quantity,
            "entry_notional": entry_fill_price * quantity,
            **stop_fields,
            "entry_portfolio_bar_index": int(source["entry_portfolio_bar_index"]),
            "exit_timestamp": exit_timestamp,
            "exit_year": int(exit_timestamp.year),
            "exit_price": exit_price,
            "exit_portfolio_bar_index": int(source["exit_portfolio_bar_index"]),
            "exit_fee": exit_fee,
            "total_fees": float(source["total_fees"]),
            "gross_pnl": float(source["gross_pnl"]),
            "net_pnl": float(source["net_pnl"]),
            "return_percent": float(source["return_percent"]),
            "holding_period_portfolio_bars": holding_portfolio_bars,
            "holding_period_ticker_bars": holding_ticker_bars,
            "holding_period_calendar_days": holding_calendar_days,
            "exit_reason": str(source["exit_reason"]),
            "exit_category": str(source["exit_category"]),
            "outcome_class": classify_outcome(float(source["net_pnl"])),
        }
        records.append(record)

    output = pd.DataFrame(records, columns=TRADE_COLUMNS)
    if output["source_trade_row"].duplicated().any() or output["trade_id"].duplicated().any():
        raise ValueError("Entry Statistics trade identifiers are not unique.")
    return output


def _finite_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric.map(lambda value: pd.isna(value) or isfinite(float(value))))


def _statistic_record(
    frame: pd.DataFrame,
    *,
    metric: str,
    entry_statistics_stamp: str,
    population: str,
    group_type: str,
    group_value: str,
) -> dict[str, Any]:
    values = _finite_numeric(frame[metric])
    valid = values.dropna()
    count = int(len(valid))
    negative_count = int((valid < 0).sum())
    flat_count = int((valid == 0).sum())
    positive_count = int((valid > 0).sum())

    def quantile(value: float) -> float | None:
        return float(valid.quantile(value, interpolation="linear")) if count else None

    return {
        "entry_statistics_stamp": entry_statistics_stamp,
        "population": population,
        "group_type": group_type,
        "group_value": str(group_value),
        "metric": metric,
        "population_trade_count": int(len(frame)),
        "count": count,
        "missing_count": int(len(frame) - count),
        "mean": float(valid.mean()) if count else None,
        "median": float(valid.median()) if count else None,
        "population_standard_deviation": float(valid.std(ddof=0)) if count else None,
        "minimum": float(valid.min()) if count else None,
        "p01": quantile(0.01),
        "p05": quantile(0.05),
        "p10": quantile(0.10),
        "p25": quantile(0.25),
        "p75": quantile(0.75),
        "p90": quantile(0.90),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "maximum": float(valid.max()) if count else None,
        "negative_count": negative_count,
        "negative_frequency": negative_count / count if count else 0.0,
        "flat_count": flat_count,
        "flat_frequency": flat_count / count if count else 0.0,
        "positive_count": positive_count,
        "positive_frequency": positive_count / count if count else 0.0,
    }


def build_statistics_table(
    trades: pd.DataFrame,
    *,
    entry_statistics_stamp: str,
    population: str,
    group_type: str,
    group_column: str | None = None,
    group_values: Iterable[Any] | None = None,
    metrics: Sequence[str] = STATISTIC_METRICS,
) -> pd.DataFrame:
    missing_metrics = set(metrics).difference(trades.columns)
    if missing_metrics:
        raise ValueError(f"Trade statistics missing metrics: {sorted(missing_metrics)}")
    groups: list[tuple[str, pd.DataFrame]] = []
    if group_column is None:
        groups.append(("ALL", trades))
    else:
        if group_column not in trades:
            raise ValueError(f"Trade statistics missing group column: {group_column}")
        values = list(group_values) if group_values is not None else sorted(
            trades[group_column].dropna().unique(), key=lambda value: str(value)
        )
        for value in values:
            groups.append((str(value), trades.loc[trades[group_column] == value]))

    records = [
        _statistic_record(
            group,
            metric=metric,
            entry_statistics_stamp=entry_statistics_stamp,
            population=population,
            group_type=group_type,
            group_value=group_value,
        )
        for group_value, group in groups
        for metric in metrics
    ]
    return pd.DataFrame(records, columns=STATISTICS_COLUMNS)


def build_aggregation_tables(
    trades: pd.DataFrame, *, entry_statistics_stamp: str
) -> dict[str, pd.DataFrame]:
    primary = "PRIMARY_ALL_COMPLETED_BASELINE_TRADES"
    sensitivity = trades.loc[trades["exit_reason"] != "FORCE_CLOSE_END"]
    return {
        "overall": build_statistics_table(
            trades,
            entry_statistics_stamp=entry_statistics_stamp,
            population=primary,
            group_type="ALL_TRADES",
        ),
        "outcomes": build_statistics_table(
            trades,
            entry_statistics_stamp=entry_statistics_stamp,
            population=primary,
            group_type="OUTCOME_CLASS",
            group_column="outcome_class",
            group_values=OUTCOME_ORDER,
        ),
        "asset_classes": build_statistics_table(
            trades,
            entry_statistics_stamp=entry_statistics_stamp,
            population=primary,
            group_type="ASSET_CLASS",
            group_column="asset_class",
            group_values=ASSET_CLASS_ORDER,
        ),
        "tickers": build_statistics_table(
            trades,
            entry_statistics_stamp=entry_statistics_stamp,
            population=primary,
            group_type="TICKER",
            group_column="ticker",
            group_values=CONTROLLED_TICKERS,
        ),
        "calendar_years": build_statistics_table(
            trades,
            entry_statistics_stamp=entry_statistics_stamp,
            population=primary,
            group_type="ENTRY_YEAR",
            group_column="entry_year",
            group_values=sorted(trades["entry_year"].unique()),
        ),
        "entry_gap_buckets": build_statistics_table(
            trades,
            entry_statistics_stamp=entry_statistics_stamp,
            population=primary,
            group_type="ENTRY_GAP_BUCKET",
            group_column="entry_gap_bucket",
            group_values=GAP_BUCKETS,
        ),
        "exit_reasons": build_statistics_table(
            trades,
            entry_statistics_stamp=entry_statistics_stamp,
            population=primary,
            group_type="EXIT_REASON",
            group_column="exit_reason",
        ),
        "exit_categories": build_statistics_table(
            trades,
            entry_statistics_stamp=entry_statistics_stamp,
            population=primary,
            group_type="EXIT_CATEGORY",
            group_column="exit_category",
        ),
        "force_close_excluded": build_statistics_table(
            sensitivity,
            entry_statistics_stamp=entry_statistics_stamp,
            population="SENSITIVITY_EXCLUDING_FORCE_CLOSE_END",
            group_type="ALL_TRADES",
        ),
    }


def _population_summary(trades: pd.DataFrame) -> dict[str, Any]:
    sensitivity = trades.loc[trades["exit_reason"] != "FORCE_CLOSE_END"]
    return {
        "primary_trade_count": int(len(trades)),
        "window_count": int(trades["window_id"].nunique()),
        "force_close_end_trade_count": int((trades["exit_reason"] == "FORCE_CLOSE_END").sum()),
        "force_close_excluded_trade_count": int(len(sensitivity)),
        "outcome_counts": {
            label: int((trades["outcome_class"] == label).sum()) for label in OUTCOME_ORDER
        },
        "asset_class_counts": {
            label: int((trades["asset_class"] == label).sum()) for label in ASSET_CLASS_ORDER
        },
        "gap_direction_counts": {
            label: int((trades["entry_gap_direction"] == label).sum())
            for label in ("ADVERSE", "FAVORABLE", "FLAT")
        },
        "mean_raw_entry_gap_percent": float(trades["raw_entry_gap_percent"].mean()),
        "median_raw_entry_gap_percent": float(trades["raw_entry_gap_percent"].median()),
        "mean_signed_entry_timing_effect_amount": float(
            trades["signed_entry_timing_effect_amount"].mean()
        ),
        "total_entry_transaction_cost_amount": float(
            trades["entry_transaction_cost_amount"].sum()
        ),
        "total_adverse_entry_timing_cost_amount": float(
            trades["adverse_entry_timing_cost_amount"].sum()
        ),
        "total_favorable_entry_timing_benefit_amount": float(
            trades["favorable_entry_timing_benefit_amount"].sum()
        ),
        "net_pnl_sum_across_reset_windows": float(trades["net_pnl"].sum()),
    }


def _field_definitions() -> dict[str, str]:
    definitions = {
        column: column.replace("_", " ") for column in TRADE_COLUMNS
    }
    definitions.update(
        {
            "source_trade_row": "Zero-based row number in the complete timing trades CSV before baseline filtering.",
            "trade_id": "Stable timing-stamp plus source-row identifier.",
            "signal_timestamp": "Immediately preceding ticker-local observation within the test window.",
            "signal_close": "Frozen snapshot Close at signal_timestamp.",
            "entry_open": "Frozen snapshot Open at entry_timestamp.",
            "entry_fill_price": "Authoritative simulated entry_price from the completed trade.",
            "entry_gap_direction": "ADVERSE above zero, FAVORABLE below zero, otherwise FLAT for a long entry.",
            "entry_gap_bucket": "Approved fixed bucket of raw_entry_gap_percent.",
            "entry_fee": "Authoritative simulated entry commission stored by the source trade.",
            "entry_commission_amount": "Alias of entry_fee for explicit cost reporting.",
            "signal_macd": "Diagnostic signal-row MACD; never an entry condition.",
            "macd_percent": "Diagnostic 100 * signal MACD / signal Close; never an entry condition.",
            "signal_regime_allowed": "Persisted final RegimeAllowed state; BTC EMA200 components are unavailable.",
            "holding_period_portfolio_bars": "Source union-portfolio timeline bar difference.",
            "holding_period_ticker_bars": "Ticker observations after entry through the last ticker row at or before exit_timestamp.",
            "holding_period_calendar_days": "Whole calendar-day difference between entry and exit timestamps.",
            "outcome_class": "WINNER, LOSER, or FLAT from authoritative rounded net_pnl.",
        }
    )
    return definitions


def build_json_payload(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": bundle["created_at"],
        "entry_statistics_stamp": bundle["entry_statistics_stamp"],
        "method": "Observational enrichment of provenanced completed baseline trades; no engine replay.",
        "population_definition": {
            "model": MODEL_BASELINE,
            "primary": "All completed baseline trades in each independently reset test window, including FORCE_CLOSE_END.",
            "sensitivity": "Primary population excluding raw exit_reason FORCE_CLOSE_END.",
            "expected_trade_count": bundle["expected_trade_count"],
            "expected_window_count": bundle["expected_window_count"],
            "controlled_tickers": list(CONTROLLED_TICKERS),
            "windows_are_separate_reset_capital_runs": True,
        },
        "source_lineage": {
            "timing_stamp": bundle["timing_stamp"],
            "source_stop_walk_forward_stamp": bundle["source_stamp"],
            "snapshot_id": bundle["snapshot_id"],
            "snapshot_fingerprint": bundle["snapshot_fingerprint"],
            "source_files": {
                label: {"path": str(path), "sha256": bundle["source_hashes"][label]}
                for label, path in bundle.get("source_files", {}).items()
            },
        },
        "field_definitions": _field_definitions(),
        "formulas": FORMULAS,
        "methodological_choices": {
            "calendar_grouping": "entry_year",
            "standard_deviation": "population standard deviation, ddof=0",
            "percentiles": "linear interpolation",
            "missing_diagnostics": "preserved as missing; no imputation and no trade removal",
            "sign_frequencies": "counts divided by non-missing metric count; exact zero is flat",
            "source_order": "preserved from the complete timing trades CSV",
            "gap_buckets": list(GAP_BUCKETS),
            "macd_role": "diagnostic only",
        },
        "trade_schema": list(TRADE_COLUMNS),
        "aggregation_schema": list(STATISTICS_COLUMNS),
        "aggregation_metrics": list(STATISTIC_METRICS),
        "summary": bundle["summary"],
        "aggregation_results": {
            name: frame.to_dict("records")
            for name, frame in bundle["aggregations"].items()
        },
        "limitations": [
            "Signal timestamp and Close are causally reconstructed because PortfolioTrade does not persist the full PortfolioSignal.",
            "Priority rank, technical score, confidence, accepted-trade SCORE_PREMIUM, and exact entry-time portfolio headroom are unsupported.",
            "RegimeAllowed is available, but exact persisted BTC EMA200 component values are not.",
            "Costs and fills are simulated; broker spread, latency, market impact, and real fills are unsupported.",
            "MACD and MACD-derived values are diagnostics only and do not affect population, filtering, or classification.",
            "Net P&L sums span independently reset windows and are not a continuously funded portfolio result.",
            "The provenance file hashes every other generated output; it cannot self-hash.",
        ],
    }


def build_entry_statistics(
    *,
    trades: pd.DataFrame,
    data_by_ticker: Mapping[str, pd.DataFrame],
    windows: pd.DataFrame,
    timing_stamp: str,
    source_stamp: str,
    snapshot_id: str,
    snapshot_fingerprint: str,
    expected_trade_count: int | None = None,
    expected_window_count: int | None = None,
    entry_statistics_stamp: str | None = None,
    source_files: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    stamp = entry_statistics_stamp or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    enriched = enrich_completed_trades(
        trades=trades,
        data_by_ticker=data_by_ticker,
        windows=windows,
        timing_stamp=timing_stamp,
        source_stamp=source_stamp,
        snapshot_id=snapshot_id,
        snapshot_fingerprint=snapshot_fingerprint,
        entry_statistics_stamp=stamp,
        expected_population=expected_trade_count,
        expected_windows=expected_window_count,
    )
    aggregations = build_aggregation_tables(enriched, entry_statistics_stamp=stamp)
    paths = {str(label): Path(path) for label, path in (source_files or {}).items()}
    hashes = {label: sha256_file(path) for label, path in paths.items()}
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "entry_statistics_stamp": stamp,
        "timing_stamp": timing_stamp,
        "source_stamp": source_stamp,
        "snapshot_id": snapshot_id,
        "snapshot_fingerprint": snapshot_fingerprint,
        "expected_trade_count": expected_trade_count,
        "expected_window_count": expected_window_count,
        "trades": enriched,
        "aggregations": aggregations,
        "summary": _population_summary(enriched),
        "source_files": paths,
        "source_hashes": hashes,
    }


def _validate_source_reconciliation(
    *,
    trades: pd.DataFrame,
    summary: pd.DataFrame,
    window_models: pd.DataFrame,
    windows: pd.DataFrame,
    expected_trade_count: int,
    expected_window_count: int,
) -> None:
    baseline = trades.loc[trades["model"].astype(str) == MODEL_BASELINE]
    baseline_summary = summary.loc[summary["model"].astype(str) == MODEL_BASELINE]
    baseline_windows = window_models.loc[
        window_models["model"].astype(str) == MODEL_BASELINE
    ]
    if len(baseline_summary) != 1:
        raise ValueError("Timing summary must have one FIXED_BASELINE row.")
    summary_row = baseline_summary.iloc[0]
    if int(summary_row["trade_count"]) != expected_trade_count:
        raise ValueError("Timing summary baseline trade count mismatch.")
    if int(summary_row["window_count"]) != expected_window_count:
        raise ValueError("Timing summary baseline window count mismatch.")
    if len(baseline) != expected_trade_count:
        raise ValueError("Timing trades baseline population mismatch.")
    if len(baseline_windows) != expected_window_count:
        raise ValueError("Timing window-model baseline count mismatch.")
    if int(pd.to_numeric(baseline_windows["completed_trades"]).sum()) != expected_trade_count:
        raise ValueError("Timing window-model completed-trade reconciliation failed.")
    snapshot_window_ids = set(windows["window_id"].astype(str))
    baseline_window_ids = set(baseline_windows["window_id"].astype(str))
    trade_window_ids = set(baseline["window_id"].astype(str))
    if len(snapshot_window_ids) != expected_window_count:
        raise ValueError("Snapshot window count mismatch.")
    if baseline_window_ids != snapshot_window_ids or trade_window_ids != snapshot_window_ids:
        raise ValueError("Baseline timing windows do not match the snapshot windows.")


def load_verified_source(
    *,
    snapshot_directory: Path,
    timing_directory: Path,
    timing_stamp: str,
    source_directory: Path = DEFAULT_SOURCE_DIRECTORY,
    project_root: Path = Path("."),
    verify_code: bool = True,
    expected_timing_stamp: str = APPROVED_TIMING_STAMP,
    expected_snapshot_id: str = APPROVED_SNAPSHOT_ID,
    expected_source_stamp: str = APPROVED_SOURCE_STAMP,
    expected_trade_count: int = EXPECTED_TRADE_COUNT,
    expected_window_count: int = EXPECTED_WINDOW_COUNT,
) -> dict[str, Any]:
    """Load and verify the complete approved Entry Statistics source chain."""

    if timing_stamp != expected_timing_stamp:
        raise ValueError(
            f"Entry Statistics requires timing stamp {expected_timing_stamp}; received {timing_stamp}."
        )
    snapshot_directory = Path(snapshot_directory)
    timing_directory = Path(timing_directory)
    source_directory = Path(source_directory)
    snapshot = load_snapshot(
        snapshot_directory,
        verify_code=verify_code,
        project_root=Path(project_root),
    )
    manifest = snapshot["manifest"]
    if manifest.get("snapshot_id") != expected_snapshot_id:
        raise ValueError("Snapshot ID does not match the approved Entry Statistics source.")
    snapshot_tickers = set(manifest.get("market_files", {}))
    if snapshot_tickers != set(CONTROLLED_TICKERS):
        raise ValueError("Snapshot does not contain exactly the controlled nine-asset basket.")

    timing_provenance_path = (
        timing_directory / f"portfolio_trade_timing_provenance_{timing_stamp}.json"
    )
    timing_provenance = _read_json(timing_provenance_path)
    if timing_provenance.get("timing_stamp") != timing_stamp:
        raise ValueError("Timing provenance stamp mismatch.")
    if timing_provenance.get("source_stamp") != expected_source_stamp:
        raise ValueError("Timing provenance source stamp mismatch.")
    if timing_provenance.get("snapshot_id") != expected_snapshot_id:
        raise ValueError("Timing provenance snapshot ID mismatch.")
    if timing_provenance.get("snapshot_fingerprint") != manifest.get("fingerprint"):
        raise ValueError("Timing provenance snapshot fingerprint mismatch.")
    timing_files = verify_provenance_result_files(
        directory=timing_directory,
        provenance=timing_provenance,
        prefix="portfolio_trade_timing_",
        stamp=timing_stamp,
    )
    required_timing_files = {"json", "trades", "summary", "window_models"}
    if not required_timing_files.issubset(timing_files):
        raise ValueError("Timing provenance is missing required result files.")

    source_stamp = str(timing_provenance["source_stamp"])
    source_provenance_path = (
        source_directory / f"portfolio_stop_walk_forward_provenance_{source_stamp}.json"
    )
    source_provenance = _read_json(source_provenance_path)
    if source_provenance.get("source_stamp") != expected_source_stamp:
        raise ValueError("Stop walk-forward provenance stamp mismatch.")
    if source_provenance.get("snapshot_id") != expected_snapshot_id:
        raise ValueError("Stop walk-forward provenance snapshot ID mismatch.")
    if source_provenance.get("snapshot_fingerprint") != manifest.get("fingerprint"):
        raise ValueError("Stop walk-forward provenance fingerprint mismatch.")
    manifest_path = snapshot_directory / "manifest.json"
    if source_provenance.get("snapshot_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("Stop walk-forward snapshot manifest hash mismatch.")
    source_files = verify_provenance_result_files(
        directory=source_directory,
        provenance=source_provenance,
        prefix="portfolio_stop_walk_forward_",
        stamp=source_stamp,
    )
    if "json" not in source_files:
        raise ValueError("Stop walk-forward provenance is missing its JSON result.")

    timing_payload = _read_json(timing_files["json"])
    if timing_payload.get("source_stamp") != source_stamp:
        raise ValueError("Timing JSON source stamp mismatch.")
    if MODEL_BASELINE not in timing_payload.get("models", []):
        raise ValueError("Timing JSON does not include FIXED_BASELINE.")
    stop_payload = _read_json(source_files["json"])
    profile = stop_payload.get("profiles", {}).get(MODEL_BASELINE, {})
    if (
        float(profile.get("stock_stop_loss_percent", -1)) != APPROVED_STOP_PERCENT
        or float(profile.get("crypto_stop_loss_percent", -1)) != APPROVED_STOP_PERCENT
    ):
        raise ValueError("Stop source FIXED_BASELINE profile is not 5%/5%.")
    config = snapshot["config"]
    if (
        float(config.stock_stop_loss_percent) != APPROVED_STOP_PERCENT
        or float(config.crypto_stop_loss_percent) != APPROVED_STOP_PERCENT
    ):
        raise ValueError("Snapshot baseline stop configuration is not 5%/5%.")

    trades = pd.read_csv(timing_files["trades"])
    trades.insert(0, "source_trade_row", range(len(trades)))
    summary = pd.read_csv(timing_files["summary"])
    window_models = pd.read_csv(timing_files["window_models"])
    _validate_source_reconciliation(
        trades=trades,
        summary=summary,
        window_models=window_models,
        windows=snapshot["windows"],
        expected_trade_count=expected_trade_count,
        expected_window_count=expected_window_count,
    )

    verified_paths: dict[str, Path] = {
        "snapshot_manifest": manifest_path,
        "snapshot_config": snapshot_directory / manifest["config"]["path"],
        "snapshot_windows": snapshot_directory / manifest["windows"]["path"],
        "timing_provenance": timing_provenance_path,
        "source_stop_walk_forward_provenance": source_provenance_path,
        "entry_statistics_code": Path(__file__).resolve(),
        "entry_feature_definition_code": Path(
            "src/backtest/run_portfolio_entry_score_feature_audit.py"
        ).resolve(),
    }
    for ticker, metadata in manifest["market_files"].items():
        verified_paths[f"snapshot_market:{ticker}"] = snapshot_directory / metadata["path"]
    verified_paths.update(
        {f"timing_result:{name}": path for name, path in timing_files.items()}
    )
    verified_paths.update(
        {f"stop_walk_forward_result:{name}": path for name, path in source_files.items()}
    )
    return {
        "snapshot": snapshot,
        "trades": trades,
        "summary": summary,
        "window_models": window_models,
        "timing_stamp": timing_stamp,
        "source_stamp": source_stamp,
        "source_files": verified_paths,
        "expected_trade_count": expected_trade_count,
        "expected_window_count": expected_window_count,
    }


def run_entry_statistics(source: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = source["snapshot"]
    manifest = snapshot["manifest"]
    return build_entry_statistics(
        trades=source["trades"],
        data_by_ticker=snapshot["data_by_ticker"],
        windows=snapshot["windows"],
        timing_stamp=str(source["timing_stamp"]),
        source_stamp=str(source["source_stamp"]),
        snapshot_id=str(manifest["snapshot_id"]),
        snapshot_fingerprint=str(manifest["fingerprint"]),
        expected_trade_count=int(source["expected_trade_count"]),
        expected_window_count=int(source["expected_window_count"]),
        source_files=source["source_files"],
    )


def save_entry_statistics(
    bundle: Mapping[str, Any],
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> dict[str, Path]:
    """Save CSV/JSON results and complete source/result provenance."""

    output_directory = Path(output_directory)
    stamp = str(bundle["entry_statistics_stamp"])
    frames = {"trades": bundle["trades"], **bundle["aggregations"]}
    paths = {
        name: output_directory / f"portfolio_entry_statistics_{name}_{stamp}.csv"
        for name in frames
    }
    paths["json"] = output_directory / f"portfolio_entry_statistics_{stamp}.json"
    paths["provenance"] = (
        output_directory / f"portfolio_entry_statistics_provenance_{stamp}.json"
    )
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(existing[0])

    for label, path in bundle.get("source_files", {}).items():
        actual_hash = sha256_file(Path(path))
        expected_hash = bundle["source_hashes"][label]
        if actual_hash != expected_hash:
            raise ValueError(f"Source changed before save: {path}")

    output_directory.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(paths[name], index=False, lineterminator="\n")
    payload = build_json_payload(bundle)
    paths["json"].write_text(
        json.dumps(_json_safe(payload), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    result_files = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in paths.items()
        if name != "provenance"
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "entry_statistics_stamp": stamp,
        "timing_stamp": bundle["timing_stamp"],
        "source_stop_walk_forward_stamp": bundle["source_stamp"],
        "snapshot_id": bundle["snapshot_id"],
        "snapshot_fingerprint": bundle["snapshot_fingerprint"],
        "source_files": {
            label: {
                "path": str(Path(path).resolve()),
                "sha256": bundle["source_hashes"][label],
            }
            for label, path in bundle.get("source_files", {}).items()
        },
        "result_files": result_files,
        "provenance_self_hash_excluded": True,
    }
    paths["provenance"].write_text(
        json.dumps(_json_safe(provenance), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths


def _print_summary(bundle: Mapping[str, Any], *, no_save: bool) -> None:
    summary = bundle["summary"]
    mode = "NO-SAVE ANALYSIS" if no_save else "SAVING ANALYSIS"
    print(f"ENTRY STATISTICS — OBSERVATIONAL {mode}")
    print("Provenance validation: PASS")
    print(f"Timing stamp: {bundle['timing_stamp']}")
    print(f"Source stop walk-forward stamp: {bundle['source_stamp']}")
    print(f"Snapshot: {bundle['snapshot_id']}")
    print(
        f"Primary population: {summary['primary_trade_count']} completed FIXED_BASELINE trades "
        f"across {summary['window_count']} reset-capital windows"
    )
    print(
        f"Outcomes: WINNER={summary['outcome_counts']['WINNER']} "
        f"LOSER={summary['outcome_counts']['LOSER']} "
        f"FLAT={summary['outcome_counts']['FLAT']}"
    )
    print(
        f"Asset classes: EQUITY={summary['asset_class_counts']['EQUITY']} "
        f"CRYPTO={summary['asset_class_counts']['CRYPTO']}"
    )
    print(
        f"Gap direction: ADVERSE={summary['gap_direction_counts']['ADVERSE']} "
        f"FAVORABLE={summary['gap_direction_counts']['FAVORABLE']} "
        f"FLAT={summary['gap_direction_counts']['FLAT']}"
    )
    print(
        "Raw entry gap: "
        f"mean={summary['mean_raw_entry_gap_percent']:.6f}% "
        f"median={summary['median_raw_entry_gap_percent']:.6f}%"
    )
    print(
        "Signed entry timing effect: "
        f"mean=${summary['mean_signed_entry_timing_effect_amount']:.6f}"
    )
    print(
        "Entry transaction costs across reset windows: "
        f"${summary['total_entry_transaction_cost_amount']:.2f}"
    )
    print(
        "Timing components across reset windows: "
        f"adverse=${summary['total_adverse_entry_timing_cost_amount']:.2f} "
        f"favorable=${summary['total_favorable_entry_timing_benefit_amount']:.2f}"
    )
    print(
        f"FORCE_CLOSE_END sensitivity: excluded={summary['force_close_end_trade_count']} "
        f"remaining={summary['force_close_excluded_trade_count']}"
    )
    if no_save:
        print("Output artifacts saved: 0 (--no-save)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Observational Entry Statistics for provenanced portfolio trades"
    )
    parser.add_argument("--snapshot-directory", type=Path, required=True)
    parser.add_argument("--timing-directory", type=Path, required=True)
    parser.add_argument("--timing-stamp", required=True)
    parser.add_argument(
        "--source-directory", type=Path, default=DEFAULT_SOURCE_DIRECTORY
    )
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--no-save", action="store_true")
    arguments = parser.parse_args()

    source = load_verified_source(
        snapshot_directory=arguments.snapshot_directory,
        timing_directory=arguments.timing_directory,
        timing_stamp=arguments.timing_stamp,
        source_directory=arguments.source_directory,
    )
    bundle = run_entry_statistics(source)
    if arguments.no_save:
        _print_summary(bundle, no_save=True)
        return
    _print_summary(bundle, no_save=False)
    for name, path in save_entry_statistics(bundle, arguments.output_directory).items():
        print(name, path.resolve())


if __name__ == "__main__":
    main()
