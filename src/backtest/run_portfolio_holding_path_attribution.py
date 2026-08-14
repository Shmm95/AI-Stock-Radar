"""Observational holding-path attribution for provenanced baseline trades.

The analyzer consumes the single approved Forward Return Statistics result and
the immutable research snapshot.  It never replays the portfolio engine and it
never changes an entry, exit, order, fee, slippage, risk, or strategy decision.

Daily OHLC cannot reveal intrabar ordering.  Consequently STOP_LOSS trades are
reported with a certain/censored path plus explicit possible bounds for the
exit bar.  Those bounds are diagnostics, not executable MFE/MAE claims.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from src.backtest.run_portfolio_entry_statistics import (
    ASSET_CLASS_ORDER,
    CONTROLLED_TICKERS,
    EXPECTED_TRADE_COUNT,
    EXPECTED_WINDOW_COUNT,
    MODEL_BASELINE,
    OUTCOME_ORDER,
    verify_provenance_result_files,
)
from src.backtest.run_portfolio_forward_return_statistics import (
    FORWARD_COLUMNS,
    _normalize_and_validate_market_frame,
    _window_lookup,
)
from src.backtest.run_research_data_snapshot import load_snapshot, sha256_file


SCHEMA_VERSION = 1
APPROVED_FORWARD_RETURN_STATISTICS_STAMP = "20260802_134341"
APPROVED_ENTRY_STATISTICS_STAMP = "20260802_103007"
APPROVED_TIMING_STAMP = "20260802_085312"
APPROVED_SOURCE_STAMP = "20260802_081449"
APPROVED_SNAPSHOT_ID = "20260802_081049_d93145cb1dcf"
APPROVED_SNAPSHOT_FINGERPRINT = (
    "d93145cb1dcf3f837415f5b28ac29eeb51c13bc8ce5b8b1187aa34ebfc01ad44"
)
APPROVED_FORWARD_CODE_SHA256 = (
    "5cc585a9653e0fcb7391769a6e1f908715a4ae7c9d8e8d6f448523310a60500f"
)
APPROVED_FORWARD_TEST_SHA256 = (
    "861a54f38fb1d93cca32b2a9ad3714a5029bc636815afe8ddeb5c2227197bce6"
)
DEFAULT_FORWARD_RETURN_STATISTICS_DIRECTORY = Path(
    "data/backtests/portfolio/forward_return_statistics"
)
DEFAULT_OUTPUT_DIRECTORY = Path("data/backtests/portfolio/holding_path_attribution")

PRIMARY_POPULATION = "PRIMARY_ALL_COMPLETED_BASELINE_TRADES"
SENSITIVITY_POPULATION = "SENSITIVITY_EXCLUDING_FORCE_CLOSE_END"
OPEN_EXIT = "OPEN_EXIT"
INTRABAR_STOP_BOUNDED = "INTRABAR_STOP_BOUNDED"
CLOSE_EXIT = "CLOSE_EXIT"
MFE_BEFORE_MAE = "MFE_BEFORE_MAE"
MAE_BEFORE_MFE = "MAE_BEFORE_MFE"
AMBIGUOUS_SAME_BAR = "AMBIGUOUS_SAME_BAR"

OFFICIAL_EXIT_REASON_COUNTS = {
    "STOP_LOSS": 114,
    "EXIT_SIGNAL_NEXT_OPEN": 85,
    "FORCE_CLOSE_END": 42,
    "GAP_STOP_LOSS": 15,
}
EXPECTED_FORWARD_RESULT_KEYS = {
    "asset_classes",
    "availability",
    "entry_gap_buckets",
    "entry_years",
    "exit_categories",
    "exit_reasons",
    "exit_relations",
    "force_close_excluded",
    "json",
    "outcomes",
    "overall",
    "tickers",
    "trades",
    "windows",
}
_STRICT_SOURCE_AUTHORIZATION = object()

REQUIRED_TRADE_COLUMNS = {
    "forward_return_statistics_stamp",
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
    "entry_timestamp",
    "entry_fill_price",
    "entry_notional",
    "quantity",
    "initial_stop_percent",
    "initial_stop_price",
    "initial_risk_amount",
    "exit_timestamp",
    "exit_price",
    "gross_pnl",
    "net_pnl",
    "return_percent",
    "holding_period_ticker_bars",
    "exit_reason",
    "exit_category",
    "outcome_class",
}

METRIC_FIELDS = (
    "effective_exit_market_timestamp",
    "exit_semantic",
    "path_ticker_bar_count",
    "full_observed_ticker_bar_count",
    "holding_bucket",
    "market_exit_reference_price",
    "market_exit_reference_return_percent",
    "net_realized_return_percent",
    "gross_realized_r_multiple",
    "net_realized_r_multiple",
    "censored_mfe_price",
    "censored_mfe_percent",
    "possible_mfe_price",
    "possible_mfe_percent",
    "mfe_bound_width_percent",
    "mfe_exact",
    "censored_mae_price",
    "censored_mae_percent",
    "possible_mae_price",
    "possible_mae_percent",
    "mae_bound_width_percent",
    "mae_exact",
    "censored_mfe_r_multiple",
    "possible_mfe_r_multiple",
    "censored_mae_r_multiple",
    "possible_mae_r_multiple",
    "ticker_bars_to_censored_mfe",
    "ticker_bars_to_censored_mae",
    "censored_mfe_timestamp",
    "censored_mae_timestamp",
    "censored_extreme_order",
    "giveback_from_censored_mfe_percent",
    "censored_mfe_capture_ratio",
    "recovery_from_censored_mae_percent",
    "exit_location_in_censored_range",
    "maximum_close_excursion_percent",
    "minimum_close_excursion_percent",
    "maximum_close_drawdown_percent",
    "underwater_close_bar_count",
    "observed_close_bar_count",
    "underwater_close_bar_fraction",
    "intrabar_order_uncertain",
)
HOLDING_TRADE_COLUMNS = ("holding_path_attribution_stamp", *FORWARD_COLUMNS, *METRIC_FIELDS)

PATH_COLUMNS = (
    "holding_path_attribution_stamp",
    "source_trade_row",
    "trade_id",
    "window_id",
    "ticker",
    "exit_reason",
    "exit_semantic",
    "ticker_bar_offset",
    "ticker_timestamp",
    "bar_role",
    "raw_open",
    "raw_high",
    "raw_low",
    "raw_close",
    "open_included",
    "high_low_close_included",
    "known_stop_touch_price",
    "censored_high_price",
    "censored_low_price",
    "censored_close_price",
    "cumulative_censored_mfe_percent",
    "cumulative_censored_mae_percent",
    "cumulative_possible_mfe_percent",
    "cumulative_possible_mae_percent",
    "is_entry_bar",
    "is_effective_exit_bar",
    "intrabar_order_uncertain",
)

STATISTIC_METRICS = (
    "net_realized_return_percent",
    "gross_realized_r_multiple",
    "net_realized_r_multiple",
    "censored_mfe_percent",
    "possible_mfe_percent",
    "mfe_bound_width_percent",
    "censored_mae_percent",
    "possible_mae_percent",
    "mae_bound_width_percent",
    "censored_mfe_r_multiple",
    "possible_mfe_r_multiple",
    "censored_mae_r_multiple",
    "possible_mae_r_multiple",
    "ticker_bars_to_censored_mfe",
    "ticker_bars_to_censored_mae",
    "giveback_from_censored_mfe_percent",
    "censored_mfe_capture_ratio",
    "recovery_from_censored_mae_percent",
    "exit_location_in_censored_range",
    "maximum_close_excursion_percent",
    "minimum_close_excursion_percent",
    "maximum_close_drawdown_percent",
    "underwater_close_bar_fraction",
    "path_ticker_bar_count",
    "full_observed_ticker_bar_count",
)
STATISTICS_COLUMNS = (
    "holding_path_attribution_stamp",
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
SCREEN_COLUMNS = (
    "holding_path_attribution_stamp",
    "check",
    "passed",
    "expected",
    "actual",
    "detail",
    "authorizes_strategy_change",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
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


def _timestamp(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    return result.tz_convert(None) if result.tzinfo is not None else result


def _finite_positive(value: Any, name: str) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def exit_semantic(exit_reason: str) -> str:
    reason = str(exit_reason)
    if reason in {"EXIT_SIGNAL_NEXT_OPEN", "GAP_STOP_LOSS"}:
        return OPEN_EXIT
    if reason == "STOP_LOSS":
        return INTRABAR_STOP_BOUNDED
    if reason == "FORCE_CLOSE_END":
        return CLOSE_EXIT
    raise ValueError(f"Unsupported exit_reason: {reason}")


def holding_bucket(holding_period_ticker_bars: Any) -> str:
    value = int(holding_period_ticker_bars)
    if value < 0 or float(value) != float(holding_period_ticker_bars):
        raise ValueError("holding_period_ticker_bars must be a non-negative integer.")
    if value == 0:
        return "0"
    if value <= 2:
        return "1_2"
    if value <= 5:
        return "3_5"
    if value <= 10:
        return "6_10"
    if value <= 20:
        return "11_20"
    if value <= 40:
        return "21_40"
    return "41_PLUS"


def _extreme_order(mfe_offset: int, mae_offset: int) -> str:
    if mfe_offset < mae_offset:
        return MFE_BEFORE_MAE
    if mae_offset < mfe_offset:
        return MAE_BEFORE_MFE
    return AMBIGUOUS_SAME_BAR


def _maximum_close_drawdown(prices: list[float]) -> float:
    peak = prices[0]
    minimum_drawdown = 0.0
    for price in prices:
        peak = max(peak, price)
        minimum_drawdown = min(minimum_drawdown, (price / peak - 1) * 100)
    return -minimum_drawdown


def holding_path_for_trade(
    *,
    trade: Mapping[str, Any],
    window_data: pd.DataFrame,
    holding_path_attribution_stamp: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return one trade attribution and its exposure-aware ticker-local path."""
    missing = REQUIRED_TRADE_COLUMNS.difference(trade)
    if missing:
        raise ValueError(f"Forward trade missing columns: {sorted(missing)}")
    ticker = str(trade["ticker"])
    data = _normalize_and_validate_market_frame(window_data, ticker)
    entry_timestamp = _timestamp(trade["entry_timestamp"])
    union_exit_timestamp = _timestamp(trade["exit_timestamp"])
    if int((data.index == entry_timestamp).sum()) != 1:
        raise ValueError(f"Expected exactly one entry bar for {trade['trade_id']}.")
    semantic = exit_semantic(str(trade["exit_reason"]))
    if semantic == CLOSE_EXIT:
        candidates = data.index[(data.index >= entry_timestamp) & (data.index <= union_exit_timestamp)]
        if len(candidates) == 0:
            raise ValueError(f"No effective FORCE_CLOSE_END ticker bar for {trade['trade_id']}.")
        effective_exit = pd.Timestamp(candidates[-1])
    else:
        if int((data.index == union_exit_timestamp).sum()) != 1:
            raise ValueError(f"Expected one market exit bar for {trade['trade_id']}.")
        effective_exit = union_exit_timestamp
    if effective_exit < entry_timestamp:
        raise ValueError(f"Exit precedes entry for {trade['trade_id']}.")
    path = data.loc[(data.index >= entry_timestamp) & (data.index <= effective_exit)]
    elapsed = int(trade["holding_period_ticker_bars"])
    if elapsed != len(path) - 1:
        raise ValueError(
            f"Ticker-bar holding reconciliation failed for {trade['trade_id']}: "
            f"source={elapsed}, reconstructed={len(path) - 1}."
        )

    entry_fill = _finite_positive(trade["entry_fill_price"], "entry_fill_price")
    quantity = _finite_positive(trade["quantity"], "quantity")
    risk_amount = _finite_positive(trade["initial_risk_amount"], "initial_risk_amount")
    stop_percent = _finite_positive(trade["initial_stop_percent"], "initial_stop_percent")
    stop_price = _finite_positive(trade["initial_stop_price"], "initial_stop_price")
    source_exit_price = _finite_positive(trade["exit_price"], "exit_price")

    certain_high = entry_fill
    certain_low = entry_fill
    certain_high_candidate = (entry_fill, 0, entry_timestamp)
    certain_low_candidate = (entry_fill, 0, entry_timestamp)
    possible_high = entry_fill
    possible_low = entry_fill
    path_rows: list[dict[str, Any]] = []
    observed_closes: list[float] = []
    full_bar_count = 0

    for offset, (bar_timestamp, row) in enumerate(path.iterrows()):
        is_entry = offset == 0
        is_exit = offset == len(path) - 1
        raw_open = float(row["Open"])
        raw_high = float(row["High"])
        raw_low = float(row["Low"])
        raw_close = float(row["Close"])
        include_full = not is_exit or semantic == CLOSE_EXIT
        stop_touch = stop_price if is_exit and semantic == INTRABAR_STOP_BOUNDED else None

        if include_full:
            full_bar_count += 1
            observed_closes.append(raw_close)
            if raw_high > certain_high:
                certain_high = raw_high
                certain_high_candidate = (raw_high, offset, pd.Timestamp(bar_timestamp))
            if raw_low < certain_low:
                certain_low = raw_low
                certain_low_candidate = (raw_low, offset, pd.Timestamp(bar_timestamp))
        elif semantic == OPEN_EXIT:
            if raw_open > certain_high:
                certain_high = raw_open
                certain_high_candidate = (raw_open, offset, pd.Timestamp(bar_timestamp))
            if raw_open < certain_low:
                certain_low = raw_open
                certain_low_candidate = (raw_open, offset, pd.Timestamp(bar_timestamp))
        else:
            for price, direction in ((raw_open, "both"), (stop_price, "low")):
                if direction == "both" and price > certain_high:
                    certain_high = price
                    certain_high_candidate = (price, offset, pd.Timestamp(bar_timestamp))
                if price < certain_low:
                    certain_low = price
                    certain_low_candidate = (price, offset, pd.Timestamp(bar_timestamp))

        possible_high = certain_high
        possible_low = certain_low
        if is_exit and semantic == INTRABAR_STOP_BOUNDED:
            possible_high = max(certain_high, raw_high)
            possible_low = min(certain_low, raw_low)

        bar_role = "HOLDING_BAR"
        if is_entry and is_exit:
            bar_role = "ENTRY_EXIT_BAR"
        elif is_entry:
            bar_role = "ENTRY_BAR"
        elif is_exit:
            bar_role = "EXIT_BAR"
        path_rows.append(
            {
                "holding_path_attribution_stamp": holding_path_attribution_stamp,
                "source_trade_row": trade["source_trade_row"],
                "trade_id": trade["trade_id"],
                "window_id": trade["window_id"],
                "ticker": ticker,
                "exit_reason": trade["exit_reason"],
                "exit_semantic": semantic,
                "ticker_bar_offset": offset,
                "ticker_timestamp": pd.Timestamp(bar_timestamp),
                "bar_role": bar_role,
                "raw_open": raw_open,
                "raw_high": raw_high,
                "raw_low": raw_low,
                "raw_close": raw_close,
                "open_included": True,
                "high_low_close_included": include_full,
                "known_stop_touch_price": stop_touch,
                "censored_high_price": raw_high if include_full else (raw_open if semantic == OPEN_EXIT else max(raw_open, stop_price)),
                "censored_low_price": raw_low if include_full else (raw_open if semantic == OPEN_EXIT else min(raw_open, stop_price)),
                "censored_close_price": raw_close if include_full else None,
                "cumulative_censored_mfe_percent": (certain_high / entry_fill - 1) * 100,
                "cumulative_censored_mae_percent": (certain_low / entry_fill - 1) * 100,
                "cumulative_possible_mfe_percent": (possible_high / entry_fill - 1) * 100,
                "cumulative_possible_mae_percent": (possible_low / entry_fill - 1) * 100,
                "is_entry_bar": is_entry,
                "is_effective_exit_bar": is_exit,
                "intrabar_order_uncertain": bool(is_exit and semantic == INTRABAR_STOP_BOUNDED),
            }
        )

    exit_row = path.iloc[-1]
    if semantic == OPEN_EXIT:
        market_exit_reference = float(exit_row["Open"])
    elif semantic == INTRABAR_STOP_BOUNDED:
        market_exit_reference = stop_price
    else:
        market_exit_reference = float(exit_row["Close"])
    tolerance = max(1e-8, market_exit_reference * 1e-8)
    if source_exit_price > market_exit_reference + tolerance:
        raise ValueError(f"Sell fill contradicts adverse slippage semantics for {trade['trade_id']}.")

    censored_mfe_percent = (certain_high / entry_fill - 1) * 100
    censored_mae_percent = (certain_low / entry_fill - 1) * 100
    possible_mfe_percent = (possible_high / entry_fill - 1) * 100
    possible_mae_percent = (possible_low / entry_fill - 1) * 100
    net_return = float(trade["return_percent"])
    closes_with_reference = [entry_fill, *observed_closes]
    close_excursions = [(price / entry_fill - 1) * 100 for price in closes_with_reference]
    underwater_count = sum(price < entry_fill for price in observed_closes)
    close_count = len(observed_closes)
    record = {
        "holding_path_attribution_stamp": holding_path_attribution_stamp,
        **{column: trade.get(column) for column in FORWARD_COLUMNS},
        "effective_exit_market_timestamp": effective_exit,
        "exit_semantic": semantic,
        "path_ticker_bar_count": len(path),
        "full_observed_ticker_bar_count": full_bar_count,
        "holding_bucket": holding_bucket(elapsed),
        "market_exit_reference_price": market_exit_reference,
        "market_exit_reference_return_percent": (market_exit_reference / entry_fill - 1) * 100,
        "net_realized_return_percent": net_return,
        "gross_realized_r_multiple": float(trade["gross_pnl"]) / risk_amount,
        "net_realized_r_multiple": float(trade["net_pnl"]) / risk_amount,
        "censored_mfe_price": certain_high,
        "censored_mfe_percent": censored_mfe_percent,
        "possible_mfe_price": possible_high,
        "possible_mfe_percent": possible_mfe_percent,
        "mfe_bound_width_percent": possible_mfe_percent - censored_mfe_percent,
        "mfe_exact": semantic != INTRABAR_STOP_BOUNDED,
        "censored_mae_price": certain_low,
        "censored_mae_percent": censored_mae_percent,
        "possible_mae_price": possible_low,
        "possible_mae_percent": possible_mae_percent,
        "mae_bound_width_percent": censored_mae_percent - possible_mae_percent,
        "mae_exact": semantic != INTRABAR_STOP_BOUNDED,
        "censored_mfe_r_multiple": censored_mfe_percent / stop_percent,
        "possible_mfe_r_multiple": possible_mfe_percent / stop_percent,
        "censored_mae_r_multiple": censored_mae_percent / stop_percent,
        "possible_mae_r_multiple": possible_mae_percent / stop_percent,
        "ticker_bars_to_censored_mfe": certain_high_candidate[1],
        "ticker_bars_to_censored_mae": certain_low_candidate[1],
        "censored_mfe_timestamp": certain_high_candidate[2],
        "censored_mae_timestamp": certain_low_candidate[2],
        "censored_extreme_order": _extreme_order(certain_high_candidate[1], certain_low_candidate[1]),
        "giveback_from_censored_mfe_percent": censored_mfe_percent - net_return,
        "censored_mfe_capture_ratio": net_return / censored_mfe_percent if censored_mfe_percent > 0 else None,
        "recovery_from_censored_mae_percent": net_return - censored_mae_percent,
        "exit_location_in_censored_range": (
            (net_return - censored_mae_percent) / (censored_mfe_percent - censored_mae_percent)
            if censored_mfe_percent > censored_mae_percent
            else None
        ),
        "maximum_close_excursion_percent": max(close_excursions),
        "minimum_close_excursion_percent": min(close_excursions),
        "maximum_close_drawdown_percent": _maximum_close_drawdown(closes_with_reference),
        "underwater_close_bar_count": underwater_count,
        "observed_close_bar_count": close_count,
        "underwater_close_bar_fraction": underwater_count / close_count if close_count else None,
        "intrabar_order_uncertain": semantic == INTRABAR_STOP_BOUNDED,
    }
    return record, path_rows


def enrich_holding_paths(
    *,
    trades: pd.DataFrame,
    data_by_ticker: Mapping[str, pd.DataFrame],
    windows: pd.DataFrame,
    holding_path_attribution_stamp: str,
    expected_trade_count: int | None = None,
    expected_window_count: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = REQUIRED_TRADE_COLUMNS.difference(trades.columns)
    if missing:
        raise ValueError(f"Forward trades missing columns: {sorted(missing)}")
    if expected_trade_count is not None and len(trades) != expected_trade_count:
        raise ValueError(f"Expected {expected_trade_count} trades; found {len(trades)}.")
    if expected_window_count is not None and trades["window_id"].astype(str).nunique() != expected_window_count:
        raise ValueError(f"Expected {expected_window_count} windows.")
    if not trades["model"].astype(str).eq(MODEL_BASELINE).all():
        raise ValueError("Holding-path attribution requires FIXED_BASELINE trades only.")
    if trades["source_trade_row"].duplicated().any() or trades["trade_id"].duplicated().any():
        raise ValueError("Forward trade identifiers must be unique.")
    lookup = _window_lookup(windows)
    normalized = {
        str(ticker).strip().upper(): _normalize_and_validate_market_frame(frame, str(ticker))
        for ticker, frame in data_by_ticker.items()
    }
    records: list[dict[str, Any]] = []
    path_records: list[dict[str, Any]] = []
    for trade in trades.to_dict("records"):
        ticker = str(trade["ticker"]).strip().upper()
        window_id = str(trade["window_id"])
        if ticker not in normalized or window_id not in lookup:
            raise ValueError(f"Missing snapshot data for {window_id}/{ticker}.")
        start, end = lookup[window_id]
        window_data = normalized[ticker].loc[
            (normalized[ticker].index >= start) & (normalized[ticker].index < end)
        ]
        record, rows = holding_path_for_trade(
            trade=trade,
            window_data=window_data,
            holding_path_attribution_stamp=holding_path_attribution_stamp,
        )
        records.append(record)
        path_records.extend(rows)
    enriched = pd.DataFrame(records, columns=HOLDING_TRADE_COLUMNS)
    paths = pd.DataFrame(path_records, columns=PATH_COLUMNS)
    if enriched["source_trade_row"].tolist() != trades["source_trade_row"].tolist():
        raise ValueError("Source trade order was not preserved.")
    return enriched, paths


def _statistic_record(
    frame: pd.DataFrame,
    *,
    metric: str,
    stamp: str,
    population: str,
    group_type: str,
    group_value: str,
) -> dict[str, Any]:
    values = pd.to_numeric(frame[metric], errors="coerce")
    values = values.where(values.map(lambda value: pd.isna(value) or isfinite(float(value))))
    valid = values.dropna()
    count = len(valid)

    def percentile(q: float) -> float | None:
        return float(valid.quantile(q, interpolation="linear")) if count else None

    negative = int((valid < 0).sum())
    flat = int((valid == 0).sum())
    positive = int((valid > 0).sum())
    return {
        "holding_path_attribution_stamp": stamp,
        "population": population,
        "group_type": group_type,
        "group_value": str(group_value),
        "metric": metric,
        "population_trade_count": len(frame),
        "count": count,
        "missing_count": len(frame) - count,
        "mean": float(valid.mean()) if count else None,
        "median": float(valid.median()) if count else None,
        "population_standard_deviation": float(valid.std(ddof=0)) if count else None,
        "minimum": float(valid.min()) if count else None,
        "p01": percentile(0.01),
        "p05": percentile(0.05),
        "p10": percentile(0.10),
        "p25": percentile(0.25),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "maximum": float(valid.max()) if count else None,
        "negative_count": negative,
        "negative_frequency": negative / count if count else 0.0,
        "flat_count": flat,
        "flat_frequency": flat / count if count else 0.0,
        "positive_count": positive,
        "positive_frequency": positive / count if count else 0.0,
    }


def build_statistics_table(
    trades: pd.DataFrame,
    *,
    stamp: str,
    population: str,
    group_type: str,
    group_column: str | None = None,
    group_values: Iterable[Any] | None = None,
) -> pd.DataFrame:
    if group_column is None:
        groups = [("ALL", trades)]
    else:
        values = list(group_values) if group_values is not None else sorted(
            trades[group_column].dropna().unique(), key=str
        )
        groups = [(value, trades.loc[trades[group_column] == value]) for value in values]
    records = [
        _statistic_record(
            group,
            metric=metric,
            stamp=stamp,
            population=population,
            group_type=group_type,
            group_value=str(value),
        )
        for value, group in groups
        for metric in STATISTIC_METRICS
    ]
    return pd.DataFrame(records, columns=STATISTICS_COLUMNS)


def _both_populations(
    trades: pd.DataFrame,
    *,
    stamp: str,
    group_type: str,
    group_column: str | None = None,
    group_values: Iterable[Any] | None = None,
) -> pd.DataFrame:
    primary = build_statistics_table(
        trades,
        stamp=stamp,
        population=PRIMARY_POPULATION,
        group_type=group_type,
        group_column=group_column,
        group_values=group_values,
    )
    sensitivity_trades = trades.loc[trades["exit_reason"] != "FORCE_CLOSE_END"]
    sensitivity = build_statistics_table(
        sensitivity_trades,
        stamp=stamp,
        population=SENSITIVITY_POPULATION,
        group_type=group_type,
        group_column=group_column,
        group_values=group_values,
    )
    return pd.concat([primary, sensitivity], ignore_index=True)


def build_aggregation_tables(trades: pd.DataFrame, *, stamp: str) -> dict[str, pd.DataFrame]:
    both = lambda name, column=None, values=None: _both_populations(
        trades,
        stamp=stamp,
        group_type=name,
        group_column=column,
        group_values=values,
    )
    sensitivity = trades.loc[trades["exit_reason"] != "FORCE_CLOSE_END"]
    return {
        "overall": both("ALL_TRADES"),
        "asset_classes": both("ASSET_CLASS", "asset_class", ASSET_CLASS_ORDER),
        "tickers": both("TICKER", "ticker", CONTROLLED_TICKERS),
        "windows": both("WINDOW_ID", "window_id", sorted(trades["window_id"].astype(str).unique())),
        "exit_reasons": both("EXIT_REASON", "exit_reason"),
        "exit_categories": both("EXIT_CATEGORY", "exit_category"),
        "outcomes": both("OUTCOME_CLASS", "outcome_class", OUTCOME_ORDER),
        "holding_buckets": both(
            "HOLDING_BUCKET", "holding_bucket", ("0", "1_2", "3_5", "6_10", "11_20", "21_40", "41_PLUS")
        ),
        "extreme_orders": both(
            "CENSORED_EXTREME_ORDER",
            "censored_extreme_order",
            (MFE_BEFORE_MAE, MAE_BEFORE_MFE, AMBIGUOUS_SAME_BAR),
        ),
        "exit_semantics": both(
            "EXIT_SEMANTIC", "exit_semantic", (OPEN_EXIT, INTRABAR_STOP_BOUNDED, CLOSE_EXIT)
        ),
        "force_close_excluded": build_statistics_table(
            sensitivity,
            stamp=stamp,
            population=SENSITIVITY_POPULATION,
            group_type="ALL_TRADES",
        ),
    }


def build_screen(
    trades: pd.DataFrame,
    paths: pd.DataFrame,
    *,
    stamp: str,
    expected_trade_count: int,
    expected_window_count: int,
    official_source: bool,
) -> pd.DataFrame:
    reason_counts = trades["exit_reason"].value_counts().to_dict()
    checks: list[tuple[str, bool, Any, Any, str]] = [
        ("trade_count", len(trades) == expected_trade_count, expected_trade_count, len(trades), "Completed baseline trade population."),
        ("window_count", trades["window_id"].nunique() == expected_window_count, expected_window_count, trades["window_id"].nunique(), "Independently reset test windows."),
        ("unique_trade_ids", not trades["trade_id"].duplicated().any(), True, not trades["trade_id"].duplicated().any(), "Trade identifiers remain unique."),
        ("path_row_reconciliation", int(trades["path_ticker_bar_count"].sum()) == len(paths), int(trades["path_ticker_bar_count"].sum()), len(paths), "Every reconstructed ticker bar has one path row."),
        ("mfe_non_negative", bool(trades["censored_mfe_percent"].ge(-1e-10).all()), True, bool(trades["censored_mfe_percent"].ge(-1e-10).all()), "Entry fill anchors MFE at zero."),
        ("mae_non_positive", bool(trades["censored_mae_percent"].le(1e-10).all()), True, bool(trades["censored_mae_percent"].le(1e-10).all()), "Entry fill anchors MAE at zero."),
        ("mfe_bounds_ordered", bool(trades["possible_mfe_percent"].ge(trades["censored_mfe_percent"] - 1e-10).all()), True, bool(trades["possible_mfe_percent"].ge(trades["censored_mfe_percent"] - 1e-10).all()), "Possible MFE is an upper bound."),
        ("mae_bounds_ordered", bool(trades["possible_mae_percent"].le(trades["censored_mae_percent"] + 1e-10).all()), True, bool(trades["possible_mae_percent"].le(trades["censored_mae_percent"] + 1e-10).all()), "Possible MAE is a lower bound."),
        ("open_exit_hlc_excluded", bool(paths.loc[(paths["is_effective_exit_bar"]) & (paths["exit_semantic"] == OPEN_EXIT), "high_low_close_included"].eq(False).all()), True, bool(paths.loc[(paths["is_effective_exit_bar"]) & (paths["exit_semantic"] == OPEN_EXIT), "high_low_close_included"].eq(False).all()), "Open exits exclude post-exit High/Low/Close."),
        ("intrabar_stop_bounded", bool(trades.loc[trades["exit_semantic"] == INTRABAR_STOP_BOUNDED, "intrabar_order_uncertain"].eq(True).all()), True, bool(trades.loc[trades["exit_semantic"] == INTRABAR_STOP_BOUNDED, "intrabar_order_uncertain"].eq(True).all()), "STOP_LOSS exit bars remain explicitly uncertain."),
        ("force_close_sensitivity_population", int((trades["exit_reason"] != "FORCE_CLOSE_END").sum()) == len(trades) - int((trades["exit_reason"] == "FORCE_CLOSE_END").sum()), True, True, "Sensitivity population excludes only FORCE_CLOSE_END."),
    ]
    if official_source:
        checks.append(("official_exit_reason_counts", reason_counts == OFFICIAL_EXIT_REASON_COUNTS, OFFICIAL_EXIT_REASON_COUNTS, reason_counts, "Official population reason reconciliation."))
    frame = pd.DataFrame(
        [
            {
                "holding_path_attribution_stamp": stamp,
                "check": name,
                "passed": passed,
                "expected": json.dumps(_json_safe(expected), sort_keys=True),
                "actual": json.dumps(_json_safe(actual), sort_keys=True),
                "detail": detail,
                "authorizes_strategy_change": False,
            }
            for name, passed, expected, actual, detail in checks
        ],
        columns=SCREEN_COLUMNS,
    )
    if not frame["passed"].all():
        failed = frame.loc[~frame["passed"], "check"].tolist()
        raise ValueError("Holding-path quality screen failed: " + ", ".join(failed))
    return frame


def _validate_lineage_frame(trades: pd.DataFrame) -> None:
    expected = {
        "forward_return_statistics_stamp": APPROVED_FORWARD_RETURN_STATISTICS_STAMP,
        "entry_statistics_stamp": APPROVED_ENTRY_STATISTICS_STAMP,
        "timing_stamp": APPROVED_TIMING_STAMP,
        "source_stop_walk_forward_stamp": APPROVED_SOURCE_STAMP,
        "snapshot_id": APPROVED_SNAPSHOT_ID,
        "snapshot_fingerprint": APPROVED_SNAPSHOT_FINGERPRINT,
        "model": MODEL_BASELINE,
    }
    for column, value in expected.items():
        if column not in trades or not trades[column].astype(str).eq(str(value)).all():
            raise ValueError(f"Forward trades {column} mismatch.")
    if len(trades) != EXPECTED_TRADE_COUNT or trades["window_id"].astype(str).nunique() != EXPECTED_WINDOW_COUNT:
        raise ValueError("Forward trade population mismatch.")
    if set(trades["ticker"].astype(str)).difference(CONTROLLED_TICKERS):
        raise ValueError("Forward trades include an uncontrolled ticker.")


def load_verified_source(
    *,
    snapshot_directory: Path,
    forward_return_statistics_directory: Path,
    forward_return_statistics_stamp: str,
    project_root: Path = Path("."),
) -> dict[str, Any]:
    """Load only the official, fully hashed FRS/snapshot chain."""
    if forward_return_statistics_stamp != APPROVED_FORWARD_RETURN_STATISTICS_STAMP:
        raise ValueError(
            "Holding-Path Attribution requires Forward Return Statistics stamp "
            f"{APPROVED_FORWARD_RETURN_STATISTICS_STAMP}."
        )
    project_root = Path(project_root).resolve()
    forward_directory = Path(forward_return_statistics_directory)
    snapshot_directory = Path(snapshot_directory)
    provenance_path = forward_directory / (
        f"portfolio_forward_return_statistics_provenance_{forward_return_statistics_stamp}.json"
    )
    provenance = _read_json(provenance_path)
    lineage = {
        "forward_return_statistics_stamp": APPROVED_FORWARD_RETURN_STATISTICS_STAMP,
        "entry_statistics_stamp": APPROVED_ENTRY_STATISTICS_STAMP,
        "timing_stamp": APPROVED_TIMING_STAMP,
        "source_stop_walk_forward_stamp": APPROVED_SOURCE_STAMP,
        "snapshot_id": APPROVED_SNAPSHOT_ID,
        "snapshot_fingerprint": APPROVED_SNAPSHOT_FINGERPRINT,
    }
    for key, expected in lineage.items():
        if provenance.get(key) != expected:
            raise ValueError(f"Forward provenance {key} mismatch.")
    result_files = verify_provenance_result_files(
        directory=forward_directory,
        provenance=provenance,
        prefix="portfolio_forward_return_statistics_",
        stamp=forward_return_statistics_stamp,
    )
    if set(result_files) != EXPECTED_FORWARD_RESULT_KEYS:
        raise ValueError("Forward provenance result coverage mismatch.")
    payload = _read_json(result_files["json"])
    if payload.get("forward_return_statistics_stamp") != forward_return_statistics_stamp:
        raise ValueError("Forward JSON stamp mismatch.")
    payload_lineage = payload.get("source_lineage", {})
    for key in ("entry_statistics_stamp", "timing_stamp", "source_stop_walk_forward_stamp", "snapshot_id", "snapshot_fingerprint"):
        if payload_lineage.get(key) != lineage[key]:
            raise ValueError(f"Forward JSON {key} mismatch.")

    declared_sources = provenance.get("source_files")
    if not isinstance(declared_sources, dict) or not declared_sources:
        raise ValueError("Forward provenance has no source_files mapping.")
    verified_upstream: dict[str, Path] = {}
    for label, metadata in declared_sources.items():
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid forward source metadata: {label}")
        path = Path(str(metadata.get("path", "")))
        expected_hash = str(metadata.get("sha256", ""))
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Forward upstream source hash mismatch: {label}")
        verified_upstream[str(label)] = path
    code_path = verified_upstream.get("forward_return_statistics_code")
    if code_path is None or sha256_file(code_path) != APPROVED_FORWARD_CODE_SHA256:
        raise ValueError("Approved Forward Return Statistics code hash mismatch.")
    forward_test_path = project_root / "tests/test_portfolio_forward_return_statistics.py"
    if not forward_test_path.is_file() or sha256_file(forward_test_path) != APPROVED_FORWARD_TEST_SHA256:
        raise ValueError("Approved Forward Return Statistics test hash mismatch.")

    snapshot = load_snapshot(snapshot_directory, verify_code=True, project_root=project_root)
    manifest = snapshot["manifest"]
    if manifest.get("snapshot_id") != APPROVED_SNAPSHOT_ID or manifest.get("fingerprint") != APPROVED_SNAPSHOT_FINGERPRINT:
        raise ValueError("Frozen snapshot lineage mismatch.")
    if set(manifest.get("market_files", {})) != set(CONTROLLED_TICKERS):
        raise ValueError("Frozen snapshot basket mismatch.")
    trades = pd.read_csv(result_files["trades"])
    _validate_lineage_frame(trades)
    if set(trades["window_id"].astype(str)) != set(snapshot["windows"]["window_id"].astype(str)):
        raise ValueError("Forward trades and snapshot windows disagree.")

    source_files: dict[str, Path] = {
        "forward_return_statistics_provenance": provenance_path,
        "forward_return_statistics_test": forward_test_path,
        **{f"forward_result:{key}": path for key, path in result_files.items()},
        **{f"forward_upstream:{key}": path for key, path in verified_upstream.items()},
    }
    source_hashes = {label: sha256_file(path) for label, path in source_files.items()}
    return {
        "snapshot": snapshot,
        "trades": trades,
        "forward_return_statistics_stamp": forward_return_statistics_stamp,
        "entry_statistics_stamp": APPROVED_ENTRY_STATISTICS_STAMP,
        "timing_stamp": APPROVED_TIMING_STAMP,
        "source_stamp": APPROVED_SOURCE_STAMP,
        "source_files": source_files,
        "source_hashes": source_hashes,
        "source_verification": {
            "verified": True,
            "code_hash_verification": True,
            "official_source": True,
            **lineage,
        },
        "save_authorization": _STRICT_SOURCE_AUTHORIZATION,
        "expected_trade_count": EXPECTED_TRADE_COUNT,
        "expected_window_count": EXPECTED_WINDOW_COUNT,
    }


def build_holding_path_attribution(
    *,
    trades: pd.DataFrame,
    data_by_ticker: Mapping[str, pd.DataFrame],
    windows: pd.DataFrame,
    forward_return_statistics_stamp: str,
    entry_statistics_stamp: str,
    timing_stamp: str,
    source_stamp: str,
    snapshot_id: str,
    snapshot_fingerprint: str,
    expected_trade_count: int,
    expected_window_count: int,
    holding_path_attribution_stamp: str | None = None,
    source_files: Mapping[str, Path] | None = None,
    source_hashes: Mapping[str, str] | None = None,
    source_verification: Mapping[str, Any] | None = None,
    save_authorization: object | None = None,
) -> dict[str, Any]:
    stamp = holding_path_attribution_stamp or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    enriched, paths = enrich_holding_paths(
        trades=trades,
        data_by_ticker=data_by_ticker,
        windows=windows,
        holding_path_attribution_stamp=stamp,
        expected_trade_count=expected_trade_count,
        expected_window_count=expected_window_count,
    )
    official = bool(source_verification and source_verification.get("official_source"))
    screen = build_screen(
        enriched,
        paths,
        stamp=stamp,
        expected_trade_count=expected_trade_count,
        expected_window_count=expected_window_count,
        official_source=official,
    )
    summary = {
        "primary_trade_count": len(enriched),
        "window_count": enriched["window_id"].nunique(),
        "path_row_count": len(paths),
        "force_close_end_trade_count": int((enriched["exit_reason"] == "FORCE_CLOSE_END").sum()),
        "force_close_excluded_trade_count": int((enriched["exit_reason"] != "FORCE_CLOSE_END").sum()),
        "intrabar_stop_bounded_trade_count": int((enriched["exit_semantic"] == INTRABAR_STOP_BOUNDED).sum()),
        "quality_screen_passed": bool(screen["passed"].all()),
        "strategy_change_authorized": False,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "holding_path_attribution_stamp": stamp,
        "forward_return_statistics_stamp": forward_return_statistics_stamp,
        "entry_statistics_stamp": entry_statistics_stamp,
        "timing_stamp": timing_stamp,
        "source_stamp": source_stamp,
        "snapshot_id": snapshot_id,
        "snapshot_fingerprint": snapshot_fingerprint,
        "expected_trade_count": expected_trade_count,
        "expected_window_count": expected_window_count,
        "trades": enriched,
        "path_rows": paths,
        "screen": screen,
        "aggregations": build_aggregation_tables(enriched, stamp=stamp),
        "summary": summary,
        "source_files": None if source_files is None else {str(k): Path(v) for k, v in source_files.items()},
        "source_hashes": None if source_hashes is None else dict(source_hashes),
        "source_verification": None if source_verification is None else dict(source_verification),
        "save_authorization": save_authorization,
    }


def run_holding_path_attribution(source: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = source["snapshot"]
    manifest = snapshot["manifest"]
    return build_holding_path_attribution(
        trades=source["trades"],
        data_by_ticker=snapshot["data_by_ticker"],
        windows=snapshot["windows"],
        forward_return_statistics_stamp=str(source["forward_return_statistics_stamp"]),
        entry_statistics_stamp=str(source["entry_statistics_stamp"]),
        timing_stamp=str(source["timing_stamp"]),
        source_stamp=str(source["source_stamp"]),
        snapshot_id=str(manifest["snapshot_id"]),
        snapshot_fingerprint=str(manifest["fingerprint"]),
        expected_trade_count=int(source["expected_trade_count"]),
        expected_window_count=int(source["expected_window_count"]),
        source_files=source["source_files"],
        source_hashes=source["source_hashes"],
        source_verification=source["source_verification"],
        save_authorization=source["save_authorization"],
    )


def _validate_saveable(bundle: Mapping[str, Any]) -> None:
    if bundle.get("save_authorization") is not _STRICT_SOURCE_AUTHORIZATION:
        raise ValueError("Cannot save: source was not loaded through strict provenance.")
    verification = bundle.get("source_verification")
    if not isinstance(verification, Mapping) or verification.get("verified") is not True or verification.get("code_hash_verification") is not True or verification.get("official_source") is not True:
        raise ValueError("Cannot save: official source verification is incomplete.")
    files = bundle.get("source_files")
    hashes = bundle.get("source_hashes")
    if not isinstance(files, Mapping) or not isinstance(hashes, Mapping) or set(files) != set(hashes):
        raise ValueError("Cannot save: consumed source mapping is incomplete.")
    for label, value in files.items():
        path = Path(value)
        if not path.is_file() or sha256_file(path) != hashes[label]:
            raise ValueError(f"Source changed before save: {label}")
    if not bundle["screen"]["passed"].all():
        raise ValueError("Cannot save: quality screen failed.")


def build_json_payload(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": bundle["created_at"],
        "holding_path_attribution_stamp": bundle["holding_path_attribution_stamp"],
        "method": "Observational exposure-aware ticker-local holding paths; no engine replay and no strategy change.",
        "source_lineage": {
            "forward_return_statistics_stamp": bundle["forward_return_statistics_stamp"],
            "entry_statistics_stamp": bundle["entry_statistics_stamp"],
            "timing_stamp": bundle["timing_stamp"],
            "source_stop_walk_forward_stamp": bundle["source_stamp"],
            "snapshot_id": bundle["snapshot_id"],
            "snapshot_fingerprint": bundle["snapshot_fingerprint"],
            "source_files": {
                label: {"path": str(path), "sha256": bundle["source_hashes"][label]}
                for label, path in bundle["source_files"].items()
            },
        },
        "population_definition": {
            "primary": PRIMARY_POPULATION,
            "sensitivity": SENSITIVITY_POPULATION,
            "windows_are_separate_reset_capital_runs": True,
        },
        "exit_semantics": {
            OPEN_EXIT: "Include exit raw Open only; exit-bar High/Low/Close are post-exit and excluded.",
            INTRABAR_STOP_BOUNDED: "Include exit Open and known stop touch as certain; report exit-bar High/Low only as possible OHLC bounds because ordering is unknown.",
            CLOSE_EXIT: "Map union-calendar exit to the last real ticker bar at or before exit and include the full bar through Close.",
        },
        "formulas": {
            "excursion_percent": "(extreme_price / entry_fill_price - 1) * 100",
            "excursion_r_multiple": "excursion_percent / initial_stop_percent",
            "giveback_percent": "censored_mfe_percent - net_realized_return_percent",
            "capture_ratio": "net_realized_return_percent / censored_mfe_percent when censored_mfe_percent > 0",
            "maximum_close_drawdown_percent": "positive magnitude of the worst peak-to-subsequent-close decline, anchored at entry fill",
        },
        "summary": bundle["summary"],
        "trade_schema": list(HOLDING_TRADE_COLUMNS),
        "path_schema": list(PATH_COLUMNS),
        "aggregation_schema": list(STATISTICS_COLUMNS),
        "quality_screen": bundle["screen"].to_dict("records"),
        "aggregation_results": {name: frame.to_dict("records") for name, frame in bundle["aggregations"].items()},
        "limitations": [
            "Daily OHLC does not identify intrabar High/Low ordering.",
            "STOP_LOSS MFE/MAE is bounded and must not be described as exact executable excursion.",
            "Open exits exclude the exit bar after Open; no post-exit High/Low/Close is attributed to the trade.",
            "FORCE_CLOSE_END sensitivity excludes those trades without changing the primary population.",
            "This data-quality screen never authorizes a strategy, baseline, parameter, or production change.",
            "The provenance file hashes every other output and cannot self-hash.",
        ],
    }


def save_holding_path_attribution(
    bundle: Mapping[str, Any], output_directory: Path = DEFAULT_OUTPUT_DIRECTORY
) -> dict[str, Path]:
    _validate_saveable(bundle)
    output_directory = Path(output_directory)
    stamp = str(bundle["holding_path_attribution_stamp"])
    frames = {
        "trades": bundle["trades"],
        "path_rows": bundle["path_rows"],
        "screen": bundle["screen"],
        **bundle["aggregations"],
    }
    paths = {
        name: output_directory / f"portfolio_holding_path_attribution_{name}_{stamp}.csv"
        for name in frames
    }
    paths["json"] = output_directory / f"portfolio_holding_path_attribution_{stamp}.json"
    paths["provenance"] = output_directory / f"portfolio_holding_path_attribution_provenance_{stamp}.json"
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(existing[0])
    output_directory.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(paths[name], index=False, lineterminator="\n")
    paths["json"].write_text(
        json.dumps(_json_safe(build_json_payload(bundle)), sort_keys=True, indent=2) + "\n",
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
        "holding_path_attribution_stamp": stamp,
        "forward_return_statistics_stamp": bundle["forward_return_statistics_stamp"],
        "entry_statistics_stamp": bundle["entry_statistics_stamp"],
        "timing_stamp": bundle["timing_stamp"],
        "source_stop_walk_forward_stamp": bundle["source_stamp"],
        "snapshot_id": bundle["snapshot_id"],
        "snapshot_fingerprint": bundle["snapshot_fingerprint"],
        "source_files": {
            label: {"path": str(Path(path).resolve()), "sha256": bundle["source_hashes"][label]}
            for label, path in bundle["source_files"].items()
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
    print("HOLDING-PATH ATTRIBUTION V1 — OBSERVATIONAL")
    print("Provenance validation: PASS")
    print("Quality screen: PASS")
    print(f"Forward Return Statistics: {bundle['forward_return_statistics_stamp']}")
    print(f"Snapshot: {bundle['snapshot_id']}")
    print(
        f"Primary: {summary['primary_trade_count']} trades, {summary['window_count']} windows, "
        f"{summary['path_row_count']} ticker-local path rows"
    )
    print(
        f"STOP_LOSS bounded: {summary['intrabar_stop_bounded_trade_count']}; "
        f"FORCE_CLOSE_END excluded sensitivity: {summary['force_close_excluded_trade_count']}"
    )
    print("Strategy change authorized: NO")
    if no_save:
        print("Output artifacts saved: 0 (--no-save)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Observational holding-path attribution for the approved baseline source"
    )
    parser.add_argument("--snapshot-directory", type=Path, required=True)
    parser.add_argument("--forward-return-statistics-directory", type=Path, required=True)
    parser.add_argument("--forward-return-statistics-stamp", required=True)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--no-save", action="store_true")
    arguments = parser.parse_args()
    source = load_verified_source(
        snapshot_directory=arguments.snapshot_directory,
        forward_return_statistics_directory=arguments.forward_return_statistics_directory,
        forward_return_statistics_stamp=arguments.forward_return_statistics_stamp,
    )
    bundle = run_holding_path_attribution(source)
    _print_summary(bundle, no_save=arguments.no_save)
    if arguments.no_save:
        return
    for name, path in save_holding_path_attribution(bundle, arguments.output_directory).items():
        print(name, path.resolve())


if __name__ == "__main__":
    main()
