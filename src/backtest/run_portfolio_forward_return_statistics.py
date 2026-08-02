"""Observational ticker-local forward returns for provenanced baseline trades.

This analyzer reads completed Entry Statistics trades and the immutable frozen
snapshot.  It neither replays the portfolio engine nor changes any trade,
portfolio, or strategy behaviour.  Forward OHLC values are outcome data only;
they never affect the already-stored entry fields or classifications.
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

from src.backtest.run_portfolio_entry_statistics import (
    APPROVED_SNAPSHOT_ID,
    APPROVED_SOURCE_STAMP,
    APPROVED_TIMING_STAMP,
    ASSET_CLASS_ORDER,
    CONTROLLED_TICKERS,
    EXPECTED_TRADE_COUNT,
    EXPECTED_WINDOW_COUNT,
    GAP_BUCKETS,
    MODEL_BASELINE,
    OUTCOME_ORDER,
    TRADE_COLUMNS as ENTRY_STATISTICS_TRADE_COLUMNS,
    verify_provenance_result_files,
)
from src.backtest.run_research_data_snapshot import load_snapshot, sha256_file


SCHEMA_VERSION = 1
APPROVED_ENTRY_STATISTICS_STAMP = "20260802_103007"
DEFAULT_ENTRY_STATISTICS_DIRECTORY = Path("data/backtests/portfolio/entry_statistics")
DEFAULT_OUTPUT_DIRECTORY = Path("data/backtests/portfolio/forward_return_statistics")
HORIZONS = (1, 3, 5, 10)
OFFICIAL_AVAILABILITY_COUNTS = {1: 256, 3: 255, 5: 251, 10: 239}
EXPECTATION_KEYS = (
    "entry_statistics_stamp",
    "timing_stamp",
    "source_stop_walk_forward_stamp",
    "snapshot_id",
    "expected_trade_count",
    "expected_window_count",
    "controlled_tickers",
)
OFFICIAL_EXPECTATIONS = {
    "entry_statistics_stamp": APPROVED_ENTRY_STATISTICS_STAMP,
    "timing_stamp": APPROVED_TIMING_STAMP,
    "source_stop_walk_forward_stamp": APPROVED_SOURCE_STAMP,
    "snapshot_id": APPROVED_SNAPSHOT_ID,
    "expected_trade_count": EXPECTED_TRADE_COUNT,
    "expected_window_count": EXPECTED_WINDOW_COUNT,
    "controlled_tickers": CONTROLLED_TICKERS,
}
_STRICT_SOURCE_AUTHORIZATION = object()

PRIMARY_POPULATION = "PRIMARY_ALL_COMPLETED_BASELINE_TRADES"
SENSITIVITY_POPULATION = "SENSITIVITY_EXCLUDING_FORCE_CLOSE_END"
AVAILABLE = "AVAILABLE"
INSUFFICIENT_BARS = "INSUFFICIENT_TICKER_BARS_IN_TEST_WINDOW"
OPEN_THROUGH_HORIZON = "OPEN_THROUGH_HORIZON"
EXITED_BEFORE_HORIZON = "EXITED_BEFORE_HORIZON"
EXITED_ON_HORIZON_TIMESTAMP = "EXITED_ON_HORIZON_TIMESTAMP"

REQUIRED_ENTRY_COLUMNS = {
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
    "entry_year",
    "entry_gap_bucket",
    "exit_timestamp",
    "exit_reason",
    "exit_category",
    "outcome_class",
}

FORWARD_METRIC_SUFFIXES = (
    "terminal_close",
    "close_to_entry_fill_return_percent",
    "maximum_observed_high",
    "observed_high_excursion_percent",
    "minimum_observed_low",
    "observed_low_excursion_percent",
)

FORWARD_COLUMNS = (
    "forward_return_statistics_stamp",
    *ENTRY_STATISTICS_TRADE_COLUMNS,
    *(
        item
        for horizon in HORIZONS
        for item in (
            f"forward_{horizon}_horizon_ticker_bars",
            f"forward_{horizon}_terminal_ticker_timestamp",
            f"forward_{horizon}_terminal_close",
            f"forward_{horizon}_available",
            f"forward_{horizon}_availability_reason",
            f"forward_{horizon}_close_to_entry_fill_return_percent",
            f"forward_{horizon}_maximum_observed_high",
            f"forward_{horizon}_observed_high_excursion_percent",
            f"forward_{horizon}_minimum_observed_low",
            f"forward_{horizon}_observed_low_excursion_percent",
            f"forward_{horizon}_exit_relation",
            f"forward_{horizon}_trade_active_through_horizon",
        )
    ),
)

STATISTICS_COLUMNS = (
    "forward_return_statistics_stamp",
    "population",
    "group_type",
    "group_value",
    "horizon_ticker_bars",
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


def _normalized_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp


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


def _resolved_expectations(expectations: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return either the complete official bundle or one complete alternate bundle."""
    if expectations is None:
        return dict(OFFICIAL_EXPECTATIONS)
    if not isinstance(expectations, Mapping):
        raise ValueError("expectations must be a complete mapping when supplied.")
    actual = set(expectations)
    required = set(EXPECTATION_KEYS)
    if actual != required:
        missing = sorted(required.difference(actual))
        extra = sorted(actual.difference(required))
        raise ValueError(
            "Alternate expectations must contain one complete expectation bundle; "
            f"missing={missing}, extra={extra}."
        )
    result = {key: expectations[key] for key in EXPECTATION_KEYS}
    if any(not isinstance(result[key], str) or not result[key] for key in EXPECTATION_KEYS[:4]):
        raise ValueError("Alternate expectation lineage fields must be non-empty strings.")
    for key in ("expected_trade_count", "expected_window_count"):
        if not isinstance(result[key], Integral) or int(result[key]) <= 0:
            raise ValueError(f"Alternate {key} must be a positive integer.")
        result[key] = int(result[key])
    tickers = tuple(str(value) for value in result["controlled_tickers"])
    if not tickers or len(tickers) != len(set(tickers)):
        raise ValueError("Alternate controlled_tickers must be a non-empty unique sequence.")
    result["controlled_tickers"] = tickers
    return result


def _normalize_and_validate_market_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    required = {"Open", "High", "Low", "Close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Snapshot missing OHLC columns for {ticker}: {sorted(missing)}")
    output = frame.copy(deep=True)
    output.index = pd.to_datetime(output.index)
    if output.index.tz is not None:
        output.index = output.index.tz_convert(None)
    if output.index.duplicated().any():
        raise ValueError(f"Snapshot has duplicate timestamps for {ticker}.")
    output = output.sort_index()
    for column in ("Open", "High", "Low", "Close"):
        values = pd.to_numeric(output[column], errors="coerce")
        invalid = values.isna() | ~values.map(lambda value: isfinite(float(value))) | values.le(0)
        if invalid.any():
            timestamp = output.index[invalid.to_numpy().nonzero()[0][0]]
            value = output.loc[timestamp, column]
            raise ValueError(
                f"Invalid {column} for ticker {ticker} at {timestamp}: {value!r}; "
                "expected a finite positive value."
            )
        output[column] = values.astype(float)
    invariants = (
        ("High >= Low", output["High"] >= output["Low"]),
        ("High >= Open", output["High"] >= output["Open"]),
        ("High >= Close", output["High"] >= output["Close"]),
        ("Low <= Open", output["Low"] <= output["Open"]),
        ("Low <= Close", output["Low"] <= output["Close"]),
    )
    for invariant, valid in invariants:
        if not valid.all():
            timestamp = output.index[(~valid).to_numpy().nonzero()[0][0]]
            row = output.loc[timestamp]
            raise ValueError(
                f"Invalid OHLC invariant {invariant} for ticker {ticker} at {timestamp}: "
                f"Open={row['Open']!r}, High={row['High']!r}, Low={row['Low']!r}, "
                f"Close={row['Close']!r}."
            )
    return output


def terminal_index(entry_index: int, horizon: int) -> int:
    """Return the inclusive terminal local-bar index for a completed-bar horizon."""
    if int(entry_index) < 0:
        raise ValueError("entry_index must be non-negative.")
    if int(horizon) not in HORIZONS:
        raise ValueError(f"Unsupported horizon: {horizon}")
    return int(entry_index) + int(horizon) - 1


def _exit_relation(exit_timestamp: pd.Timestamp, terminal_timestamp: pd.Timestamp) -> tuple[str, bool | None]:
    if exit_timestamp > terminal_timestamp:
        return OPEN_THROUGH_HORIZON, True
    if exit_timestamp < terminal_timestamp:
        return EXITED_BEFORE_HORIZON, False
    return EXITED_ON_HORIZON_TIMESTAMP, None


def _validate_entry_population(
    trades: pd.DataFrame,
    *,
    expected_trade_count: int | None,
    expected_window_count: int | None,
) -> pd.DataFrame:
    missing = REQUIRED_ENTRY_COLUMNS.difference(trades.columns)
    if missing:
        raise ValueError(f"Entry Statistics trades missing columns: {sorted(missing)}")
    output = trades.copy(deep=True)
    if not output["model"].astype(str).eq(MODEL_BASELINE).all():
        raise ValueError("Forward Return Statistics requires FIXED_BASELINE trades only.")
    if output.empty:
        raise ValueError("Entry Statistics source has no completed baseline trades.")
    if output["source_trade_row"].duplicated().any() or output["trade_id"].duplicated().any():
        raise ValueError("Entry Statistics trade identifiers are not unique.")
    tickers = set(output["ticker"].astype(str))
    if not tickers.issubset(set(CONTROLLED_TICKERS)):
        raise ValueError("Entry Statistics trades include a ticker outside the controlled basket.")
    if expected_trade_count is not None and len(output) != expected_trade_count:
        raise ValueError(f"Expected {expected_trade_count} baseline trades; found {len(output)}.")
    windows = output["window_id"].astype(str).nunique()
    if expected_window_count is not None and windows != expected_window_count:
        raise ValueError(f"Expected {expected_window_count} baseline windows; found {windows}.")
    return output


def forward_fields_for_trade(
    *,
    window_data: pd.DataFrame,
    entry_timestamp: Any,
    exit_timestamp: Any,
    entry_fill_price: float,
) -> dict[str, Any]:
    """Compute inclusive, ticker-local observational outcomes for one trade."""
    entry_timestamp = _normalized_timestamp(entry_timestamp)
    exit_timestamp = _normalized_timestamp(exit_timestamp)
    if not isfinite(float(entry_fill_price)) or float(entry_fill_price) <= 0:
        raise ValueError("entry_fill_price must be finite and positive.")
    matches = int((window_data.index == entry_timestamp).sum())
    if matches != 1:
        raise ValueError(f"Expected exactly one entry row at {entry_timestamp}; found {matches}.")
    entry_index = int(window_data.index.get_loc(entry_timestamp))
    output: dict[str, Any] = {}
    for horizon in HORIZONS:
        prefix = f"forward_{horizon}_"
        terminal = terminal_index(entry_index, horizon)
        output[f"{prefix}horizon_ticker_bars"] = horizon
        if terminal >= len(window_data):
            output.update(
                {
                    f"{prefix}terminal_ticker_timestamp": None,
                    f"{prefix}terminal_close": None,
                    f"{prefix}available": False,
                    f"{prefix}availability_reason": INSUFFICIENT_BARS,
                    f"{prefix}close_to_entry_fill_return_percent": None,
                    f"{prefix}maximum_observed_high": None,
                    f"{prefix}observed_high_excursion_percent": None,
                    f"{prefix}minimum_observed_low": None,
                    f"{prefix}observed_low_excursion_percent": None,
                    f"{prefix}exit_relation": None,
                    f"{prefix}trade_active_through_horizon": None,
                }
            )
            continue
        terminal_row = window_data.iloc[terminal]
        observed = window_data.iloc[entry_index : terminal + 1]
        terminal_timestamp = pd.Timestamp(window_data.index[terminal])
        close = float(terminal_row["Close"])
        maximum_high = float(observed["High"].max())
        minimum_low = float(observed["Low"].min())
        relation, active = _exit_relation(exit_timestamp, terminal_timestamp)
        output.update(
            {
                f"{prefix}terminal_ticker_timestamp": terminal_timestamp,
                f"{prefix}terminal_close": close,
                f"{prefix}available": True,
                f"{prefix}availability_reason": AVAILABLE,
                f"{prefix}close_to_entry_fill_return_percent": (close / entry_fill_price - 1) * 100,
                f"{prefix}maximum_observed_high": maximum_high,
                f"{prefix}observed_high_excursion_percent": (maximum_high / entry_fill_price - 1) * 100,
                f"{prefix}minimum_observed_low": minimum_low,
                f"{prefix}observed_low_excursion_percent": (minimum_low / entry_fill_price - 1) * 100,
                f"{prefix}exit_relation": relation,
                f"{prefix}trade_active_through_horizon": active,
            }
        )
    return output


def enrich_forward_returns(
    *,
    trades: pd.DataFrame,
    data_by_ticker: Mapping[str, pd.DataFrame],
    windows: pd.DataFrame,
    forward_return_statistics_stamp: str,
    expected_trade_count: int | None = None,
    expected_window_count: int | None = None,
) -> pd.DataFrame:
    """Append only forward outcome fields while preserving source trade fields."""
    source = _validate_entry_population(
        trades,
        expected_trade_count=expected_trade_count,
        expected_window_count=expected_window_count,
    )
    window_by_id = _window_lookup(windows)
    normalized_data = {
        str(ticker).strip().upper(): _normalize_and_validate_market_frame(frame, str(ticker))
        for ticker, frame in data_by_ticker.items()
    }
    records: list[dict[str, Any]] = []
    for source_row in source.to_dict("records"):
        ticker = str(source_row["ticker"]).strip().upper()
        window_id = str(source_row["window_id"])
        if ticker not in normalized_data:
            raise ValueError(f"Snapshot missing trade ticker: {ticker}")
        if window_id not in window_by_id:
            raise ValueError(f"Snapshot missing trade window: {window_id}")
        start, end = window_by_id[window_id]
        window_data = normalized_data[ticker].loc[
            (normalized_data[ticker].index >= start) & (normalized_data[ticker].index < end)
        ]
        if window_data.empty:
            raise ValueError(f"No snapshot rows for {window_id}/{ticker}.")
        entry_timestamp = _normalized_timestamp(source_row["entry_timestamp"])
        if entry_timestamp < start or entry_timestamp >= end:
            raise ValueError(f"Entry timestamp outside test window for {window_id}/{ticker}.")
        record = {
            "forward_return_statistics_stamp": forward_return_statistics_stamp,
            **source_row,
            **forward_fields_for_trade(
                window_data=window_data,
                entry_timestamp=entry_timestamp,
                exit_timestamp=source_row["exit_timestamp"],
                entry_fill_price=float(source_row["entry_fill_price"]),
            ),
        }
        records.append(record)
    output = pd.DataFrame(records, columns=FORWARD_COLUMNS)
    if output["source_trade_row"].tolist() != source["source_trade_row"].tolist():
        raise ValueError("Forward Return Statistics did not preserve source-trade order.")
    return output


def _finite_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric.map(lambda value: pd.isna(value) or isfinite(float(value))))


def _statistic_record(
    frame: pd.DataFrame,
    *,
    metric: str,
    forward_return_statistics_stamp: str,
    population: str,
    group_type: str,
    group_value: str,
    horizon: int,
) -> dict[str, Any]:
    values = _finite_numeric(frame[metric])
    valid = values.dropna()
    count = int(len(valid))
    negative = int((valid < 0).sum())
    flat = int((valid == 0).sum())
    positive = int((valid > 0).sum())

    def percentile(value: float) -> float | None:
        return float(valid.quantile(value, interpolation="linear")) if count else None

    return {
        "forward_return_statistics_stamp": forward_return_statistics_stamp,
        "population": population,
        "group_type": group_type,
        "group_value": str(group_value),
        "horizon_ticker_bars": horizon,
        "metric": metric,
        "population_trade_count": int(len(frame)),
        "count": count,
        "missing_count": int(len(frame) - count),
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
    forward_return_statistics_stamp: str,
    population: str,
    group_type: str,
    group_column: str | None = None,
    group_values: Iterable[Any] | None = None,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        active_column = group_column.format(horizon=horizon) if group_column else None
        if active_column is None:
            groups = [("ALL", trades)]
        else:
            if active_column not in trades.columns:
                raise ValueError(f"Forward trades missing group column: {active_column}")
            values = list(group_values) if group_values is not None else sorted(
                trades[active_column].dropna().unique(), key=lambda value: str(value)
            )
            groups = [(str(value), trades.loc[trades[active_column] == value]) for value in values]
        metrics = [f"forward_{horizon}_{suffix}" for suffix in FORWARD_METRIC_SUFFIXES]
        for metric in metrics:
            if metric not in trades.columns:
                raise ValueError(f"Forward trades missing metric: {metric}")
            for group_value, group in groups:
                records.append(
                    _statistic_record(
                        group,
                        metric=metric,
                        forward_return_statistics_stamp=forward_return_statistics_stamp,
                        population=population,
                        group_type=group_type,
                        group_value=group_value,
                        horizon=horizon,
                    )
                )
    return pd.DataFrame(records, columns=STATISTICS_COLUMNS)


def _all_populations_table(
    trades: pd.DataFrame,
    *,
    forward_return_statistics_stamp: str,
    group_type: str,
    group_column: str | None = None,
    group_values: Iterable[Any] | None = None,
) -> pd.DataFrame:
    primary = build_statistics_table(
        trades,
        forward_return_statistics_stamp=forward_return_statistics_stamp,
        population=PRIMARY_POPULATION,
        group_type=group_type,
        group_column=group_column,
        group_values=group_values,
    )
    sensitivity = build_statistics_table(
        trades.loc[trades["exit_reason"] != "FORCE_CLOSE_END"],
        forward_return_statistics_stamp=forward_return_statistics_stamp,
        population=SENSITIVITY_POPULATION,
        group_type=group_type,
        group_column=group_column,
        group_values=group_values,
    )
    return pd.concat([primary, sensitivity], ignore_index=True)


def build_aggregation_tables(
    trades: pd.DataFrame, *, forward_return_statistics_stamp: str
) -> dict[str, pd.DataFrame]:
    """Build deterministic long-form summaries for both approved populations."""
    static = lambda group_type, column=None, values=None: _all_populations_table(
        trades,
        forward_return_statistics_stamp=forward_return_statistics_stamp,
        group_type=group_type,
        group_column=column,
        group_values=values,
    )
    sensitivity = trades.loc[trades["exit_reason"] != "FORCE_CLOSE_END"]
    return {
        "overall": static("ALL_TRADES"),
        "outcomes": static("OUTCOME_CLASS", "outcome_class", OUTCOME_ORDER),
        "asset_classes": static("ASSET_CLASS", "asset_class", ASSET_CLASS_ORDER),
        "tickers": static("TICKER", "ticker", CONTROLLED_TICKERS),
        "windows": static("WINDOW_ID", "window_id", sorted(trades["window_id"].astype(str).unique())),
        "entry_years": static("ENTRY_YEAR", "entry_year", sorted(trades["entry_year"].unique())),
        "entry_gap_buckets": static("ENTRY_GAP_BUCKET", "entry_gap_bucket", GAP_BUCKETS),
        "exit_reasons": static("EXIT_REASON", "exit_reason"),
        "exit_categories": static("EXIT_CATEGORY", "exit_category"),
        "availability": static("HORIZON_AVAILABILITY_REASON", "forward_{horizon}_availability_reason"),
        "exit_relations": static("HORIZON_EXIT_RELATION", "forward_{horizon}_exit_relation"),
        "force_close_excluded": build_statistics_table(
            sensitivity,
            forward_return_statistics_stamp=forward_return_statistics_stamp,
            population=SENSITIVITY_POPULATION,
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
        "horizon_availability_counts": {
            str(horizon): {
                AVAILABLE: int(trades[f"forward_{horizon}_available"].sum()),
                INSUFFICIENT_BARS: int((~trades[f"forward_{horizon}_available"]).sum()),
            }
            for horizon in HORIZONS
        },
    }


def _reconcile_official_availability(
    *,
    summary: Mapping[str, Any],
    source_verification: Mapping[str, Any] | None,
) -> None:
    """Lock availability counts only for the approved, strictly verified source."""
    if not isinstance(source_verification, Mapping) or not source_verification.get("official_source"):
        return
    counts = summary["horizon_availability_counts"]
    for horizon, expected in OFFICIAL_AVAILABILITY_COUNTS.items():
        actual = int(counts[str(horizon)][AVAILABLE])
        if actual != expected:
            raise ValueError(
                f"Approved official availability mismatch for H{horizon}: "
                f"expected {expected}; found {actual}."
            )


def _validate_saveable_provenance(bundle: Mapping[str, Any]) -> None:
    """Fail closed unless a bundle carries complete strict source verification."""
    source_files = bundle.get("source_files")
    source_hashes = bundle.get("source_hashes")
    verification = bundle.get("source_verification")
    if not isinstance(source_files, Mapping):
        raise ValueError("Cannot save: consumed source metadata is missing.")
    if not source_files:
        raise ValueError("Cannot save: consumed source mapping is empty.")
    if not isinstance(source_hashes, Mapping):
        raise ValueError("Cannot save: consumed source hashes are missing.")
    if set(source_files) != set(source_hashes):
        raise ValueError("Cannot save: every consumed source requires exactly one recorded hash.")
    if bundle.get("save_authorization") is not _STRICT_SOURCE_AUTHORIZATION:
        raise ValueError("Cannot save: source was not loaded through strict verified provenance.")
    required_lineage = (
        "entry_statistics_stamp",
        "timing_stamp",
        "source_stamp",
        "snapshot_id",
        "snapshot_fingerprint",
    )
    for field in required_lineage:
        value = bundle.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Cannot save: required result lineage field {field} is missing.")
    if not isinstance(verification, Mapping):
        raise ValueError("Cannot save: verified source lineage metadata is missing.")
    if verification.get("verified") is not True or verification.get("code_hash_verification") is not True:
        raise ValueError("Cannot save: source verification is incomplete or weakened.")
    verification_fields = {
        "entry_statistics_stamp": bundle["entry_statistics_stamp"],
        "timing_stamp": bundle["timing_stamp"],
        "source_stop_walk_forward_stamp": bundle["source_stamp"],
        "snapshot_id": bundle["snapshot_id"],
        "snapshot_fingerprint": bundle["snapshot_fingerprint"],
    }
    for field, expected in verification_fields.items():
        if verification.get(field) != expected:
            raise ValueError(f"Cannot save: result lineage disagrees with verified source {field}.")
    for label, path_value in source_files.items():
        path = Path(path_value)
        recorded_hash = source_hashes.get(label)
        if not isinstance(recorded_hash, str) or not recorded_hash:
            raise ValueError(f"Cannot save: recorded source hash is missing for {label}.")
        if not path.is_file():
            raise ValueError(f"Cannot save: consumed source path is missing for {label}: {path}")
        if sha256_file(path) != recorded_hash:
            raise ValueError(f"Source changed before save: {path}")


def _field_definitions() -> dict[str, str]:
    definitions = {column: column.replace("_", " ") for column in FORWARD_COLUMNS}
    definitions.update(
        {
            "forward_return_statistics_stamp": "Stable stamp for this observational Forward Return Statistics result.",
            "forward_*_horizon_ticker_bars": "Completed ticker-local bar count from entry; H1 terminates on the entry bar.",
            "forward_*_availability_reason": "AVAILABLE or INSUFFICIENT_TICKER_BARS_IN_TEST_WINDOW; unavailable rows retain the trade with missing outcome values.",
            "forward_*_maximum_observed_high": "Maximum OHLC High over the inclusive outcome interval; observational only, not executable MFE.",
            "forward_*_observed_high_excursion_percent": "Observed High excursion from simulated entry fill; not executable MFE.",
            "forward_*_minimum_observed_low": "Minimum OHLC Low over the inclusive outcome interval; observational only, not executable MAE.",
            "forward_*_observed_low_excursion_percent": "Observed Low excursion from simulated entry fill; not executable MAE.",
            "forward_*_exit_relation": "For available horizons: OPEN_THROUGH_HORIZON, EXITED_BEFORE_HORIZON, or EXITED_ON_HORIZON_TIMESTAMP.",
            "forward_*_trade_active_through_horizon": "True only when exit is later than terminal timestamp; False when earlier; missing when timestamps are equal or horizon unavailable.",
        }
    )
    return definitions


def _formulas() -> dict[str, str]:
    return {
        "terminal_index": "entry_index + horizon - 1",
        "close_to_entry_fill_return_percent": "(Close[terminal_index] / entry_fill_price - 1) * 100",
        "observed_high_excursion_percent": "(max(High[entry_index:terminal_index inclusive]) / entry_fill_price - 1) * 100",
        "observed_low_excursion_percent": "(min(Low[entry_index:terminal_index inclusive]) / entry_fill_price - 1) * 100",
    }


def build_json_payload(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": bundle["created_at"],
        "forward_return_statistics_stamp": bundle["forward_return_statistics_stamp"],
        "method": "Observational ticker-local forward OHLC outcomes from provenanced Entry Statistics trades and an immutable snapshot; no engine replay.",
        "population_definition": {
            "model": MODEL_BASELINE,
            "primary": "All completed baseline trades in independently reset test windows, including FORCE_CLOSE_END.",
            "sensitivity": "Primary population excluding raw exit_reason FORCE_CLOSE_END.",
            "expected_trade_count": bundle["expected_trade_count"],
            "expected_window_count": bundle["expected_window_count"],
            "controlled_tickers": list(CONTROLLED_TICKERS),
            "windows_are_separate_reset_capital_runs": True,
        },
        "source_lineage": {
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
        "field_definitions": _field_definitions(),
        "formulas": _formulas(),
        "methodological_choices": {
            "horizons": list(HORIZONS),
            "horizon_semantics": "H1 terminal row is entry row; terminal_index = entry_index + horizon - 1.",
            "outcome_interval": "Inclusive entry row through terminal row on the ticker-local calendar.",
            "post_exit_outcomes": "Retained as observational counterfactual market outcomes and never treated as realized trade returns.",
            "standard_deviation": "population standard deviation, ddof=0",
            "percentiles": "linear interpolation",
            "source_order": "Preserved from provenanced Entry Statistics trades CSV.",
            "macd_role": "Diagnostic-only source field; unused by this analyzer.",
        },
        "trade_schema": list(FORWARD_COLUMNS),
        "aggregation_schema": list(STATISTICS_COLUMNS),
        "summary": bundle["summary"],
        "aggregation_results": {name: frame.to_dict("records") for name, frame in bundle["aggregations"].items()},
        "limitations": [
            "High and Low excursions are observational OHLC ranges, not executable MFE or MAE; intrabar ordering is unknown.",
            "Unavailable horizons remain as missing outcomes and do not cross test-window boundaries.",
            "FORCE_CLOSE_END may use a union-calendar timestamp without a matching ticker bar; no ticker bar is fabricated.",
            "Forward outcomes after an exit are counterfactual market observations, not realized returns.",
            "The provenance file hashes every other generated output and cannot self-hash.",
        ],
    }


def build_forward_return_statistics(
    *,
    trades: pd.DataFrame,
    data_by_ticker: Mapping[str, pd.DataFrame],
    windows: pd.DataFrame,
    entry_statistics_stamp: str,
    timing_stamp: str,
    source_stamp: str,
    snapshot_id: str,
    snapshot_fingerprint: str,
    expected_trade_count: int | None = None,
    expected_window_count: int | None = None,
    forward_return_statistics_stamp: str | None = None,
    source_files: Mapping[str, Path] | None = None,
    source_hashes: Mapping[str, str] | None = None,
    source_verification: Mapping[str, Any] | None = None,
    save_authorization: object | None = None,
) -> dict[str, Any]:
    stamp = forward_return_statistics_stamp or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    enriched = enrich_forward_returns(
        trades=trades,
        data_by_ticker=data_by_ticker,
        windows=windows,
        forward_return_statistics_stamp=stamp,
        expected_trade_count=expected_trade_count,
        expected_window_count=expected_window_count,
    )
    paths = (
        None
        if source_files is None
        else {str(label): Path(path) for label, path in source_files.items()}
    )
    hashes = (
        None
        if source_hashes is None
        else {str(label): value for label, value in source_hashes.items()}
    )
    summary = _population_summary(enriched)
    _reconcile_official_availability(
        summary=summary,
        source_verification=source_verification,
    )
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "forward_return_statistics_stamp": stamp,
        "entry_statistics_stamp": entry_statistics_stamp,
        "timing_stamp": timing_stamp,
        "source_stamp": source_stamp,
        "snapshot_id": snapshot_id,
        "snapshot_fingerprint": snapshot_fingerprint,
        "expected_trade_count": expected_trade_count,
        "expected_window_count": expected_window_count,
        "trades": enriched,
        "aggregations": build_aggregation_tables(
            enriched, forward_return_statistics_stamp=stamp
        ),
        "summary": summary,
        "source_files": paths,
        "source_hashes": hashes,
        "source_verification": None if source_verification is None else dict(source_verification),
        "save_authorization": save_authorization,
    }


def load_verified_source(
    *,
    snapshot_directory: Path,
    entry_statistics_directory: Path,
    entry_statistics_stamp: str,
    project_root: Path = Path("."),
    verify_code: bool = True,
    expectations: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load the single approved Entry Statistics source chain and snapshot."""
    active = _resolved_expectations(expectations)
    if entry_statistics_stamp != active["entry_statistics_stamp"]:
        raise ValueError(
            "Forward Return Statistics requires Entry Statistics stamp "
            f"{active['entry_statistics_stamp']}; received {entry_statistics_stamp}."
        )
    snapshot_directory = Path(snapshot_directory)
    entry_statistics_directory = Path(entry_statistics_directory)
    snapshot = load_snapshot(
        snapshot_directory, verify_code=verify_code, project_root=Path(project_root)
    )
    manifest = snapshot["manifest"]
    if manifest.get("snapshot_id") != active["snapshot_id"]:
        raise ValueError("Snapshot ID does not match the approved Forward Return Statistics source.")
    if set(manifest.get("market_files", {})) != set(active["controlled_tickers"]):
        raise ValueError("Snapshot does not contain exactly the expected controlled basket.")

    provenance_path = entry_statistics_directory / (
        f"portfolio_entry_statistics_provenance_{entry_statistics_stamp}.json"
    )
    provenance = _read_json(provenance_path)
    expected_lineage = {
        "entry_statistics_stamp": entry_statistics_stamp,
        "timing_stamp": active["timing_stamp"],
        "source_stop_walk_forward_stamp": active["source_stop_walk_forward_stamp"],
        "snapshot_id": active["snapshot_id"],
        "snapshot_fingerprint": manifest.get("fingerprint"),
    }
    for key, expected in expected_lineage.items():
        if provenance.get(key) != expected:
            raise ValueError(f"Entry Statistics provenance {key} mismatch.")
    result_files = verify_provenance_result_files(
        directory=entry_statistics_directory,
        provenance=provenance,
        prefix="portfolio_entry_statistics_",
        stamp=entry_statistics_stamp,
    )
    if not {"trades", "json"}.issubset(result_files):
        raise ValueError("Entry Statistics provenance is missing required trades or JSON results.")
    payload = _read_json(result_files["json"])
    if payload.get("entry_statistics_stamp") != entry_statistics_stamp:
        raise ValueError("Entry Statistics JSON entry_statistics_stamp mismatch.")
    payload_lineage = payload.get("source_lineage")
    if not isinstance(payload_lineage, dict):
        raise ValueError("Entry Statistics JSON has no source_lineage object.")
    for key in (
        "timing_stamp",
        "source_stop_walk_forward_stamp",
        "snapshot_id",
        "snapshot_fingerprint",
    ):
        if payload_lineage.get(key) != expected_lineage[key]:
            raise ValueError(f"Entry Statistics JSON {key} mismatch.")
    trades = pd.read_csv(result_files["trades"])
    _validate_entry_population(
        trades,
        expected_trade_count=active["expected_trade_count"],
        expected_window_count=active["expected_window_count"],
    )
    if set(trades["window_id"].astype(str)) != set(snapshot["windows"]["window_id"].astype(str)):
        raise ValueError("Entry Statistics windows do not match frozen snapshot windows.")

    source_files: dict[str, Path] = {
        "snapshot_manifest": snapshot_directory / "manifest.json",
        "snapshot_config": snapshot_directory / manifest["config"]["path"],
        "snapshot_windows": snapshot_directory / manifest["windows"]["path"],
        "entry_statistics_provenance": provenance_path,
        "forward_return_statistics_code": Path(__file__).resolve(),
    }
    for ticker, metadata in manifest["market_files"].items():
        source_files[f"snapshot_market:{ticker}"] = snapshot_directory / metadata["path"]
    source_files.update({f"entry_statistics_result:{name}": path for name, path in result_files.items()})
    source_hashes = {label: sha256_file(path) for label, path in source_files.items()}
    official_source = active == _resolved_expectations(None) and bool(verify_code)
    source_verification = {
        "verified": True,
        "code_hash_verification": bool(verify_code),
        "official_source": official_source,
        "entry_statistics_stamp": entry_statistics_stamp,
        "timing_stamp": active["timing_stamp"],
        "source_stop_walk_forward_stamp": active["source_stop_walk_forward_stamp"],
        "snapshot_id": active["snapshot_id"],
        "snapshot_fingerprint": str(manifest["fingerprint"]),
    }
    return {
        "snapshot": snapshot,
        "trades": trades,
        "entry_statistics_stamp": entry_statistics_stamp,
        "timing_stamp": active["timing_stamp"],
        "source_stamp": active["source_stop_walk_forward_stamp"],
        "source_files": source_files,
        "source_hashes": source_hashes,
        "source_verification": source_verification,
        "save_authorization": _STRICT_SOURCE_AUTHORIZATION if verify_code else None,
        "expected_trade_count": active["expected_trade_count"],
        "expected_window_count": active["expected_window_count"],
    }


def run_forward_return_statistics(source: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = source["snapshot"]
    manifest = snapshot["manifest"]
    return build_forward_return_statistics(
        trades=source["trades"],
        data_by_ticker=snapshot["data_by_ticker"],
        windows=snapshot["windows"],
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


def save_forward_return_statistics(
    bundle: Mapping[str, Any], output_directory: Path = DEFAULT_OUTPUT_DIRECTORY
) -> dict[str, Path]:
    """Save future analysis outputs only after an explicitly approved saving run."""
    output_directory = Path(output_directory)
    stamp = str(bundle["forward_return_statistics_stamp"])
    frames = {"trades": bundle["trades"], **bundle["aggregations"]}
    paths = {
        name: output_directory / f"portfolio_forward_return_statistics_{name}_{stamp}.csv"
        for name in frames
    }
    paths["json"] = output_directory / f"portfolio_forward_return_statistics_{stamp}.json"
    paths["provenance"] = output_directory / f"portfolio_forward_return_statistics_provenance_{stamp}.json"
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(existing[0])
    _validate_saveable_provenance(bundle)
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
        "forward_return_statistics_stamp": stamp,
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
    mode = "NO-SAVE ANALYSIS" if no_save else "SAVING ANALYSIS"
    print(f"FORWARD RETURN STATISTICS — OBSERVATIONAL {mode}")
    print("Provenance validation: PASS")
    print(f"Entry Statistics stamp: {bundle['entry_statistics_stamp']}")
    print(f"Timing stamp: {bundle['timing_stamp']}")
    print(f"Snapshot: {bundle['snapshot_id']}")
    print(
        f"Primary population: {summary['primary_trade_count']} completed FIXED_BASELINE trades "
        f"across {summary['window_count']} reset-capital windows"
    )
    for horizon in HORIZONS:
        counts = summary["horizon_availability_counts"][str(horizon)]
        print(
            f"H{horizon} availability: {counts[AVAILABLE]} available; "
            f"{counts[INSUFFICIENT_BARS]} insufficient ticker-local rows"
        )
    print(
        f"FORCE_CLOSE_END sensitivity: excluded={summary['force_close_end_trade_count']} "
        f"remaining={summary['force_close_excluded_trade_count']}"
    )
    if no_save:
        print("Output artifacts saved: 0 (--no-save)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Observational Forward Return Statistics for provenanced portfolio trades"
    )
    parser.add_argument("--snapshot-directory", type=Path, required=True)
    parser.add_argument("--entry-statistics-directory", type=Path, required=True)
    parser.add_argument("--entry-statistics-stamp", required=True)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--no-save", action="store_true")
    arguments = parser.parse_args()
    source = load_verified_source(
        snapshot_directory=arguments.snapshot_directory,
        entry_statistics_directory=arguments.entry_statistics_directory,
        entry_statistics_stamp=arguments.entry_statistics_stamp,
    )
    bundle = run_forward_return_statistics(source)
    if arguments.no_save:
        _print_summary(bundle, no_save=True)
        return
    _print_summary(bundle, no_save=False)
    for name, path in save_forward_return_statistics(bundle, arguments.output_directory).items():
        print(name, path.resolve())


if __name__ == "__main__":
    main()
