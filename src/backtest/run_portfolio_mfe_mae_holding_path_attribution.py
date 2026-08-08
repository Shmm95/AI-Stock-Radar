"""Observed fully-held MFE/MAE and Close paths for completed portfolio trades.

The analyzer consumes the provenanced Forward Return Statistics population and
the immutable research snapshot.  It does not import or replay the portfolio
engine.  Only ticker-local bars held from Open through Close enter full-held
metrics; exit-Open observations and partial intrabar stop bars stay separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

SCHEMA_VERSION = 1
MODEL_BASELINE = "FIXED_BASELINE"
APPROVED_ENTRY_STATISTICS_STAMP = "20260802_103007"
APPROVED_TIMING_STAMP = "20260802_085312"
APPROVED_SOURCE_STAMP = "20260802_081449"
APPROVED_SNAPSHOT_ID = "20260802_081049_d93145cb1dcf"
APPROVED_SNAPSHOT_FINGERPRINT = (
    "d93145cb1dcf3f837415f5b28ac29eeb51c13bc8ce5b8b1187aa34ebfc01ad44"
)
APPROVED_FORWARD_RETURN_STATISTICS_STAMP = "20260802_134341"
EXPECTED_TRADE_COUNT = 256
EXPECTED_WINDOW_COUNT = 13
EXPECTED_FORCE_CLOSE_COUNT = 42
EXPECTED_SENSITIVITY_COUNT = 214
APPROVED_SLIPPAGE_BPS = 5.0
APPROVED_STOP_PERCENT = 5.0
CONTROLLED_TICKERS = (
    "AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA", "BTC-USD", "ETH-USD"
)
OUTCOME_ORDER = ("WINNER", "LOSER", "FLAT")
ASSET_CLASS_ORDER = ("EQUITY", "CRYPTO")
GAP_BUCKETS = (
    "LT_NEG_2", "NEG_2_TO_NEG_1", "NEG_1_TO_NEG_0P5", "NEG_0P5_TO_0",
    "ZERO_TO_POS_0P5", "POS_0P5_TO_POS_1", "POS_1_TO_POS_2", "GE_POS_2",
)
DEFAULT_OUTPUT_DIRECTORY = Path(
    "data/backtests/portfolio/mfe_mae_holding_path_attribution"
)

PRIMARY_POPULATION = "PRIMARY_ALL_COMPLETED_BASELINE_TRADES"
SENSITIVITY_POPULATION = "SENSITIVITY_EXCLUDING_FORCE_CLOSE_END"
FULL_HELD_AVAILABLE = "FULLY_HELD_TICKER_BARS_AVAILABLE"
FULL_HELD_UNAVAILABLE = "NO_FULLY_HELD_TICKER_BARS"

OPEN_PENDING_EXIT = "OPEN_PENDING_EXIT"
OPEN_GAP_STOP = "OPEN_GAP_STOP"
INTRABAR_INITIAL_STOP = "INTRABAR_INITIAL_STOP"
FORCE_CLOSE_LOCAL_CLOSE = "FORCE_CLOSE_LOCAL_CLOSE"

OPEN_EXIT_EXCLUDED = "OPEN_EXIT_EXCLUDED"
INTRABAR_PARTIAL_AMBIGUOUS = "INTRABAR_PARTIAL_AMBIGUOUS"

NO_AMBIGUITY = "NONE"
HIGH_LOW_ORDER_UNKNOWN = "HIGH_LOW_ORDER_UNKNOWN"
EXIT_BAR_HIGH_LOW_POST_EXIT_UNKNOWN = "EXIT_BAR_HIGH_LOW_POST_EXIT_UNKNOWN"

EXPECTATION_KEYS = (
    "forward_return_statistics_stamp",
    "entry_statistics_stamp",
    "timing_stamp",
    "source_stop_walk_forward_stamp",
    "snapshot_id",
    "expected_trade_count",
    "expected_window_count",
    "expected_force_close_count",
    "controlled_tickers",
)
OFFICIAL_EXPECTATIONS = {
    "forward_return_statistics_stamp": APPROVED_FORWARD_RETURN_STATISTICS_STAMP,
    "entry_statistics_stamp": APPROVED_ENTRY_STATISTICS_STAMP,
    "timing_stamp": APPROVED_TIMING_STAMP,
    "source_stop_walk_forward_stamp": APPROVED_SOURCE_STAMP,
    "snapshot_id": APPROVED_SNAPSHOT_ID,
    "expected_trade_count": EXPECTED_TRADE_COUNT,
    "expected_window_count": EXPECTED_WINDOW_COUNT,
    "expected_force_close_count": EXPECTED_FORCE_CLOSE_COUNT,
    "controlled_tickers": CONTROLLED_TICKERS,
}
OFFICIAL_EXIT_PHASE_COUNTS = {
    OPEN_PENDING_EXIT: 85,
    OPEN_GAP_STOP: 15,
    INTRABAR_INITIAL_STOP: 114,
    FORCE_CLOSE_LOCAL_CLOSE: 42,
}
OFFICIAL_FULL_HELD_AVAILABILITY_COUNTS = {
    FULL_HELD_AVAILABLE: 235,
    FULL_HELD_UNAVAILABLE: 21,
}
OFFICIAL_FORWARD_RESULT_KEYS = {
    "asset_classes", "availability", "entry_gap_buckets", "entry_years",
    "exit_categories", "exit_reasons", "exit_relations", "force_close_excluded",
    "json", "outcomes", "overall", "tickers", "trades", "windows",
}

@dataclass(frozen=True)
class _StrictOfficialBundleRegistration:
    bundle: Mapping[str, Any]
    source: Mapping[str, Any]
    stamp: str
    created_at: str
    trades_csv: str
    statistics_csv: str
    json_payload: str


_STRICT_OFFICIAL_SOURCES: list[Mapping[str, Any]] = []
_STRICT_OFFICIAL_BUNDLES: list[_StrictOfficialBundleRegistration] = []

FORWARD_STATISTICS_TRADE_COLUMNS = (
    "forward_return_statistics_stamp", "entry_statistics_stamp", "timing_stamp",
    "source_stop_walk_forward_stamp", "snapshot_id", "snapshot_fingerprint",
    "source_trade_row", "trade_id", "window_id", "model", "ticker", "asset_class",
    "stock_stop_loss_percent", "crypto_stop_loss_percent", "signal_timestamp",
    "signal_year", "signal_close", "entry_timestamp", "entry_year", "entry_open",
    "entry_fill_price", "raw_entry_gap_percent", "raw_entry_gap_amount",
    "entry_gap_direction", "entry_gap_bucket", "entry_slippage_per_unit",
    "entry_slippage_percent", "entry_slippage_amount", "entry_fee",
    "entry_commission_amount", "entry_commission_percent",
    "entry_transaction_cost_amount", "entry_transaction_cost_percent",
    "signed_entry_timing_effect_amount", "signed_entry_timing_effect_percent",
    "adverse_entry_timing_cost_amount", "favorable_entry_timing_benefit_amount",
    "signal_score", "signal_reason", "signal_ema20", "signal_ema50", "signal_rsi14",
    "signal_macd", "signal_volume", "signal_regime_allowed", "trend_spread_percent",
    "price_extension_percent", "rsi_quality", "momentum_20_percent", "macd_percent",
    "atr14_percent", "low_atr14_quality", "quantity", "entry_notional",
    "initial_stop_percent", "initial_stop_price", "initial_stop_distance",
    "initial_risk_amount", "entry_portfolio_bar_index", "exit_timestamp", "exit_year",
    "exit_price", "exit_portfolio_bar_index", "exit_fee", "total_fees", "gross_pnl",
    "net_pnl", "return_percent", "holding_period_portfolio_bars",
    "holding_period_ticker_bars", "holding_period_calendar_days", "exit_reason",
    "exit_category", "outcome_class", "forward_1_horizon_ticker_bars",
    "forward_1_terminal_ticker_timestamp", "forward_1_terminal_close",
    "forward_1_available", "forward_1_availability_reason",
    "forward_1_close_to_entry_fill_return_percent", "forward_1_maximum_observed_high",
    "forward_1_observed_high_excursion_percent", "forward_1_minimum_observed_low",
    "forward_1_observed_low_excursion_percent", "forward_1_exit_relation",
    "forward_1_trade_active_through_horizon", "forward_3_horizon_ticker_bars",
    "forward_3_terminal_ticker_timestamp", "forward_3_terminal_close",
    "forward_3_available", "forward_3_availability_reason",
    "forward_3_close_to_entry_fill_return_percent", "forward_3_maximum_observed_high",
    "forward_3_observed_high_excursion_percent", "forward_3_minimum_observed_low",
    "forward_3_observed_low_excursion_percent", "forward_3_exit_relation",
    "forward_3_trade_active_through_horizon", "forward_5_horizon_ticker_bars",
    "forward_5_terminal_ticker_timestamp", "forward_5_terminal_close",
    "forward_5_available", "forward_5_availability_reason",
    "forward_5_close_to_entry_fill_return_percent", "forward_5_maximum_observed_high",
    "forward_5_observed_high_excursion_percent", "forward_5_minimum_observed_low",
    "forward_5_observed_low_excursion_percent", "forward_5_exit_relation",
    "forward_5_trade_active_through_horizon", "forward_10_horizon_ticker_bars",
    "forward_10_terminal_ticker_timestamp", "forward_10_terminal_close",
    "forward_10_available", "forward_10_availability_reason",
    "forward_10_close_to_entry_fill_return_percent", "forward_10_maximum_observed_high",
    "forward_10_observed_high_excursion_percent", "forward_10_minimum_observed_low",
    "forward_10_observed_low_excursion_percent", "forward_10_exit_relation",
    "forward_10_trade_active_through_horizon",
)

REQUIRED_FORWARD_COLUMNS = {
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
    "entry_year",
    "entry_gap_bucket",
    "quantity",
    "initial_stop_price",
    "exit_timestamp",
    "exit_price",
    "gross_pnl",
    "net_pnl",
    "return_percent",
    "holding_period_portfolio_bars",
    "exit_reason",
    "exit_category",
    "outcome_class",
}

PATH_COLUMNS = (
    "mfe_mae_holding_path_stamp",
    "source_forward_trade_row",
    "holding_path_trade_id",
    "source_exit_timestamp",
    "effective_local_exit_timestamp",
    "exit_execution_phase",
    "exit_bar_treatment",
    "intrabar_order_ambiguity",
    "exit_open",
    "exit_open_return_percent",
    "simulated_exit_fill_return_percent",
    "realized_gross_return_percent",
    "realized_net_return_percent",
    "full_held_ticker_bar_count",
    "full_held_bar_available",
    "full_held_bar_availability_reason",
    "observed_fully_held_maximum_high",
    "observed_fully_held_mfe_percent",
    "observed_fully_held_mfe_amount",
    "observed_fully_held_mfe_timestamp",
    "observed_fully_held_mfe_offset_from_entry",
    "observed_fully_held_mfe_completed_ticker_bar_number",
    "observed_fully_held_minimum_low",
    "observed_fully_held_mae_percent",
    "observed_fully_held_mae_amount",
    "observed_fully_held_mae_timestamp",
    "observed_fully_held_mae_offset_from_entry",
    "observed_fully_held_mae_completed_ticker_bar_number",
    "minimum_observed_fully_held_return_percent",
    "maximum_observed_fully_held_return_percent",
    "maximum_fully_held_close",
    "maximum_fully_held_close_return_percent",
    "maximum_fully_held_close_timestamp",
    "maximum_fully_held_close_offset_from_entry",
    "maximum_fully_held_close_completed_ticker_bar_number",
    "minimum_fully_held_close",
    "minimum_fully_held_close_return_percent",
    "minimum_fully_held_close_timestamp",
    "minimum_fully_held_close_offset_from_entry",
    "minimum_fully_held_close_completed_ticker_bar_number",
    "first_fully_held_close_above_entry_available",
    "first_fully_held_close_above_entry_timestamp",
    "first_fully_held_close_above_entry_offset_from_entry",
    "first_fully_held_close_above_entry_completed_ticker_bar_number",
    "first_fully_held_close_below_entry_available",
    "first_fully_held_close_below_entry_timestamp",
    "first_fully_held_close_below_entry_offset_from_entry",
    "first_fully_held_close_below_entry_completed_ticker_bar_number",
    "first_fully_held_close_equal_entry_available",
    "first_fully_held_close_equal_entry_timestamp",
    "first_fully_held_close_equal_entry_offset_from_entry",
    "first_fully_held_close_equal_entry_completed_ticker_bar_number",
    "first_fully_held_close_at_or_above_entry_available",
    "first_fully_held_close_at_or_above_entry_timestamp",
    "first_fully_held_close_at_or_above_entry_offset_from_entry",
    "first_fully_held_close_at_or_above_entry_completed_ticker_bar_number",
    "price_breakeven_available",
    "price_breakeven_timestamp",
    "price_breakeven_offset_from_entry",
    "bars_to_price_breakeven",
    "fully_held_close_above_entry_count",
    "fully_held_close_below_entry_count",
    "fully_held_close_at_entry_count",
    "fully_held_close_above_entry_frequency",
    "fully_held_close_below_entry_frequency",
    "fully_held_close_at_entry_frequency",
    "intrabar_stop_price",
    "intrabar_stop_return_percent",
    "partial_exit_bar_timestamp",
    "partial_exit_bar_open",
    "partial_exit_bar_high",
    "partial_exit_bar_low",
    "partial_exit_bar_close",
    "partial_exit_bar_high_return_percent",
    "partial_exit_bar_low_return_percent",
    "partial_exit_bar_observation_available",
    "partial_exit_bar_contains_unknown_post_exit_price_action",
)
TRADE_COLUMNS = (*FORWARD_STATISTICS_TRADE_COLUMNS, *PATH_COLUMNS)

SIGNED_METRICS = (
    "minimum_observed_fully_held_return_percent",
    "maximum_fully_held_close_return_percent",
    "minimum_fully_held_close_return_percent",
    "exit_open_return_percent",
    "simulated_exit_fill_return_percent",
    "realized_gross_return_percent",
    "realized_net_return_percent",
    "intrabar_stop_return_percent",
    "partial_exit_bar_high_return_percent",
    "partial_exit_bar_low_return_percent",
)
NONNEGATIVE_FAVORABLE_MAXIMUM_METRICS = (
    "maximum_observed_fully_held_return_percent",
)
MAGNITUDE_METRICS = (
    "observed_fully_held_mfe_percent",
    "observed_fully_held_mfe_amount",
    "observed_fully_held_mae_percent",
    "observed_fully_held_mae_amount",
)
NONNEGATIVE_METRICS = (
    "full_held_ticker_bar_count",
    "observed_fully_held_mfe_offset_from_entry",
    "observed_fully_held_mfe_completed_ticker_bar_number",
    "observed_fully_held_mae_offset_from_entry",
    "observed_fully_held_mae_completed_ticker_bar_number",
    "maximum_fully_held_close_offset_from_entry",
    "maximum_fully_held_close_completed_ticker_bar_number",
    "minimum_fully_held_close_offset_from_entry",
    "minimum_fully_held_close_completed_ticker_bar_number",
    "bars_to_price_breakeven",
    "fully_held_close_above_entry_count",
    "fully_held_close_below_entry_count",
    "fully_held_close_at_entry_count",
    "fully_held_close_above_entry_frequency",
    "fully_held_close_below_entry_frequency",
    "fully_held_close_at_entry_frequency",
)
STATISTIC_METRICS = (
    *MAGNITUDE_METRICS,
    "minimum_observed_fully_held_return_percent",
    "maximum_observed_fully_held_return_percent",
    *SIGNED_METRICS[1:],
    *NONNEGATIVE_METRICS,
)

STATISTICS_COLUMNS = (
    "mfe_mae_holding_path_stamp",
    "population",
    "group_type",
    "group_value",
    "metric",
    "metric_semantics",
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
    "zero_count",
    "zero_frequency",
    "positive_count",
    "positive_frequency",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _expected_result_name(prefix: str, key: str, stamp: str) -> str:
    if key == "json":
        return f"{prefix}{stamp}.json"
    return f"{prefix}{key}_{stamp}.csv"


def _verify_provenance_result_files(
    *, directory: Path, provenance: Mapping[str, Any], prefix: str, stamp: str
) -> dict[str, Path]:
    records = provenance.get("result_files")
    if not isinstance(records, Mapping) or not records:
        raise ValueError("Forward Return provenance has no result_files mapping.")
    verified: dict[str, Path] = {}
    for key, metadata in records.items():
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Invalid Forward Return result provenance metadata for {key}.")
        expected_name = _expected_result_name(prefix, str(key), stamp)
        declared = Path(str(metadata.get("path", "")))
        if declared.name != expected_name:
            raise ValueError(f"Unexpected provenanced filename for {key}: {declared.name}")
        path = Path(directory) / expected_name
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_hash = metadata.get("sha256")
        if not isinstance(expected_hash, str) or not expected_hash:
            raise ValueError(f"Missing Forward Return result hash for {key}.")
        if sha256_file(path) != expected_hash:
            raise ValueError(f"Forward Return result hash mismatch for {path}.")
        verified[str(key)] = path
    return verified


def _restore_market_frame(path: Path, metadata: Mapping[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "timestamp" not in frame.columns:
        raise ValueError(f"Snapshot market file has no timestamp column: {path}")
    timestamps = pd.to_datetime(frame.pop("timestamp"), errors="raise")
    frame.index = timestamps
    dtypes = metadata.get("dtypes")
    if not isinstance(dtypes, Mapping):
        raise ValueError(f"Snapshot market metadata has no dtypes mapping: {path}")
    for column, dtype_value in dtypes.items():
        if column not in frame.columns:
            raise ValueError(f"Snapshot column missing for {path}: {column}")
        dtype = str(dtype_value)
        if dtype in {"bool", "boolean"}:
            values = frame[column].astype(str).str.lower().map({"true": True, "false": False})
            if values.isna().any():
                raise ValueError(f"Invalid boolean values in {column}: {path}")
            frame[column] = values
        elif dtype.startswith(("float", "int", "uint")):
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def _load_verified_snapshot(
    snapshot_directory: Path, *, verify_code: bool, project_root: Path
) -> dict[str, Any]:
    snapshot_directory = Path(snapshot_directory)
    project_root = Path(project_root).resolve()
    manifest_path = snapshot_directory / "manifest.json"
    manifest = _read_json(manifest_path)
    fingerprint_core = {
        key: value
        for key, value in manifest.items()
        if key not in {"snapshot_id", "fingerprint", "created_at"}
    }
    computed_fingerprint = sha256_json(fingerprint_core)
    if manifest.get("fingerprint") != computed_fingerprint:
        raise ValueError(
            "Snapshot fingerprint mismatch: "
            f"declared={manifest.get('fingerprint')!r}, computed={computed_fingerprint}."
        )
    artifact_records: dict[str, Mapping[str, Any]] = {
        "config": manifest.get("config", {}),
        "windows": manifest.get("windows", {}),
    }
    market_files = manifest.get("market_files")
    if not isinstance(market_files, Mapping) or not market_files:
        raise ValueError("Snapshot manifest has no market_files mapping.")
    artifact_records.update({f"market:{ticker}": metadata for ticker, metadata in market_files.items()})
    artifact_paths: dict[str, Path] = {}
    for label, metadata in artifact_records.items():
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Invalid snapshot metadata for {label}.")
        relative = Path(str(metadata.get("path", "")))
        path = snapshot_directory / relative
        expected_hash = metadata.get("sha256")
        if not isinstance(expected_hash, str) or not expected_hash:
            raise ValueError(f"Snapshot hash missing for {label}.")
        if not path.is_file():
            raise ValueError(f"Snapshot artifact missing for {label}: {path}")
        if sha256_file(path) != expected_hash:
            raise ValueError(f"Snapshot hash mismatch for {label}: {path}")
        artifact_paths[label] = path
    if verify_code:
        code_files = manifest.get("code_files")
        if not isinstance(code_files, Mapping) or not code_files:
            raise ValueError("Snapshot manifest has no code_files mapping.")
        for relative, expected_hash in code_files.items():
            path = project_root / str(relative)
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise ValueError(f"Snapshot code hash mismatch: {relative}")
    config = _read_json(artifact_paths["config"])
    windows = pd.read_csv(artifact_paths["windows"])
    data_by_ticker = {
        str(ticker): _restore_market_frame(
            snapshot_directory / str(metadata["path"]), metadata
        )
        for ticker, metadata in market_files.items()
    }
    return {
        "manifest": manifest,
        "config": config,
        "windows": windows,
        "data_by_ticker": data_by_ticker,
        "artifact_paths": artifact_paths,
        "snapshot_hashes_verified": True,
        "snapshot_code_hashes_verified": bool(verify_code),
    }


def _verify_declared_source_files(
    provenance: Mapping[str, Any], *, project_root: Path
) -> dict[str, Path]:
    records = provenance.get("source_files")
    if not isinstance(records, Mapping) or not records:
        raise ValueError("Forward Return provenance has no source_files mapping.")
    verified: dict[str, Path] = {}
    for label, metadata in records.items():
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Invalid Forward Return source provenance metadata for {label}.")
        declared = Path(str(metadata.get("path", "")))
        path = declared if declared.is_absolute() else Path(project_root) / declared
        expected_hash = metadata.get("sha256")
        if not isinstance(expected_hash, str) or not expected_hash:
            raise ValueError(f"Missing Forward Return source hash for {label}.")
        if not path.is_file():
            raise ValueError(f"Forward Return declared source is missing for {label}: {path}")
        if sha256_file(path) != expected_hash:
            raise ValueError(f"Forward Return declared source hash mismatch for {label}: {path}")
        verified[str(label)] = path
    return verified


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
    if pd.isna(timestamp):
        raise ValueError(f"Invalid missing timestamp: {value!r}")
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


def _normalize_and_validate_market_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    required = {"Open", "High", "Low", "Close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Snapshot missing OHLC columns for {ticker}: {sorted(missing)}")
    output = frame.copy(deep=True)
    output.index = pd.to_datetime(output.index, errors="raise")
    if output.index.tz is not None:
        output.index = output.index.tz_convert(None)
    if output.index.duplicated().any():
        raise ValueError(f"Snapshot has duplicate timestamps for {ticker}.")
    output = output.sort_index()
    for column in ("Open", "High", "Low", "Close"):
        values = pd.to_numeric(output[column], errors="coerce")
        invalid = values.isna() | ~values.map(lambda item: isfinite(float(item))) | values.le(0)
        if invalid.any():
            position = int(invalid.to_numpy().nonzero()[0][0])
            timestamp = output.index[position]
            raise ValueError(
                f"Invalid {column} for ticker {ticker} at {timestamp}: "
                f"{output.iloc[position][column]!r}; expected a finite positive value."
            )
        output[column] = values.astype(float)
    invariants = (
        ("High >= Low", output["High"] >= output["Low"]),
        ("High >= Open", output["High"] >= output["Open"]),
        ("High >= Close", output["High"] >= output["Close"]),
        ("Low <= Open", output["Low"] <= output["Open"]),
        ("Low <= Close", output["Low"] <= output["Close"]),
    )
    for name, valid in invariants:
        if not valid.all():
            position = int((~valid).to_numpy().nonzero()[0][0])
            timestamp = output.index[position]
            row = output.iloc[position]
            raise ValueError(
                f"Invalid OHLC invariant {name} for ticker {ticker} at {timestamp}: "
                f"Open={row['Open']!r}, High={row['High']!r}, Low={row['Low']!r}, "
                f"Close={row['Close']!r}."
            )
    return output


def _finite_positive(value: Any, field: str, trade_id: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {field} for trade {trade_id}: {value!r}") from error
    if not isfinite(result) or result <= 0:
        raise ValueError(f"Invalid {field} for trade {trade_id}: {value!r}")
    return result


def _validated_nonnegative_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or value is None or value is pd.NA:
        raise ValueError(f"Invalid nonnegative integer {field}: {value!r}")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid nonnegative integer {field}: {value!r}") from error
    if not isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise ValueError(f"Invalid nonnegative integer {field}: {value!r}")
    return int(numeric)


def _sell_fill(raw_price: float, slippage_bps: float) -> float:
    return round(float(raw_price) * (1 - float(slippage_bps) / 10_000), 8)


def _validate_exit_fill(
    *, raw_price: float, actual_fill: float, slippage_bps: float, trade_id: str, context: str
) -> None:
    expected = _sell_fill(raw_price, slippage_bps)
    if abs(float(actual_fill) - expected) > 5e-9:
        raise ValueError(
            f"Exit fill mismatch for trade {trade_id} ({context}): "
            f"expected {expected}; found {actual_fill}."
        )


def _exit_semantics(exit_reason: str) -> tuple[str, str]:
    if exit_reason == "EXIT_SIGNAL_NEXT_OPEN":
        return OPEN_PENDING_EXIT, OPEN_EXIT_EXCLUDED
    if exit_reason == "GAP_STOP_LOSS":
        return OPEN_GAP_STOP, OPEN_EXIT_EXCLUDED
    if exit_reason == "STOP_LOSS":
        return INTRABAR_INITIAL_STOP, INTRABAR_PARTIAL_AMBIGUOUS
    if exit_reason == "FORCE_CLOSE_END":
        return FORCE_CLOSE_LOCAL_CLOSE, FORCE_CLOSE_LOCAL_CLOSE
    normalized = exit_reason.upper()
    if "TRAIL" in normalized and ("INTRABAR" in normalized or "STOP" in normalized):
        raise ValueError(
            f"Unsupported intrabar trailing-stop exit semantics: {exit_reason!r}."
        )
    raise ValueError(f"Unsupported exit_reason for holding-path attribution: {exit_reason!r}.")


def _occurrence_fields(
    index: pd.DatetimeIndex, position: int | None, entry_position: int, prefix: str
) -> dict[str, Any]:
    if position is None:
        return {
            f"{prefix}_timestamp": None,
            f"{prefix}_offset_from_entry": None,
            f"{prefix}_completed_ticker_bar_number": None,
        }
    offset = int(position - entry_position)
    return {
        f"{prefix}_timestamp": pd.Timestamp(index[position]),
        f"{prefix}_offset_from_entry": offset,
        f"{prefix}_completed_ticker_bar_number": offset + 1,
    }


def _first_position(mask: pd.Series) -> int | None:
    positions = mask.to_numpy().nonzero()[0]
    return int(positions[0]) if len(positions) else None


def _qualifying_close_fields(
    *,
    full_data: pd.DataFrame,
    entry_position: int,
    mask: pd.Series,
    prefix: str,
) -> dict[str, Any]:
    local = _first_position(mask)
    absolute = None if local is None else entry_position + local
    fields = _occurrence_fields(full_data.index, absolute, entry_position, prefix)
    return {f"{prefix}_available": absolute is not None, **fields}


def holding_path_fields_for_trade(
    *,
    window_data: pd.DataFrame,
    trade: Mapping[str, Any],
    slippage_bps: float = APPROVED_SLIPPAGE_BPS,
) -> dict[str, Any]:
    """Calculate one trade's full-held path without using a post-exit bar."""
    trade_id = str(trade.get("trade_id", "<unknown>"))
    entry_timestamp = _normalized_timestamp(trade["entry_timestamp"])
    source_exit_timestamp = _normalized_timestamp(trade["exit_timestamp"])
    entry_fill = _finite_positive(trade["entry_fill_price"], "entry_fill_price", trade_id)
    exit_fill = _finite_positive(trade["exit_price"], "exit_price", trade_id)
    quantity = _finite_positive(trade["quantity"], "quantity", trade_id)
    if int((window_data.index == entry_timestamp).sum()) != 1:
        raise ValueError(f"Expected exactly one entry row for trade {trade_id} at {entry_timestamp}.")
    entry_position = int(window_data.index.get_loc(entry_timestamp))
    phase, treatment = _exit_semantics(str(trade["exit_reason"]))

    exit_position: int
    if phase == FORCE_CLOSE_LOCAL_CLOSE:
        eligible = window_data.index[window_data.index <= source_exit_timestamp]
        if len(eligible) == 0:
            raise ValueError(f"No local force-close row for trade {trade_id}.")
        effective_exit_timestamp = pd.Timestamp(eligible[-1])
        exit_position = int(window_data.index.get_loc(effective_exit_timestamp))
        if exit_position < entry_position:
            raise ValueError(f"Force-close row precedes entry for trade {trade_id}.")
        _validate_exit_fill(
            raw_price=float(window_data.iloc[exit_position]["Close"]),
            actual_fill=exit_fill,
            slippage_bps=slippage_bps,
            trade_id=trade_id,
            context="FORCE_CLOSE_END local Close",
        )
        full_data = window_data.iloc[entry_position : exit_position + 1]
        exit_open = None
        partial_row = None
    else:
        if int((window_data.index == source_exit_timestamp).sum()) != 1:
            raise ValueError(
                f"Expected exactly one local exit row for trade {trade_id} at {source_exit_timestamp}."
            )
        effective_exit_timestamp = source_exit_timestamp
        exit_position = int(window_data.index.get_loc(source_exit_timestamp))
        if exit_position < entry_position:
            raise ValueError(f"Exit row precedes entry for trade {trade_id}.")
        exit_row = window_data.iloc[exit_position]
        full_data = window_data.iloc[entry_position:exit_position]
        partial_row = exit_row if phase == INTRABAR_INITIAL_STOP else None
        exit_open = float(exit_row["Open"]) if phase in {OPEN_PENDING_EXIT, OPEN_GAP_STOP} else None
        if phase in {OPEN_PENDING_EXIT, OPEN_GAP_STOP}:
            _validate_exit_fill(
                raw_price=float(exit_row["Open"]), actual_fill=exit_fill,
                slippage_bps=slippage_bps, trade_id=trade_id, context=phase,
            )
        else:
            stop_price = _finite_positive(trade["initial_stop_price"], "initial_stop_price", trade_id)
            _validate_exit_fill(
                raw_price=stop_price, actual_fill=exit_fill,
                slippage_bps=slippage_bps, trade_id=trade_id, context=phase,
            )

    result: dict[str, Any] = {
        "source_exit_timestamp": source_exit_timestamp,
        "effective_local_exit_timestamp": effective_exit_timestamp,
        "exit_execution_phase": phase,
        "exit_bar_treatment": treatment,
        "intrabar_order_ambiguity": (
            EXIT_BAR_HIGH_LOW_POST_EXIT_UNKNOWN
            if phase == INTRABAR_INITIAL_STOP
            else HIGH_LOW_ORDER_UNKNOWN if len(full_data) else NO_AMBIGUITY
        ),
        "exit_open": exit_open,
        "exit_open_return_percent": (
            None if exit_open is None else (exit_open / entry_fill - 1) * 100
        ),
        "simulated_exit_fill_return_percent": (exit_fill / entry_fill - 1) * 100,
        "realized_gross_return_percent": float(trade["gross_pnl"]) / (entry_fill * quantity) * 100,
        "realized_net_return_percent": float(trade["return_percent"]),
        "full_held_ticker_bar_count": int(len(full_data)),
        "full_held_bar_available": bool(len(full_data)),
        "full_held_bar_availability_reason": (
            FULL_HELD_AVAILABLE if len(full_data) else FULL_HELD_UNAVAILABLE
        ),
    }

    partial_available = partial_row is not None
    stop_price = float(trade["initial_stop_price"]) if phase == INTRABAR_INITIAL_STOP else None
    result.update(
        {
            "intrabar_stop_price": stop_price,
            "intrabar_stop_return_percent": (
                None if stop_price is None else (stop_price / entry_fill - 1) * 100
            ),
            "partial_exit_bar_timestamp": effective_exit_timestamp if partial_available else None,
            "partial_exit_bar_open": float(partial_row["Open"]) if partial_available else None,
            "partial_exit_bar_high": float(partial_row["High"]) if partial_available else None,
            "partial_exit_bar_low": float(partial_row["Low"]) if partial_available else None,
            "partial_exit_bar_close": float(partial_row["Close"]) if partial_available else None,
            "partial_exit_bar_high_return_percent": (
                (float(partial_row["High"]) / entry_fill - 1) * 100 if partial_available else None
            ),
            "partial_exit_bar_low_return_percent": (
                (float(partial_row["Low"]) / entry_fill - 1) * 100 if partial_available else None
            ),
            "partial_exit_bar_observation_available": partial_available,
            "partial_exit_bar_contains_unknown_post_exit_price_action": partial_available,
        }
    )
    if full_data.empty:
        for column in PATH_COLUMNS:
            if column not in result and column not in {
                "mfe_mae_holding_path_stamp", "source_forward_trade_row", "holding_path_trade_id"
            }:
                result[column] = None
        for column in (
            "first_fully_held_close_above_entry_available",
            "first_fully_held_close_below_entry_available",
            "first_fully_held_close_equal_entry_available",
            "first_fully_held_close_at_or_above_entry_available",
            "price_breakeven_available",
        ):
            result[column] = False
        return result

    highs = full_data["High"]
    lows = full_data["Low"]
    closes = full_data["Close"]
    max_high = float(highs.max())
    min_low = float(lows.min())
    max_high_local = _first_position(highs == max_high)
    min_low_local = _first_position(lows == min_low)
    assert max_high_local is not None and min_low_local is not None
    max_high_position = entry_position + max_high_local
    min_low_position = entry_position + min_low_local
    mfe_percent = max((max_high / entry_fill - 1) * 100, 0.0)
    mae_percent = max((1 - min_low / entry_fill) * 100, 0.0)
    result.update(
        {
            "observed_fully_held_maximum_high": max_high,
            "observed_fully_held_mfe_percent": mfe_percent,
            "observed_fully_held_mfe_amount": max((max_high - entry_fill) * quantity, 0.0),
            "observed_fully_held_minimum_low": min_low,
            "observed_fully_held_mae_percent": mae_percent,
            "observed_fully_held_mae_amount": max((entry_fill - min_low) * quantity, 0.0),
            "minimum_observed_fully_held_return_percent": min(
                (min_low / entry_fill - 1) * 100, 0.0
            ),
            "maximum_observed_fully_held_return_percent": max(
                (max_high / entry_fill - 1) * 100, 0.0
            ),
            **_occurrence_fields(
                window_data.index, max_high_position, entry_position,
                "observed_fully_held_mfe",
            ),
            **_occurrence_fields(
                window_data.index, min_low_position, entry_position,
                "observed_fully_held_mae",
            ),
        }
    )

    max_close = float(closes.max())
    min_close = float(closes.min())
    max_close_position = entry_position + int(_first_position(closes == max_close))
    min_close_position = entry_position + int(_first_position(closes == min_close))
    result.update(
        {
            "maximum_fully_held_close": max_close,
            "maximum_fully_held_close_return_percent": (max_close / entry_fill - 1) * 100,
            "minimum_fully_held_close": min_close,
            "minimum_fully_held_close_return_percent": (min_close / entry_fill - 1) * 100,
            **_occurrence_fields(
                window_data.index, max_close_position, entry_position, "maximum_fully_held_close"
            ),
            **_occurrence_fields(
                window_data.index, min_close_position, entry_position, "minimum_fully_held_close"
            ),
        }
    )

    above = closes > entry_fill
    below = closes < entry_fill
    equal = closes == entry_fill
    at_or_above = closes >= entry_fill
    result.update(
        _qualifying_close_fields(
            full_data=window_data, entry_position=entry_position, mask=above,
            prefix="first_fully_held_close_above_entry",
        )
    )
    result.update(
        _qualifying_close_fields(
            full_data=window_data, entry_position=entry_position, mask=below,
            prefix="first_fully_held_close_below_entry",
        )
    )
    result.update(
        _qualifying_close_fields(
            full_data=window_data, entry_position=entry_position, mask=equal,
            prefix="first_fully_held_close_equal_entry",
        )
    )
    result.update(
        _qualifying_close_fields(
            full_data=window_data, entry_position=entry_position, mask=at_or_above,
            prefix="first_fully_held_close_at_or_above_entry",
        )
    )
    breakeven_local = _first_position(at_or_above)
    result.update(
        {
            "price_breakeven_available": breakeven_local is not None,
            "price_breakeven_timestamp": (
                None if breakeven_local is None else pd.Timestamp(full_data.index[breakeven_local])
            ),
            "price_breakeven_offset_from_entry": breakeven_local,
            "bars_to_price_breakeven": None if breakeven_local is None else breakeven_local + 1,
            "fully_held_close_above_entry_count": int(above.sum()),
            "fully_held_close_below_entry_count": int(below.sum()),
            "fully_held_close_at_entry_count": int(equal.sum()),
            "fully_held_close_above_entry_frequency": float(above.mean()),
            "fully_held_close_below_entry_frequency": float(below.mean()),
            "fully_held_close_at_entry_frequency": float(equal.mean()),
        }
    )
    return result


def _validate_forward_population(
    trades: pd.DataFrame,
    *,
    expected_trade_count: int | None,
    expected_window_count: int | None,
    expected_force_close_count: int | None,
    controlled_tickers: Iterable[str] = CONTROLLED_TICKERS,
) -> pd.DataFrame:
    missing = REQUIRED_FORWARD_COLUMNS.difference(trades.columns)
    if missing:
        raise ValueError(f"Forward Return Statistics trades missing columns: {sorted(missing)}")
    output = trades.copy(deep=True)
    if output.empty:
        raise ValueError("Forward Return Statistics source has no completed trades.")
    if not output["model"].astype(str).eq(MODEL_BASELINE).all():
        raise ValueError("MFE/MAE attribution requires FIXED_BASELINE trades only.")
    if output["source_trade_row"].duplicated().any() or output["trade_id"].duplicated().any():
        raise ValueError("Forward Return Statistics trade identifiers are not unique.")
    for value in output["source_trade_row"].tolist():
        _validated_nonnegative_integer(value, "source_trade_row")
    tickers = set(output["ticker"].astype(str))
    if not tickers.issubset(set(controlled_tickers)):
        raise ValueError("Forward Return Statistics includes a ticker outside the controlled basket.")
    if expected_trade_count is not None and len(output) != expected_trade_count:
        raise ValueError(f"Expected {expected_trade_count} completed trades; found {len(output)}.")
    if expected_window_count is not None and output["window_id"].astype(str).nunique() != expected_window_count:
        raise ValueError(
            f"Expected {expected_window_count} reset-capital windows; "
            f"found {output['window_id'].astype(str).nunique()}."
        )
    force_count = int(output["exit_reason"].astype(str).eq("FORCE_CLOSE_END").sum())
    if expected_force_close_count is not None and force_count != expected_force_close_count:
        raise ValueError(
            f"Expected {expected_force_close_count} FORCE_CLOSE_END trades; found {force_count}."
        )
    return output


def enrich_holding_paths(
    *,
    trades: pd.DataFrame,
    data_by_ticker: Mapping[str, pd.DataFrame],
    windows: pd.DataFrame,
    mfe_mae_holding_path_stamp: str,
    slippage_bps: float = APPROVED_SLIPPAGE_BPS,
    expected_trade_count: int | None = None,
    expected_window_count: int | None = None,
    expected_force_close_count: int | None = None,
    controlled_tickers: Iterable[str] = CONTROLLED_TICKERS,
) -> pd.DataFrame:
    source = _validate_forward_population(
        trades,
        expected_trade_count=expected_trade_count,
        expected_window_count=expected_window_count,
        expected_force_close_count=expected_force_close_count,
        controlled_tickers=controlled_tickers,
    )
    if not isfinite(float(slippage_bps)) or float(slippage_bps) < 0:
        raise ValueError("slippage_bps must be finite and nonnegative.")
    window_by_id = _window_lookup(windows)
    normalized_data = {
        str(ticker).strip().upper(): _normalize_and_validate_market_frame(frame, str(ticker))
        for ticker, frame in data_by_ticker.items()
    }
    records: list[dict[str, Any]] = []
    for source_forward_trade_row, row in enumerate(source.to_dict("records")):
        ticker = str(row["ticker"]).strip().upper()
        window_id = str(row["window_id"])
        if ticker not in normalized_data:
            raise ValueError(f"Snapshot missing trade ticker: {ticker}")
        if window_id not in window_by_id:
            raise ValueError(f"Snapshot missing trade window: {window_id}")
        start, end = window_by_id[window_id]
        market = normalized_data[ticker]
        window_data = market.loc[(market.index >= start) & (market.index < end)]
        if window_data.empty:
            raise ValueError(f"No snapshot rows for {window_id}/{ticker}.")
        entry_timestamp = _normalized_timestamp(row["entry_timestamp"])
        if not start <= entry_timestamp < end:
            raise ValueError(f"Entry timestamp outside test window for {window_id}/{ticker}.")
        source_exit_timestamp = _normalized_timestamp(row["exit_timestamp"])
        if not start <= source_exit_timestamp < end:
            raise ValueError(
                f"Exit timestamp outside test window for {window_id}/{ticker}: "
                f"{source_exit_timestamp}."
            )
        fields = holding_path_fields_for_trade(
            window_data=window_data,
            trade=row,
            slippage_bps=slippage_bps,
        )
        records.append(
            {
                **row,
                "mfe_mae_holding_path_stamp": mfe_mae_holding_path_stamp,
                "source_forward_trade_row": source_forward_trade_row,
                "holding_path_trade_id": (
                    f"{row['forward_return_statistics_stamp']}:"
                    f"{_validated_nonnegative_integer(row['source_trade_row'], 'source_trade_row'):06d}"
                ),
                **fields,
            }
        )
    output = pd.DataFrame(records, columns=TRADE_COLUMNS)
    if output["source_trade_row"].tolist() != source["source_trade_row"].tolist():
        raise ValueError("Holding-path attribution did not preserve source-trade order.")
    if output["trade_id"].tolist() != source["trade_id"].tolist():
        raise ValueError("Holding-path attribution did not preserve stable trade identifiers.")
    return output


def _finite_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric.map(lambda value: pd.isna(value) or isfinite(float(value))))


def _metric_semantics(metric: str) -> str:
    if metric in MAGNITUDE_METRICS:
        return "NONNEGATIVE_EXCURSION_MAGNITUDE"
    if metric in NONNEGATIVE_FAVORABLE_MAXIMUM_METRICS:
        return "NONNEGATIVE_FAVORABLE_DIRECTIONAL_MAXIMUM"
    if metric in SIGNED_METRICS:
        return "SIGNED_DIRECTIONAL_VALUE"
    return "NONNEGATIVE_COUNT_OFFSET_OR_FREQUENCY"


def _statistic_record(
    frame: pd.DataFrame,
    *,
    metric: str,
    stamp: str,
    population: str,
    group_type: str,
    group_value: str,
) -> dict[str, Any]:
    values = _finite_numeric(frame[metric])
    valid = values.dropna()
    count = int(len(valid))
    negative = int((valid < 0).sum()) if metric in SIGNED_METRICS else 0
    zero = int((valid == 0).sum())
    positive = int((valid > 0).sum())

    def percentile(value: float) -> float | None:
        return float(valid.quantile(value, interpolation="linear")) if count else None

    return {
        "mfe_mae_holding_path_stamp": stamp,
        "population": population,
        "group_type": group_type,
        "group_value": str(group_value),
        "metric": metric,
        "metric_semantics": _metric_semantics(metric),
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
        "negative_count": negative if metric in SIGNED_METRICS else None,
        "negative_frequency": negative / count if count and metric in SIGNED_METRICS else None,
        "zero_count": zero,
        "zero_frequency": zero / count if count else 0.0,
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
        if group_column not in trades.columns:
            raise ValueError(f"Holding-path trades missing group column: {group_column}")
        values = list(group_values) if group_values is not None else sorted(
            trades[group_column].dropna().unique(), key=lambda item: str(item)
        )
        groups = [(str(value), trades.loc[trades[group_column] == value]) for value in values]
    records = [
        _statistic_record(
            group,
            metric=metric,
            stamp=stamp,
            population=population,
            group_type=group_type,
            group_value=group_value,
        )
        for metric in STATISTIC_METRICS
        for group_value, group in groups
    ]
    return pd.DataFrame(records, columns=STATISTICS_COLUMNS)


def build_aggregation_table(trades: pd.DataFrame, *, stamp: str) -> pd.DataFrame:
    groupings: tuple[tuple[str, str | None, Iterable[Any] | None], ...] = (
        ("ALL_TRADES", None, None),
        ("OUTCOME_CLASS", "outcome_class", OUTCOME_ORDER),
        ("ASSET_CLASS", "asset_class", ASSET_CLASS_ORDER),
        ("TICKER", "ticker", CONTROLLED_TICKERS),
        ("WINDOW_ID", "window_id", sorted(trades["window_id"].astype(str).unique())),
        ("ENTRY_YEAR", "entry_year", sorted(trades["entry_year"].dropna().unique())),
        ("ENTRY_GAP_BUCKET", "entry_gap_bucket", GAP_BUCKETS),
        ("EXIT_REASON", "exit_reason", None),
        ("EXIT_CATEGORY", "exit_category", None),
        ("EXIT_EXECUTION_PHASE", "exit_execution_phase", None),
        ("EXIT_BAR_TREATMENT", "exit_bar_treatment", None),
        (
            "FULL_HELD_BAR_AVAILABILITY",
            "full_held_bar_availability_reason",
            (FULL_HELD_AVAILABLE, FULL_HELD_UNAVAILABLE),
        ),
    )
    frames: list[pd.DataFrame] = []
    for population, population_frame in (
        (PRIMARY_POPULATION, trades),
        (SENSITIVITY_POPULATION, trades.loc[trades["exit_reason"] != "FORCE_CLOSE_END"]),
    ):
        for group_type, column, values in groupings:
            frames.append(
                build_statistics_table(
                    population_frame,
                    stamp=stamp,
                    population=population,
                    group_type=group_type,
                    group_column=column,
                    group_values=values,
                )
            )
    return pd.concat(frames, ignore_index=True)


def _population_summary(trades: pd.DataFrame) -> dict[str, Any]:
    sensitivity = trades.loc[trades["exit_reason"] != "FORCE_CLOSE_END"]
    return {
        "primary_trade_count": int(len(trades)),
        "window_count": int(trades["window_id"].astype(str).nunique()),
        "force_close_end_trade_count": int(trades["exit_reason"].eq("FORCE_CLOSE_END").sum()),
        "force_close_excluded_trade_count": int(len(sensitivity)),
        "exit_execution_phase_counts": {
            str(key): int(value)
            for key, value in trades["exit_execution_phase"].value_counts(sort=False).items()
        },
        "exit_bar_treatment_counts": {
            str(key): int(value)
            for key, value in trades["exit_bar_treatment"].value_counts(sort=False).items()
        },
        "full_held_bar_availability_counts": {
            str(key): int(value)
            for key, value in trades["full_held_bar_availability_reason"].value_counts(sort=False).items()
        },
    }


def _validate_official_distributions(summary: Mapping[str, Any]) -> None:
    phase_counts = summary.get("exit_execution_phase_counts")
    if phase_counts != OFFICIAL_EXIT_PHASE_COUNTS:
        raise ValueError(
            "Approved official exit-phase distribution mismatch: "
            f"expected {OFFICIAL_EXIT_PHASE_COUNTS}; found {phase_counts}."
        )
    availability_counts = summary.get("full_held_bar_availability_counts")
    if availability_counts != OFFICIAL_FULL_HELD_AVAILABILITY_COUNTS:
        raise ValueError(
            "Approved official full-held availability distribution mismatch: "
            f"expected {OFFICIAL_FULL_HELD_AVAILABILITY_COUNTS}; found {availability_counts}."
        )


def _validate_strict_official_summary(summary: Mapping[str, Any]) -> None:
    if summary.get("primary_trade_count") != EXPECTED_TRADE_COUNT:
        raise ValueError("Approved official primary population mismatch.")
    if summary.get("window_count") != EXPECTED_WINDOW_COUNT:
        raise ValueError("Approved official window count mismatch.")
    if summary.get("force_close_end_trade_count") != EXPECTED_FORCE_CLOSE_COUNT:
        raise ValueError("Approved official FORCE_CLOSE_END population mismatch.")
    if summary.get("force_close_excluded_trade_count") != EXPECTED_SENSITIVITY_COUNT:
        raise ValueError("Approved official sensitivity population mismatch.")
    _validate_official_distributions(summary)


def _field_definitions() -> dict[str, str]:
    definitions = {column: column.replace("_", " ") for column in TRADE_COLUMNS}
    definitions.update(
        {
            "full_held_ticker_bar_count": "Ticker-local bars held from that bar's Open through its Close.",
            "observed_fully_held_mfe_percent": "Nonnegative favorable High excursion magnitude over fully held bars.",
            "observed_fully_held_mae_percent": "Nonnegative adverse Low excursion magnitude over fully held bars.",
            "intrabar_order_ambiguity": "Daily OHLC path-order limitation; partial stop bars may contain post-exit action.",
            "price_breakeven_available": "Whether a fully held Close was at or above actual entry fill; fees are excluded.",
            "partial_exit_bar_high": "Ambiguous stop-bar High, excluded from all fully held extrema and Close metrics.",
            "partial_exit_bar_low": "Ambiguous stop-bar Low, excluded from all fully held extrema and Close metrics.",
        }
    )
    return definitions


def _formulas() -> dict[str, str]:
    return {
        "signed_return_percent": "(P / entry_fill_price - 1) * 100",
        "signed_amount": "(P - entry_fill_price) * quantity",
        "observed_fully_held_mfe_percent": "max((max(High[F]) / entry_fill_price - 1) * 100, 0)",
        "observed_fully_held_mfe_amount": "max((max(High[F]) - entry_fill_price) * quantity, 0)",
        "observed_fully_held_mae_percent": "max((1 - min(Low[F]) / entry_fill_price) * 100, 0)",
        "observed_fully_held_mae_amount": "max((entry_fill_price - min(Low[F])) * quantity, 0)",
        "minimum_observed_fully_held_return_percent": "min((min(Low[F]) / entry_fill_price - 1) * 100, 0)",
        "maximum_observed_fully_held_return_percent": "max((max(High[F]) / entry_fill_price - 1) * 100, 0)",
        "price_breakeven": "first fully held Close >= entry_fill_price; fees excluded",
    }


def build_json_payload(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": bundle["created_at"],
        "mfe_mae_holding_path_stamp": bundle["mfe_mae_holding_path_stamp"],
        "method": "Observational ticker-local fully-held MFE/MAE and Close paths; no engine replay.",
        "population_definition": {
            "model": MODEL_BASELINE,
            "primary": "All completed provenanced baseline trades, including FORCE_CLOSE_END.",
            "sensitivity": "Primary population excluding only FORCE_CLOSE_END.",
            "expected_trade_count": bundle["expected_trade_count"],
            "expected_window_count": bundle["expected_window_count"],
            "windows_are_separate_reset_capital_runs": True,
            "controlled_tickers": list(CONTROLLED_TICKERS),
        },
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
        "field_definitions": _field_definitions(),
        "formulas": _formulas(),
        "methodological_choices": {
            "fully_held_set": "Ticker-local bars held from Open through Close; exit-Open and partial stop bars excluded; force-close local Close included.",
            "entry_bar": "Included when held through its Close.",
            "extremum_ties": "First ticker-local occurrence.",
            "offsets": "Zero-based from entry row; completed ticker-bar numbers are one-based.",
            "close_comparison": "Exact deterministic floating-point comparison to stored entry_fill_price.",
            "standard_deviation": "population standard deviation, ddof=0",
            "percentiles": "linear interpolation",
            "source_order": "Preserved from provenanced Forward Return Statistics trades CSV.",
            "macd_role": "Preserved source diagnostic only; unused by this analyzer.",
        },
        "trade_schema": list(TRADE_COLUMNS),
        "statistics_schema": list(STATISTICS_COLUMNS),
        "summary": bundle["summary"],
        "aggregation_results": bundle["statistics"].to_dict("records"),
        "limitations": [
            "Daily OHLC cannot establish High/Low ordering.",
            "Partial intrabar stop-bar High, Low, and Close may contain post-exit price action and are excluded from fully held metrics.",
            "Observed excursion values are not guaranteed executable prices.",
            "Price breakeven uses entry fill only and excludes commissions and hypothetical exit costs.",
            "No full per-bar path table is produced in V1.",
            "The provenance file hashes every generated output except itself.",
        ],
    }


def _serialized_table(frame: pd.DataFrame) -> str:
    buffer = StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue()


def _serialized_json_payload(bundle: Mapping[str, Any]) -> str:
    return json.dumps(
        _json_safe(build_json_payload(bundle)), sort_keys=True, indent=2
    ) + "\n"


def build_mfe_mae_holding_path_attribution(
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
    slippage_bps: float = APPROVED_SLIPPAGE_BPS,
    expected_trade_count: int | None = None,
    expected_window_count: int | None = None,
    expected_force_close_count: int | None = None,
    controlled_tickers: Iterable[str] = CONTROLLED_TICKERS,
    mfe_mae_holding_path_stamp: str | None = None,
    source_files: Mapping[str, Path] | None = None,
    source_hashes: Mapping[str, str] | None = None,
    source_verification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    stamp = mfe_mae_holding_path_stamp or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    enriched = enrich_holding_paths(
        trades=trades,
        data_by_ticker=data_by_ticker,
        windows=windows,
        mfe_mae_holding_path_stamp=stamp,
        slippage_bps=slippage_bps,
        expected_trade_count=expected_trade_count,
        expected_window_count=expected_window_count,
        expected_force_close_count=expected_force_close_count,
        controlled_tickers=controlled_tickers,
    )
    summary = _population_summary(enriched)
    public_verification = (
        None if source_verification is None else dict(source_verification)
    )
    if public_verification is not None:
        public_verification["official_source"] = False
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "mfe_mae_holding_path_stamp": stamp,
        "forward_return_statistics_stamp": forward_return_statistics_stamp,
        "entry_statistics_stamp": entry_statistics_stamp,
        "timing_stamp": timing_stamp,
        "source_stamp": source_stamp,
        "snapshot_id": snapshot_id,
        "snapshot_fingerprint": snapshot_fingerprint,
        "expected_trade_count": expected_trade_count,
        "expected_window_count": expected_window_count,
        "trades": enriched,
        "statistics": build_aggregation_table(enriched, stamp=stamp),
        "summary": summary,
        "source_files": None if source_files is None else {str(k): Path(v) for k, v in source_files.items()},
        "source_hashes": None if source_hashes is None else dict(source_hashes),
        "source_verification": public_verification,
    }


def _resolved_expectations(expectations: Mapping[str, Any] | None) -> dict[str, Any]:
    if expectations is None:
        return dict(OFFICIAL_EXPECTATIONS)
    if not isinstance(expectations, Mapping) or set(expectations) != set(EXPECTATION_KEYS):
        actual = set(expectations) if isinstance(expectations, Mapping) else set()
        raise ValueError(
            "Alternate expectations must contain one complete expectation bundle; "
            f"missing={sorted(set(EXPECTATION_KEYS) - actual)}, "
            f"extra={sorted(actual - set(EXPECTATION_KEYS))}."
        )
    result = {key: expectations[key] for key in EXPECTATION_KEYS}
    for key in EXPECTATION_KEYS[:5]:
        if not isinstance(result[key], str) or not result[key]:
            raise ValueError(f"Alternate {key} must be a non-empty string.")
    for key in ("expected_trade_count", "expected_window_count", "expected_force_close_count"):
        if not isinstance(result[key], Integral) or int(result[key]) < 0:
            raise ValueError(f"Alternate {key} must be a nonnegative integer.")
        result[key] = int(result[key])
    tickers = tuple(str(value) for value in result["controlled_tickers"])
    if not tickers or len(tickers) != len(set(tickers)):
        raise ValueError("Alternate controlled_tickers must be a non-empty unique sequence.")
    result["controlled_tickers"] = tickers
    return result


def load_verified_source(
    *,
    snapshot_directory: Path,
    forward_return_statistics_directory: Path,
    forward_return_statistics_stamp: str,
    project_root: Path = Path("."),
    verify_code: bool = True,
    expectations: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and verify the complete approved Forward Return source chain."""
    active = _resolved_expectations(expectations)
    if forward_return_statistics_stamp != active["forward_return_statistics_stamp"]:
        raise ValueError(
            "MFE/MAE attribution requires Forward Return Statistics stamp "
            f"{active['forward_return_statistics_stamp']}; received {forward_return_statistics_stamp}."
        )
    snapshot_directory = Path(snapshot_directory)
    source_directory = Path(forward_return_statistics_directory)
    snapshot = _load_verified_snapshot(
        snapshot_directory, verify_code=verify_code, project_root=Path(project_root)
    )
    manifest = snapshot["manifest"]
    if manifest.get("snapshot_id") != active["snapshot_id"]:
        raise ValueError("Snapshot ID does not match the approved holding-path source.")
    if (
        active == _resolved_expectations(None)
        and manifest.get("fingerprint") != APPROVED_SNAPSHOT_FINGERPRINT
    ):
        raise ValueError("Snapshot fingerprint does not match the approved holding-path source.")
    if set(manifest.get("market_files", {})) != set(active["controlled_tickers"]):
        raise ValueError("Snapshot does not contain exactly the expected controlled basket.")
    config = snapshot["config"]
    if float(config.get("slippage_bps", float("nan"))) != APPROVED_SLIPPAGE_BPS:
        raise ValueError("Frozen snapshot slippage contradicts the approved 5 bps baseline.")
    if (
        float(config.get("stock_stop_loss_percent", float("nan"))) != APPROVED_STOP_PERCENT
        or float(config.get("crypto_stop_loss_percent", float("nan"))) != APPROVED_STOP_PERCENT
    ):
        raise ValueError("Frozen snapshot stop percentages contradict the approved 5% baseline.")

    provenance_path = source_directory / (
        f"portfolio_forward_return_statistics_provenance_{forward_return_statistics_stamp}.json"
    )
    provenance = _read_json(provenance_path)
    expected_lineage = {
        "forward_return_statistics_stamp": forward_return_statistics_stamp,
        "entry_statistics_stamp": active["entry_statistics_stamp"],
        "timing_stamp": active["timing_stamp"],
        "source_stop_walk_forward_stamp": active["source_stop_walk_forward_stamp"],
        "snapshot_id": active["snapshot_id"],
        "snapshot_fingerprint": manifest.get("fingerprint"),
    }
    for key, expected in expected_lineage.items():
        if provenance.get(key) != expected:
            raise ValueError(f"Forward Return Statistics provenance {key} mismatch.")
    declared_source_files = _verify_declared_source_files(
        provenance, project_root=Path(project_root)
    )
    result_files = _verify_provenance_result_files(
        directory=source_directory,
        provenance=provenance,
        prefix="portfolio_forward_return_statistics_",
        stamp=forward_return_statistics_stamp,
    )
    if not {"trades", "json"}.issubset(result_files):
        raise ValueError("Forward Return provenance is missing required trades or JSON results.")
    if active == _resolved_expectations(None) and set(result_files) != OFFICIAL_FORWARD_RESULT_KEYS:
        raise ValueError(
            "Forward Return provenance does not declare the exact official result filenames."
        )
    payload = _read_json(result_files["json"])
    if payload.get("forward_return_statistics_stamp") != forward_return_statistics_stamp:
        raise ValueError("Forward Return JSON stamp mismatch.")
    payload_lineage = payload.get("source_lineage")
    if not isinstance(payload_lineage, Mapping):
        raise ValueError("Forward Return JSON has no source_lineage object.")
    for key in (
        "entry_statistics_stamp", "timing_stamp", "source_stop_walk_forward_stamp",
        "snapshot_id", "snapshot_fingerprint",
    ):
        if payload_lineage.get(key) != expected_lineage[key]:
            raise ValueError(f"Forward Return JSON {key} mismatch.")

    trades = pd.read_csv(result_files["trades"])
    trades = _validate_forward_population(
        trades,
        expected_trade_count=active["expected_trade_count"],
        expected_window_count=active["expected_window_count"],
        expected_force_close_count=active["expected_force_close_count"],
        controlled_tickers=active["controlled_tickers"],
    )
    if not trades["forward_return_statistics_stamp"].astype(str).eq(forward_return_statistics_stamp).all():
        raise ValueError("Forward Return trades contain a mismatched result stamp.")
    lineage_columns = {
        "entry_statistics_stamp": active["entry_statistics_stamp"],
        "timing_stamp": active["timing_stamp"],
        "source_stop_walk_forward_stamp": active["source_stop_walk_forward_stamp"],
        "snapshot_id": active["snapshot_id"],
        "snapshot_fingerprint": str(manifest["fingerprint"]),
    }
    for column, expected in lineage_columns.items():
        if not trades[column].astype(str).eq(expected).all():
            raise ValueError(f"Forward Return trades {column} mismatch.")
    source_rows = [
        _validated_nonnegative_integer(value, "source_trade_row")
        for value in trades["source_trade_row"].tolist()
    ]
    if source_rows != sorted(source_rows):
        raise ValueError("Forward Return source order is not deterministic source_trade_row order.")
    expected_ids = [f"{active['timing_stamp']}:{row:06d}" for row in source_rows]
    if trades["trade_id"].astype(str).tolist() != expected_ids:
        raise ValueError("Forward Return stable trade identifiers do not match approved lineage.")
    if set(trades["ticker"].astype(str)) != set(active["controlled_tickers"]):
        raise ValueError("Forward Return trades do not cover exactly the controlled basket.")
    if set(trades["window_id"].astype(str)) != set(snapshot["windows"]["window_id"].astype(str)):
        raise ValueError("Forward Return windows do not match frozen snapshot windows.")

    source_files: dict[str, Path] = {
        "snapshot_manifest": snapshot_directory / "manifest.json",
        "snapshot_config": snapshot_directory / manifest["config"]["path"],
        "snapshot_windows": snapshot_directory / manifest["windows"]["path"],
        "forward_return_statistics_provenance": provenance_path,
        "mfe_mae_holding_path_code": Path(__file__).resolve(),
    }
    for ticker, metadata in manifest["market_files"].items():
        source_files[f"snapshot_market:{ticker}"] = snapshot_directory / metadata["path"]
    source_files.update(
        {f"forward_return_statistics_result:{name}": path for name, path in result_files.items()}
    )
    source_files.update(
        {f"forward_return_statistics_source:{name}": path for name, path in declared_source_files.items()}
    )
    source_hashes = {label: sha256_file(path) for label, path in source_files.items()}
    official_source = (
        active == _resolved_expectations(None)
        and bool(verify_code)
        and manifest.get("fingerprint") == APPROVED_SNAPSHOT_FINGERPRINT
    )
    verification = {
        "verified": True,
        "code_hash_verification": bool(verify_code),
        "official_source": official_source,
        "forward_provenance_verified": True,
        "forward_result_hashes_verified": True,
        "forward_source_hashes_verified": True,
        "snapshot_hashes_verified": bool(snapshot["snapshot_hashes_verified"]),
        "snapshot_code_hashes_verified": bool(snapshot["snapshot_code_hashes_verified"]),
        **expected_lineage,
    }
    source = {
        "snapshot": snapshot,
        "trades": trades,
        "forward_return_statistics_stamp": forward_return_statistics_stamp,
        "entry_statistics_stamp": active["entry_statistics_stamp"],
        "timing_stamp": active["timing_stamp"],
        "source_stamp": active["source_stop_walk_forward_stamp"],
        "source_files": source_files,
        "source_hashes": source_hashes,
        "source_verification": verification,
        "expected_trade_count": active["expected_trade_count"],
        "expected_window_count": active["expected_window_count"],
        "expected_force_close_count": active["expected_force_close_count"],
        "controlled_tickers": active["controlled_tickers"],
        "snapshot_directory": snapshot_directory.resolve(),
        "forward_return_statistics_directory": source_directory.resolve(),
        "project_root": Path(project_root).resolve(),
    }
    if official_source:
        _STRICT_OFFICIAL_SOURCES.append(source)
    return source


def _build_bundle_from_source(
    source: Mapping[str, Any], *, mfe_mae_holding_path_stamp: str | None = None
) -> dict[str, Any]:
    snapshot = source["snapshot"]
    manifest = snapshot["manifest"]
    return build_mfe_mae_holding_path_attribution(
        trades=source["trades"],
        data_by_ticker=snapshot["data_by_ticker"],
        windows=snapshot["windows"],
        forward_return_statistics_stamp=str(source["forward_return_statistics_stamp"]),
        entry_statistics_stamp=str(source["entry_statistics_stamp"]),
        timing_stamp=str(source["timing_stamp"]),
        source_stamp=str(source["source_stamp"]),
        snapshot_id=str(manifest["snapshot_id"]),
        snapshot_fingerprint=str(manifest["fingerprint"]),
        slippage_bps=float(snapshot["config"]["slippage_bps"]),
        expected_trade_count=int(source["expected_trade_count"]),
        expected_window_count=int(source["expected_window_count"]),
        expected_force_close_count=int(source["expected_force_close_count"]),
        controlled_tickers=source["controlled_tickers"],
        mfe_mae_holding_path_stamp=mfe_mae_holding_path_stamp,
        source_files=source["source_files"],
        source_hashes=source["source_hashes"],
        source_verification=source["source_verification"],
    )


def run_mfe_mae_holding_path_attribution(source: Mapping[str, Any]) -> dict[str, Any]:
    bundle = _build_bundle_from_source(source)
    if any(candidate is source for candidate in _STRICT_OFFICIAL_SOURCES):
        bundle["source_verification"] = dict(source["source_verification"])
        _validate_strict_official_summary(bundle["summary"])
        _STRICT_OFFICIAL_BUNDLES.append(
            _StrictOfficialBundleRegistration(
                bundle=bundle,
                source=source,
                stamp=str(bundle["mfe_mae_holding_path_stamp"]),
                created_at=str(bundle["created_at"]),
                trades_csv=_serialized_table(bundle["trades"]),
                statistics_csv=_serialized_table(bundle["statistics"]),
                json_payload=_serialized_json_payload(bundle),
            )
        )
    return bundle


def _strict_registration_for_bundle(
    bundle: Mapping[str, Any],
) -> _StrictOfficialBundleRegistration:
    for registration in _STRICT_OFFICIAL_BUNDLES:
        if registration.bundle is bundle:
            return registration
    raise ValueError(
        "Cannot save: bundle was not returned by the strict official loader and analysis path."
    )


def _validate_saveable_provenance(bundle: Mapping[str, Any]) -> None:
    source_files = bundle.get("source_files")
    source_hashes = bundle.get("source_hashes")
    verification = bundle.get("source_verification")
    if not isinstance(source_files, Mapping) or not source_files:
        raise ValueError("Cannot save: complete consumed source metadata is missing.")
    if not isinstance(source_hashes, Mapping) or set(source_files) != set(source_hashes):
        raise ValueError("Cannot save: every consumed source requires exactly one recorded hash.")
    for label, path_value in source_files.items():
        if not isinstance(source_hashes.get(label), str) or not source_hashes[label]:
            raise ValueError(f"Cannot save: recorded source hash is missing for {label}.")
        if not Path(path_value).is_file():
            raise ValueError(f"Cannot save: consumed source path is missing for {label}: {path_value}")
    registration = _strict_registration_for_bundle(bundle)
    registered_source = registration.source
    if not isinstance(verification, Mapping):
        raise ValueError("Cannot save: verified source lineage metadata is missing.")
    required_verification_flags = (
        "verified", "code_hash_verification", "official_source",
        "forward_provenance_verified", "forward_result_hashes_verified",
        "forward_source_hashes_verified", "snapshot_hashes_verified",
        "snapshot_code_hashes_verified",
    )
    if any(verification.get(field) is not True for field in required_verification_flags):
        raise ValueError("Cannot save: source verification is incomplete or weakened.")
    verified_fields = {
        "forward_return_statistics_stamp": APPROVED_FORWARD_RETURN_STATISTICS_STAMP,
        "entry_statistics_stamp": APPROVED_ENTRY_STATISTICS_STAMP,
        "timing_stamp": APPROVED_TIMING_STAMP,
        "source_stop_walk_forward_stamp": APPROVED_SOURCE_STAMP,
        "snapshot_id": APPROVED_SNAPSHOT_ID,
        "snapshot_fingerprint": APPROVED_SNAPSHOT_FINGERPRINT,
    }
    for field, expected in verified_fields.items():
        bundle_field = "source_stamp" if field == "source_stop_walk_forward_stamp" else field
        if bundle.get(bundle_field) != expected:
            raise ValueError(f"Cannot save: result does not use the approved official {field}.")
        if verification.get(field) != expected:
            raise ValueError(f"Cannot save: result lineage disagrees with verified source {field}.")
    if (
        bundle.get("expected_trade_count") != EXPECTED_TRADE_COUNT
        or bundle.get("expected_window_count") != EXPECTED_WINDOW_COUNT
    ):
        raise ValueError("Cannot save: result expectations are not the approved official counts.")
    trades = bundle.get("trades")
    if not isinstance(trades, pd.DataFrame):
        raise ValueError("Cannot save: trade-level result is missing.")
    _validate_forward_population(
        trades,
        expected_trade_count=EXPECTED_TRADE_COUNT,
        expected_window_count=EXPECTED_WINDOW_COUNT,
        expected_force_close_count=EXPECTED_FORCE_CLOSE_COUNT,
        controlled_tickers=CONTROLLED_TICKERS,
    )
    if set(trades["ticker"].astype(str)) != set(CONTROLLED_TICKERS):
        raise ValueError("Cannot save: result does not cover the exact official ticker basket.")
    if not trades["model"].astype(str).eq(MODEL_BASELINE).all():
        raise ValueError("Cannot save: result is not FIXED_BASELINE-only.")
    summary = bundle.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("Cannot save: result summary is missing.")
    if (
        summary.get("primary_trade_count") != EXPECTED_TRADE_COUNT
        or summary.get("window_count") != EXPECTED_WINDOW_COUNT
        or summary.get("force_close_end_trade_count") != EXPECTED_FORCE_CLOSE_COUNT
        or summary.get("force_close_excluded_trade_count") != EXPECTED_SENSITIVITY_COUNT
    ):
        raise ValueError("Cannot save: official population reconciliation failed.")
    _validate_strict_official_summary(summary)
    if set(source_files) != set(registered_source["source_files"]):
        raise ValueError("Cannot save: consumed-source mapping differs from the strict official load.")
    for label, expected_path in registered_source["source_files"].items():
        if Path(source_files[label]).resolve() != Path(expected_path).resolve():
            raise ValueError(f"Cannot save: consumed-source path mismatch for {label}.")
        if source_hashes.get(label) != registered_source["source_hashes"].get(label):
            raise ValueError(f"Cannot save: consumed-source hash mismatch for {label}.")

    fresh_source = load_verified_source(
        snapshot_directory=Path(registered_source["snapshot_directory"]),
        forward_return_statistics_directory=Path(
            registered_source["forward_return_statistics_directory"]
        ),
        forward_return_statistics_stamp=APPROVED_FORWARD_RETURN_STATISTICS_STAMP,
        project_root=Path(registered_source["project_root"]),
        verify_code=True,
    )
    if set(fresh_source["source_files"]) != set(source_files):
        raise ValueError("Cannot save: canonical source mapping changed before saving.")
    for label, path_value in source_files.items():
        path = Path(path_value)
        expected_hash = source_hashes.get(label)
        if not isinstance(expected_hash, str) or not expected_hash:
            raise ValueError(f"Cannot save: recorded source hash is missing for {label}.")
        if not path.is_file():
            raise ValueError(f"Cannot save: consumed source path is missing for {label}: {path}")
        if sha256_file(path) != expected_hash:
            raise ValueError(f"Source changed before save: {path}")

    canonical_bundle = _build_bundle_from_source(
        fresh_source, mfe_mae_holding_path_stamp=registration.stamp
    )
    canonical_bundle["created_at"] = registration.created_at
    canonical_bundle["source_files"] = dict(registered_source["source_files"])
    canonical_bundle["source_hashes"] = dict(registered_source["source_hashes"])
    canonical_bundle["source_verification"] = dict(fresh_source["source_verification"])
    _validate_strict_official_summary(canonical_bundle["summary"])

    canonical_trades_csv = _serialized_table(canonical_bundle["trades"])
    canonical_statistics_csv = _serialized_table(canonical_bundle["statistics"])
    canonical_json_payload = _serialized_json_payload(canonical_bundle)
    sealed_outputs = (
        ("trade table", canonical_trades_csv, registration.trades_csv),
        ("statistics table", canonical_statistics_csv, registration.statistics_csv),
        ("JSON payload", canonical_json_payload, registration.json_payload),
    )
    for label, recomputed, sealed in sealed_outputs:
        if recomputed != sealed:
            raise ValueError(
                f"Cannot save: freshly recomputed official {label} differs from "
                "the strict analysis result."
            )

    statistics = bundle.get("statistics")
    if not isinstance(statistics, pd.DataFrame):
        raise ValueError("Cannot save: statistics result is missing.")
    caller_outputs = (
        ("trade table", _serialized_table(trades), canonical_trades_csv),
        ("statistics table", _serialized_table(statistics), canonical_statistics_csv),
        ("JSON payload", _serialized_json_payload(bundle), canonical_json_payload),
    )
    for label, supplied, recomputed in caller_outputs:
        if supplied != recomputed:
            raise ValueError(
                f"Cannot save: caller {label} differs from the freshly recomputed "
                "official result."
            )


def save_mfe_mae_holding_path_attribution(
    bundle: Mapping[str, Any], output_directory: Path = DEFAULT_OUTPUT_DIRECTORY
) -> dict[str, Path]:
    output_directory = Path(output_directory)
    stamp = str(bundle["mfe_mae_holding_path_stamp"])
    paths = {
        "trades": output_directory / f"portfolio_mfe_mae_holding_path_trades_{stamp}.csv",
        "statistics": output_directory / f"portfolio_mfe_mae_holding_path_statistics_{stamp}.csv",
        "json": output_directory / f"portfolio_mfe_mae_holding_path_{stamp}.json",
        "provenance": output_directory / f"portfolio_mfe_mae_holding_path_provenance_{stamp}.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(existing[0])
    _validate_saveable_provenance(bundle)
    output_directory.mkdir(parents=True, exist_ok=True)
    bundle["trades"].to_csv(paths["trades"], index=False, lineterminator="\n")
    bundle["statistics"].to_csv(paths["statistics"], index=False, lineterminator="\n")
    paths["json"].write_text(_serialized_json_payload(bundle), encoding="utf-8")
    result_files = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in paths.items()
        if name != "provenance"
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "mfe_mae_holding_path_stamp": stamp,
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
    mode = "NO-SAVE ANALYSIS" if no_save else "SAVING ANALYSIS"
    print(f"PORTFOLIO MFE/MAE AND HOLDING-PATH ATTRIBUTION — OBSERVATIONAL {mode}")
    print("Provenance validation: PASS")
    print(f"Forward Return Statistics stamp: {bundle['forward_return_statistics_stamp']}")
    print(f"Entry Statistics stamp: {bundle['entry_statistics_stamp']}")
    print(f"Timing stamp: {bundle['timing_stamp']}")
    print(f"Snapshot: {bundle['snapshot_id']}")
    print(
        f"Primary population: {summary['primary_trade_count']} completed FIXED_BASELINE trades "
        f"across {summary['window_count']} reset-capital windows"
    )
    print(
        f"FORCE_CLOSE_END sensitivity: excluded={summary['force_close_end_trade_count']} "
        f"remaining={summary['force_close_excluded_trade_count']}"
    )
    print(f"Exit execution phases: {summary['exit_execution_phase_counts']}")
    print(f"Full-held-bar availability: {summary['full_held_bar_availability_counts']}")
    if no_save:
        print("Output artifacts saved: 0 (--no-save)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Observational MFE/MAE and holding paths for provenanced portfolio trades"
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
    bundle = run_mfe_mae_holding_path_attribution(source)
    if arguments.no_save:
        _print_summary(bundle, no_save=True)
        return
    _print_summary(bundle, no_save=False)
    for name, path in save_mfe_mae_holding_path_attribution(
        bundle, arguments.output_directory
    ).items():
        print(name, path.resolve())


if __name__ == "__main__":
    main()
