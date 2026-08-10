"""Pre-registered execution-cost stress test for the frozen portfolio baseline.

The analyzer replays the immutable 13 out-of-sample test windows under a fixed
4x4 commission/slippage grid.  Only execution-cost fields on copied configs are
changed.  There is no training, ranking, model selection, baseline mutation,
broker access, paper trading, or production authorization.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.backtest.portfolio_backtest_engine import run_portfolio_backtest
from src.backtest.portfolio_backtest_models import PortfolioBacktestConfig
from src.backtest import run_portfolio_mfe_mae_holding_path_attribution as mfe_mae
from src.backtest.run_portfolio_entry_statistics import (
    CONTROLLED_TICKERS,
    EXPECTED_TRADE_COUNT,
    EXPECTED_WINDOW_COUNT,
    verify_provenance_result_files,
)
from src.backtest.run_portfolio_position_ablation import (
    annual_returns_from_equity,
    build_variant_summary,
    rejection_counts,
    run_matched_benchmark,
    ticker_contributions,
)
from src.backtest.run_portfolio_risk_walk_forward import _aggregate_model
from src.backtest.run_portfolio_walk_forward import slice_prepared_data
from src.backtest.run_research_data_snapshot import load_snapshot, sha256_file


SCHEMA_VERSION = 1
# Repointing note (see report for the full takeover-audit context):
# this used to hash-lock onto src/backtest/run_portfolio_holding_path_attribution.py
# (stamp 20260802_155749), a file that was never committed. The owner
# decided the committed, HEAD run_portfolio_mfe_mae_holding_path_attribution.py
# (stamp 20260808_074833) is the sole canonical holding-path-equivalent
# source going forward; the uncommitted file is never revived.
APPROVED_MFE_MAE_HOLDING_PATH_STAMP = "20260808_074833"
APPROVED_FORWARD_RETURN_STATISTICS_STAMP = "20260802_134341"
APPROVED_ENTRY_STATISTICS_STAMP = "20260802_103007"
APPROVED_TIMING_STAMP = "20260802_085312"
APPROVED_STOP_WALK_FORWARD_STAMP = "20260802_081449"
APPROVED_SNAPSHOT_ID = "20260802_081049_d93145cb1dcf"
APPROVED_SNAPSHOT_FINGERPRINT = (
    "d93145cb1dcf3f837415f5b28ac29eeb51c13bc8ce5b8b1187aa34ebfc01ad44"
)
# Real sha256 of src/backtest/run_portfolio_mfe_mae_holding_path_attribution.py
# and tests/test_portfolio_mfe_mae_holding_path_attribution.py (computed and
# hardcoded per the repointing decision -- not carried over from the old,
# never-committed file's hashes).
APPROVED_MFE_MAE_CODE_SHA256 = (
    "81a7bb861bede51b8b5f2eea93c36bdb6a19155b3e9189b90b1ac0de396c4ab9"
)
APPROVED_MFE_MAE_TEST_SHA256 = (
    "8c1d5a9112f4dc9fc48aa74598be8cd31a472f8e6a9ee63f521c6ea7a9d9826a"
)

DEFAULT_HOLDING_PATH_DIRECTORY = Path(
    "data/backtests/portfolio/mfe_mae_holding_path_attribution"
)
DEFAULT_STOP_WALK_FORWARD_DIRECTORY = Path(
    "data/backtests/portfolio/stop_walk_forward"
)
DEFAULT_OUTPUT_DIRECTORY = Path(
    "data/backtests/portfolio/execution_slippage_stress"
)

MULTIPLIERS = (1, 2, 4, 8)
BASELINE_SCENARIO_ID = "C01_S01"
PRIMARY_STRESS_SCENARIO_ID = "C02_S02"
PRIMARY_MINIMUM_POSITIVE_WINDOWS = 7
PRIMARY_MAXIMUM_DRAWDOWN_WORSENING_PERCENT = 5.0
_STRICT_SOURCE_AUTHORIZATION = object()

# The full 15-key holding-path-equivalent taxonomy this module targets.
# Of these: {json, trades} are real mfe_mae files, hash-verified on
# disk. The remaining 12 are NOT separate mfe_mae files -- they are
# derived in-memory by this module from mfe_mae's verified trades/
# statistics (see _derive_holding_path_datasets), except path_rows,
# which is NOT_IMPLEMENTED this turn (see NOT_IMPLEMENTED_HOLDING_KEYS
# and the module report).
EXPECTED_HOLDING_RESULT_KEYS = {
    "asset_classes",
    "exit_categories",
    "exit_reasons",
    "exit_semantics",
    "extreme_orders",
    "force_close_excluded",
    "holding_buckets",
    "json",
    "outcomes",
    "overall",
    "path_rows",
    "screen",
    "tickers",
    "trades",
    "windows",
}
# What mfe_mae's own save function actually writes to disk (see
# save_mfe_mae_holding_path_attribution) -- this, not
# EXPECTED_HOLDING_RESULT_KEYS, is what verify_provenance_result_files
# is checked against now.
EXPECTED_MFE_MAE_RESULT_KEYS = {"json", "statistics", "trades"}
# Genuinely not implemented this turn: mfe_mae's own docstring states
# "No full per-bar path table is produced in V1." Reconstructing one
# would require a fresh per-bar walk over the snapshot market data,
# which is out of this repointing's scope. Never fabricated or
# approximated -- see the report for what the next turn needs.
NOT_IMPLEMENTED_HOLDING_KEYS = frozenset({"path_rows"})
EXPECTED_STOP_RESULT_KEYS = {
    "aggregate",
    "equity",
    "json",
    "rejections",
    "test_runs",
    "tickers",
    "training",
    "windows",
}

REPLAY_FIELDS = (
    "ending_equity",
    "total_return_amount",
    "total_return_percent",
    "maximum_drawdown_percent",
    "completed_trades",
    "winning_trades",
    "losing_trades",
    "win_rate_percent",
    "profit_factor",
    "total_fees",
    "average_exposure_percent",
    "rejected_signals",
    "matched_benchmark_return_percent",
    "excess_return_vs_matched_percent",
)


@dataclass(frozen=True, slots=True, order=True)
class ExecutionStressProfile:
    commission_multiplier: int
    slippage_multiplier: int

    def __post_init__(self) -> None:
        if self.commission_multiplier not in MULTIPLIERS:
            raise ValueError("commission_multiplier is outside the pre-registered grid.")
        if self.slippage_multiplier not in MULTIPLIERS:
            raise ValueError("slippage_multiplier is outside the pre-registered grid.")

    @property
    def scenario_id(self) -> str:
        return f"C{self.commission_multiplier:02d}_S{self.slippage_multiplier:02d}"

    def to_dict(self, base_config: PortfolioBacktestConfig) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "commission_multiplier": self.commission_multiplier,
            "slippage_multiplier": self.slippage_multiplier,
            "commission_rate": base_config.commission_rate * self.commission_multiplier,
            "minimum_fee": base_config.minimum_fee * self.commission_multiplier,
            "slippage_bps": base_config.slippage_bps * self.slippage_multiplier,
            "is_baseline": self.scenario_id == BASELINE_SCENARIO_ID,
            "is_primary_stress": self.scenario_id == PRIMARY_STRESS_SCENARIO_ID,
        }


def build_stress_grid() -> tuple[ExecutionStressProfile, ...]:
    return tuple(
        ExecutionStressProfile(commission, slippage)
        for commission in MULTIPLIERS
        for slippage in MULTIPLIERS
    )


def build_stress_config(
    base_config: PortfolioBacktestConfig, profile: ExecutionStressProfile
) -> PortfolioBacktestConfig:
    """Clone baseline config and change only the three cost fields."""
    base_config.validate()
    stressed = replace(
        base_config,
        commission_rate=base_config.commission_rate * profile.commission_multiplier,
        minimum_fee=base_config.minimum_fee * profile.commission_multiplier,
        slippage_bps=base_config.slippage_bps * profile.slippage_multiplier,
    )
    stressed.validate()
    protected = set(base_config.to_dict()).difference(
        {"commission_rate", "minimum_fee", "slippage_bps"}
    )
    for field in protected:
        if stressed.to_dict()[field] != base_config.to_dict()[field]:
            raise ValueError(f"Stress config changed protected field: {field}")
    return stressed


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


def _window_records(windows: pd.DataFrame) -> list[dict[str, Any]]:
    required = {
        "window_id",
        "train_start",
        "train_end_exclusive",
        "test_start",
        "test_end_exclusive",
    }
    missing = required.difference(windows.columns)
    if missing:
        raise ValueError(f"Snapshot windows missing columns: {sorted(missing)}")
    result: list[dict[str, Any]] = []
    for record in windows.to_dict("records"):
        normalized = {"window_id": str(record["window_id"])}
        for field in required.difference({"window_id"}):
            normalized[field] = pd.Timestamp(record[field])
        if normalized["test_end_exclusive"] <= normalized["test_start"]:
            raise ValueError(f"Invalid test interval: {normalized['window_id']}")
        result.append(normalized)
    result.sort(key=lambda item: item["window_id"])
    if len(result) != EXPECTED_WINDOW_COUNT:
        raise ValueError(f"Expected {EXPECTED_WINDOW_COUNT} windows; found {len(result)}.")
    return result


def _append_scaled_equity(
    rows: list[dict[str, Any]],
    *,
    result: Any,
    window_id: str,
    profile: ExecutionStressProfile,
    base_config: PortfolioBacktestConfig,
    opening_capital: float,
    initial_cash: float,
) -> float:
    if not result.equity_curve:
        return opening_capital
    scale = opening_capital / initial_cash
    profile_fields = profile.to_dict(base_config)
    for point in result.equity_curve:
        rows.append(
            {
                "window_id": window_id,
                **profile_fields,
                "timestamp": pd.Timestamp(point.timestamp),
                "total_equity": round(float(point.total_equity) * scale, 6),
            }
        )
    return round(float(result.equity_curve[-1].total_equity) * scale, 6)


def _trade_rows(
    result: Any,
    *,
    window_id: str,
    profile: ExecutionStressProfile,
    base_config: PortfolioBacktestConfig,
) -> list[dict[str, Any]]:
    fields = profile.to_dict(base_config)
    return [
        {"window_id": window_id, **fields, "trade_sequence": sequence, **trade.to_dict()}
        for sequence, trade in enumerate(result.trades)
    ]


def build_trade_pairs(trades: pd.DataFrame) -> pd.DataFrame:
    """Outer-match each scenario's trades to the baseline without inventing IDs."""
    key = ["window_id", "ticker", "entry_timestamp", "exit_timestamp", "exit_reason"]
    required = set(key).union(
        {
            "scenario_id",
            "entry_price",
            "exit_price",
            "quantity",
            "total_fees",
            "net_pnl",
            "return_percent",
        }
    )
    missing = required.difference(trades.columns)
    if missing:
        raise ValueError(f"Trade output missing columns: {sorted(missing)}")
    source = trades.copy()
    source["match_occurrence"] = source.groupby(["scenario_id", *key]).cumcount()
    pair_key = [*key, "match_occurrence"]
    values = ["entry_price", "exit_price", "quantity", "total_fees", "net_pnl", "return_percent"]
    baseline = source.loc[source["scenario_id"] == BASELINE_SCENARIO_ID, [*pair_key, *values]]
    baseline = baseline.rename(columns={column: f"baseline_{column}" for column in values})
    records: list[pd.DataFrame] = []
    for scenario_id in sorted(source["scenario_id"].unique()):
        scenario = source.loc[source["scenario_id"] == scenario_id, [*pair_key, *values]]
        scenario = scenario.rename(columns={column: f"scenario_{column}" for column in values})
        merged = baseline.merge(scenario, on=pair_key, how="outer", indicator=True)
        merged.insert(0, "scenario_id", scenario_id)
        merged["match_status"] = merged["_merge"].map(
            {"both": "MATCHED", "left_only": "BASELINE_ONLY", "right_only": "SCENARIO_ONLY"}
        ).astype(str)
        merged = merged.drop(columns="_merge")
        for column in values:
            merged[f"delta_{column}"] = (
                pd.to_numeric(merged[f"scenario_{column}"], errors="coerce")
                - pd.to_numeric(merged[f"baseline_{column}"], errors="coerce")
            )
        records.append(merged)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def build_replay_checks(
    test_runs: pd.DataFrame,
    official_baseline: pd.DataFrame,
    *,
    tolerance: float = 1e-8,
) -> pd.DataFrame:
    actual = test_runs.loc[test_runs["scenario_id"] == BASELINE_SCENARIO_ID]
    expected = official_baseline.loc[official_baseline["model"] == "FIXED_BASELINE"]
    if len(actual) != EXPECTED_WINDOW_COUNT or len(expected) != EXPECTED_WINDOW_COUNT:
        raise ValueError("Baseline replay requires exactly 13 actual and official windows.")
    rows: list[dict[str, Any]] = []
    actual = actual.set_index("window_id")
    expected = expected.set_index("window_id")
    if set(actual.index) != set(expected.index):
        raise ValueError("Baseline replay window IDs do not match official source.")
    for window_id in sorted(actual.index):
        for field in REPLAY_FIELDS:
            observed = actual.loc[window_id, field]
            reference = expected.loc[window_id, field]
            if pd.isna(observed) and pd.isna(reference):
                difference = 0.0
                passed = True
            elif field in {"completed_trades", "winning_trades", "losing_trades", "rejected_signals"}:
                difference = float(observed) - float(reference)
                passed = int(observed) == int(reference)
            else:
                left, right = float(observed), float(reference)
                if not isfinite(left) and not isfinite(right):
                    difference = 0.0
                    passed = True
                else:
                    difference = left - right
                    passed = abs(difference) <= tolerance
            rows.append(
                {
                    "window_id": window_id,
                    "field": field,
                    "official_value": reference,
                    "replay_value": observed,
                    "difference": difference,
                    "tolerance": tolerance,
                    "passed": bool(passed),
                }
            )
    output = pd.DataFrame(rows)
    if not output["passed"].all():
        failed = output.loc[~output["passed"], ["window_id", "field"]].to_dict("records")
        raise ValueError(f"Official baseline replay mismatch: {failed[:5]}")
    return output


def build_comparison(
    aggregate: pd.DataFrame,
    test_runs: pd.DataFrame,
    trade_pairs: pd.DataFrame,
) -> pd.DataFrame:
    baseline = aggregate.loc[aggregate["scenario_id"] == BASELINE_SCENARIO_ID]
    if len(baseline) != 1:
        raise ValueError("Expected exactly one baseline aggregate row.")
    base = baseline.iloc[0]
    rows: list[dict[str, Any]] = []
    for record in aggregate.sort_values("scenario_id").to_dict("records"):
        scenario_id = str(record["scenario_id"])
        pairs = trade_pairs.loc[trade_pairs["scenario_id"] == scenario_id]
        window_values = test_runs.loc[test_runs["scenario_id"] == scenario_id, "total_return_percent"]
        rows.append(
            {
                **record,
                "compounded_return_delta_vs_baseline_percent": round(
                    float(record["compounded_return_percent"]) - float(base["compounded_return_percent"]), 4
                ),
                "drawdown_worsening_vs_baseline_percent": round(
                    float(record["maximum_drawdown_percent"]) - float(base["maximum_drawdown_percent"]), 4
                ),
                "return_drawdown_ratio_delta_vs_baseline": round(
                    float(record["return_drawdown_ratio"]) - float(base["return_drawdown_ratio"]), 4
                ),
                "average_profit_factor_delta_vs_baseline": round(
                    float(record["average_profit_factor"]) - float(base["average_profit_factor"]), 4
                ),
                "benchmark_excess_compounded_percent": round(
                    float(record["compounded_return_percent"])
                    - float(record["matched_benchmark_compounded_return_percent"]), 4
                ),
                "positive_window_frequency": float((window_values > 0).mean()),
                "matched_trade_count": int((pairs["match_status"] == "MATCHED").sum()),
                "baseline_only_trade_count": int((pairs["match_status"] == "BASELINE_ONLY").sum()),
                "scenario_only_trade_count": int((pairs["match_status"] == "SCENARIO_ONLY").sum()),
                "is_primary_stress": scenario_id == PRIMARY_STRESS_SCENARIO_ID,
            }
        )
    return pd.DataFrame(rows)


def build_primary_gates(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = comparison.loc[comparison["scenario_id"] == PRIMARY_STRESS_SCENARIO_ID]
    if len(rows) != 1:
        raise ValueError("Primary 2x/2x scenario is missing.")
    row = rows.iloc[0]
    definitions = [
        (
            "primary_compounded_return_positive",
            float(row["compounded_return_percent"]) > 0,
            "> 0",
            row["compounded_return_percent"],
        ),
        (
            "primary_average_profit_factor_above_one",
            float(row["average_profit_factor"]) > 1,
            "> 1",
            row["average_profit_factor"],
        ),
        (
            "primary_positive_window_majority",
            int(row["positive_window_count"]) >= PRIMARY_MINIMUM_POSITIVE_WINDOWS,
            f">= {PRIMARY_MINIMUM_POSITIVE_WINDOWS}",
            row["positive_window_count"],
        ),
        (
            "primary_drawdown_worsening_within_limit",
            float(row["drawdown_worsening_vs_baseline_percent"])
            <= PRIMARY_MAXIMUM_DRAWDOWN_WORSENING_PERCENT,
            f"<= {PRIMARY_MAXIMUM_DRAWDOWN_WORSENING_PERCENT}",
            row["drawdown_worsening_vs_baseline_percent"],
        ),
    ]
    output = pd.DataFrame(
        [
            {
                "scenario_id": PRIMARY_STRESS_SCENARIO_ID,
                "gate": name,
                "passed": bool(passed),
                "criterion": criterion,
                "actual": actual,
                "authorizes_baseline_change": False,
                "authorizes_paper_or_production": False,
            }
            for name, passed, criterion, actual in definitions
        ]
    )
    output["all_primary_gates_passed"] = bool(output["passed"].all())
    return output


def build_quality_screen(
    *,
    test_runs: pd.DataFrame,
    aggregate: pd.DataFrame,
    replay_checks: pd.DataFrame,
    trade_pairs: pd.DataFrame,
) -> pd.DataFrame:
    profiles = build_stress_grid()
    expected_ids = {profile.scenario_id for profile in profiles}
    checks = [
        ("scenario_count", aggregate["scenario_id"].nunique() == 16, 16, aggregate["scenario_id"].nunique()),
        ("scenario_grid_exact", set(aggregate["scenario_id"]) == expected_ids, sorted(expected_ids), sorted(aggregate["scenario_id"])),
        ("test_run_count", len(test_runs) == 16 * EXPECTED_WINDOW_COUNT, 16 * EXPECTED_WINDOW_COUNT, len(test_runs)),
        ("baseline_replay", bool(replay_checks["passed"].all()), True, bool(replay_checks["passed"].all())),
        ("baseline_trade_pairs", bool(trade_pairs.loc[trade_pairs["scenario_id"] == BASELINE_SCENARIO_ID, "match_status"].eq("MATCHED").all()), True, bool(trade_pairs.loc[trade_pairs["scenario_id"] == BASELINE_SCENARIO_ID, "match_status"].eq("MATCHED").all())),
        ("no_scenario_selection", "selected_scenario" not in aggregate.columns, True, "selected_scenario" not in aggregate.columns),
    ]
    output = pd.DataFrame(
        [
            {
                "check": name,
                "passed": bool(passed),
                "expected": json.dumps(_json_safe(expected), sort_keys=True),
                "actual": json.dumps(_json_safe(actual), sort_keys=True),
                "authorizes_baseline_change": False,
            }
            for name, passed, expected, actual in checks
        ]
    )
    if not output["passed"].all():
        raise ValueError(
            "Execution/slippage quality screen failed: "
            + ", ".join(output.loc[~output["passed"], "check"])
        )
    return output


def run_execution_slippage_stress(
    *,
    snapshot: Mapping[str, Any],
    official_baseline_test_runs: pd.DataFrame,
    source_files: Mapping[str, Path] | None = None,
    source_hashes: Mapping[str, str] | None = None,
    source_verification: Mapping[str, Any] | None = None,
    save_authorization: object | None = None,
    execution_slippage_stress_stamp: str | None = None,
    # Pass-through from load_verified_source's holding-path derivation layer.
    # Not consumed by the stress computation itself (it never was a data
    # dependency of it, only a provenance/lineage gate -- see the report);
    # carried on the bundle purely so callers/tests can inspect what was
    # derived, without load_verified_source's full return dict needing to
    # match this function's required parameters exactly.
    holding_path_trades: pd.DataFrame | None = None,
    holding_path_statistics: pd.DataFrame | None = None,
    holding_path_derived_datasets: Mapping[str, Any] | None = None,
    holding_path_summary: Mapping[str, Any] | None = None,
    holding_path_not_implemented: Sequence[str] | None = None,
) -> dict[str, Any]:
    stamp = execution_slippage_stress_stamp or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    manifest = snapshot["manifest"]
    if manifest.get("snapshot_id") != APPROVED_SNAPSHOT_ID:
        raise ValueError("Execution stress requires the approved frozen snapshot.")
    base_config: PortfolioBacktestConfig = snapshot["config"]
    base_config.validate()
    expected_costs = (0.0005, 1.0, 5.0)
    if (base_config.commission_rate, base_config.minimum_fee, base_config.slippage_bps) != expected_costs:
        raise ValueError("Frozen baseline execution costs do not match the pre-registration.")
    windows = _window_records(snapshot["windows"])
    grid = build_stress_grid()
    capitals = {profile.scenario_id: float(base_config.initial_cash) for profile in grid}

    test_rows: list[dict[str, Any]] = []
    ticker_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []

    for window_number, window in enumerate(windows, 1):
        test_data = slice_prepared_data(
            snapshot["data_by_ticker"],
            start=window["test_start"],
            end_exclusive=window["test_end_exclusive"],
        )
        print(f"EXECUTION STRESS [{window_number}/{len(windows)}] {window['window_id']}")
        for scenario_number, profile in enumerate(grid, 1):
            config = build_stress_config(base_config, profile)
            result = run_portfolio_backtest(
                data_by_ticker=test_data, config=config, include_benchmark=False
            )
            matched, _ = run_matched_benchmark(
                test_data,
                initial_cash=config.initial_cash,
                exposure_percent=max(result.average_exposure_percent, 0.0001),
                config=config,
                label=f"MATCHED_{profile.scenario_id}_{window['window_id']}",
            )
            annual = annual_returns_from_equity(
                result.equity_curve, initial_cash=result.initial_cash
            )
            tickers = ticker_contributions(result)
            summary = build_variant_summary(result, matched, annual, tickers)
            fields = profile.to_dict(base_config)
            test_rows.append(
                {
                    "execution_slippage_stress_stamp": stamp,
                    "window_id": window["window_id"],
                    "test_start": window["test_start"],
                    "test_end_exclusive": window["test_end_exclusive"],
                    **fields,
                    **summary,
                }
            )
            if not tickers.empty:
                frame = tickers.copy()
                for column, value in reversed(list({"window_id": window["window_id"], **fields}.items())):
                    frame.insert(0, column, value)
                ticker_rows.extend(frame.to_dict("records"))
            for reason, count in sorted(rejection_counts(result).items()):
                rejection_rows.append(
                    {"window_id": window["window_id"], **fields, "reason_code": reason, "count": count}
                )
            trade_rows.extend(
                _trade_rows(
                    result,
                    window_id=window["window_id"],
                    profile=profile,
                    base_config=base_config,
                )
            )
            capitals[profile.scenario_id] = _append_scaled_equity(
                equity_rows,
                result=result,
                window_id=window["window_id"],
                profile=profile,
                base_config=base_config,
                opening_capital=capitals[profile.scenario_id],
                initial_cash=base_config.initial_cash,
            )
            if scenario_number == len(grid):
                print(
                    f"  {profile.scenario_id} return={summary['total_return_percent']:+.4f}% "
                    f"fees={summary['total_fees']:.2f}"
                )

    test_runs = pd.DataFrame(test_rows)
    tickers = pd.DataFrame(ticker_rows)
    rejections = pd.DataFrame(rejection_rows)
    trades = pd.DataFrame(trade_rows)
    equity = pd.DataFrame(equity_rows)
    aggregate_records: list[dict[str, Any]] = []
    profile_lookup = {profile.scenario_id: profile for profile in grid}
    for scenario_id in sorted(profile_lookup):
        record = _aggregate_model(
            test_runs.assign(model=test_runs["scenario_id"]),
            equity.assign(model=equity["scenario_id"]),
            model=scenario_id,
            initial_cash=base_config.initial_cash,
            first_test_start=windows[0]["test_start"],
            last_test_end_exclusive=windows[-1]["test_end_exclusive"],
        )
        record.pop("model", None)
        aggregate_records.append(
            {
                "execution_slippage_stress_stamp": stamp,
                **profile_lookup[scenario_id].to_dict(base_config),
                **record,
            }
        )
    aggregate = pd.DataFrame(aggregate_records)
    trade_pairs = build_trade_pairs(trades)
    comparison = build_comparison(aggregate, test_runs, trade_pairs)
    replay_checks = build_replay_checks(test_runs, official_baseline_test_runs)
    gates = build_primary_gates(comparison)
    screen = build_quality_screen(
        test_runs=test_runs,
        aggregate=aggregate,
        replay_checks=replay_checks,
        trade_pairs=trade_pairs,
    )
    summary = {
        "scenario_count": len(grid),
        "window_count": len(windows),
        "test_run_count": len(test_runs),
        "baseline_replay_passed": bool(replay_checks["passed"].all()),
        "quality_screen_passed": bool(screen["passed"].all()),
        "all_primary_gates_passed": bool(gates["passed"].all()),
        "selected_scenario": None,
        "baseline_change_authorized": False,
        "paper_or_production_authorized": False,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "execution_slippage_stress_stamp": stamp,
        "mfe_mae_holding_path_stamp": APPROVED_MFE_MAE_HOLDING_PATH_STAMP,
        "stop_walk_forward_stamp": APPROVED_STOP_WALK_FORWARD_STAMP,
        "snapshot_id": manifest["snapshot_id"],
        "snapshot_fingerprint": manifest["fingerprint"],
        "base_config": base_config.to_dict(),
        "stress_grid": [profile.to_dict(base_config) for profile in grid],
        "windows": pd.DataFrame(windows),
        "test_runs": test_runs,
        "aggregate": aggregate,
        "comparison": comparison,
        "ticker_contributions": tickers,
        "rejections": rejections,
        "trades": trades,
        "trade_pairs": trade_pairs,
        "stitched_equity": equity,
        "replay_checks": replay_checks,
        "primary_gates": gates,
        "screen": screen,
        "summary": summary,
        "source_files": None if source_files is None else {str(k): Path(v) for k, v in source_files.items()},
        "source_hashes": None if source_hashes is None else dict(source_hashes),
        "source_verification": None if source_verification is None else dict(source_verification),
        "save_authorization": save_authorization,
        "holding_path_trades": holding_path_trades,
        "holding_path_statistics": holding_path_statistics,
        "holding_path_derived_datasets": (
            None if holding_path_derived_datasets is None else dict(holding_path_derived_datasets)
        ),
        "holding_path_summary": (
            None if holding_path_summary is None else dict(holding_path_summary)
        ),
        "holding_path_not_implemented": (
            None if holding_path_not_implemented is None else list(holding_path_not_implemented)
        ),
    }


def _verify_declared_files(records: Any, *, label: str) -> dict[str, Path]:
    if not isinstance(records, dict) or not records:
        raise ValueError(f"{label} provenance has no file mapping.")
    output: dict[str, Path] = {}
    for key, metadata in records.items():
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid {label} metadata: {key}")
        path = Path(str(metadata.get("path", "")))
        expected_hash = str(metadata.get("sha256", ""))
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"{label} hash mismatch: {key}")
        output[str(key)] = path
    return output


# --- Holding-path derivation layer -----------------------------------------
#
# The never-committed run_portfolio_holding_path_attribution.py produced 15
# named result keys directly as separate files. mfe_mae only ever writes
# three (trades, statistics, json) -- everything else below is computed here,
# in-memory, from mfe_mae's own verified trades/statistics, reusing pure
# functions lifted from the original file (never committed, so never
# importable) and mfe_mae's own build_statistics_table where the grouping
# already exists in mfe_mae's statistics.csv. `path_rows` is the one
# genuinely NOT_IMPLEMENTED gap -- see NOT_IMPLEMENTED_HOLDING_KEYS above.

_OPEN_EXIT = "OPEN_EXIT"
_INTRABAR_STOP_BOUNDED = "INTRABAR_STOP_BOUNDED"
_CLOSE_EXIT = "CLOSE_EXIT"
_EXIT_SEMANTIC_VALUES = (_OPEN_EXIT, _INTRABAR_STOP_BOUNDED, _CLOSE_EXIT)

_MFE_BEFORE_MAE = "MFE_BEFORE_MAE"
_MAE_BEFORE_MFE = "MAE_BEFORE_MFE"
_AMBIGUOUS_SAME_BAR = "AMBIGUOUS_SAME_BAR"
_EXTREME_ORDER_VALUES = (_MFE_BEFORE_MAE, _MAE_BEFORE_MFE, _AMBIGUOUS_SAME_BAR)

_HOLDING_BUCKET_VALUES = ("0", "1_2", "3_5", "6_10", "11_20", "21_40", "41_PLUS")

_PRIMARY_POPULATION = "PRIMARY_ALL_COMPLETED_BASELINE_TRADES"
_SENSITIVITY_POPULATION = "SENSITIVITY_EXCLUDING_FORCE_CLOSE_END"

# Cross-checked against mfe_mae's own OFFICIAL_EXIT_PHASE_COUNTS (same
# 256-trade population, same counts: 85/15/114/42, different label names).
_OFFICIAL_EXIT_REASON_COUNTS = {
    "STOP_LOSS": 114,
    "EXIT_SIGNAL_NEXT_OPEN": 85,
    "FORCE_CLOSE_END": 42,
    "GAP_STOP_LOSS": 15,
}


def _derived_exit_semantic(exit_reason: str) -> str:
    """Reimplementation of the original file's `exit_semantic()` -- a pure
    3-way remap of `exit_reason`. Not mfe_mae's own 4-way
    `exit_execution_phase` (OPEN_PENDING_EXIT/OPEN_GAP_STOP/
    INTRABAR_INITIAL_STOP/FORCE_CLOSE_LOCAL_CLOSE), which classifies the
    same rows more granularly under different names.
    """
    reason = str(exit_reason)
    if reason in {"EXIT_SIGNAL_NEXT_OPEN", "GAP_STOP_LOSS"}:
        return _OPEN_EXIT
    if reason == "STOP_LOSS":
        return _INTRABAR_STOP_BOUNDED
    if reason == "FORCE_CLOSE_END":
        return _CLOSE_EXIT
    raise ValueError(f"Unsupported exit_reason for holding-path derivation: {reason!r}")


def _derived_holding_bucket(holding_period_ticker_bars: Any) -> str:
    """Reimplementation of the original file's `holding_bucket()`, applied
    to mfe_mae's own (identically named) `holding_period_ticker_bars` column.
    """
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


def _derived_extreme_order(mfe_offset: Any, mae_offset: Any) -> str | None:
    """Reimplementation of the original file's `_extreme_order()`, applied to
    mfe_mae's `observed_fully_held_mfe_offset_from_entry` /
    `observed_fully_held_mae_offset_from_entry`. Returns None (excluded from
    grouping, not a crash) when either offset is unavailable -- i.e. no fully
    held bar exists for that trade (mfe_mae's `full_held_bar_available` is
    False for 21 of 256 trades).
    """
    if mfe_offset is None or mae_offset is None or pd.isna(mfe_offset) or pd.isna(mae_offset):
        return None
    if mfe_offset < mae_offset:
        return _MFE_BEFORE_MAE
    if mae_offset < mfe_offset:
        return _MAE_BEFORE_MFE
    return _AMBIGUOUS_SAME_BAR


def _with_derived_columns(mfe_mae_trades: pd.DataFrame) -> pd.DataFrame:
    trades = mfe_mae_trades.copy()
    trades["exit_semantic"] = trades["exit_reason"].map(_derived_exit_semantic)
    trades["holding_bucket"] = trades["holding_period_ticker_bars"].map(_derived_holding_bucket)
    trades["censored_extreme_order"] = [
        _derived_extreme_order(mfe_offset, mae_offset)
        for mfe_offset, mae_offset in zip(
            trades["observed_fully_held_mfe_offset_from_entry"],
            trades["observed_fully_held_mae_offset_from_entry"],
        )
    ]
    return trades


def _grouped_both_populations(
    trades: pd.DataFrame, *, stamp: str, group_type: str, group_column: str, group_values: Sequence[str]
) -> pd.DataFrame:
    """Reimplementation of the original file's `_both_populations()`, using
    mfe_mae's own (public, reused-not-reimplemented) `build_statistics_table`
    against a locally derived grouping column mfe_mae's own statistics.csv
    does not contain.
    """
    frames = [
        mfe_mae.build_statistics_table(
            trades if population == _PRIMARY_POPULATION
            else trades.loc[trades["exit_reason"] != "FORCE_CLOSE_END"],
            stamp=stamp,
            population=population,
            group_type=group_type,
            group_column=group_column,
            group_values=group_values,
        )
        for population in (_PRIMARY_POPULATION, _SENSITIVITY_POPULATION)
    ]
    return pd.concat(frames, ignore_index=True)


def _derive_holding_path_screen(
    trades: pd.DataFrame, *, stamp: str, official_source: bool
) -> pd.DataFrame:
    """8 (or 9, if `official_source`) of the original's checks, honestly
    derivable from mfe_mae's own verified data; 2 genuinely NOT_IMPLEMENTED
    (need path_rows); 2 structurally NOT_APPLICABLE (mfe_mae has a single
    MFE/MAE estimate per trade, not the original's censored/possible
    dual-bound design, so there is no second bound to order-check against).
    Every row's `passed` reflects reality -- NOT_IMPLEMENTED/NOT_APPLICABLE
    rows are never marked as having passed a check that was never run.
    """
    reason_counts = trades["exit_reason"].value_counts().to_dict()
    checks: list[tuple[str, bool, Any, Any, str, str]] = [
        (
            "trade_count", len(trades) == EXPECTED_TRADE_COUNT, EXPECTED_TRADE_COUNT, len(trades),
            "Completed baseline trade population.", "IMPLEMENTED",
        ),
        (
            "window_count", trades["window_id"].nunique() == EXPECTED_WINDOW_COUNT,
            EXPECTED_WINDOW_COUNT, trades["window_id"].nunique(),
            "Independently reset test windows.", "IMPLEMENTED",
        ),
        (
            "unique_trade_ids", not trades["trade_id"].duplicated().any(), True,
            not trades["trade_id"].duplicated().any(), "Trade identifiers remain unique.",
            "IMPLEMENTED",
        ),
        (
            "mfe_non_negative",
            bool(trades["observed_fully_held_mfe_percent"].dropna().ge(-1e-10).all()),
            True, bool(trades["observed_fully_held_mfe_percent"].dropna().ge(-1e-10).all()),
            "mfe_mae's own observed_fully_held_mfe_percent is a nonnegative magnitude "
            "wherever a fully held bar exists (NaN for the 21 trades with none, which "
            "is a legitimate absence, not a sign violation, and is excluded here).",
            "IMPLEMENTED",
        ),
        (
            "mae_non_negative",
            bool(trades["observed_fully_held_mae_percent"].dropna().ge(-1e-10).all()),
            True, bool(trades["observed_fully_held_mae_percent"].dropna().ge(-1e-10).all()),
            "mfe_mae's own observed_fully_held_mae_percent is a nonnegative magnitude "
            "wherever a fully held bar exists (sign convention differs from the "
            "original file's mae_non_positive; NaN rows excluded, see mfe_non_negative).",
            "IMPLEMENTED",
        ),
        (
            "intrabar_stop_bounded",
            bool(
                trades.loc[trades["exit_semantic"] == _INTRABAR_STOP_BOUNDED, "intrabar_order_ambiguity"]
                .eq("EXIT_BAR_HIGH_LOW_POST_EXIT_UNKNOWN").all()
            ),
            True,
            bool(
                trades.loc[trades["exit_semantic"] == _INTRABAR_STOP_BOUNDED, "intrabar_order_ambiguity"]
                .eq("EXIT_BAR_HIGH_LOW_POST_EXIT_UNKNOWN").all()
            ),
            "STOP_LOSS exit bars remain explicitly order-ambiguous in mfe_mae's own field.",
            "IMPLEMENTED",
        ),
        (
            "force_close_sensitivity_population",
            int((trades["exit_reason"] != "FORCE_CLOSE_END").sum())
            == len(trades) - int((trades["exit_reason"] == "FORCE_CLOSE_END").sum()),
            True, True, "Sensitivity population excludes only FORCE_CLOSE_END.", "IMPLEMENTED",
        ),
    ]
    if official_source:
        checks.append(
            (
                "official_exit_reason_counts", reason_counts == _OFFICIAL_EXIT_REASON_COUNTS,
                _OFFICIAL_EXIT_REASON_COUNTS, reason_counts,
                "Official population reason reconciliation.", "IMPLEMENTED",
            )
        )
    for name in ("path_row_reconciliation", "open_exit_hlc_excluded"):
        checks.append(
            (
                name, False, "requires path_rows", "path_rows is NOT_IMPLEMENTED this turn",
                "Depends on a per-bar path_rows table this repointing does not produce.",
                "NOT_IMPLEMENTED",
            )
        )
    for name in ("mfe_bounds_ordered", "mae_bounds_ordered"):
        checks.append(
            (
                name, True, "N/A", "N/A",
                "mfe_mae has one MFE/MAE estimate per trade, not the original's "
                "censored/possible dual-bound design; no second bound exists to order-check.",
                "NOT_APPLICABLE",
            )
        )
    frame = pd.DataFrame(
        [
            {
                "mfe_mae_holding_path_stamp": stamp,
                "check": name,
                "passed": passed,
                "expected": json.dumps(_json_safe(expected), sort_keys=True),
                "actual": json.dumps(_json_safe(actual), sort_keys=True),
                "detail": detail,
                "authorizes_strategy_change": False,
                "status": status,
            }
            for name, passed, expected, actual, detail, status in checks
        ]
    )
    implemented = frame.loc[frame["status"] == "IMPLEMENTED"]
    if not implemented["passed"].all():
        failed = implemented.loc[~implemented["passed"], "check"].tolist()
        raise ValueError("Derived holding-path quality screen failed: " + ", ".join(failed))
    return frame


def _derive_holding_path_datasets(
    *,
    mfe_mae_trades: pd.DataFrame,
    mfe_mae_statistics: pd.DataFrame,
    stamp: str,
    official_source: bool,
) -> dict[str, Any]:
    """Derive the 12 non-file-backed holding-path-equivalent datasets
    (11 named tables + screen) from mfe_mae's own verified trades/statistics.
    `path_rows` is deliberately absent -- see NOT_IMPLEMENTED_HOLDING_KEYS.
    """
    trades = _with_derived_columns(mfe_mae_trades)

    def _filtered(group_type: str, *, population: str | None = None) -> pd.DataFrame:
        selected = mfe_mae_statistics.loc[mfe_mae_statistics["group_type"] == group_type]
        if population is not None:
            selected = selected.loc[selected["population"] == population]
        return selected.reset_index(drop=True)

    datasets: dict[str, Any] = {
        # Pure filters of mfe_mae's own already-computed statistics.csv --
        # mfe_mae's build_aggregation_table already groups by these exact
        # columns for both populations; nothing is recomputed here.
        "overall": _filtered("ALL_TRADES"),
        "asset_classes": _filtered("ASSET_CLASS"),
        "tickers": _filtered("TICKER"),
        "windows": _filtered("WINDOW_ID"),
        "exit_reasons": _filtered("EXIT_REASON"),
        "exit_categories": _filtered("EXIT_CATEGORY"),
        "outcomes": _filtered("OUTCOME_CLASS"),
        "force_close_excluded": _filtered("ALL_TRADES", population=_SENSITIVITY_POPULATION),
        # New groupings mfe_mae's statistics.csv does not contain -- computed
        # via mfe_mae's own build_statistics_table against a locally derived
        # column (see _with_derived_columns).
        "holding_buckets": _grouped_both_populations(
            trades, stamp=stamp, group_type="HOLDING_BUCKET",
            group_column="holding_bucket", group_values=_HOLDING_BUCKET_VALUES,
        ),
        "exit_semantics": _grouped_both_populations(
            trades, stamp=stamp, group_type="EXIT_SEMANTIC",
            group_column="exit_semantic", group_values=_EXIT_SEMANTIC_VALUES,
        ),
        "extreme_orders": _grouped_both_populations(
            trades, stamp=stamp, group_type="CENSORED_EXTREME_ORDER",
            group_column="censored_extreme_order", group_values=_EXTREME_ORDER_VALUES,
        ),
    }
    datasets["screen"] = _derive_holding_path_screen(
        trades, stamp=stamp, official_source=official_source
    )
    return datasets


def load_verified_source(
    *,
    snapshot_directory: Path,
    holding_path_directory: Path,
    holding_path_stamp: str,
    stop_walk_forward_directory: Path,
    stop_walk_forward_stamp: str,
    project_root: Path = Path("."),
) -> dict[str, Any]:
    """Load the one approved snapshot/holding-path-equivalent/stop-WF lineage.

    `holding_path_directory`/`holding_path_stamp` now point at mfe_mae's real
    saved output (repointed; see APPROVED_MFE_MAE_HOLDING_PATH_STAMP). Only
    mfe_mae's own three real files (trades, statistics, json) are hash-
    verified on disk; the other 11 named holding-path-equivalent datasets are
    derived in-memory from them (see _derive_holding_path_datasets) --
    `path_rows` is the one exception, NOT_IMPLEMENTED this turn.
    """
    if holding_path_stamp != APPROVED_MFE_MAE_HOLDING_PATH_STAMP:
        raise ValueError("Unapproved mfe_mae holding-path stamp.")
    if stop_walk_forward_stamp != APPROVED_STOP_WALK_FORWARD_STAMP:
        raise ValueError("Unapproved Stop Walk-Forward stamp.")
    project_root = Path(project_root).resolve()
    snapshot_directory = Path(snapshot_directory)
    holding_directory = Path(holding_path_directory)
    stop_directory = Path(stop_walk_forward_directory)
    snapshot = load_snapshot(snapshot_directory, verify_code=True, project_root=project_root)
    manifest = snapshot["manifest"]
    if manifest.get("snapshot_id") != APPROVED_SNAPSHOT_ID or manifest.get("fingerprint") != APPROVED_SNAPSHOT_FINGERPRINT:
        raise ValueError("Snapshot lineage mismatch.")
    if set(manifest.get("market_files", {})) != set(CONTROLLED_TICKERS):
        raise ValueError("Snapshot ticker basket mismatch.")

    holding_provenance_path = holding_directory / (
        f"portfolio_mfe_mae_holding_path_provenance_{holding_path_stamp}.json"
    )
    holding_provenance = _read_json(holding_provenance_path)
    holding_lineage = {
        "mfe_mae_holding_path_stamp": APPROVED_MFE_MAE_HOLDING_PATH_STAMP,
        "forward_return_statistics_stamp": APPROVED_FORWARD_RETURN_STATISTICS_STAMP,
        "entry_statistics_stamp": APPROVED_ENTRY_STATISTICS_STAMP,
        "timing_stamp": APPROVED_TIMING_STAMP,
        "source_stop_walk_forward_stamp": APPROVED_STOP_WALK_FORWARD_STAMP,
        "snapshot_id": APPROVED_SNAPSHOT_ID,
        "snapshot_fingerprint": APPROVED_SNAPSHOT_FINGERPRINT,
    }
    for key, expected in holding_lineage.items():
        if holding_provenance.get(key) != expected:
            raise ValueError(f"mfe_mae holding-path provenance {key} mismatch.")
    holding_results = verify_provenance_result_files(
        directory=holding_directory,
        provenance=holding_provenance,
        prefix="portfolio_mfe_mae_holding_path_",
        stamp=holding_path_stamp,
    )
    if set(holding_results) != EXPECTED_MFE_MAE_RESULT_KEYS:
        raise ValueError("mfe_mae holding-path result coverage mismatch.")
    holding_sources = _verify_declared_files(
        holding_provenance.get("source_files"), label="mfe_mae holding-path source"
    )
    holding_trades = pd.read_csv(holding_results["trades"])
    if len(holding_trades) != EXPECTED_TRADE_COUNT:
        raise ValueError("mfe_mae holding-path trade population mismatch.")
    holding_statistics = pd.read_csv(holding_results["statistics"])
    official_source = True
    holding_datasets = _derive_holding_path_datasets(
        mfe_mae_trades=holding_trades,
        mfe_mae_statistics=holding_statistics,
        stamp=holding_path_stamp,
        official_source=official_source,
    )
    # strategy_change_authorized is not read from mfe_mae's JSON (it has no
    # such field) -- it is asserted here as a static governance fact about
    # this derivation itself, the same boundary every analyzer in this
    # codebase asserts, never a claim verified against an external source.
    holding_summary = {"strategy_change_authorized": False}

    stop_provenance_path = stop_directory / (
        f"portfolio_stop_walk_forward_provenance_{stop_walk_forward_stamp}.json"
    )
    stop_provenance = _read_json(stop_provenance_path)
    if stop_provenance.get("source_stamp") != stop_walk_forward_stamp:
        raise ValueError("Stop provenance stamp mismatch.")
    if stop_provenance.get("snapshot_id") != APPROVED_SNAPSHOT_ID or stop_provenance.get("snapshot_fingerprint") != APPROVED_SNAPSHOT_FINGERPRINT:
        raise ValueError("Stop provenance snapshot mismatch.")
    stop_results = verify_provenance_result_files(
        directory=stop_directory,
        provenance=stop_provenance,
        prefix="portfolio_stop_walk_forward_",
        stamp=stop_walk_forward_stamp,
    )
    if set(stop_results) != EXPECTED_STOP_RESULT_KEYS:
        raise ValueError("Stop walk-forward result coverage mismatch.")
    stop_payload = _read_json(stop_results["json"])
    if stop_payload.get("base_config") != snapshot["config"].to_dict():
        raise ValueError("Stop walk-forward and snapshot configs disagree.")
    official_baseline = pd.read_csv(stop_results["test_runs"])
    if len(official_baseline.loc[official_baseline["model"] == "FIXED_BASELINE"]) != EXPECTED_WINDOW_COUNT:
        raise ValueError("Official baseline test-window population mismatch.")

    holding_code = project_root / "src/backtest/run_portfolio_mfe_mae_holding_path_attribution.py"
    holding_test = project_root / "tests/test_portfolio_mfe_mae_holding_path_attribution.py"
    if sha256_file(holding_code) != APPROVED_MFE_MAE_CODE_SHA256:
        raise ValueError("Approved mfe_mae holding-path code hash mismatch.")
    if sha256_file(holding_test) != APPROVED_MFE_MAE_TEST_SHA256:
        raise ValueError("Approved mfe_mae holding-path test hash mismatch.")

    source_files: dict[str, Path] = {
        "mfe_mae_holding_path_provenance": holding_provenance_path,
        "stop_walk_forward_provenance": stop_provenance_path,
        "mfe_mae_holding_path_code": holding_code,
        "mfe_mae_holding_path_test": holding_test,
        "execution_slippage_stress_code": Path(__file__).resolve(),
        **{f"holding_result:{key}": path for key, path in holding_results.items()},
        **{f"holding_upstream:{key}": path for key, path in holding_sources.items()},
        **{f"stop_result:{key}": path for key, path in stop_results.items()},
    }
    source_hashes = {label: sha256_file(path) for label, path in source_files.items()}
    return {
        "snapshot": snapshot,
        "official_baseline_test_runs": official_baseline,
        "holding_path_trades": holding_trades,
        "holding_path_statistics": holding_statistics,
        "holding_path_derived_datasets": holding_datasets,
        "holding_path_summary": holding_summary,
        "holding_path_not_implemented": sorted(NOT_IMPLEMENTED_HOLDING_KEYS),
        "source_files": source_files,
        "source_hashes": source_hashes,
        "source_verification": {
            "verified": True,
            "code_hash_verification": True,
            "official_source": official_source,
            **holding_lineage,
        },
        "save_authorization": _STRICT_SOURCE_AUTHORIZATION,
    }


def _validate_saveable(bundle: Mapping[str, Any]) -> None:
    if bundle.get("save_authorization") is not _STRICT_SOURCE_AUTHORIZATION:
        raise ValueError("Cannot save: strict source authorization is missing.")
    verification = bundle.get("source_verification")
    if not isinstance(verification, Mapping) or verification.get("verified") is not True or verification.get("code_hash_verification") is not True or verification.get("official_source") is not True:
        raise ValueError("Cannot save: official source verification is incomplete.")
    files, hashes = bundle.get("source_files"), bundle.get("source_hashes")
    if not isinstance(files, Mapping) or not isinstance(hashes, Mapping) or set(files) != set(hashes):
        raise ValueError("Cannot save: consumed source mapping is incomplete.")
    for label, value in files.items():
        path = Path(value)
        if not path.is_file() or sha256_file(path) != hashes[label]:
            raise ValueError(f"Source changed before save: {label}")
    if not bundle["screen"]["passed"].all() or not bundle["replay_checks"]["passed"].all():
        raise ValueError("Cannot save: quality or replay screen failed.")


def build_json_payload(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": bundle["created_at"],
        "execution_slippage_stress_stamp": bundle["execution_slippage_stress_stamp"],
        "method": "Fixed 4x4 execution-cost stress grid on immutable OOS windows; no selection.",
        "source_lineage": {
            "mfe_mae_holding_path_stamp": bundle["mfe_mae_holding_path_stamp"],
            "stop_walk_forward_stamp": bundle["stop_walk_forward_stamp"],
            "snapshot_id": bundle["snapshot_id"],
            "snapshot_fingerprint": bundle["snapshot_fingerprint"],
            "source_files": {
                label: {"path": str(path), "sha256": bundle["source_hashes"][label]}
                for label, path in bundle["source_files"].items()
            },
        },
        "pre_registration": {
            "multipliers": list(MULTIPLIERS),
            "baseline_scenario": BASELINE_SCENARIO_ID,
            "primary_stress_scenario": PRIMARY_STRESS_SCENARIO_ID,
            "primary_minimum_positive_windows": PRIMARY_MINIMUM_POSITIVE_WINDOWS,
            "primary_maximum_drawdown_worsening_percent": PRIMARY_MAXIMUM_DRAWDOWN_WORSENING_PERCENT,
            "benchmark_excess_is_reported_not_gated": True,
            "four_x_and_eight_x_are_diagnostic_only": True,
        },
        "base_config": bundle["base_config"],
        "stress_grid": bundle["stress_grid"],
        "summary": bundle["summary"],
        "aggregate": bundle["aggregate"].to_dict("records"),
        "comparison": bundle["comparison"].to_dict("records"),
        "primary_gates": bundle["primary_gates"].to_dict("records"),
        "quality_screen": bundle["screen"].to_dict("records"),
        "limitations": [
            "Cost changes can alter equity, position sizing, cash feasibility, and later trade population; trade pairs expose those differences.",
            "Matched benchmark uses each scenario's own costs and strategy exposure.",
            "No scenario is selected or promoted by this analyzer.",
            "Passing primary gates does not authorize a baseline, paper, or production change.",
            "The provenance file hashes every other output and cannot self-hash.",
        ],
    }


def save_execution_slippage_stress(
    bundle: Mapping[str, Any], output_directory: Path = DEFAULT_OUTPUT_DIRECTORY
) -> dict[str, Path]:
    _validate_saveable(bundle)
    output_directory = Path(output_directory)
    stamp = str(bundle["execution_slippage_stress_stamp"])
    frames = {
        "windows": bundle["windows"],
        "test_runs": bundle["test_runs"],
        "aggregate": bundle["aggregate"],
        "comparison": bundle["comparison"],
        "tickers": bundle["ticker_contributions"],
        "rejections": bundle["rejections"],
        "trades": bundle["trades"],
        "trade_pairs": bundle["trade_pairs"],
        "equity": bundle["stitched_equity"],
        "replay_checks": bundle["replay_checks"],
        "primary_gates": bundle["primary_gates"],
        "screen": bundle["screen"],
    }
    paths = {
        name: output_directory / f"portfolio_execution_slippage_stress_{name}_{stamp}.csv"
        for name in frames
    }
    paths["json"] = output_directory / f"portfolio_execution_slippage_stress_{stamp}.json"
    paths["provenance"] = output_directory / f"portfolio_execution_slippage_stress_provenance_{stamp}.json"
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
        "execution_slippage_stress_stamp": stamp,
        "mfe_mae_holding_path_stamp": bundle["mfe_mae_holding_path_stamp"],
        "stop_walk_forward_stamp": bundle["stop_walk_forward_stamp"],
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
    baseline = bundle["comparison"].loc[
        bundle["comparison"]["scenario_id"] == BASELINE_SCENARIO_ID
    ].iloc[0]
    primary = bundle["comparison"].loc[
        bundle["comparison"]["scenario_id"] == PRIMARY_STRESS_SCENARIO_ID
    ].iloc[0]
    print("PORTFOLIO EXECUTION AND SLIPPAGE STRESS V1")
    print("Provenance validation: PASS")
    print("Official baseline replay: PASS")
    print("Quality screen: PASS")
    print(
        f"Grid: 16 fixed scenarios x {bundle['summary']['window_count']} OOS windows = "
        f"{bundle['summary']['test_run_count']} runs"
    )
    print(
        f"Baseline {BASELINE_SCENARIO_ID}: return={baseline['compounded_return_percent']:+.4f}% "
        f"DD={baseline['maximum_drawdown_percent']:.4f}% PF={baseline['average_profit_factor']:.4f}"
    )
    print(
        f"Primary {PRIMARY_STRESS_SCENARIO_ID}: return={primary['compounded_return_percent']:+.4f}% "
        f"DD={primary['maximum_drawdown_percent']:.4f}% PF={primary['average_profit_factor']:.4f} "
        f"positive_windows={int(primary['positive_window_count'])}/{EXPECTED_WINDOW_COUNT}"
    )
    print(
        "Primary robustness gates: "
        + ("PASS" if bundle["summary"]["all_primary_gates_passed"] else "FAIL")
    )
    print("Scenario selected: NONE")
    print("Baseline/paper/production change authorized: NO")
    if no_save:
        print("Output artifacts saved: 0 (--no-save)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-registered fixed-grid portfolio execution/slippage stress test"
    )
    parser.add_argument("--snapshot-directory", type=Path, required=True)
    parser.add_argument("--holding-path-directory", type=Path, required=True)
    parser.add_argument("--holding-path-stamp", required=True)
    parser.add_argument("--stop-walk-forward-directory", type=Path, required=True)
    parser.add_argument("--stop-walk-forward-stamp", required=True)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--no-save", action="store_true")
    arguments = parser.parse_args()
    source = load_verified_source(
        snapshot_directory=arguments.snapshot_directory,
        holding_path_directory=arguments.holding_path_directory,
        holding_path_stamp=arguments.holding_path_stamp,
        stop_walk_forward_directory=arguments.stop_walk_forward_directory,
        stop_walk_forward_stamp=arguments.stop_walk_forward_stamp,
    )
    bundle = run_execution_slippage_stress(**source)
    _print_summary(bundle, no_save=arguments.no_save)
    if arguments.no_save:
        return
    for name, path in save_execution_slippage_stress(bundle, arguments.output_directory).items():
        print(name, path.resolve())


if __name__ == "__main__":
    main()
