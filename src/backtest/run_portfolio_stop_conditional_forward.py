"""Preregister and evaluate a forward-only stop-conditional RSI hypothesis.

The protocol is deliberately non-operational.  Registration freezes the
post-hoc interaction finding, the last observed date, the future horizon, and
all decision gates.  Evaluation then runs four deterministic research arms on
data created after registration:

* 5%/5% stop with no replacement;
* 5%/5% stop with RSI_Q12_LOSER_ONLY;
* 3.5%/3.5% stop with no replacement;
* 3.5%/3.5% stop with RSI_Q12_LOSER_ONLY.

No result from this module authorizes baseline, paper, shadow, or production
changes.  A successful completed evaluation is only ready for human review.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.backtest.portfolio_backtest_models import PortfolioBacktestConfig
from src.backtest.run_portfolio_rsi_replacement_ablation import (
    CONTROL_POLICY,
    REPLACEMENT_POLICIES,
    STOP_PROFILES,
    ReplacementPolicy,
)
from src.backtest.run_portfolio_rsi_replacement_walk_forward import _run_policy
from src.backtest.run_research_data_snapshot import (
    load_snapshot,
    sha256_file,
    sha256_json,
)
from src.backtest.run_portfolio_walk_forward import slice_prepared_data


SCHEMA_VERSION = 1
PROTOCOL_NAME = "PREREGISTERED_FORWARD_ONLY_STOP_CONDITIONAL_HYPOTHESIS"
DEFAULT_INTERACTION_DIRECTORY = Path(
    "data/backtests/portfolio/replacement_stop_interaction_audit"
)
DEFAULT_OUTPUT_DIRECTORY = Path(
    "data/backtests/portfolio/stop_conditional_forward"
)
SOURCE_PREFIX = "portfolio_replacement_stop_interaction_audit"
OUTPUT_PREFIX = "portfolio_stop_conditional_forward"

STOP_BASELINE = "FIXED_BASELINE"
STOP_MAX_RETURN = "FIXED_MAX_RETURN"
STOP_MODELS = (STOP_BASELINE, STOP_MAX_RETURN)
MODEL_CONTROL = "FIXED_CONTROL"
MODEL_REPLACEMENT = "FIXED_RSI_Q12_LOSER_ONLY"
MODELS = (MODEL_CONTROL, MODEL_REPLACEMENT)
REPLACEMENT_POLICY = "RSI_Q12_LOSER_ONLY"

OBSERVED_CUTOFF = pd.Timestamp("2026-08-02T00:00:00")
FORWARD_START = pd.Timestamp("2026-08-03T00:00:00")
FORWARD_END_EXCLUSIVE = pd.Timestamp("2028-08-03T00:00:00")
BLOCK_MONTHS = 6
EXPECTED_BLOCK_COUNT = 4
MINIMUM_CROSS_STOP_EVENT_PAIRS = 5
MAXIMUM_DRAWDOWN_WORSENING_PERCENT = 2.5
REPLAY_ANCHOR_BARS = 200
REPLAY_FLOAT_TOLERANCE = 1e-10

SOURCE_FRAME_NAMES = (
    "summary",
    "windows",
    "events",
    "event_pairs",
    "trade_pairs",
    "tickers",
    "exit_reasons",
    "rejections",
    "leave_one_window_out",
    "screen",
)
EVALUATION_FRAME_NAMES = (
    "summary",
    "strata",
    "blocks",
    "block_leave_one_out",
    "ticker_leave_one_out",
    "events",
    "event_pairs",
    "trades",
    "tickers",
    "rejections",
    "screen",
)


def _safe(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, float) and not isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1"}


def _policy(name: str) -> ReplacementPolicy:
    matches = [policy for policy in REPLACEMENT_POLICIES if policy.name == name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one replacement policy definition: {name}")
    return matches[0]


POLICIES = {
    MODEL_CONTROL: _policy(CONTROL_POLICY),
    MODEL_REPLACEMENT: _policy(REPLACEMENT_POLICY),
}


def _interaction_paths(directory: Path, stamp: str) -> dict[str, Path]:
    directory = Path(directory)
    paths = {
        name: directory / f"{SOURCE_PREFIX}_{name}_{stamp}.csv"
        for name in SOURCE_FRAME_NAMES
    }
    paths["json"] = directory / f"{SOURCE_PREFIX}_{stamp}.json"
    paths["provenance"] = (
        directory / f"{SOURCE_PREFIX}_provenance_{stamp}.json"
    )
    return paths


def validate_interaction_decision(
    payload: Mapping[str, Any], screen: pd.DataFrame
) -> None:
    if len(screen) != 1:
        raise ValueError("Interaction audit must have exactly one screen row.")
    row = screen.iloc[0]
    required = {
        "stop_interaction_supported",
        "conditional_policy_authorized",
        "production_authorized",
        "recommended_next_stage",
    }
    missing = sorted(required.difference(screen.columns))
    if missing:
        raise ValueError(f"Interaction screen columns are missing: {missing}")
    if not _truthy(row["stop_interaction_supported"]):
        raise ValueError("Source audit did not support the stop interaction.")
    if _truthy(row["conditional_policy_authorized"]):
        raise ValueError("Source audit unexpectedly authorized a conditional policy.")
    if _truthy(row["production_authorized"]):
        raise ValueError("Source audit unexpectedly authorized production.")
    if str(row["recommended_next_stage"]) != PROTOCOL_NAME:
        raise ValueError("Source audit did not recommend this forward protocol.")
    if not bool(payload.get("stop_interaction_supported")):
        raise ValueError("Interaction JSON decision mismatch.")
    if bool(payload.get("conditional_policy_authorized")):
        raise ValueError("Interaction JSON authorized a conditional policy.")
    if bool(payload.get("production_authorized")):
        raise ValueError("Interaction JSON authorized production.")
    if payload.get("recommended_next_stage") != PROTOCOL_NAME:
        raise ValueError("Interaction JSON next-stage mismatch.")


def verify_interaction_source(
    *, directory: Path, stamp: str
) -> dict[str, Any]:
    paths = _interaction_paths(Path(directory), stamp)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing interaction-audit files: " + ", ".join(missing)
        )
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    if provenance.get("audit_stamp") != stamp:
        raise ValueError("Interaction provenance stamp mismatch.")
    expected = set(paths).difference({"provenance"})
    recorded = provenance.get("result_files", {})
    if set(recorded) != expected:
        raise ValueError("Interaction provenance does not cover every result.")
    for name in sorted(expected):
        if sha256_file(paths[name]) != recorded[name].get("sha256"):
            raise ValueError(f"Interaction result hash mismatch: {name}")
    code = provenance.get("audit_code")
    if not code:
        raise ValueError("Interaction provenance has no audit-code hash.")
    code_path = Path(code["path"])
    if not code_path.exists() or sha256_file(code_path) != code.get("sha256"):
        raise ValueError("Interaction audit-code hash mismatch.")
    for name, metadata in provenance.get("source_files", {}).items():
        source_path = Path(metadata["path"])
        if (
            not source_path.exists()
            or sha256_file(source_path) != metadata.get("sha256")
        ):
            raise ValueError(f"Interaction predecessor hash mismatch: {name}")
    manifest_path = Path(provenance["snapshot_manifest_path"])
    if (
        not manifest_path.exists()
        or sha256_file(manifest_path)
        != provenance.get("snapshot_manifest_sha256")
    ):
        raise ValueError("Interaction snapshot-manifest hash mismatch.")
    screen = pd.read_csv(paths["screen"])
    validate_interaction_decision(payload, screen)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("fingerprint") != provenance.get("snapshot_fingerprint"):
        raise ValueError("Interaction snapshot fingerprint mismatch.")
    return {
        "paths": paths,
        "provenance": provenance,
        "payload": payload,
        "screen": screen,
        "manifest": manifest,
        "manifest_path": manifest_path,
    }


def _market_end(manifest: Mapping[str, Any]) -> pd.Timestamp:
    values = [
        pd.Timestamp(metadata["end"])
        for metadata in manifest["market_files"].values()
    ]
    if not values:
        raise ValueError("Snapshot manifest has no market files.")
    return max(values)


def _protocol_core(source: Mapping[str, Any]) -> dict[str, Any]:
    manifest = source["manifest"]
    market_end = _market_end(manifest)
    if market_end != OBSERVED_CUTOFF:
        raise ValueError(
            f"Source market cutoff must be {OBSERVED_CUTOFF.isoformat()}, "
            f"got {market_end.isoformat()}."
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "source": {
            "interaction_stamp": source["provenance"]["audit_stamp"],
            "interaction_snapshot_id": source["provenance"]["snapshot_id"],
            "interaction_snapshot_fingerprint": source["provenance"][
                "snapshot_fingerprint"
            ],
            "snapshot_manifest_path": str(source["manifest_path"].resolve()),
            "snapshot_manifest_sha256": sha256_file(source["manifest_path"]),
            "config_sha256": manifest["config"]["sha256"],
            "tickers": sorted(manifest["market_files"]),
            "code_files": dict(sorted(manifest.get("code_files", {}).items())),
            "observed_cutoff": OBSERVED_CUTOFF.isoformat(),
        },
        "future_snapshot_rules": {
            "must_be_created_after_registration": True,
            "same_config_hash_required": True,
            "same_ticker_universe_required": True,
            "same_frozen_code_hashes_required": True,
            "replay_anchor_bars": REPLAY_ANCHOR_BARS,
            "replay_float_tolerance": REPLAY_FLOAT_TOLERANCE,
        },
        "evaluation_period": {
            "start_inclusive": FORWARD_START.isoformat(),
            "end_exclusive": FORWARD_END_EXCLUSIVE.isoformat(),
            "months": 24,
            "interim_results_are_monitoring_only": True,
        },
        "research_arms": [
            {
                "stop_model": stop_model,
                "stock_stop_loss_percent": STOP_PROFILES[stop_model][0],
                "crypto_stop_loss_percent": STOP_PROFILES[stop_model][1],
                "model": model,
                "replacement_policy": POLICIES[model].name,
            }
            for stop_model in STOP_MODELS
            for model in MODELS
        ],
        "primary_hypothesis": (
            "The RSI_Q12_LOSER_ONLY return effect versus no replacement is "
            "positive under 5%/5% stops and greater than its effect under "
            "3.5%/3.5% stops on pristine forward data."
        ),
        "success_gates": {
            "complete_fixed_24_month_horizon": True,
            "minimum_cross_stop_event_pairs": MINIMUM_CROSS_STOP_EVENT_PAIRS,
            "baseline_return_delta_strictly_positive": True,
            "interaction_spread_strictly_positive": True,
            "baseline_return_drawdown_ratio_delta_nonnegative": True,
            "baseline_profit_factor_delta_nonnegative": True,
            "baseline_matched_benchmark_excess_delta_nonnegative": True,
            "maximum_baseline_drawdown_worsening_percent": (
                MAXIMUM_DRAWDOWN_WORSENING_PERCENT
            ),
            "baseline_leave_one_6m_block_out_always_positive": True,
        },
        "robustness_reports": {
            "fixed_block_months": BLOCK_MONTHS,
            "expected_block_count": EXPECTED_BLOCK_COUNT,
            "exact_leave_one_ticker_out": True,
            "ticker_leave_one_out_is_not_an_optional_stopping_gate": True,
        },
        "authorization": {
            "baseline_change_authorized": False,
            "paper_trading_authorized": False,
            "shadow_trading_authorized": False,
            "production_authorized": False,
            "successful_result_requires_human_review": True,
        },
    }


def build_preregistration(source: Mapping[str, Any]) -> dict[str, Any]:
    core = _protocol_core(source)
    protocol_hash = sha256_json(core)
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "registration_id": f"STOP_CONDITIONAL_FORWARD_V1_{protocol_hash[:12]}",
        "protocol_hash": protocol_hash,
        **core,
    }


def _registration_core(registration: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"created_at", "registration_id", "protocol_hash"}
    return {key: value for key, value in registration.items() if key not in excluded}


def validate_preregistration(registration: Mapping[str, Any]) -> None:
    if registration.get("protocol_hash") != sha256_json(
        _registration_core(registration)
    ):
        raise ValueError("Preregistration protocol hash mismatch.")
    if registration.get("protocol_name") != PROTOCOL_NAME:
        raise ValueError("Unexpected preregistration protocol.")
    period = registration.get("evaluation_period", {})
    if period.get("start_inclusive") != FORWARD_START.isoformat():
        raise ValueError("Preregistration forward start was altered.")
    if period.get("end_exclusive") != FORWARD_END_EXCLUSIVE.isoformat():
        raise ValueError("Preregistration forward end was altered.")
    gates = registration.get("success_gates", {})
    if gates.get("minimum_cross_stop_event_pairs") != MINIMUM_CROSS_STOP_EVENT_PAIRS:
        raise ValueError("Preregistration event threshold was altered.")
    if (
        gates.get("maximum_baseline_drawdown_worsening_percent")
        != MAXIMUM_DRAWDOWN_WORSENING_PERCENT
    ):
        raise ValueError("Preregistration drawdown threshold was altered.")
    authorization = registration.get("authorization", {})
    forbidden = (
        "baseline_change_authorized",
        "paper_trading_authorized",
        "shadow_trading_authorized",
        "production_authorized",
    )
    if any(bool(authorization.get(name)) for name in forbidden):
        raise ValueError("Preregistration contains unauthorized execution authority.")


def save_preregistration(
    registration: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> dict[str, Path]:
    validate_preregistration(registration)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    registration_path = output / f"{OUTPUT_PREFIX}_registration_{stamp}.json"
    registration_path.write_text(
        json.dumps(_safe(registration), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    code_path = Path(__file__).resolve()
    provenance = {
        "created_at": datetime.now(UTC).isoformat(),
        "registration_stamp": stamp,
        "registration_id": registration["registration_id"],
        "protocol_hash": registration["protocol_hash"],
        "audit_code": {
            "path": str(code_path),
            "sha256": sha256_file(code_path),
        },
        "source_files": {
            name: {
                "path": str(Path(path).resolve()),
                "sha256": sha256_file(path),
            }
            for name, path in source["paths"].items()
        },
        "result_files": {
            "registration": {
                "path": str(registration_path.resolve()),
                "sha256": sha256_file(registration_path),
            }
        },
    }
    provenance_path = (
        output / f"{OUTPUT_PREFIX}_registration_provenance_{stamp}.json"
    )
    provenance_path.write_text(
        json.dumps(provenance, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"registration": registration_path, "provenance": provenance_path}


def verify_preregistration(
    *, registration_path: Path, provenance_path: Path
) -> dict[str, Any]:
    registration_path = Path(registration_path)
    provenance_path = Path(provenance_path)
    if not registration_path.exists():
        raise FileNotFoundError(registration_path)
    if not provenance_path.exists():
        raise FileNotFoundError(provenance_path)
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    validate_preregistration(registration)
    recorded = provenance.get("result_files", {}).get("registration", {})
    if sha256_file(registration_path) != recorded.get("sha256"):
        raise ValueError("Preregistration file hash mismatch.")
    if provenance.get("registration_id") != registration.get("registration_id"):
        raise ValueError("Preregistration ID mismatch.")
    if provenance.get("protocol_hash") != registration.get("protocol_hash"):
        raise ValueError("Preregistration provenance protocol mismatch.")
    code = provenance.get("audit_code", {})
    code_path = Path(code.get("path", ""))
    if not code_path.exists() or sha256_file(code_path) != code.get("sha256"):
        raise ValueError("Preregistration audit-code hash mismatch.")
    for name, metadata in provenance.get("source_files", {}).items():
        path = Path(metadata["path"])
        if not path.exists() or sha256_file(path) != metadata.get("sha256"):
            raise ValueError(f"Preregistration source hash mismatch: {name}")
    return {
        "registration": registration,
        "provenance": provenance,
        "registration_path": registration_path,
        "provenance_path": provenance_path,
    }


def validate_future_manifest(
    registration: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    future_manifest: Mapping[str, Any],
) -> None:
    source = registration["source"]
    if future_manifest.get("fingerprint") == source.get(
        "interaction_snapshot_fingerprint"
    ):
        raise ValueError("Evaluation requires a newly created future snapshot.")
    if pd.Timestamp(future_manifest["created_at"]) <= pd.Timestamp(
        registration["created_at"]
    ):
        raise ValueError("Future snapshot must be created after preregistration.")
    if future_manifest.get("config", {}).get("sha256") != source["config_sha256"]:
        raise ValueError("Future snapshot config hash drifted from registration.")
    future_tickers = sorted(future_manifest.get("market_files", {}))
    if future_tickers != source["tickers"]:
        raise ValueError("Future snapshot ticker universe drifted from registration.")
    if future_manifest.get("code_files", {}) != source_manifest.get(
        "code_files", {}
    ):
        raise ValueError("Future snapshot frozen code hashes drifted from source.")
    if _market_end(future_manifest) <= OBSERVED_CUTOFF:
        raise ValueError("Future snapshot contains no bars after the cutoff.")


def verify_replay_anchors(
    source_data: Mapping[str, pd.DataFrame],
    future_data: Mapping[str, pd.DataFrame],
    *,
    bars: int = REPLAY_ANCHOR_BARS,
    tolerance: float = REPLAY_FLOAT_TOLERANCE,
) -> None:
    if sorted(source_data) != sorted(future_data):
        raise ValueError("Replay ticker coverage mismatch.")
    for ticker in sorted(source_data):
        source = source_data[ticker].loc[
            source_data[ticker].index <= OBSERVED_CUTOFF
        ].tail(int(bars))
        if len(source) != int(bars):
            raise ValueError(f"{ticker} has fewer than {bars} replay-anchor bars.")
        missing = source.index.difference(future_data[ticker].index)
        if len(missing):
            raise ValueError(f"{ticker} future snapshot misses replay-anchor dates.")
        current = future_data[ticker].loc[source.index, source.columns]
        try:
            pd.testing.assert_frame_equal(
                source,
                current,
                check_dtype=False,
                check_exact=False,
                rtol=float(tolerance),
                atol=float(tolerance),
            )
        except AssertionError as exc:
            raise ValueError(f"{ticker} replay-anchor drift detected.") from exc


def _decorate_frame(
    frame: pd.DataFrame, *, stop_model: str, model: str, period_id: str
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return [
        {
            "period_id": period_id,
            "stop_model": stop_model,
            "model": model,
            "actual_policy": POLICIES[model].name,
            **record,
        }
        for record in frame.to_dict("records")
    ]


def _run_arms(
    *,
    data_by_ticker: dict[str, pd.DataFrame],
    base_config: PortfolioBacktestConfig,
    period_id: str,
    include_details: bool,
) -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    ticker_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    for stop_model in STOP_MODELS:
        for model in MODELS:
            bundle = _run_policy(
                data_by_ticker=data_by_ticker,
                base_config=base_config,
                stop_model=stop_model,
                policy=POLICIES[model],
            )
            summary_rows.append(
                {
                    "period_id": period_id,
                    "stop_model": stop_model,
                    "model": model,
                    "actual_policy": POLICIES[model].name,
                    **bundle["summary"],
                }
            )
            if include_details:
                event_rows.extend(
                    _decorate_frame(
                        bundle["events"],
                        stop_model=stop_model,
                        model=model,
                        period_id=period_id,
                    )
                )
                trade_rows.extend(
                    _decorate_frame(
                        bundle["trades"],
                        stop_model=stop_model,
                        model=model,
                        period_id=period_id,
                    )
                )
                ticker_rows.extend(
                    _decorate_frame(
                        bundle["tickers"],
                        stop_model=stop_model,
                        model=model,
                        period_id=period_id,
                    )
                )
                rejection_rows.extend(
                    {
                        "period_id": period_id,
                        "stop_model": stop_model,
                        "model": model,
                        "actual_policy": POLICIES[model].name,
                        "reason_code": reason,
                        "count": int(count),
                    }
                    for reason, count in sorted(bundle["rejections"].items())
                )
    return {
        "summary": pd.DataFrame(summary_rows),
        "events": pd.DataFrame(event_rows),
        "trades": pd.DataFrame(trade_rows),
        "tickers": pd.DataFrame(ticker_rows),
        "rejections": pd.DataFrame(rejection_rows),
    }


def build_strata_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stop_model in STOP_MODELS:
        group = summary.loc[summary["stop_model"] == stop_model].set_index("model")
        if set(group.index) != set(MODELS):
            raise ValueError(f"Research-arm coverage mismatch: {stop_model}")
        control = group.loc[MODEL_CONTROL]
        candidate = group.loc[MODEL_REPLACEMENT]
        rows.append(
            {
                "stop_model": stop_model,
                "control_return_percent": control["total_return_percent"],
                "candidate_return_percent": candidate["total_return_percent"],
                "return_delta_vs_control_percent": round(
                    float(candidate["total_return_percent"])
                    - float(control["total_return_percent"]),
                    6,
                ),
                "control_maximum_drawdown_percent": control[
                    "maximum_drawdown_percent"
                ],
                "candidate_maximum_drawdown_percent": candidate[
                    "maximum_drawdown_percent"
                ],
                "drawdown_worsening_percent": round(
                    float(candidate["maximum_drawdown_percent"])
                    - float(control["maximum_drawdown_percent"]),
                    6,
                ),
                "return_drawdown_ratio_delta": round(
                    float(candidate["return_drawdown_ratio"])
                    - float(control["return_drawdown_ratio"]),
                    6,
                ),
                "profit_factor_delta": round(
                    float(candidate["profit_factor"])
                    - float(control["profit_factor"]),
                    6,
                ),
                "matched_benchmark_excess_delta_percent": round(
                    float(candidate["excess_return_vs_matched_percent"])
                    - float(control["excess_return_vs_matched_percent"]),
                    6,
                ),
                "control_completed_trades": int(control["completed_trades"]),
                "candidate_completed_trades": int(candidate["completed_trades"]),
                "executed_replacements": int(candidate["executed_replacements"]),
            }
        )
    output = pd.DataFrame(rows)
    baseline_delta = float(
        output.loc[
            output["stop_model"] == STOP_BASELINE,
            "return_delta_vs_control_percent",
        ].iloc[0]
    )
    maximum_delta = float(
        output.loc[
            output["stop_model"] == STOP_MAX_RETURN,
            "return_delta_vs_control_percent",
        ].iloc[0]
    )
    output["interaction_spread_baseline_minus_max_return_percent"] = round(
        baseline_delta - maximum_delta, 6
    )
    return output


def _compound(values: Sequence[float]) -> float:
    capital = 1.0
    for value in values:
        capital *= 1 + float(value) / 100
    return round((capital - 1) * 100, 6)


def fixed_blocks() -> tuple[tuple[str, pd.Timestamp, pd.Timestamp], ...]:
    blocks = []
    start = FORWARD_START
    for index in range(EXPECTED_BLOCK_COUNT):
        end = start + pd.DateOffset(months=BLOCK_MONTHS)
        blocks.append((f"B{index + 1:02d}", start, end))
        start = end
    if start != FORWARD_END_EXCLUSIVE:
        raise RuntimeError("Fixed forward blocks do not cover the horizon exactly.")
    return tuple(blocks)


def build_block_leave_one_out(blocks: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "stop_model",
        "excluded_block_id",
        "candidate_compounded_return_percent",
        "control_compounded_return_percent",
        "return_delta_vs_control_percent",
        "positive_after_exclusion",
    ]
    if blocks.empty:
        return pd.DataFrame(columns=columns)
    expected = {block[0] for block in fixed_blocks()}
    if set(blocks["period_id"].unique()) != expected:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for stop_model in STOP_MODELS:
        group = blocks.loc[blocks["stop_model"] == stop_model]
        pivot = group.pivot(
            index="period_id", columns="model", values="total_return_percent"
        ).sort_index()
        if set(pivot.columns) != set(MODELS) or set(pivot.index) != expected:
            raise ValueError(f"Block research-arm coverage mismatch: {stop_model}")
        for excluded in sorted(expected):
            reduced = pivot.drop(index=excluded)
            candidate = _compound(reduced[MODEL_REPLACEMENT].tolist())
            control = _compound(reduced[MODEL_CONTROL].tolist())
            delta = round(candidate - control, 6)
            rows.append(
                {
                    "stop_model": stop_model,
                    "excluded_block_id": excluded,
                    "candidate_compounded_return_percent": candidate,
                    "control_compounded_return_percent": control,
                    "return_delta_vs_control_percent": delta,
                    "positive_after_exclusion": delta > 1e-9,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_forward_event_pairs(events: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "event_pair_id",
        "timestamp",
        "candidate_ticker",
        "victim_ticker_baseline",
        "victim_ticker_max_return",
        "present_in_both_stops",
        "same_victim",
    ]
    if events.empty or "status" not in events:
        return pd.DataFrame(columns=columns)
    executed = events.loc[
        (events["model"] == MODEL_REPLACEMENT)
        & (events["status"].astype(str) == "EXECUTED")
    ]
    keys = ["timestamp", "candidate_ticker"]
    values = keys + ["victim_ticker"]
    baseline = executed.loc[
        executed["stop_model"] == STOP_BASELINE, values
    ].drop_duplicates(keys, keep="last")
    maximum = executed.loc[
        executed["stop_model"] == STOP_MAX_RETURN, values
    ].drop_duplicates(keys, keep="last")
    paired = baseline.merge(
        maximum,
        on=keys,
        how="outer",
        suffixes=("_baseline", "_max_return"),
        indicator=True,
    )
    paired.insert(
        0,
        "event_pair_id",
        [f"FEP{index:04d}" for index in range(1, len(paired) + 1)],
    )
    paired["present_in_both_stops"] = paired["_merge"] == "both"
    paired["same_victim"] = (
        paired["victim_ticker_baseline"]
        == paired["victim_ticker_max_return"]
    ) & paired["present_in_both_stops"]
    return paired.drop(columns="_merge").reindex(columns=columns)


def _common_market_end(manifest: Mapping[str, Any]) -> pd.Timestamp:
    return min(
        pd.Timestamp(metadata["end"])
        for metadata in manifest["market_files"].values()
    )


def _horizon_complete(manifest: Mapping[str, Any]) -> bool:
    final_inclusive = FORWARD_END_EXCLUSIVE - pd.Timedelta(days=1)
    return _common_market_end(manifest) >= final_inclusive


def build_forward_screen(
    *,
    strata: pd.DataFrame,
    block_leave_one_out: pd.DataFrame,
    ticker_leave_one_out: pd.DataFrame,
    event_pairs: pd.DataFrame,
    horizon_complete: bool,
) -> pd.DataFrame:
    by_stop = strata.set_index("stop_model")
    baseline = by_stop.loc[STOP_BASELINE]
    maximum = by_stop.loc[STOP_MAX_RETURN]
    paired = int(
        event_pairs.get("present_in_both_stops", pd.Series(dtype=bool)).sum()
    )
    baseline_low = block_leave_one_out.loc[
        block_leave_one_out.get(
            "stop_model", pd.Series(index=block_leave_one_out.index, dtype=str)
        )
        == STOP_BASELINE,
        "return_delta_vs_control_percent",
    ]
    ticker_low = ticker_leave_one_out.loc[
        ticker_leave_one_out.get(
            "stop_model", pd.Series(index=ticker_leave_one_out.index, dtype=str)
        )
        == STOP_BASELINE,
        "return_delta_vs_control_percent",
    ]
    criteria = {
        "complete_fixed_24_month_horizon": bool(horizon_complete),
        "at_least_5_cross_stop_event_pairs": paired
        >= MINIMUM_CROSS_STOP_EVENT_PAIRS,
        "baseline_return_delta_strictly_positive": baseline[
            "return_delta_vs_control_percent"
        ]
        > 1e-9,
        "interaction_spread_strictly_positive": baseline[
            "return_delta_vs_control_percent"
        ]
        - maximum["return_delta_vs_control_percent"]
        > 1e-9,
        "baseline_return_drawdown_ratio_delta_nonnegative": baseline[
            "return_drawdown_ratio_delta"
        ]
        >= -1e-9,
        "baseline_profit_factor_delta_nonnegative": baseline[
            "profit_factor_delta"
        ]
        >= -1e-9,
        "baseline_matched_benchmark_excess_delta_nonnegative": baseline[
            "matched_benchmark_excess_delta_percent"
        ]
        >= -1e-9,
        "baseline_drawdown_worsening_within_2_5_percent": baseline[
            "drawdown_worsening_percent"
        ]
        <= MAXIMUM_DRAWDOWN_WORSENING_PERCENT + 1e-9,
        "baseline_leave_one_6m_block_out_always_positive": (
            len(baseline_low) == EXPECTED_BLOCK_COUNT
            and bool((baseline_low > 1e-9).all())
        ),
    }
    passed = bool(all(criteria.values()))
    if not horizon_complete:
        status = "MONITORING"
    elif paired < MINIMUM_CROSS_STOP_EVENT_PAIRS:
        status = "INCONCLUSIVE_INSUFFICIENT_EVENTS"
    elif passed:
        status = "HYPOTHESIS_SUPPORTED_READY_FOR_HUMAN_REVIEW"
    else:
        status = "HYPOTHESIS_NOT_SUPPORTED"
    return pd.DataFrame(
        [
            {
                **criteria,
                "criteria_passed": int(sum(bool(value) for value in criteria.values())),
                "criteria_total": len(criteria),
                "cross_stop_event_pair_count": paired,
                "same_victim_cross_stop_event_pair_count": int(
                    event_pairs.get("same_victim", pd.Series(dtype=bool)).sum()
                ),
                "baseline_return_delta_percent": baseline[
                    "return_delta_vs_control_percent"
                ],
                "max_return_return_delta_percent": maximum[
                    "return_delta_vs_control_percent"
                ],
                "interaction_spread_percent": round(
                    float(baseline["return_delta_vs_control_percent"])
                    - float(maximum["return_delta_vs_control_percent"]),
                    6,
                ),
                "baseline_block_loo_min_return_delta_percent": (
                    float(baseline_low.min()) if len(baseline_low) else None
                ),
                "baseline_ticker_loo_min_return_delta_percent": (
                    float(ticker_low.min()) if len(ticker_low) else None
                ),
                "research_hypothesis_passed": passed,
                "decision_status": status,
                "conditional_policy_authorized": False,
                "baseline_change_authorized": False,
                "paper_trading_authorized": False,
                "shadow_trading_authorized": False,
                "production_authorized": False,
            }
        ]
    )


def _empty_ticker_loo() -> pd.DataFrame:
    return pd.DataFrame(
        columns=(
            "excluded_ticker",
            "stop_model",
            "control_return_percent",
            "candidate_return_percent",
            "return_delta_vs_control_percent",
        )
    )


def run_forward_evaluation(
    *,
    registration_path: Path,
    registration_provenance_path: Path,
    future_snapshot_path: Path,
    project_root: Path = Path("."),
) -> dict[str, Any]:
    locked = verify_preregistration(
        registration_path=registration_path,
        provenance_path=registration_provenance_path,
    )
    registration = locked["registration"]
    source_manifest_path = Path(registration["source"]["snapshot_manifest_path"])
    if (
        not source_manifest_path.exists()
        or sha256_file(source_manifest_path)
        != registration["source"]["snapshot_manifest_sha256"]
    ):
        raise ValueError("Registered source snapshot manifest hash mismatch.")
    source_snapshot = load_snapshot(
        source_manifest_path.parent,
        verify_code=True,
        project_root=Path(project_root),
    )
    future_snapshot = load_snapshot(
        Path(future_snapshot_path),
        verify_code=True,
        project_root=Path(project_root),
    )
    validate_future_manifest(
        registration,
        source_snapshot["manifest"],
        future_snapshot["manifest"],
    )
    verify_replay_anchors(
        source_snapshot["data_by_ticker"], future_snapshot["data_by_ticker"]
    )

    common_end = _common_market_end(future_snapshot["manifest"])
    observed_end_exclusive = min(
        FORWARD_END_EXCLUSIVE, common_end.normalize() + pd.Timedelta(days=1)
    )
    if observed_end_exclusive <= FORWARD_START:
        raise ValueError("Future snapshot has no common post-cutoff interval.")
    future_data = slice_prepared_data(
        future_snapshot["data_by_ticker"],
        start=FORWARD_START,
        end_exclusive=observed_end_exclusive,
    )
    full = _run_arms(
        data_by_ticker=future_data,
        base_config=future_snapshot["config"],
        period_id="FORWARD_TO_DATE",
        include_details=True,
    )
    strata = build_strata_comparison(full["summary"])
    event_pairs = build_forward_event_pairs(full["events"])

    block_summaries: list[pd.DataFrame] = []
    for block_id, start, end in fixed_blocks():
        if end > observed_end_exclusive:
            continue
        block_data = slice_prepared_data(
            future_snapshot["data_by_ticker"], start=start, end_exclusive=end
        )
        block_summaries.append(
            _run_arms(
                data_by_ticker=block_data,
                base_config=future_snapshot["config"],
                period_id=block_id,
                include_details=False,
            )["summary"]
        )
    blocks = (
        pd.concat(block_summaries, ignore_index=True)
        if block_summaries
        else pd.DataFrame(columns=full["summary"].columns)
    )
    block_loo = build_block_leave_one_out(blocks)

    complete = _horizon_complete(future_snapshot["manifest"])
    ticker_loo = _empty_ticker_loo()
    if complete:
        rows: list[dict[str, Any]] = []
        for excluded in sorted(future_data):
            reduced = {
                ticker: frame
                for ticker, frame in future_data.items()
                if ticker != excluded
            }
            result = _run_arms(
                data_by_ticker=reduced,
                base_config=future_snapshot["config"],
                period_id=f"WITHOUT_{excluded}",
                include_details=False,
            )
            comparison = build_strata_comparison(result["summary"])
            for record in comparison.to_dict("records"):
                rows.append({"excluded_ticker": excluded, **record})
        ticker_loo = pd.DataFrame(rows)

    screen = build_forward_screen(
        strata=strata,
        block_leave_one_out=block_loo,
        ticker_leave_one_out=ticker_loo,
        event_pairs=event_pairs,
        horizon_complete=complete,
    )
    summary = screen.copy()
    summary.insert(0, "protocol_hash", registration["protocol_hash"])
    summary.insert(1, "forward_start", FORWARD_START.isoformat())
    summary.insert(2, "observed_end_exclusive", observed_end_exclusive.isoformat())
    summary.insert(3, "fixed_end_exclusive", FORWARD_END_EXCLUSIVE.isoformat())
    return {
        "summary": summary,
        "strata": strata,
        "blocks": blocks,
        "block_leave_one_out": block_loo,
        "ticker_leave_one_out": ticker_loo,
        "events": full["events"],
        "event_pairs": event_pairs,
        "trades": full["trades"],
        "tickers": full["tickers"],
        "rejections": full["rejections"],
        "screen": screen,
        "registration": registration,
        "registration_path": Path(registration_path),
        "registration_provenance_path": Path(registration_provenance_path),
        "future_snapshot_id": future_snapshot["manifest"]["snapshot_id"],
        "future_snapshot_fingerprint": future_snapshot["manifest"]["fingerprint"],
        "future_snapshot_manifest_path": Path(future_snapshot_path) / "manifest.json",
        "observed_end_exclusive": observed_end_exclusive,
        "horizon_complete": complete,
    }


def save_forward_evaluation(
    bundle: Mapping[str, Any],
    *,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> dict[str, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    paths: dict[str, Path] = {}
    for name in EVALUATION_FRAME_NAMES:
        path = output / f"{OUTPUT_PREFIX}_{name}_{stamp}.csv"
        bundle[name].to_csv(path, index=False, lineterminator="\n")
        paths[name] = path
    screen_row = bundle["screen"].iloc[0].to_dict()
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_name": PROTOCOL_NAME,
        "protocol_hash": bundle["registration"]["protocol_hash"],
        "registration_id": bundle["registration"]["registration_id"],
        "future_snapshot_id": bundle["future_snapshot_id"],
        "future_snapshot_fingerprint": bundle["future_snapshot_fingerprint"],
        "forward_start": FORWARD_START.isoformat(),
        "observed_end_exclusive": bundle["observed_end_exclusive"].isoformat(),
        "fixed_end_exclusive": FORWARD_END_EXCLUSIVE.isoformat(),
        "horizon_complete": bundle["horizon_complete"],
        "screen": _safe(screen_row),
        "conditional_policy_authorized": False,
        "baseline_change_authorized": False,
        "paper_trading_authorized": False,
        "shadow_trading_authorized": False,
        "production_authorized": False,
        "limitations": [
            "Interim results are monitoring only and cannot stop the protocol early.",
            "An insufficient event count is inconclusive, not a pass.",
            "A supported result is only ready for separate human review.",
            "This module contains no broker or order-routing integration.",
        ],
    }
    json_path = output / f"{OUTPUT_PREFIX}_{stamp}.json"
    json_path.write_text(
        json.dumps(_safe(payload), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["json"] = json_path
    code_path = Path(__file__).resolve()
    provenance = {
        "created_at": datetime.now(UTC).isoformat(),
        "audit_stamp": stamp,
        "protocol_hash": bundle["registration"]["protocol_hash"],
        "audit_code": {
            "path": str(code_path),
            "sha256": sha256_file(code_path),
        },
        "registration_files": {
            "registration": {
                "path": str(bundle["registration_path"].resolve()),
                "sha256": sha256_file(bundle["registration_path"]),
            },
            "provenance": {
                "path": str(bundle["registration_provenance_path"].resolve()),
                "sha256": sha256_file(bundle["registration_provenance_path"]),
            },
        },
        "future_snapshot_manifest": {
            "path": str(bundle["future_snapshot_manifest_path"].resolve()),
            "sha256": sha256_file(bundle["future_snapshot_manifest_path"]),
        },
        "result_files": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    provenance_path = output / f"{OUTPUT_PREFIX}_provenance_{stamp}.json"
    provenance_path.write_text(
        json.dumps(provenance, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["provenance"] = provenance_path
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    register = subparsers.add_parser("register", help="Freeze the protocol now.")
    register.add_argument(
        "--interaction-source-directory",
        type=Path,
        default=DEFAULT_INTERACTION_DIRECTORY,
    )
    register.add_argument("--interaction-stamp", required=True)
    register.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    register.add_argument("--no-save", action="store_true")

    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate a post-registration future snapshot."
    )
    evaluate.add_argument("--registration", type=Path, required=True)
    evaluate.add_argument(
        "--registration-provenance", type=Path, required=True
    )
    evaluate.add_argument("--snapshot", type=Path, required=True)
    evaluate.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    evaluate.add_argument("--no-save", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "register":
        source = verify_interaction_source(
            directory=args.interaction_source_directory,
            stamp=args.interaction_stamp,
        )
        registration = build_preregistration(source)
        print(json.dumps(_safe(registration), sort_keys=True, indent=2))
        if args.no_save:
            print("NO_SAVE")
            return
        paths = save_preregistration(
            registration, source=source, output_directory=args.output_directory
        )
    else:
        bundle = run_forward_evaluation(
            registration_path=args.registration,
            registration_provenance_path=args.registration_provenance,
            future_snapshot_path=args.snapshot,
            project_root=Path("."),
        )
        print("\nSTRATA")
        print(bundle["strata"].to_string(index=False))
        print("\nSCREEN")
        print(bundle["screen"].to_string(index=False))
        if args.no_save:
            print("NO_SAVE")
            return
        paths = save_forward_evaluation(
            bundle, output_directory=args.output_directory
        )
    for name, path in paths.items():
        print(name, path.resolve())


if __name__ == "__main__":
    main()
