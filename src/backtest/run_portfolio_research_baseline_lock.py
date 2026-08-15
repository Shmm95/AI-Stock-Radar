"""Verify and certify the immutable AI-Stock-Radar research baseline.

This module does not run a strategy search or change trading behavior.  It
verifies a pre-registered baseline manifest, the approved project files, and
the complete Execution and Slippage Stress V1 provenance chain.  A successful
certificate locks only a historical research reference for the controlled
nine-asset basket; it grants no paper, production, broker, or automation
authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from math import isclose, isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.backtest.portfolio_backtest_models import PortfolioBacktestConfig


SCHEMA_VERSION = 1
LOCK_NAME = "AI_STOCK_RADAR_RESEARCH_BASELINE_V1"
LOCK_STATUS = "RESEARCH_REFERENCE_LOCKED"
APPROVED_EXECUTION_STRESS_STAMP = "20260814_054701"
APPROVED_SNAPSHOT_ID = "20260802_081049_d93145cb1dcf"
APPROVED_SNAPSHOT_FINGERPRINT = (
    "d93145cb1dcf3f837415f5b28ac29eeb51c13bc8ce5b8b1187aa34ebfc01ad44"
)
APPROVED_HOLDING_PATH_STAMP = "20260802_155749"
APPROVED_STOP_WALK_FORWARD_STAMP = "20260802_081449"
BASELINE_SCENARIO_ID = "C01_S01"
PRIMARY_STRESS_SCENARIO_ID = "C02_S02"
CONTROLLED_TICKERS = (
    "AAPL",
    "AMZN",
    "BTC-USD",
    "ETH-USD",
    "GOOGL",
    "META",
    "MSFT",
    "NVDA",
    "TSLA",
)
DEFAULT_REGISTRY_PATH = Path("config/research_baseline_lock_v1.json")
DEFAULT_EXECUTION_STRESS_DIRECTORY = Path(
    "data/backtests/portfolio/execution_slippage_stress"
)
DEFAULT_OUTPUT_DIRECTORY = Path(
    "data/backtests/portfolio/research_baseline_lock"
)
FLOAT_TOLERANCE = 1e-8
_STRICT_SOURCE_AUTHORIZATION = object()

STRESS_RESULT_KEYS = (
    "aggregate",
    "comparison",
    "equity",
    "json",
    "primary_gates",
    "rejections",
    "replay_checks",
    "screen",
    "test_runs",
    "tickers",
    "trade_pairs",
    "trades",
    "windows",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _float_equal(actual: Any, expected: Any) -> bool:
    return isclose(
        float(actual),
        float(expected),
        rel_tol=0.0,
        abs_tol=FLOAT_TOLERANCE,
    )


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def validate_registry(registry: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "lock_name",
        "lock_id",
        "status",
        "scope",
        "lineage",
        "controlled_tickers",
        "baseline_config",
        "approved_project_files",
        "execution_stress",
        "reference_metrics",
        "known_limitations",
        "authorization",
        "following_stage",
    }
    missing = sorted(required.difference(registry))
    if missing:
        raise ValueError(f"Baseline registry fields are missing: {missing}")
    if int(registry["schema_version"]) != SCHEMA_VERSION:
        raise ValueError("Baseline registry schema version mismatch.")
    if registry["lock_name"] != LOCK_NAME:
        raise ValueError("Baseline registry name mismatch.")
    if registry["status"] != LOCK_STATUS:
        raise ValueError("Baseline registry status mismatch.")
    if registry["scope"] != "CONTROLLED_NINE_ASSET_HISTORICAL_RESEARCH_ONLY":
        raise ValueError("Baseline registry scope mismatch.")
    if tuple(sorted(registry["controlled_tickers"])) != CONTROLLED_TICKERS:
        raise ValueError("Baseline registry controlled universe mismatch.")

    lineage = registry["lineage"]
    expected_lineage = {
        "snapshot_id": APPROVED_SNAPSHOT_ID,
        "snapshot_fingerprint": APPROVED_SNAPSHOT_FINGERPRINT,
        "holding_path_attribution_stamp": APPROVED_HOLDING_PATH_STAMP,
        "stop_walk_forward_stamp": APPROVED_STOP_WALK_FORWARD_STAMP,
        "execution_slippage_stress_stamp": APPROVED_EXECUTION_STRESS_STAMP,
    }
    for field, expected in expected_lineage.items():
        if lineage.get(field) != expected:
            raise ValueError(f"Baseline registry lineage mismatch: {field}")

    authorization = registry["authorization"]
    forbidden = (
        "baseline_parameter_change_authorized",
        "broader_universe_validated",
        "paper_trading_authorized",
        "production_authorized",
        "broker_access_authorized",
        "automation_authorized",
    )
    if any(bool(authorization.get(field)) for field in forbidden):
        raise ValueError("Baseline registry grants forbidden execution authority.")
    if registry["following_stage"] != "BROADER_UNIVERSE_ROBUSTNESS_SPECIFICATION_ONLY":
        raise ValueError("Baseline registry following-stage mismatch.")

    project_files = registry["approved_project_files"]
    if not isinstance(project_files, Mapping) or not project_files:
        raise ValueError("Baseline registry has no approved project-file hashes.")
    if any(not _is_sha256(value) for value in project_files.values()):
        raise ValueError("Baseline registry contains an invalid project-file hash.")

    stress = registry["execution_stress"]
    if stress.get("stamp") != APPROVED_EXECUTION_STRESS_STAMP:
        raise ValueError("Baseline registry execution-stress stamp mismatch.")
    if set(stress.get("result_hashes", {})) != set(STRESS_RESULT_KEYS):
        raise ValueError("Baseline registry execution-stress result set mismatch.")
    if not _is_sha256(stress.get("provenance_sha256", "")):
        raise ValueError("Baseline registry execution-stress provenance hash is invalid.")


def load_registry(path: Path) -> dict[str, Any]:
    registry = _read_json(Path(path))
    validate_registry(registry)
    return registry


def _stress_paths(directory: Path, stamp: str) -> dict[str, Path]:
    directory = Path(directory)
    prefix = "portfolio_execution_slippage_stress"
    paths = {
        name: directory / f"{prefix}_{name}_{stamp}.csv"
        for name in STRESS_RESULT_KEYS
        if name != "json"
    }
    paths["json"] = directory / f"{prefix}_{stamp}.json"
    paths["provenance"] = directory / f"{prefix}_provenance_{stamp}.json"
    return paths


def verify_project_files(
    registry: Mapping[str, Any], project_root: Path
) -> dict[str, dict[str, str]]:
    verified: dict[str, dict[str, str]] = {}
    for relative, expected in sorted(registry["approved_project_files"].items()):
        path = Path(project_root) / str(relative)
        if not path.is_file():
            raise FileNotFoundError(f"Approved project file is missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Approved project file hash mismatch: {relative}")
        verified[f"project:{relative}"] = {
            "path": str(path.resolve()),
            "sha256": actual,
        }
    return verified


def _verify_recorded_file(
    *, name: str, path: Path, recorded: Mapping[str, Any]
) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Recorded source is missing: {name}: {path}")
    actual = sha256_file(path)
    if actual != recorded.get("sha256"):
        raise ValueError(f"Recorded source hash mismatch: {name}")
    return {"path": str(path.resolve()), "sha256": actual}


def verify_execution_stress(
    *, registry: Mapping[str, Any], directory: Path, stamp: str
) -> dict[str, Any]:
    if stamp != APPROVED_EXECUTION_STRESS_STAMP:
        raise ValueError("Only the approved execution-stress stamp may be locked.")
    paths = _stress_paths(Path(directory), stamp)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Execution-stress files are missing: " + ", ".join(missing))

    stress_registration = registry["execution_stress"]
    provenance_path = paths["provenance"]
    if sha256_file(provenance_path) != stress_registration["provenance_sha256"]:
        raise ValueError("Execution-stress provenance hash mismatch.")
    provenance = _read_json(provenance_path)
    if provenance.get("execution_slippage_stress_stamp") != stamp:
        raise ValueError("Execution-stress provenance stamp mismatch.")
    if provenance.get("snapshot_id") != APPROVED_SNAPSHOT_ID:
        raise ValueError("Execution-stress snapshot id mismatch.")
    if provenance.get("snapshot_fingerprint") != APPROVED_SNAPSHOT_FINGERPRINT:
        raise ValueError("Execution-stress snapshot fingerprint mismatch.")
    if provenance.get("holding_path_attribution_stamp") != APPROVED_HOLDING_PATH_STAMP:
        raise ValueError("Execution-stress holding-path stamp mismatch.")
    if provenance.get("stop_walk_forward_stamp") != APPROVED_STOP_WALK_FORWARD_STAMP:
        raise ValueError("Execution-stress stop-walk-forward stamp mismatch.")

    recorded_results = provenance.get("result_files", {})
    if set(recorded_results) != set(STRESS_RESULT_KEYS):
        raise ValueError("Execution-stress provenance result set mismatch.")
    if recorded_results != stress_registration["result_hashes"]:
        raise ValueError("Execution-stress registered result hashes mismatch.")

    verified_files: dict[str, dict[str, str]] = {
        "stress:provenance": {
            "path": str(provenance_path.resolve()),
            "sha256": sha256_file(provenance_path),
        }
    }
    for name in STRESS_RESULT_KEYS:
        path = paths[name]
        record = recorded_results[name]
        actual = sha256_file(path)
        if actual != record.get("sha256"):
            raise ValueError(f"Execution-stress result hash mismatch: {name}")
        verified_files[f"stress:result:{name}"] = {
            "path": str(path.resolve()),
            "sha256": actual,
        }

    source_records = provenance.get("source_files", {})
    if not isinstance(source_records, Mapping) or len(source_records) < 1:
        raise ValueError("Execution-stress provenance has no upstream sources.")
    for name, record in sorted(source_records.items()):
        path = Path(record["path"])
        verified_files[f"stress:source:{name}"] = _verify_recorded_file(
            name=name, path=path, recorded=record
        )

    return {
        "paths": paths,
        "provenance": provenance,
        "payload": _read_json(paths["json"]),
        "aggregate": pd.read_csv(paths["aggregate"]),
        "comparison": pd.read_csv(paths["comparison"]),
        "primary_gates": pd.read_csv(paths["primary_gates"]),
        "replay_checks": pd.read_csv(paths["replay_checks"]),
        "screen": pd.read_csv(paths["screen"]),
        "test_runs": pd.read_csv(paths["test_runs"]),
        "tickers": pd.read_csv(paths["tickers"]),
        "trades": pd.read_csv(paths["trades"]),
        "windows": pd.read_csv(paths["windows"]),
        "verified_files": verified_files,
    }


def verify_runtime_config(registry: Mapping[str, Any]) -> None:
    actual = PortfolioBacktestConfig().to_dict()
    expected = dict(registry["baseline_config"])
    if set(actual) != set(expected):
        raise ValueError("Runtime baseline config field set drifted.")
    for field, expected_value in expected.items():
        actual_value = actual[field]
        if isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool):
            if not _float_equal(actual_value, expected_value):
                raise ValueError(f"Runtime baseline config drifted: {field}")
        elif actual_value != expected_value:
            raise ValueError(f"Runtime baseline config drifted: {field}")


def _one_scenario(frame: pd.DataFrame, scenario_id: str) -> pd.Series:
    rows = frame.loc[frame["scenario_id"] == scenario_id]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one aggregate row: {scenario_id}")
    return rows.iloc[0]


def _verify_metric_set(
    *, row: pd.Series, expected: Mapping[str, Any], label: str
) -> None:
    for field, expected_value in expected.items():
        if field not in row.index:
            raise ValueError(f"Registered metric is missing for {label}: {field}")
        actual = row[field]
        if isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool):
            if not _float_equal(actual, expected_value):
                raise ValueError(f"Registered metric mismatch for {label}: {field}")
        elif actual != expected_value:
            raise ValueError(f"Registered metric mismatch for {label}: {field}")


def validate_stress_decision(
    registry: Mapping[str, Any], source: Mapping[str, Any]
) -> pd.DataFrame:
    payload = source["payload"]
    summary = payload.get("summary", {})
    expected_summary = {
        "scenario_count": 16,
        "window_count": 13,
        "test_run_count": 208,
        "baseline_replay_passed": True,
        "quality_screen_passed": True,
        "all_primary_gates_passed": True,
        "selected_scenario": None,
        "baseline_change_authorized": False,
        "paper_or_production_authorized": False,
    }
    for field, expected in expected_summary.items():
        if summary.get(field) != expected:
            raise ValueError(f"Execution-stress summary mismatch: {field}")

    if payload.get("base_config") != registry["baseline_config"]:
        raise ValueError("Execution-stress baseline config mismatches the registry.")
    screen = source["screen"]
    if len(screen) != 6 or not bool(screen["passed"].map(_truthy).all()):
        raise ValueError("Execution-stress quality screen did not pass exactly.")
    if bool(screen["authorizes_baseline_change"].map(_truthy).any()):
        raise ValueError("Execution-stress quality screen authorized a baseline change.")
    gates = source["primary_gates"]
    if len(gates) != 4 or not bool(gates["passed"].map(_truthy).all()):
        raise ValueError("Execution-stress primary gates did not all pass.")
    if bool(gates["authorizes_baseline_change"].map(_truthy).any()):
        raise ValueError("Execution-stress primary gates authorized a baseline change.")
    if bool(gates["authorizes_paper_or_production"].map(_truthy).any()):
        raise ValueError("Execution-stress primary gates authorized deployment.")

    replay = source["replay_checks"]
    if len(replay) != 182 or not bool(replay["passed"].map(_truthy).all()):
        raise ValueError("Execution-stress replay checks did not all pass.")
    replay_differences = pd.to_numeric(replay["difference"], errors="coerce")
    if replay_differences.isna().any():
        raise ValueError("Execution-stress replay differences are not finite numeric values.")
    if float(replay_differences.abs().max()) > FLOAT_TOLERANCE:
        raise ValueError("Execution-stress replay tolerance was exceeded.")
    if len(source["windows"]) != 13 or len(source["test_runs"]) != 208:
        raise ValueError("Execution-stress window/run coverage mismatch.")
    if set(source["aggregate"]["scenario_id"]) != set(
        registry["execution_stress"]["scenario_ids"]
    ):
        raise ValueError("Execution-stress scenario grid mismatch.")
    observed_tickers = tuple(sorted(source["trades"]["ticker"].unique()))
    if observed_tickers != CONTROLLED_TICKERS:
        raise ValueError("Execution-stress trade universe mismatch.")

    metrics = registry["reference_metrics"]
    baseline = _one_scenario(source["comparison"], BASELINE_SCENARIO_ID)
    primary = _one_scenario(source["comparison"], PRIMARY_STRESS_SCENARIO_ID)
    _verify_metric_set(row=baseline, expected=metrics["baseline"], label="baseline")
    _verify_metric_set(row=primary, expected=metrics["primary_stress"], label="primary")

    checks = [
        ("registry_contract", "static registry schema, scope and authority"),
        ("project_file_hashes", "approved project revisions"),
        ("stress_provenance_hash", "registered stress provenance"),
        ("stress_result_hashes", "all 13 stress result files"),
        ("upstream_source_hashes", "full recorded predecessor chain"),
        ("snapshot_lineage", "snapshot, holding-path and stop stamps"),
        ("runtime_baseline_config", "runtime defaults equal registered config"),
        ("official_baseline_replay", "182 of 182 replay comparisons"),
        ("fixed_stress_grid", "16 scenarios and 208 test runs"),
        ("primary_robustness_gates", "four of four primary gates"),
        ("controlled_universe", "exact controlled nine-asset basket"),
        ("reference_metrics", "baseline and primary reference metrics"),
        ("no_selection_or_execution_authority", "research-only decision boundary"),
    ]
    return pd.DataFrame(
        [
            {
                "check": name,
                "passed": True,
                "detail": detail,
                "authorizes_baseline_parameter_change": False,
                "authorizes_paper_or_production": False,
            }
            for name, detail in checks
        ]
    )


def build_lock_record(
    *,
    registry: Mapping[str, Any],
    registry_path: Path,
    stress_source: Mapping[str, Any],
    checks: pd.DataFrame,
    stamp: str,
) -> dict[str, Any]:
    if not bool(checks["passed"].all()):
        raise ValueError("Research baseline lock checks did not all pass.")
    if bool(checks["authorizes_baseline_parameter_change"].any()):
        raise ValueError("Research baseline lock attempted to authorize a parameter change.")
    if bool(checks["authorizes_paper_or_production"].any()):
        raise ValueError("Research baseline lock attempted to authorize deployment.")
    return {
        "schema_version": SCHEMA_VERSION,
        "lock_name": LOCK_NAME,
        "lock_id": registry["lock_id"],
        "verification_stamp": stamp,
        "verified_at": datetime.now(UTC).isoformat(),
        "status": LOCK_STATUS,
        "scope": registry["scope"],
        "registry": {
            "path": str(Path(registry_path).resolve()),
            "sha256": sha256_file(Path(registry_path)),
        },
        "lineage": dict(registry["lineage"]),
        "controlled_tickers": list(registry["controlled_tickers"]),
        "baseline_config": dict(registry["baseline_config"]),
        "reference_metrics": dict(registry["reference_metrics"]),
        "verification": {
            "all_checks_passed": True,
            "check_count": int(len(checks)),
            "approved_project_file_count": len(registry["approved_project_files"]),
            "stress_result_file_count": len(STRESS_RESULT_KEYS),
            "upstream_source_file_count": len(
                stress_source["provenance"]["source_files"]
            ),
            "baseline_replay_check_count": int(len(stress_source["replay_checks"])),
        },
        "known_limitations": list(registry["known_limitations"]),
        "authorization": dict(registry["authorization"]),
        "following_stage": registry["following_stage"],
    }


def load_verified_source(
    *,
    registry_path: Path,
    execution_stress_directory: Path,
    execution_stress_stamp: str,
    project_root: Path,
) -> dict[str, Any]:
    registry_path = Path(registry_path).resolve()
    project_root = Path(project_root).resolve()
    registry = load_registry(registry_path)
    project_files = verify_project_files(registry, project_root)
    verify_runtime_config(registry)
    stress = verify_execution_stress(
        registry=registry,
        directory=execution_stress_directory,
        stamp=execution_stress_stamp,
    )
    verified_files = {
        "registry": {
            "path": str(registry_path),
            "sha256": sha256_file(registry_path),
        },
        **project_files,
        **stress["verified_files"],
    }
    return {
        "registry": registry,
        "registry_path": registry_path,
        "project_root": project_root,
        "stress": stress,
        "verified_files": verified_files,
        "verified_hashes": {
            name: metadata["sha256"] for name, metadata in verified_files.items()
        },
        "save_authorization": _STRICT_SOURCE_AUTHORIZATION,
    }


def run_research_baseline_lock(source: Mapping[str, Any]) -> dict[str, Any]:
    checks = validate_stress_decision(source["registry"], source["stress"])
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    record = build_lock_record(
        registry=source["registry"],
        registry_path=source["registry_path"],
        stress_source=source["stress"],
        checks=checks,
        stamp=stamp,
    )
    return {
        "stamp": stamp,
        "checks": checks,
        "record": record,
        "source_files": source["verified_files"],
        "source_hashes": source["verified_hashes"],
        "save_authorization": source.get("save_authorization"),
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_write_text(path, frame.to_csv(index=False))


def save_research_baseline_lock(
    bundle: Mapping[str, Any], output_directory: Path
) -> dict[str, Path]:
    if bundle.get("save_authorization") is not _STRICT_SOURCE_AUTHORIZATION:
        raise PermissionError("Only a strictly verified official baseline may be saved.")
    for name, metadata in bundle["source_files"].items():
        path = Path(metadata["path"])
        expected = bundle["source_hashes"].get(name)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Verified source mutated before save: {name}")

    output_directory = Path(output_directory)
    stamp = str(bundle["stamp"])
    prefix = "portfolio_research_baseline_lock"
    paths = {
        "checks": output_directory / f"{prefix}_checks_{stamp}.csv",
        "json": output_directory / f"{prefix}_{stamp}.json",
        "provenance": output_directory / f"{prefix}_provenance_{stamp}.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite baseline-lock results: " + ", ".join(existing))

    _atomic_write_csv(paths["checks"], bundle["checks"])
    _atomic_write_text(
        paths["json"], json.dumps(_safe(bundle["record"]), indent=2, sort_keys=True) + "\n"
    )
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "lock_name": LOCK_NAME,
        "lock_id": bundle["record"]["lock_id"],
        "verification_stamp": stamp,
        "created_at": datetime.now(UTC).isoformat(),
        "source_files": bundle["source_files"],
        "result_files": {
            "checks": {
                "path": str(paths["checks"].resolve()),
                "sha256": sha256_file(paths["checks"]),
            },
            "json": {
                "path": str(paths["json"].resolve()),
                "sha256": sha256_file(paths["json"]),
            },
        },
        "provenance_self_hash_excluded": True,
        "authorization": bundle["record"]["authorization"],
    }
    _atomic_write_text(
        paths["provenance"], json.dumps(_safe(provenance), indent=2, sort_keys=True) + "\n"
    )
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify and certify the immutable research baseline"
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument(
        "--execution-stress-directory",
        type=Path,
        default=DEFAULT_EXECUTION_STRESS_DIRECTORY,
    )
    parser.add_argument(
        "--execution-stress-stamp",
        default=APPROVED_EXECUTION_STRESS_STAMP,
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--no-save", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    source = load_verified_source(
        registry_path=arguments.registry,
        execution_stress_directory=arguments.execution_stress_directory,
        execution_stress_stamp=arguments.execution_stress_stamp,
        project_root=arguments.project_root,
    )
    bundle = run_research_baseline_lock(source)
    record = bundle["record"]
    print("PORTFOLIO RESEARCH BASELINE LOCK V1")
    print("Registry validation: PASS")
    print("Project file hashes: PASS")
    print("Execution-stress provenance and results: PASS")
    print("Upstream source hashes: PASS")
    print("Official baseline replay: PASS")
    print(f"Lock checks: PASS ({len(bundle['checks'])}/{len(bundle['checks'])})")
    print(f"Status: {record['status']}")
    print("Scope: CONTROLLED NINE-ASSET HISTORICAL RESEARCH ONLY")
    print("Baseline parameter change authorized: NO")
    print("Paper/production/broker/automation authorized: NO")
    print("Following stage: BROADER-UNIVERSE ROBUSTNESS — SPECIFICATION ONLY")
    if arguments.no_save:
        print("Output artifacts saved: 0 (--no-save)")
        return
    paths = save_research_baseline_lock(bundle, arguments.output_directory)
    for name, path in paths.items():
        print(name, path.resolve())


if __name__ == "__main__":
    main()
