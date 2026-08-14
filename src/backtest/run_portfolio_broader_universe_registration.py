"""Register and audit a point-in-time broader research universe.

This module is deliberately registration-only.  It verifies the locked
research comparator, a point-in-time cohort manifest, and an immutable market
data snapshot.  It does not run a backtest or change any trading behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SCHEMA_VERSION = 1
DEFAULT_SPEC = Path("config/broader_universe_cohort_v1.json")
DEFAULT_BASELINE_LOCK = Path("config/research_baseline_lock_v1.json")
DEFAULT_OUTPUT_DIRECTORY = Path(
    "data/backtests/portfolio/broader_universe_registration"
)
REQUIRED_COHORT_COLUMNS = (
    "cohort_date",
    "ticker",
    "asset_type",
    "region",
    "liquidity_rank",
    "eligible_universe_count",
    "liquidity_measure",
    "liquidity_value",
    "selection_source",
    "source_as_of",
    "source_snapshot_path",
    "source_snapshot_sha256",
    "eligible_from",
    "eligible_to",
    "status_at_source",
    "price_adjustment",
)
DATE_COLUMNS = ("cohort_date", "source_as_of", "eligible_from", "eligible_to")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
PROHIBITED_SOURCE_TOKENS = (
    "controlled_basket",
    "post_outcome",
    "current_constituent_backfill",
)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 hash."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value with deterministic serialization."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _normal_ticker(value: Any) -> str:
    return str(value).strip().upper()


def _parse_iso_date(value: Any, *, allow_empty: bool = False) -> pd.Timestamp:
    if allow_empty and (pd.isna(value) or str(value).strip() == ""):
        return pd.NaT
    parsed = pd.to_datetime(value, format="%Y-%m-%d", errors="raise")
    if isinstance(parsed, pd.DatetimeIndex):
        raise TypeError("Expected one date value.")
    return pd.Timestamp(parsed).normalize()


def validate_spec(spec: dict[str, Any]) -> None:
    """Validate the immutable registration specification."""

    if int(spec.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("Unsupported broader-universe specification schema.")
    if spec.get("stage") != (
        "BROADER_UNIVERSE_COHORT_REGISTRATION_AND_DATA_AUDIT_ONLY"
    ):
        raise ValueError("Specification stage is not registration-only.")

    policies = spec.get("cohort_policy")
    if not isinstance(policies, dict) or set(policies) != {"equity", "crypto"}:
        raise ValueError("Specification must define equity and crypto policies.")
    for asset_type, policy in policies.items():
        if policy.get("asset_type") != asset_type:
            raise ValueError(f"Asset policy mismatch: {asset_type}")
        target = int(policy.get("target_count", 0))
        minimum = int(policy.get("minimum_count", 0))
        if target <= 0 or minimum <= 0 or minimum > target:
            raise ValueError(f"Invalid cohort counts for {asset_type}.")

    source = spec.get("source_policy", {})
    required_true = (
        "require_point_in_time_source",
        "require_source_snapshot_file",
        "require_source_snapshot_sha256",
        "require_source_as_of_not_after_cohort_date",
    )
    if any(source.get(key) is not True for key in required_true):
        raise ValueError("Specification omits required point-in-time source evidence.")
    required_false = (
        "allow_current_constituent_backfill",
        "allow_controlled_basket_prefilter",
        "allow_post_outcome_selection",
        "ticker_specific_parameters_allowed",
    )
    if any(bool(source.get(key, True)) for key in required_false):
        raise ValueError("Specification permits a prohibited selection behavior.")
    if source.get("inactive_security_policy") != (
        "RETAIN_WHEN_POINT_IN_TIME_ELIGIBLE"
    ):
        raise ValueError("Inactive-security policy is not fail-closed.")

    data = spec.get("data_policy", {})
    if tuple(data.get("required_columns", ())) == ():
        raise ValueError("No required market-data columns were registered.")
    if int(data.get("minimum_indicator_history_rows_at_selection", 0)) < 50:
        raise ValueError("Indicator warm-up threshold is too small.")

    authorization = spec.get("authorization", {})
    allowed_true = {"cohort_registration_may_be_authorized"}
    unexpected = [
        key
        for key, value in authorization.items()
        if bool(value) and key not in allowed_true
    ]
    if unexpected:
        raise ValueError(
            "Specification grants prohibited authority: " + ", ".join(unexpected)
        )


def load_spec(path: Path) -> dict[str, Any]:
    spec = _read_json(path)
    validate_spec(spec)
    return spec


def load_cohort_manifest(path: Path) -> pd.DataFrame:
    """Load and normalize the point-in-time cohort manifest."""

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if tuple(frame.columns) != REQUIRED_COHORT_COLUMNS:
        raise ValueError(
            "Cohort manifest columns or order differ from the registered schema. "
            f"Expected: {list(REQUIRED_COHORT_COLUMNS)}"
        )
    if frame.empty:
        raise ValueError("Cohort manifest cannot be empty.")

    output = frame.copy()
    output["ticker"] = output["ticker"].map(_normal_ticker)
    output["asset_type"] = output["asset_type"].str.strip().str.lower()
    output["region"] = output["region"].str.strip().str.upper()
    output["liquidity_measure"] = output["liquidity_measure"].str.strip()
    output["selection_source"] = output["selection_source"].str.strip()
    source_paths = []
    for value in output["source_snapshot_path"]:
        candidate = Path(str(value).strip()).expanduser()
        if not candidate.is_absolute():
            candidate = Path(path).resolve().parent / candidate
        source_paths.append(str(candidate.resolve()))
    output["source_snapshot_path"] = source_paths
    output["source_snapshot_sha256"] = (
        output["source_snapshot_sha256"].str.strip().str.lower()
    )
    output["status_at_source"] = output["status_at_source"].str.strip().str.lower()
    output["price_adjustment"] = output["price_adjustment"].str.strip()

    for column in DATE_COLUMNS:
        output[column] = [
            _parse_iso_date(value, allow_empty=column == "eligible_to")
            for value in output[column]
        ]
    for column in ("liquidity_rank", "eligible_universe_count"):
        output[column] = pd.to_numeric(output[column], errors="raise").astype(int)
    output["liquidity_value"] = pd.to_numeric(
        output["liquidity_value"], errors="raise"
    ).astype(float)
    return output


def verify_baseline_lock(
    spec: dict[str, Any], baseline_path: Path, project_root: Path
) -> dict[str, Any]:
    """Verify registry identity, authority boundaries, and registered code."""

    baseline_path = Path(baseline_path).resolve()
    lock = _read_json(baseline_path)
    policy = spec["baseline_lock"]
    actual_hash = sha256_file(baseline_path)
    hash_passed = actual_hash == str(policy["registry_sha256"])
    identity_passed = (
        lock.get("lock_id") == policy.get("lock_id")
        and lock.get("lock_name") == policy.get("lock_name")
    )
    authorization = lock.get("authorization", {})
    authority_passed = (
        lock.get("status") == "RESEARCH_REFERENCE_LOCKED"
        and authorization.get("research_reference_locked") is True
        and all(
            authorization.get(key) is False
            for key in (
                "baseline_parameter_change_authorized",
                "broader_universe_validated",
                "paper_trading_authorized",
                "production_authorized",
                "broker_access_authorized",
                "automation_authorized",
            )
        )
    )

    project_rows: list[dict[str, Any]] = []
    for relative, expected in sorted(lock.get("approved_project_files", {}).items()):
        path = Path(project_root) / relative
        actual = sha256_file(path) if path.is_file() else None
        project_rows.append(
            {
                "relative_path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "passed": actual == expected,
            }
        )
    project_passed = bool(project_rows) and all(row["passed"] for row in project_rows)
    return {
        "lock": lock,
        "path": baseline_path,
        "sha256": actual_hash,
        "hash_passed": hash_passed,
        "identity_passed": identity_passed,
        "authority_passed": authority_passed,
        "project_rows": project_rows,
        "project_passed": project_passed,
    }


def _snapshot_core(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest.items()
        if key not in {"snapshot_id", "fingerprint", "created_at"}
    }


def verify_snapshot(snapshot_path: Path) -> dict[str, Any]:
    """Independently verify the immutable research snapshot."""

    snapshot_path = Path(snapshot_path).resolve()
    manifest_path = snapshot_path / "manifest.json"
    manifest = _read_json(manifest_path)
    fingerprint_passed = (
        sha256_json(_snapshot_core(manifest)) == manifest.get("fingerprint")
    )
    rows: list[dict[str, Any]] = []
    entries: dict[str, dict[str, Any]] = {}
    for label in ("config", "windows"):
        metadata = manifest.get(label)
        if isinstance(metadata, dict):
            entries[label] = metadata
    for ticker, metadata in manifest.get("market_files", {}).items():
        if isinstance(metadata, dict):
            entries[f"market:{ticker}"] = metadata

    root = snapshot_path.resolve()
    for label, metadata in sorted(entries.items()):
        declared = str(metadata.get("path", ""))
        path = (snapshot_path / declared).resolve()
        inside = path == root or root in path.parents
        actual = sha256_file(path) if inside and path.is_file() else None
        expected = metadata.get("sha256")
        rows.append(
            {
                "artifact": label,
                "path": str(path),
                "inside_snapshot": inside,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "passed": inside and actual == expected,
            }
        )
    files_passed = bool(rows) and all(row["passed"] for row in rows)
    return {
        "path": snapshot_path,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "fingerprint_passed": fingerprint_passed,
        "file_rows": rows,
        "files_passed": files_passed,
        "passed": fingerprint_passed and files_passed,
    }


def audit_cohort(
    frame: pd.DataFrame, spec: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, bool]]:
    """Audit counts, ranks, dates, and point-in-time source evidence."""

    study = spec["study"]
    policies = spec["cohort_policy"]
    dates = sorted(pd.Timestamp(value) for value in frame["cohort_date"].unique())
    study_start = _parse_iso_date(study["study_start"])
    study_end = _parse_iso_date(study["study_end"])
    tolerance = int(study["endpoint_tolerance_days"])
    maximum_gap = int(study["maximum_days_between_cohorts"])
    gaps = [(right - left).days for left, right in zip(dates, dates[1:])]
    schedule_passed = (
        dates
        and abs((dates[0] - study_start).days) <= tolerance
        and abs((dates[-1] - study_end).days) <= tolerance
        and all(1 <= gap <= maximum_gap for gap in gaps)
    )

    domain_passed = (
        frame["ticker"].ne("").all()
        and frame["asset_type"].isin(["equity", "crypto"]).all()
        and frame["status_at_source"].isin(["active", "inactive", "delisted"]).all()
        and (frame["liquidity_rank"] > 0).all()
        and (frame["eligible_universe_count"] > 0).all()
        and frame["liquidity_value"].map(isfinite).all()
        and (frame["liquidity_value"] >= 0).all()
    )
    eligibility_passed = (
        (frame["eligible_from"] <= frame["cohort_date"]).all()
        and (
            frame["eligible_to"].isna()
            | (frame["eligible_to"] >= frame["cohort_date"])
        ).all()
    )
    source_passed = (
        frame["selection_source"].ne("").all()
        and (frame["source_as_of"] <= frame["cohort_date"]).all()
        and frame["source_snapshot_sha256"].map(
            lambda value: bool(SHA256_PATTERN.fullmatch(value))
        ).all()
    )
    source_file_passed = frame.apply(
        lambda row: Path(row["source_snapshot_path"]).is_file()
        and sha256_file(Path(row["source_snapshot_path"]))
        == row["source_snapshot_sha256"],
        axis=1,
    )
    no_prohibited_source = ~frame["selection_source"].str.lower().map(
        lambda value: any(token in value for token in PROHIBITED_SOURCE_TOKENS)
    )
    source_passed = bool(
        source_passed and source_file_passed.all() and no_prohibited_source.all()
    )

    duplicate_passed = not frame.duplicated(
        ["cohort_date", "asset_type", "ticker"]
    ).any()
    rows: list[dict[str, Any]] = []
    counts_passed = True
    ranks_passed = True
    policy_passed = True

    for cohort_date in dates:
        for asset_type in ("equity", "crypto"):
            group = frame.loc[
                (frame["cohort_date"] == cohort_date)
                & (frame["asset_type"] == asset_type)
            ].copy()
            policy = policies[asset_type]
            eligible_counts = sorted(group["eligible_universe_count"].unique())
            eligible_count = eligible_counts[0] if len(eligible_counts) == 1 else None
            target = int(policy["target_count"])
            minimum = int(policy["minimum_count"])
            expected_count = (
                target
                if asset_type == "equity"
                else min(target, eligible_count or 0)
            )
            count_passed = (
                eligible_count is not None
                and eligible_count >= len(group)
                and len(group) == expected_count
                and len(group) >= minimum
            )
            expected_ranks = list(range(1, len(group) + 1))
            rank_values = sorted(group["liquidity_rank"].tolist())
            rank_sequence_passed = rank_values == expected_ranks
            deterministic = group.sort_values(
                ["liquidity_value", "ticker"],
                ascending=[False, True],
                kind="mergesort",
            )
            deterministic_rank_passed = deterministic["liquidity_rank"].tolist() == (
                expected_ranks
            )
            group_policy_passed = (
                group["region"].eq(str(policy["region"])).all()
                and group["liquidity_measure"]
                .eq(str(policy["liquidity_measure"]))
                .all()
                and group["price_adjustment"]
                .eq(str(policy["price_adjustment"]))
                .all()
            )
            counts_passed = counts_passed and count_passed
            ranks_passed = (
                ranks_passed and rank_sequence_passed and deterministic_rank_passed
            )
            policy_passed = policy_passed and group_policy_passed
            rows.append(
                {
                    "cohort_date": cohort_date.date().isoformat(),
                    "asset_type": asset_type,
                    "selected_count": len(group),
                    "eligible_universe_count": eligible_count,
                    "target_count": target,
                    "minimum_count": minimum,
                    "expected_selected_count": expected_count,
                    "count_passed": count_passed,
                    "rank_sequence_passed": rank_sequence_passed,
                    "deterministic_rank_passed": deterministic_rank_passed,
                    "policy_values_passed": group_policy_passed,
                }
            )

    flags = {
        "domain_passed": bool(domain_passed),
        "eligibility_passed": bool(eligibility_passed),
        "source_passed": bool(source_passed),
        "duplicate_passed": bool(duplicate_passed),
        "schedule_passed": bool(schedule_passed),
        "counts_passed": bool(counts_passed),
        "ranks_passed": bool(ranks_passed),
        "policy_passed": bool(policy_passed),
    }
    return pd.DataFrame(rows), flags


def _membership_end(
    row: pd.Series, cohort_dates: list[pd.Timestamp], study_end: pd.Timestamp
) -> pd.Timestamp:
    cohort_date = pd.Timestamp(row["cohort_date"])
    later = [value for value in cohort_dates if value > cohort_date]
    candidates = [study_end]
    if later:
        candidates.append(later[0] - pd.Timedelta(days=1))
    if not pd.isna(row["eligible_to"]):
        candidates.append(pd.Timestamp(row["eligible_to"]))
    return min(candidates)


def _missing_fraction(
    timestamps: pd.DatetimeIndex, asset_type: str
) -> tuple[int, int, float]:
    if timestamps.empty:
        return 0, 0, 1.0
    if asset_type == "equity":
        expected = pd.bdate_range(timestamps.min().normalize(), timestamps.max().normalize())
    else:
        expected = pd.date_range(
            timestamps.min().normalize(), timestamps.max().normalize(), freq="D"
        )
    actual_dates = pd.DatetimeIndex(timestamps.normalize().unique())
    observed = len(expected.intersection(actual_dates))
    missing = max(0, len(expected) - observed)
    fraction = missing / len(expected) if len(expected) else 0.0
    return len(expected), missing, fraction


def audit_market_data(
    cohort: pd.DataFrame,
    spec: dict[str, Any],
    snapshot: dict[str, Any],
) -> pd.DataFrame:
    """Audit every selected ticker's prepared data and membership coverage."""

    policy = spec["data_policy"]
    required = set(policy["required_columns"])
    warmup = int(policy["minimum_indicator_history_rows_at_selection"])
    study_end = _parse_iso_date(spec["study"]["study_end"])
    cohort_dates = sorted(
        pd.Timestamp(value) for value in cohort["cohort_date"].unique()
    )
    market_files = snapshot["manifest"].get("market_files", {})
    snapshot_root = snapshot["path"].resolve()
    rows: list[dict[str, Any]] = []

    for ticker in sorted(cohort["ticker"].unique()):
        memberships = cohort.loc[cohort["ticker"] == ticker].copy()
        asset_types = sorted(memberships["asset_type"].unique())
        asset_type = asset_types[0] if len(asset_types) == 1 else "invalid"
        metadata = market_files.get(ticker)
        declared_path = str(metadata.get("path", "")) if isinstance(metadata, dict) else ""
        path = (snapshot_root / declared_path).resolve()
        inside = path == snapshot_root or snapshot_root in path.parents
        file_present = bool(metadata) and inside and path.is_file()

        row: dict[str, Any] = {
            "ticker": ticker,
            "asset_type": asset_type,
            "membership_count": len(memberships),
            "path": str(path),
            "file_present": file_present,
            "hash_passed": False,
            "required_columns_passed": False,
            "timestamps_passed": False,
            "finite_numeric_passed": False,
            "positive_ohlc_passed": False,
            "ohlc_consistency_passed": False,
            "volume_passed": False,
            "missing_fraction": 1.0,
            "missing_fraction_passed": False,
            "warmup_passed": False,
            "membership_coverage_passed": False,
            "rows": 0,
            "start": None,
            "end": None,
        }
        if not file_present:
            row["passed"] = False
            rows.append(row)
            continue

        row["hash_passed"] = sha256_file(path) == metadata.get("sha256")
        try:
            frame = pd.read_csv(path)
        except Exception:
            row["passed"] = False
            rows.append(row)
            continue
        row["rows"] = len(frame)
        row["required_columns_passed"] = required.issubset(frame.columns)
        if not row["required_columns_passed"]:
            row["passed"] = False
            rows.append(row)
            continue

        timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
        timestamps_passed = (
            timestamps.notna().all()
            and timestamps.is_unique
            and timestamps.is_monotonic_increasing
        )
        row["timestamps_passed"] = bool(timestamps_passed)
        valid_timestamps = pd.DatetimeIndex(timestamps.dropna())
        if not valid_timestamps.empty:
            row["start"] = valid_timestamps.min().isoformat()
            row["end"] = valid_timestamps.max().isoformat()

        numeric_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "EMA20",
            "EMA50",
            "RSI14",
            "MACD",
        ]
        numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
        finite_numeric = numeric.notna().all().all() and numeric.map(isfinite).all().all()
        row["finite_numeric_passed"] = bool(finite_numeric)
        ohlc = numeric[["Open", "High", "Low", "Close"]]
        row["positive_ohlc_passed"] = bool((ohlc > 0).all().all())
        row["ohlc_consistency_passed"] = bool(
            (numeric["High"] >= numeric[["Open", "Close", "Low"]].max(axis=1)).all()
            and (
                numeric["Low"]
                <= numeric[["Open", "Close", "High"]].min(axis=1)
            ).all()
        )
        row["volume_passed"] = bool((numeric["Volume"] >= 0).all())

        expected_days, missing_days, missing_fraction = _missing_fraction(
            valid_timestamps, asset_type
        )
        row["expected_calendar_rows"] = expected_days
        row["missing_calendar_rows"] = missing_days
        row["missing_fraction"] = missing_fraction
        threshold = (
            float(policy["maximum_missing_business_day_fraction_equity"])
            if asset_type == "equity"
            else float(policy["maximum_missing_calendar_day_fraction_crypto"])
        )
        row["missing_fraction_passed"] = bool(missing_fraction <= threshold)

        warmup_passed = True
        coverage_passed = bool(not valid_timestamps.empty)
        for _, membership in memberships.iterrows():
            cohort_date = pd.Timestamp(membership["cohort_date"])
            required_end = _membership_end(membership, cohort_dates, study_end)
            available_by_selection = int((valid_timestamps <= cohort_date).sum())
            warmup_passed = warmup_passed and available_by_selection >= warmup
            coverage_passed = coverage_passed and (
                valid_timestamps.min().normalize() <= cohort_date
                and valid_timestamps.max().normalize() >= required_end
            )
        row["warmup_passed"] = bool(warmup_passed)
        row["membership_coverage_passed"] = bool(coverage_passed)
        row["passed"] = all(
            bool(row[key])
            for key in (
                "file_present",
                "hash_passed",
                "required_columns_passed",
                "timestamps_passed",
                "finite_numeric_passed",
                "positive_ohlc_passed",
                "ohlc_consistency_passed",
                "volume_passed",
                "missing_fraction_passed",
                "warmup_passed",
                "membership_coverage_passed",
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _check_row(name: str, detail: str, passed: bool) -> dict[str, Any]:
    return {
        "check": name,
        "detail": detail,
        "passed": bool(passed),
        "authorizes_baseline_parameter_change": False,
        "authorizes_broader_universe_backtest": False,
        "authorizes_paper_or_production": False,
    }


def build_checks(
    *,
    spec: dict[str, Any],
    baseline: dict[str, Any],
    cohort_flags: dict[str, bool],
    snapshot: dict[str, Any],
    data_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Build the fixed fail-closed registration gate table."""

    source_policy = spec["source_policy"]
    no_prefilter = (
        source_policy.get("allow_controlled_basket_prefilter") is False
        and source_policy.get("allow_post_outcome_selection") is False
        and source_policy.get("allow_current_constituent_backfill") is False
    )
    data_exists = not data_audit.empty
    rows = [
        _check_row("specification_contract", "registration-only schema", True),
        _check_row(
            "baseline_lock_hash",
            "exact immutable research registry",
            baseline["hash_passed"] and baseline["identity_passed"],
        ),
        _check_row(
            "baseline_lock_status",
            "locked reference and authority boundary",
            baseline["authority_passed"],
        ),
        _check_row(
            "baseline_project_files",
            "all registry-approved project hashes",
            baseline["project_passed"],
        ),
        _check_row(
            "cohort_schema_domains",
            "ticker, asset, status, eligibility value domains",
            cohort_flags["domain_passed"]
            and cohort_flags["eligibility_passed"]
            and cohort_flags["policy_passed"],
        ),
        _check_row(
            "cohort_schedule",
            "annual schedule and study endpoints",
            cohort_flags["schedule_passed"],
        ),
        _check_row(
            "cohort_counts",
            "registered equity and dynamic crypto counts",
            cohort_flags["counts_passed"],
        ),
        _check_row(
            "point_in_time_sources",
            "source dates, immutable hashes, and eligibility",
            cohort_flags["source_passed"],
        ),
        _check_row(
            "duplicates_and_ranks",
            "unique membership and deterministic liquidity ranks",
            cohort_flags["duplicate_passed"] and cohort_flags["ranks_passed"],
        ),
        _check_row(
            "snapshot_integrity",
            "fingerprint and every declared snapshot file",
            snapshot["passed"],
        ),
        _check_row(
            "prepared_data_columns",
            "all registered prepared-data columns",
            data_exists and bool(data_audit["required_columns_passed"].all()),
        ),
        _check_row(
            "market_data_quality",
            "hash, timestamp, OHLCV, finite values, missing dates",
            data_exists
            and bool(
                data_audit[
                    [
                        "file_present",
                        "hash_passed",
                        "timestamps_passed",
                        "finite_numeric_passed",
                        "positive_ohlc_passed",
                        "ohlc_consistency_passed",
                        "volume_passed",
                        "missing_fraction_passed",
                    ]
                ].all(axis=None)
            ),
        ),
        _check_row(
            "membership_data_coverage",
            "indicator warm-up and full membership intervals",
            data_exists
            and bool(
                data_audit[["warmup_passed", "membership_coverage_passed"]].all(
                    axis=None
                )
            ),
        ),
        _check_row(
            "no_outcome_or_controlled_prefilter",
            "selection policy excludes outcome and development-basket filters",
            no_prefilter and cohort_flags["source_passed"],
        ),
        _check_row(
            "no_additional_authority",
            "registration does not authorize backtest, parameter, or deployment changes",
            baseline["authority_passed"]
            and all(
                spec["authorization"].get(key) is False
                for key in (
                    "broader_universe_backtest_authorized",
                    "baseline_parameter_change_authorized",
                    "broader_universe_validated",
                    "paper_trading_authorized",
                    "production_authorized",
                    "broker_access_authorized",
                    "automation_authorized",
                )
            ),
        ),
    ]
    return pd.DataFrame(rows)


def register_cohort(
    *,
    spec_path: Path,
    baseline_lock_path: Path,
    cohort_manifest_path: Path,
    snapshot_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Run all registration and data gates without saving outputs."""

    spec_path = Path(spec_path).resolve()
    cohort_manifest_path = Path(cohort_manifest_path).resolve()
    spec = load_spec(spec_path)
    cohort = load_cohort_manifest(cohort_manifest_path)
    baseline = verify_baseline_lock(spec, baseline_lock_path, project_root)
    snapshot = verify_snapshot(snapshot_path)
    cohort_summary, cohort_flags = audit_cohort(cohort, spec)
    data_audit = audit_market_data(cohort, spec, snapshot)
    checks = build_checks(
        spec=spec,
        baseline=baseline,
        cohort_flags=cohort_flags,
        snapshot=snapshot,
        data_audit=data_audit,
    )
    passed = bool(checks["passed"].all())
    core = {
        "schema_version": SCHEMA_VERSION,
        "registration_name": spec["registration_name"],
        "baseline_lock_id": baseline["lock"].get("lock_id"),
        "baseline_lock_sha256": baseline["sha256"],
        "spec_sha256": sha256_file(spec_path),
        "cohort_manifest_sha256": sha256_file(cohort_manifest_path),
        "snapshot_id": snapshot["manifest"].get("snapshot_id"),
        "snapshot_fingerprint": snapshot["manifest"].get("fingerprint"),
        "study_start": spec["study"]["study_start"],
        "study_end": spec["study"]["study_end"],
    }
    fingerprint = sha256_json(core)
    registration_id = (
        "BURC_V1_"
        + str(spec["study"]["study_end"]).replace("-", "")
        + "_"
        + fingerprint[:12].upper()
    )
    screen = pd.DataFrame(
        [
            {
                "registration_id": registration_id,
                "status": (
                    "BROADER_UNIVERSE_COHORT_REGISTERED"
                    if passed
                    else "REGISTRATION_REJECTED"
                ),
                "all_checks_passed": passed,
                "passed_check_count": int(checks["passed"].sum()),
                "check_count": len(checks),
                "cohort_date_count": int(cohort["cohort_date"].nunique()),
                "membership_row_count": len(cohort),
                "unique_equity_count": int(
                    cohort.loc[cohort["asset_type"] == "equity", "ticker"].nunique()
                ),
                "unique_crypto_count": int(
                    cohort.loc[cohort["asset_type"] == "crypto", "ticker"].nunique()
                ),
                "audited_market_file_count": len(data_audit),
                "failed_market_file_count": int((~data_audit["passed"]).sum()),
                "broader_universe_backtest_authorized": False,
                "baseline_parameter_change_authorized": False,
                "paper_or_production_authorized": False,
            }
        ]
    )
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "registration_id": registration_id,
        "registration_fingerprint": fingerprint,
        "status": screen.iloc[0]["status"],
        "passed": passed,
        "core": core,
        "authorization": {
            "cohort_registered": passed,
            "broader_universe_backtest_authorized": False,
            "baseline_parameter_change_authorized": False,
            "broader_universe_validated": False,
            "paper_trading_authorized": False,
            "production_authorized": False,
            "broker_access_authorized": False,
            "automation_authorized": False,
        },
        "paths": {
            "spec": spec_path,
            "baseline_lock": Path(baseline_lock_path).resolve(),
            "cohort_manifest": cohort_manifest_path,
            "snapshot": Path(snapshot_path).resolve(),
            "project_root": Path(project_root).resolve(),
        },
        "spec": spec,
        "baseline": baseline,
        "snapshot": snapshot,
        "cohort": cohort,
        "cohort_summary": cohort_summary,
        "data_audit": data_audit,
        "checks": checks,
        "screen": screen,
    }


def _public_registration(result: dict[str, Any]) -> dict[str, Any]:
    failed_checks = result["checks"].loc[~result["checks"]["passed"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": result["created_at"],
        "registration_id": result["registration_id"],
        "registration_fingerprint": result["registration_fingerprint"],
        "status": result["status"],
        "all_checks_passed": result["passed"],
        "core": result["core"],
        "verification": {
            "check_count": len(result["checks"]),
            "passed_check_count": int(result["checks"]["passed"].sum()),
            "cohort_date_count": int(result["cohort"]["cohort_date"].nunique()),
            "membership_row_count": len(result["cohort"]),
            "audited_market_file_count": len(result["data_audit"]),
            "failed_market_file_count": int((~result["data_audit"]["passed"]).sum()),
            "failed_checks": failed_checks.to_dict("records"),
        },
        "authorization": result["authorization"],
        "following_stage": (
            "BROADER_UNIVERSE_BACKTEST_PROTOCOL_SPECIFICATION_ONLY"
            if result["passed"]
            else "REMEDIATE_COHORT_OR_DATA_AND_RERUN_REGISTRATION"
        ),
    }


def _source_files(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for name in ("spec", "baseline_lock", "cohort_manifest"):
        path = result["paths"][name]
        sources[name] = {"path": str(path), "sha256": sha256_file(path)}
    snapshot = result["snapshot"]
    sources["snapshot_manifest"] = {
        "path": str(snapshot["manifest_path"]),
        "sha256": sha256_file(snapshot["manifest_path"]),
    }
    for row in snapshot["file_rows"]:
        if row["actual_sha256"] is not None:
            sources[f"snapshot:{row['artifact']}"] = {
                "path": row["path"],
                "sha256": row["actual_sha256"],
            }
    for row in result["baseline"]["project_rows"]:
        if row["actual_sha256"] is not None:
            sources[f"project:{row['relative_path']}"] = {
                "path": str(result["paths"]["project_root"] / row["relative_path"]),
                "sha256": row["actual_sha256"],
            }
    unique_membership_sources = result["cohort"][
        ["source_snapshot_path", "source_snapshot_sha256"]
    ].drop_duplicates()
    for index, row in unique_membership_sources.reset_index(drop=True).iterrows():
        path = Path(row["source_snapshot_path"])
        if path.is_file():
            sources[f"membership_source:{index + 1:03d}"] = {
                "path": str(path),
                "sha256": sha256_file(path),
            }
    return dict(sorted(sources.items()))


def save_registration(
    result: dict[str, Any], output_directory: Path = DEFAULT_OUTPUT_DIRECTORY
) -> dict[str, Path]:
    """Save registration audit artifacts and a complete provenance record."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    prefix = "portfolio_broader_universe_registration"
    paths = {
        "screen": output_directory / f"{prefix}_screen_{stamp}.csv",
        "cohorts": output_directory / f"{prefix}_cohorts_{stamp}.csv",
        "data_audit": output_directory / f"{prefix}_data_audit_{stamp}.csv",
        "checks": output_directory / f"{prefix}_checks_{stamp}.csv",
        "json": output_directory / f"{prefix}_{stamp}.json",
        "provenance": output_directory / f"{prefix}_provenance_{stamp}.json",
    }
    result["screen"].to_csv(paths["screen"], index=False, lineterminator="\n")
    result["cohort_summary"].to_csv(
        paths["cohorts"], index=False, lineterminator="\n"
    )
    result["data_audit"].to_csv(
        paths["data_audit"], index=False, lineterminator="\n"
    )
    result["checks"].to_csv(paths["checks"], index=False, lineterminator="\n")

    public = _public_registration(result)
    public["result_files"] = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in paths.items()
        if name in {"screen", "cohorts", "data_audit", "checks"}
    }
    paths["json"].write_text(
        json.dumps(_json_safe(public), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "registration_id": result["registration_id"],
        "registration_fingerprint": result["registration_fingerprint"],
        "status": result["status"],
        "source_files": _source_files(result),
        "result_files": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in paths.items()
            if name != "provenance"
        },
        "authorization": result["authorization"],
    }
    paths["provenance"].write_text(
        json.dumps(_json_safe(provenance), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths


def _print_result(result: dict[str, Any], paths: dict[str, Path] | None) -> None:
    print("=" * 72)
    print("BROADER-UNIVERSE COHORT REGISTRATION & DATA AUDIT V1")
    print("=" * 72)
    print("Registration ID:", result["registration_id"])
    print("Lock verification:", "PASS" if result["baseline"]["hash_passed"] else "FAIL")
    print("Snapshot integrity:", "PASS" if result["snapshot"]["passed"] else "FAIL")
    print(
        "Registration checks:",
        f"{int(result['checks']['passed'].sum())}/{len(result['checks'])}",
    )
    print("Status:", result["status"])
    print("Broader-universe backtest authorized: NO")
    print("Baseline parameter change authorized: NO")
    print("Paper/production/broker/automation authorized: NO")
    if paths is None:
        print("Output artifacts saved: 0 (--no-save)")
    else:
        for name, path in paths.items():
            print(name, path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register and audit a point-in-time broader research universe"
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--baseline-lock", type=Path, default=DEFAULT_BASELINE_LOCK)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--no-save", action="store_true")
    arguments = parser.parse_args()
    result = register_cohort(
        spec_path=arguments.spec,
        baseline_lock_path=arguments.baseline_lock,
        cohort_manifest_path=arguments.cohort_manifest,
        snapshot_path=arguments.snapshot,
        project_root=arguments.project_root.resolve(),
    )
    output = arguments.output_directory
    if output is None:
        output = Path(result["spec"]["output"]["directory"])
    paths = None if arguments.no_save else save_registration(result, output)
    _print_result(result, paths)
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
