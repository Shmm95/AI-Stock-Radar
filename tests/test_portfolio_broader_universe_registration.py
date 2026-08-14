from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtest.run_portfolio_broader_universe_registration import (
    REQUIRED_COHORT_COLUMNS,
    load_cohort_manifest,
    register_cohort,
    save_registration,
    sha256_file,
    sha256_json,
    validate_spec,
    verify_snapshot,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _market_frame(asset_type: str, *, bad_ohlc: bool = False) -> pd.DataFrame:
    frequency = "B" if asset_type == "equity" else "D"
    dates = pd.date_range("2019-01-01", "2020-01-31", freq=frequency)
    x = np.arange(len(dates), dtype=float)
    close = 100.0 + x * 0.05
    high = close * 1.01
    if bad_ohlc:
        high[5] = close[5] * 0.90
    return pd.DataFrame(
        {
            "timestamp": dates.strftime("%Y-%m-%dT00:00:00"),
            "Open": close * 0.999,
            "High": high,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1_000_000 + x,
            "EMA20": close * 0.99,
            "EMA50": close * 0.98,
            "RSI14": 55.0,
            "MACD": 1.0,
            "RegimeAllowed": True,
        }
    )


def _snapshot(root: Path, *, bad_ohlc: bool = False, missing_column: bool = False) -> Path:
    snapshot = root / "snapshot"
    market = snapshot / "market"
    market.mkdir(parents=True)
    market_files = {}
    for ticker, asset_type in (("AAA", "equity"), ("BBB", "equity"), ("CCC-USD", "crypto")):
        frame = _market_frame(asset_type, bad_ohlc=bad_ohlc and ticker == "AAA")
        if missing_column and ticker == "AAA":
            frame = frame.drop(columns="MACD")
        path = market / f"{ticker}.csv"
        frame.to_csv(path, index=False, lineterminator="\n")
        timestamps = pd.to_datetime(frame["timestamp"])
        market_files[ticker] = {
            "path": f"market/{ticker}.csv",
            "sha256": sha256_file(path),
            "rows": len(frame),
            "start": timestamps.min().isoformat(),
            "end": timestamps.max().isoformat(),
            "columns": [column for column in frame.columns if column != "timestamp"],
            "dtypes": {
                column: str(frame[column].dtype)
                for column in frame.columns
                if column != "timestamp"
            },
        }
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    (snapshot / "windows.csv").write_text("window_id\nW01\n", encoding="utf-8")
    core = {
        "schema_version": 1,
        "period": "provided",
        "regime_period": "provided",
        "use_crypto_regime": True,
        "train_months": 24,
        "test_months": 6,
        "step_months": 6,
        "tickers": sorted(market_files),
        "market_files": market_files,
        "config": {
            "path": "config.json",
            "sha256": sha256_file(snapshot / "config.json"),
        },
        "windows": {
            "path": "windows.csv",
            "sha256": sha256_file(snapshot / "windows.csv"),
            "count": 1,
        },
        "code_files": {},
        "git_commit": None,
    }
    fingerprint = sha256_json(core)
    _write_json(
        snapshot / "manifest.json",
        {
            **core,
            "snapshot_id": f"TEST_{fingerprint[:12]}",
            "fingerprint": fingerprint,
            "created_at": "2020-02-01T00:00:00+00:00",
        },
    )
    return snapshot


def _baseline(root: Path) -> tuple[Path, Path]:
    project = root / "project"
    target = project / "src/backtest/locked.py"
    target.parent.mkdir(parents=True)
    target.write_text("LOCKED = True\n", encoding="utf-8")
    baseline = root / "baseline.json"
    _write_json(
        baseline,
        {
            "schema_version": 1,
            "lock_name": "TEST_LOCK",
            "lock_id": "TEST_LOCK_ID",
            "status": "RESEARCH_REFERENCE_LOCKED",
            "approved_project_files": {
                "src/backtest/locked.py": sha256_file(target),
            },
            "authorization": {
                "research_reference_locked": True,
                "baseline_parameter_change_authorized": False,
                "broader_universe_validated": False,
                "paper_trading_authorized": False,
                "production_authorized": False,
                "broker_access_authorized": False,
                "automation_authorized": False,
            },
        },
    )
    return baseline, project


def _spec(root: Path, baseline: Path, **overrides) -> Path:
    value = {
        "schema_version": 1,
        "registration_name": "TEST_REGISTRATION",
        "stage": "BROADER_UNIVERSE_COHORT_REGISTRATION_AND_DATA_AUDIT_ONLY",
        "baseline_lock": {
            "lock_id": "TEST_LOCK_ID",
            "lock_name": "TEST_LOCK",
            "registry_sha256": sha256_file(baseline),
        },
        "study": {
            "study_start": "2020-01-02",
            "study_end": "2020-01-31",
            "maximum_days_between_cohorts": 370,
            "endpoint_tolerance_days": 31,
            "membership_rebalance": "ANNUAL_POINT_IN_TIME",
            "selection_rule": "DESCENDING_POINT_IN_TIME_DOLLAR_LIQUIDITY_WITH_TICKER_ASC_TIEBREAK",
        },
        "cohort_policy": {
            "equity": {
                "asset_type": "equity",
                "region": "US",
                "target_count": 2,
                "minimum_count": 2,
                "liquidity_measure": "median_daily_dollar_volume_90d",
                "price_adjustment": "split_dividend_adjusted",
            },
            "crypto": {
                "asset_type": "crypto",
                "region": "GLOBAL",
                "target_count": 1,
                "minimum_count": 1,
                "liquidity_measure": "median_daily_dollar_volume_90d",
                "price_adjustment": "raw_spot",
            },
        },
        "source_policy": {
            "require_point_in_time_source": True,
            "require_source_snapshot_file": True,
            "require_source_snapshot_sha256": True,
            "require_source_as_of_not_after_cohort_date": True,
            "allow_current_constituent_backfill": False,
            "allow_controlled_basket_prefilter": False,
            "allow_post_outcome_selection": False,
            "inactive_security_policy": "RETAIN_WHEN_POINT_IN_TIME_ELIGIBLE",
            "ticker_specific_parameters_allowed": False,
        },
        "data_policy": {
            "required_columns": [
                "timestamp",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "EMA20",
                "EMA50",
                "RSI14",
                "MACD",
                "RegimeAllowed",
            ],
            "minimum_indicator_history_rows_at_selection": 200,
            "maximum_missing_business_day_fraction_equity": 0.10,
            "maximum_missing_calendar_day_fraction_crypto": 0.03,
            "require_unique_sorted_timestamps": True,
            "require_positive_ohlc": True,
            "require_nonnegative_volume": True,
            "require_membership_interval_coverage": True,
        },
        "output": {"directory": "unused", "save_failed_audits": True},
        "authorization": {
            "cohort_registration_may_be_authorized": True,
            "broader_universe_backtest_authorized": False,
            "baseline_parameter_change_authorized": False,
            "broader_universe_validated": False,
            "paper_trading_authorized": False,
            "production_authorized": False,
            "broker_access_authorized": False,
            "automation_authorized": False,
        },
    }
    for section, updates in overrides.items():
        value[section].update(updates)
    path = root / "spec.json"
    _write_json(path, value)
    return path


def _cohort(root: Path, *, source="point_in_time_vendor", source_after=False) -> Path:
    date = "2020-01-02"
    source_as_of = "2020-01-03" if source_after else "2020-01-01"
    membership_source = root / "point_in_time_membership.csv"
    membership_source.write_text("ticker,liquidity\nAAA,200\nBBB,100\nCCC-USD,50\n")
    common = {
        "cohort_date": date,
        "liquidity_measure": "median_daily_dollar_volume_90d",
        "selection_source": source,
        "source_as_of": source_as_of,
        "source_snapshot_path": str(membership_source),
        "source_snapshot_sha256": sha256_file(membership_source),
        "eligible_from": "2010-01-01",
        "eligible_to": "",
        "status_at_source": "active",
    }
    rows = [
        {
            **common,
            "ticker": "AAA",
            "asset_type": "equity",
            "region": "US",
            "liquidity_rank": 1,
            "eligible_universe_count": 2,
            "liquidity_value": 200,
            "price_adjustment": "split_dividend_adjusted",
        },
        {
            **common,
            "ticker": "BBB",
            "asset_type": "equity",
            "region": "US",
            "liquidity_rank": 2,
            "eligible_universe_count": 2,
            "liquidity_value": 100,
            "price_adjustment": "split_dividend_adjusted",
        },
        {
            **common,
            "ticker": "CCC-USD",
            "asset_type": "crypto",
            "region": "GLOBAL",
            "liquidity_rank": 1,
            "eligible_universe_count": 1,
            "liquidity_value": 50,
            "price_adjustment": "raw_spot",
        },
    ]
    path = root / "cohort.csv"
    pd.DataFrame(rows)[list(REQUIRED_COHORT_COLUMNS)].to_csv(
        path, index=False, lineterminator="\n"
    )
    return path


def _bundle(tmp_path: Path, **snapshot_options):
    baseline, project = _baseline(tmp_path)
    spec = _spec(tmp_path, baseline)
    cohort = _cohort(tmp_path)
    snapshot = _snapshot(tmp_path, **snapshot_options)
    return spec, baseline, cohort, snapshot, project


def _run(tmp_path: Path, **snapshot_options):
    spec, baseline, cohort, snapshot, project = _bundle(tmp_path, **snapshot_options)
    return register_cohort(
        spec_path=spec,
        baseline_lock_path=baseline,
        cohort_manifest_path=cohort,
        snapshot_path=snapshot,
        project_root=project,
    )


def test_json_hash_is_order_independent():
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})


def test_spec_rejects_execution_authority(tmp_path):
    baseline, _ = _baseline(tmp_path)
    path = _spec(tmp_path, baseline)
    value = json.loads(path.read_text())
    value["authorization"]["paper_trading_authorized"] = True
    with pytest.raises(ValueError, match="prohibited authority"):
        validate_spec(value)


def test_cohort_schema_and_order_are_exact(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"ticker": "AAA"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="columns or order"):
        load_cohort_manifest(path)


def test_complete_registration_passes_all_fixed_gates(tmp_path):
    result = _run(tmp_path)
    assert result["passed"]
    assert result["status"] == "BROADER_UNIVERSE_COHORT_REGISTERED"
    assert len(result["checks"]) == 15
    assert result["checks"]["passed"].all()
    assert not result["authorization"]["broader_universe_backtest_authorized"]


def test_registration_fingerprint_is_deterministic(tmp_path):
    spec, baseline, cohort, snapshot, project = _bundle(tmp_path)
    first = register_cohort(
        spec_path=spec,
        baseline_lock_path=baseline,
        cohort_manifest_path=cohort,
        snapshot_path=snapshot,
        project_root=project,
    )
    second = register_cohort(
        spec_path=spec,
        baseline_lock_path=baseline,
        cohort_manifest_path=cohort,
        snapshot_path=snapshot,
        project_root=project,
    )
    assert first["registration_fingerprint"] == second["registration_fingerprint"]
    assert first["registration_id"] == second["registration_id"]


def test_source_after_cohort_date_is_rejected(tmp_path):
    spec, baseline, _, snapshot, project = _bundle(tmp_path)
    cohort = _cohort(tmp_path, source_after=True)
    result = register_cohort(
        spec_path=spec,
        baseline_lock_path=baseline,
        cohort_manifest_path=cohort,
        snapshot_path=snapshot,
        project_root=project,
    )
    assert not result["passed"]
    assert not result["checks"].set_index("check").loc["point_in_time_sources", "passed"]


def test_membership_source_tamper_is_rejected(tmp_path):
    spec, baseline, cohort, snapshot, project = _bundle(tmp_path)
    (tmp_path / "point_in_time_membership.csv").write_text("tampered\n")
    result = register_cohort(
        spec_path=spec,
        baseline_lock_path=baseline,
        cohort_manifest_path=cohort,
        snapshot_path=snapshot,
        project_root=project,
    )
    assert not result["checks"].set_index("check").loc[
        "point_in_time_sources", "passed"
    ]


def test_current_constituent_backfill_label_is_rejected(tmp_path):
    spec, baseline, _, snapshot, project = _bundle(tmp_path)
    cohort = _cohort(tmp_path, source="current_constituent_backfill")
    result = register_cohort(
        spec_path=spec,
        baseline_lock_path=baseline,
        cohort_manifest_path=cohort,
        snapshot_path=snapshot,
        project_root=project,
    )
    assert not result["passed"]
    assert not result["checks"].set_index("check").loc[
        "no_outcome_or_controlled_prefilter", "passed"
    ]


def test_rank_drift_is_rejected(tmp_path):
    spec, baseline, cohort, snapshot, project = _bundle(tmp_path)
    frame = pd.read_csv(cohort)
    frame.loc[frame["ticker"] == "AAA", "liquidity_rank"] = 2
    frame.loc[frame["ticker"] == "BBB", "liquidity_rank"] = 1
    frame.to_csv(cohort, index=False)
    result = register_cohort(
        spec_path=spec,
        baseline_lock_path=baseline,
        cohort_manifest_path=cohort,
        snapshot_path=snapshot,
        project_root=project,
    )
    assert not result["checks"].set_index("check").loc["duplicates_and_ranks", "passed"]


def test_snapshot_manifest_tamper_is_detected(tmp_path):
    _, _, _, snapshot, _ = _bundle(tmp_path)
    manifest = json.loads((snapshot / "manifest.json").read_text())
    manifest["period"] = "tampered"
    _write_json(snapshot / "manifest.json", manifest)
    assert not verify_snapshot(snapshot)["passed"]


def test_market_file_tamper_is_detected(tmp_path):
    spec, baseline, cohort, snapshot, project = _bundle(tmp_path)
    path = snapshot / "market/AAA.csv"
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    result = register_cohort(
        spec_path=spec,
        baseline_lock_path=baseline,
        cohort_manifest_path=cohort,
        snapshot_path=snapshot,
        project_root=project,
    )
    assert not result["checks"].set_index("check").loc["snapshot_integrity", "passed"]


def test_missing_prepared_column_is_rejected(tmp_path):
    result = _run(tmp_path, missing_column=True)
    assert not result["passed"]
    assert not result["checks"].set_index("check").loc["prepared_data_columns", "passed"]


def test_invalid_ohlc_is_rejected(tmp_path):
    result = _run(tmp_path, bad_ohlc=True)
    assert not result["passed"]
    assert not result["checks"].set_index("check").loc["market_data_quality", "passed"]


def test_locked_project_file_drift_is_rejected(tmp_path):
    spec, baseline, cohort, snapshot, project = _bundle(tmp_path)
    (project / "src/backtest/locked.py").write_text("LOCKED = False\n", encoding="utf-8")
    result = register_cohort(
        spec_path=spec,
        baseline_lock_path=baseline,
        cohort_manifest_path=cohort,
        snapshot_path=snapshot,
        project_root=project,
    )
    assert not result["checks"].set_index("check").loc[
        "baseline_project_files", "passed"
    ]


def test_save_writes_hashed_results_and_provenance(tmp_path):
    result = _run(tmp_path)
    paths = save_registration(result, tmp_path / "results")
    assert set(paths) == {"screen", "cohorts", "data_audit", "checks", "json", "provenance"}
    provenance = json.loads(paths["provenance"].read_text())
    assert provenance["status"] == "BROADER_UNIVERSE_COHORT_REGISTERED"
    for metadata in provenance["result_files"].values():
        assert sha256_file(Path(metadata["path"])) == metadata["sha256"]
    assert not provenance["authorization"]["paper_trading_authorized"]
