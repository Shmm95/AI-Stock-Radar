from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import src.backtest.run_portfolio_research_baseline_lock as lock
from src.backtest.run_portfolio_research_baseline_lock import (
    APPROVED_EXECUTION_STRESS_STAMP,
    APPROVED_HOLDING_PATH_STAMP,
    APPROVED_SNAPSHOT_FINGERPRINT,
    APPROVED_SNAPSHOT_ID,
    APPROVED_STOP_WALK_FORWARD_STAMP,
    BASELINE_SCENARIO_ID,
    CONTROLLED_TICKERS,
    LOCK_NAME,
    LOCK_STATUS,
    PRIMARY_STRESS_SCENARIO_ID,
    STRESS_RESULT_KEYS,
    _STRICT_SOURCE_AUTHORIZATION,
    build_lock_record,
    load_registry,
    load_verified_source,
    run_research_baseline_lock,
    save_research_baseline_lock,
    validate_registry,
    validate_stress_decision,
    verify_project_files,
    verify_runtime_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "config/research_baseline_lock_v1.json"
STRESS_DIRECTORY = PROJECT_ROOT / "data/backtests/portfolio/execution_slippage_stress"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def registered() -> dict[str, object]:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "lock_name": LOCK_NAME,
        "lock_id": "TEST_LOCK",
        "status": LOCK_STATUS,
        "scope": "CONTROLLED_NINE_ASSET_HISTORICAL_RESEARCH_ONLY",
        "lineage": {
            "snapshot_id": APPROVED_SNAPSHOT_ID,
            "snapshot_fingerprint": APPROVED_SNAPSHOT_FINGERPRINT,
            "holding_path_attribution_stamp": APPROVED_HOLDING_PATH_STAMP,
            "stop_walk_forward_stamp": APPROVED_STOP_WALK_FORWARD_STAMP,
            "execution_slippage_stress_stamp": APPROVED_EXECUTION_STRESS_STAMP,
        },
        "controlled_tickers": list(CONTROLLED_TICKERS),
        "baseline_config": lock.PortfolioBacktestConfig().to_dict(),
        "approved_project_files": {"source.py": "a" * 64},
        "execution_stress": {
            "stamp": APPROVED_EXECUTION_STRESS_STAMP,
            "provenance_sha256": "b" * 64,
            "scenario_ids": [
                f"C{commission:02d}_S{slippage:02d}"
                for commission in (1, 2, 4, 8)
                for slippage in (1, 2, 4, 8)
            ],
            "result_hashes": {
                name: {"path": f"/{name}", "sha256": "c" * 64}
                for name in STRESS_RESULT_KEYS
            },
        },
        "reference_metrics": {
            "baseline": {"compounded_return_percent": 10.0},
            "primary_stress": {"compounded_return_percent": 8.0},
        },
        "known_limitations": ["test"],
        "authorization": {
            "research_reference_locked": True,
            "baseline_parameter_change_authorized": False,
            "broader_universe_validated": False,
            "paper_trading_authorized": False,
            "production_authorized": False,
            "broker_access_authorized": False,
            "automation_authorized": False,
        },
        "following_stage": "BROADER_UNIVERSE_ROBUSTNESS_SPECIFICATION_ONLY",
    }


def test_registry_is_research_only_and_fail_closed():
    registry = registered()
    validate_registry(registry)
    damaged = copy.deepcopy(registry)
    damaged["authorization"]["paper_trading_authorized"] = True
    with pytest.raises(ValueError, match="forbidden execution authority"):
        validate_registry(damaged)


def test_registry_rejects_lineage_or_universe_drift():
    registry = registered()
    damaged = copy.deepcopy(registry)
    damaged["lineage"]["snapshot_id"] = "changed"
    with pytest.raises(ValueError, match="lineage mismatch"):
        validate_registry(damaged)
    damaged = copy.deepcopy(registry)
    damaged["controlled_tickers"] = damaged["controlled_tickers"][:-1]
    with pytest.raises(ValueError, match="controlled universe"):
        validate_registry(damaged)


def test_project_hash_verification_rejects_mutation(tmp_path: Path):
    source = tmp_path / "source.py"
    source.write_text("approved\n", encoding="utf-8")
    registry = registered()
    registry["approved_project_files"] = {"source.py": file_hash(source)}
    verified = verify_project_files(registry, tmp_path)
    assert verified["project:source.py"]["sha256"] == file_hash(source)
    source.write_text("mutated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="project file hash mismatch"):
        verify_project_files(registry, tmp_path)


def test_runtime_config_rejects_any_default_drift(monkeypatch: pytest.MonkeyPatch):
    registry = registered()
    verify_runtime_config(registry)

    class ChangedConfig:
        def to_dict(self):
            value = dict(registry["baseline_config"])
            value["slippage_bps"] = 6.0
            return value

    monkeypatch.setattr(lock, "PortfolioBacktestConfig", ChangedConfig)
    with pytest.raises(ValueError, match="slippage_bps"):
        verify_runtime_config(registry)


def synthetic_stress_source(registry: dict[str, object]) -> dict[str, object]:
    scenario_ids = registry["execution_stress"]["scenario_ids"]
    baseline = {
        "scenario_id": BASELINE_SCENARIO_ID,
        **registry["reference_metrics"]["baseline"],
    }
    primary = {
        "scenario_id": PRIMARY_STRESS_SCENARIO_ID,
        **registry["reference_metrics"]["primary_stress"],
    }
    comparison = pd.DataFrame([baseline, primary])
    aggregate = pd.DataFrame({"scenario_id": scenario_ids})
    trades = pd.DataFrame(
        {
            "ticker": list(CONTROLLED_TICKERS),
            "scenario_id": [BASELINE_SCENARIO_ID] * len(CONTROLLED_TICKERS),
        }
    )
    return {
        "payload": {
            "summary": {
                "scenario_count": 16,
                "window_count": 13,
                "test_run_count": 208,
                "baseline_replay_passed": True,
                "quality_screen_passed": True,
                "all_primary_gates_passed": True,
                "selected_scenario": None,
                "baseline_change_authorized": False,
                "paper_or_production_authorized": False,
            },
            "base_config": registry["baseline_config"],
        },
        "screen": pd.DataFrame(
            {
                "passed": [True] * 6,
                "authorizes_baseline_change": [False] * 6,
            }
        ),
        "primary_gates": pd.DataFrame(
            {
                "passed": [True] * 4,
                "authorizes_baseline_change": [False] * 4,
                "authorizes_paper_or_production": [False] * 4,
            }
        ),
        "replay_checks": pd.DataFrame(
            {"passed": [True] * 182, "difference": [0.0] * 182}
        ),
        "windows": pd.DataFrame({"window_id": [f"W{i:02d}" for i in range(1, 14)]}),
        "test_runs": pd.DataFrame({"row": list(range(208))}),
        "aggregate": aggregate,
        "comparison": comparison,
        "trades": trades,
    }


def test_stress_decision_requires_exact_replay_grid_metrics_and_no_authority():
    registry = registered()
    source = synthetic_stress_source(registry)
    checks = validate_stress_decision(registry, source)
    assert len(checks) == 13
    assert bool(checks["passed"].all()) is True
    assert bool(checks["authorizes_paper_or_production"].any()) is False
    damaged = copy.deepcopy(source)
    damaged["payload"]["summary"]["selected_scenario"] = PRIMARY_STRESS_SCENARIO_ID
    with pytest.raises(ValueError, match="selected_scenario"):
        validate_stress_decision(registry, damaged)


def test_stress_decision_rejects_metric_or_replay_drift():
    registry = registered()
    source = synthetic_stress_source(registry)
    source["comparison"].loc[
        source["comparison"]["scenario_id"] == BASELINE_SCENARIO_ID,
        "compounded_return_percent",
    ] += 0.01
    with pytest.raises(ValueError, match="Registered metric mismatch"):
        validate_stress_decision(registry, source)
    source = synthetic_stress_source(registry)
    source["replay_checks"].loc[0, "difference"] = 1e-4
    with pytest.raises(ValueError, match="replay tolerance"):
        validate_stress_decision(registry, source)


def bundle_for_save(tmp_path: Path) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.txt"
    source.write_text("locked\n", encoding="utf-8")
    registry_path = tmp_path / "registry.json"
    registry_path.write_text("{}\n", encoding="utf-8")
    registry = registered()
    stress_source = {
        "provenance": {"source_files": {"only": {"path": str(source)}}},
        "replay_checks": pd.DataFrame({"passed": [True] * 182}),
    }
    checks = pd.DataFrame(
        {
            "check": ["all"],
            "passed": [True],
            "detail": ["synthetic"],
            "authorizes_baseline_parameter_change": [False],
            "authorizes_paper_or_production": [False],
        }
    )
    record = build_lock_record(
        registry=registry,
        registry_path=registry_path,
        stress_source=stress_source,
        checks=checks,
        stamp="TEST",
    )
    return {
        "stamp": "TEST",
        "checks": checks,
        "record": record,
        "source_files": {"source": {"path": str(source), "sha256": file_hash(source)}},
        "source_hashes": {"source": file_hash(source)},
        "save_authorization": _STRICT_SOURCE_AUTHORIZATION,
    }


def test_save_writes_three_hash_linked_artifacts_and_never_overwrites(tmp_path: Path):
    bundle = bundle_for_save(tmp_path)
    paths = save_research_baseline_lock(bundle, tmp_path / "out")
    assert set(paths) == {"checks", "json", "provenance"}
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    assert file_hash(paths["checks"]) == provenance["result_files"]["checks"]["sha256"]
    assert file_hash(paths["json"]) == provenance["result_files"]["json"]["sha256"]
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_research_baseline_lock(bundle, tmp_path / "out")


def test_save_rechecks_sources_and_requires_strict_authorization(tmp_path: Path):
    bundle = bundle_for_save(tmp_path)
    Path(bundle["source_files"]["source"]["path"]).write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mutated before save"):
        save_research_baseline_lock(bundle, tmp_path / "mutated")
    bundle = bundle_for_save(tmp_path / "second")
    bundle["save_authorization"] = None
    with pytest.raises(PermissionError, match="strictly verified"):
        save_research_baseline_lock(bundle, tmp_path / "unauthorized")


def test_no_save_cli_never_calls_saver(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    bundle = {
        "checks": pd.DataFrame({"passed": [True]}),
        "record": {"status": LOCK_STATUS},
    }
    monkeypatch.setattr(lock, "load_verified_source", lambda **_: {})
    monkeypatch.setattr(lock, "run_research_baseline_lock", lambda _: bundle)
    monkeypatch.setattr(
        lock,
        "save_research_baseline_lock",
        lambda *_: pytest.fail("saver must not run under --no-save"),
    )
    lock.main(["--no-save"])
    assert "Output artifacts saved: 0" in capsys.readouterr().out


def test_official_registry_and_full_lineage_pass():
    if not REGISTRY_PATH.exists() or not STRESS_DIRECTORY.exists():
        pytest.skip("Approved untracked research artifacts are unavailable.")
    registry = load_registry(REGISTRY_PATH)
    assert registry["lock_name"] == LOCK_NAME
    source = load_verified_source(
        registry_path=REGISTRY_PATH,
        execution_stress_directory=STRESS_DIRECTORY,
        execution_stress_stamp=APPROVED_EXECUTION_STRESS_STAMP,
        project_root=PROJECT_ROOT,
    )
    bundle = run_research_baseline_lock(source)
    assert len(bundle["checks"]) == 13
    assert bundle["record"]["status"] == LOCK_STATUS
    assert bundle["record"]["authorization"]["paper_trading_authorized"] is False


# --- Regression: registry references to execution_slippage_stress.py ------
# Today's mfe_mae repointing + path_rows implementation changed the real
# content of run_portfolio_execution_slippage_stress.py and its test file
# twice, making the registry's approved_project_files hashes for those two
# files stale (confirmed stale before this fix, via a direct sha256
# comparison against the then-current files). APPROVED_EXECUTION_STRESS_STAMP
# and APPROVED_HOLDING_PATH_STAMP are deliberately NOT touched by that fix:
# they describe the one real saved execution-stress run (20260802_192047),
# produced by the pre-repointing code, which genuinely recorded
# holding_path_attribution_stamp=20260802_155749 in its own provenance --
# changing them without a new saved run would make the registry assert a
# provenance value no real file actually has. Gated on REGISTRY_PATH only
# (not STRESS_DIRECTORY, which is a separate, larger gap -- see the report):
# this test only needs the registry JSON and the two live source files,
# both present locally, and does not require the missing saved
# execution-stress output directory.
_registry_required = pytest.mark.skipif(
    not REGISTRY_PATH.exists(), reason="config/research_baseline_lock_v1.json is not installed."
)


@_registry_required
@pytest.mark.parametrize(
    "relative_path",
    [
        "src/backtest/run_portfolio_execution_slippage_stress.py",
        "tests/test_portfolio_execution_slippage_stress.py",
    ],
)
def test_registry_execution_slippage_stress_references_match_current_file(relative_path: str):
    registry = load_registry(REGISTRY_PATH)
    expected = registry["approved_project_files"][relative_path]
    actual = file_hash(PROJECT_ROOT / relative_path)
    assert actual == expected, (
        f"{relative_path}'s registered hash is stale relative to its current "
        "content -- update config/research_baseline_lock_v1.json."
    )


@_registry_required
def test_registry_stamps_still_describe_the_one_real_saved_stress_run():
    registry = load_registry(REGISTRY_PATH)
    assert registry["lineage"]["execution_slippage_stress_stamp"] == APPROVED_EXECUTION_STRESS_STAMP
    assert registry["lineage"]["holding_path_attribution_stamp"] == APPROVED_HOLDING_PATH_STAMP
    assert registry["execution_stress"]["stamp"] == APPROVED_EXECUTION_STRESS_STAMP
