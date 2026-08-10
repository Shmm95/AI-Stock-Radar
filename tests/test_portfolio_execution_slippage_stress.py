from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import src.backtest.run_portfolio_execution_slippage_stress as stress
from src.backtest.portfolio_backtest_models import PortfolioBacktestConfig
from src.backtest.run_portfolio_execution_slippage_stress import (
    APPROVED_MFE_MAE_HOLDING_PATH_STAMP,
    APPROVED_SNAPSHOT_ID,
    APPROVED_STOP_WALK_FORWARD_STAMP,
    BASELINE_SCENARIO_ID,
    EXPECTED_MFE_MAE_RESULT_KEYS,
    MULTIPLIERS,
    NOT_IMPLEMENTED_HOLDING_KEYS,
    PRIMARY_STRESS_SCENARIO_ID,
    ExecutionStressProfile,
    _STRICT_SOURCE_AUTHORIZATION,
    _derive_holding_path_datasets,
    _derived_exit_semantic,
    _derived_extreme_order,
    _derived_holding_bucket,
    build_primary_gates,
    build_quality_screen,
    build_replay_checks,
    build_stress_config,
    build_stress_grid,
    build_trade_pairs,
    load_verified_source,
    run_execution_slippage_stress,
    save_execution_slippage_stress,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = PROJECT_ROOT / "data/backtests/portfolio/research_snapshots" / APPROVED_SNAPSHOT_ID
HOLDING_DIRECTORY = PROJECT_ROOT / "data/backtests/portfolio/mfe_mae_holding_path_attribution"
STOP_DIRECTORY = PROJECT_ROOT / "data/backtests/portfolio/stop_walk_forward"
REAL_MFE_MAE_TRADES_PATH = (
    HOLDING_DIRECTORY
    / f"portfolio_mfe_mae_holding_path_trades_{APPROVED_MFE_MAE_HOLDING_PATH_STAMP}.csv"
)
REAL_MFE_MAE_STATISTICS_PATH = (
    HOLDING_DIRECTORY
    / f"portfolio_mfe_mae_holding_path_statistics_{APPROVED_MFE_MAE_HOLDING_PATH_STAMP}.csv"
)


def base_config() -> PortfolioBacktestConfig:
    return PortfolioBacktestConfig()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_grid_is_exact_pre_registered_cartesian_product():
    grid = build_stress_grid()
    assert len(grid) == 16
    assert {(item.commission_multiplier, item.slippage_multiplier) for item in grid} == {
        (commission, slippage)
        for commission in MULTIPLIERS
        for slippage in MULTIPLIERS
    }
    assert grid[0].scenario_id == BASELINE_SCENARIO_ID
    assert ExecutionStressProfile(2, 2).scenario_id == PRIMARY_STRESS_SCENARIO_ID
    assert grid[-1].scenario_id == "C08_S08"


@pytest.mark.parametrize("commission,slippage", [(0, 1), (3, 1), (1, 16), (-1, 2)])
def test_profile_rejects_any_unregistered_multiplier(commission: int, slippage: int):
    with pytest.raises(ValueError, match="pre-registered grid"):
        ExecutionStressProfile(commission, slippage)


def test_stress_config_changes_only_cost_fields():
    baseline = base_config()
    profile = ExecutionStressProfile(4, 8)
    result = build_stress_config(baseline, profile)
    assert result.commission_rate == pytest.approx(0.002)
    assert result.minimum_fee == pytest.approx(4.0)
    assert result.slippage_bps == pytest.approx(40.0)
    protected = set(baseline.to_dict()).difference(
        {"commission_rate", "minimum_fee", "slippage_bps"}
    )
    assert all(result.to_dict()[field] == baseline.to_dict()[field] for field in protected)
    assert baseline.commission_rate == 0.0005
    assert baseline.minimum_fee == 1.0
    assert baseline.slippage_bps == 5.0


def trade_frame() -> pd.DataFrame:
    rows = []
    for scenario in (BASELINE_SCENARIO_ID, PRIMARY_STRESS_SCENARIO_ID):
        rows.append(
            {
                "scenario_id": scenario,
                "window_id": "W01",
                "ticker": "AAPL",
                "entry_timestamp": "2024-01-02",
                "exit_timestamp": "2024-01-03",
                "exit_reason": "STOP_LOSS",
                "entry_price": 100.0 if scenario == BASELINE_SCENARIO_ID else 100.1,
                "exit_price": 95.0 if scenario == BASELINE_SCENARIO_ID else 94.9,
                "quantity": 10.0,
                "total_fees": 2.0 if scenario == BASELINE_SCENARIO_ID else 4.0,
                "net_pnl": -52.0 if scenario == BASELINE_SCENARIO_ID else -56.0,
                "return_percent": -5.2 if scenario == BASELINE_SCENARIO_ID else -5.6,
            }
        )
    rows.append(
        {
            "scenario_id": BASELINE_SCENARIO_ID,
            "window_id": "W01",
            "ticker": "MSFT",
            "entry_timestamp": "2024-01-04",
            "exit_timestamp": "2024-01-05",
            "exit_reason": "FORCE_CLOSE_END",
            "entry_price": 200.0,
            "exit_price": 210.0,
            "quantity": 5.0,
            "total_fees": 2.0,
            "net_pnl": 48.0,
            "return_percent": 4.8,
        }
    )
    return pd.DataFrame(rows)


def test_trade_pairs_expose_matched_and_cost_driven_population_changes():
    pairs = build_trade_pairs(trade_frame())
    primary = pairs.loc[pairs["scenario_id"] == PRIMARY_STRESS_SCENARIO_ID]
    assert set(primary["match_status"]) == {"MATCHED", "BASELINE_ONLY"}
    matched = primary.loc[primary["match_status"] == "MATCHED"].iloc[0]
    assert matched["delta_total_fees"] == pytest.approx(2.0)
    assert matched["delta_net_pnl"] == pytest.approx(-4.0)
    baseline = pairs.loc[pairs["scenario_id"] == BASELINE_SCENARIO_ID]
    assert baseline["match_status"].eq("MATCHED").all()


def official_rows() -> pd.DataFrame:
    rows = []
    for index in range(1, 14):
        row = {"window_id": f"W{index:02d}", "model": "FIXED_BASELINE"}
        for field in stress.REPLAY_FIELDS:
            if field in {"completed_trades", "winning_trades", "losing_trades", "rejected_signals"}:
                row[field] = 1 if field != "losing_trades" else 0
            elif field == "profit_factor":
                row[field] = 2.0
            elif field == "average_exposure_percent":
                row[field] = 50.0
            elif field == "matched_benchmark_return_percent":
                row[field] = 5.0
            elif field == "excess_return_vs_matched_percent":
                row[field] = 4.0
            elif field == "maximum_drawdown_percent":
                row[field] = 3.0
            elif field == "total_fees":
                row[field] = 2.0
            elif field in {"ending_equity", "total_return_amount"}:
                row[field] = 10900.0 if field == "ending_equity" else 900.0
            else:
                row[field] = 9.0 if field != "win_rate_percent" else 100.0
        rows.append(row)
    return pd.DataFrame(rows)


def replay_actual() -> pd.DataFrame:
    frame = official_rows().drop(columns="model")
    frame.insert(1, "scenario_id", BASELINE_SCENARIO_ID)
    return frame


def test_replay_checks_require_exact_official_window_parity():
    checks = build_replay_checks(replay_actual(), official_rows())
    assert len(checks) == 13 * len(stress.REPLAY_FIELDS)
    assert bool(checks["passed"].all()) is True
    changed = replay_actual()
    changed.loc[0, "total_return_percent"] += 0.01
    with pytest.raises(ValueError, match="Official baseline replay mismatch"):
        build_replay_checks(changed, official_rows())


def comparison_frame(*, return_percent: float = 25.0, profit_factor: float = 1.5,
                     positive_windows: int = 8, drawdown_worsening: float = 2.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario_id": PRIMARY_STRESS_SCENARIO_ID,
                "compounded_return_percent": return_percent,
                "average_profit_factor": profit_factor,
                "positive_window_count": positive_windows,
                "drawdown_worsening_vs_baseline_percent": drawdown_worsening,
            }
        ]
    )


def test_primary_gates_are_fixed_and_never_authorize_changes():
    gates = build_primary_gates(comparison_frame())
    assert len(gates) == 4
    assert bool(gates["passed"].all()) is True
    assert bool(gates["all_primary_gates_passed"].all()) is True
    assert bool(gates["authorizes_baseline_change"].any()) is False
    assert bool(gates["authorizes_paper_or_production"].any()) is False
    failed = build_primary_gates(comparison_frame(return_percent=-1.0))
    assert bool(failed["all_primary_gates_passed"].any()) is False


def test_quality_screen_rejects_missing_scenarios_and_never_selects():
    aggregate = pd.DataFrame({"scenario_id": [item.scenario_id for item in build_stress_grid()]})
    test_runs = pd.DataFrame(
        {
            "scenario_id": [item.scenario_id for item in build_stress_grid() for _ in range(13)]
        }
    )
    replay = pd.DataFrame({"passed": [True]})
    pairs = pd.DataFrame(
        {
            "scenario_id": [BASELINE_SCENARIO_ID],
            "match_status": ["MATCHED"],
        }
    )
    screen = build_quality_screen(
        test_runs=test_runs,
        aggregate=aggregate,
        replay_checks=replay,
        trade_pairs=pairs,
    )
    assert bool(screen["passed"].all()) is True
    assert bool(screen["authorizes_baseline_change"].any()) is False
    with pytest.raises(ValueError, match="quality screen failed"):
        build_quality_screen(
            test_runs=test_runs.iloc[:-1],
            aggregate=aggregate,
            replay_checks=replay,
            trade_pairs=pairs,
        )


class FakeTrade:
    def __init__(self, cost: float):
        self.cost = cost

    def to_dict(self):
        return {
            "ticker": "AAPL",
            "asset_class": "EQUITY",
            "entry_timestamp": "2024-01-01",
            "exit_timestamp": "2024-01-02",
            "entry_portfolio_bar_index": 0,
            "exit_portfolio_bar_index": 1,
            "quantity": 10.0,
            "entry_price": 100.0 + self.cost,
            "exit_price": 110.0 - self.cost,
            "entry_fee": self.cost,
            "exit_fee": self.cost,
            "total_fees": self.cost * 2,
            "gross_pnl": 100.0,
            "net_pnl": 100.0 - self.cost * 2,
            "return_percent": 10.0 - self.cost,
            "holding_period_bars": 1,
            "exit_reason": "EXIT_SIGNAL_NEXT_OPEN",
            "signal_score": 80.0,
            "signal_reason": "TEST",
        }


def fake_snapshot() -> dict[str, object]:
    dates = pd.date_range("2024-01-01", periods=28, freq="D")
    data = pd.DataFrame(
        {
            "Open": 100.0,
            "High": 110.0,
            "Low": 90.0,
            "Close": 105.0,
        },
        index=dates,
    )
    windows = []
    for index in range(13):
        start = dates[index * 2]
        windows.append(
            {
                "window_id": f"W{index + 1:02d}",
                "train_start": start - pd.Timedelta(days=2),
                "train_end_exclusive": start,
                "test_start": start,
                "test_end_exclusive": start + pd.Timedelta(days=2),
            }
        )
    return {
        "manifest": {
            "snapshot_id": APPROVED_SNAPSHOT_ID,
            "fingerprint": stress.APPROVED_SNAPSHOT_FINGERPRINT,
        },
        "config": base_config(),
        "windows": pd.DataFrame(windows),
        "data_by_ticker": {"AAPL": data},
    }


def install_fake_engine(monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    def runner(*, data_by_ticker, config, include_benchmark):
        cost = config.commission_rate * 1000 + config.slippage_bps / 100
        ret = 10.0 - cost
        point1 = SimpleNamespace(timestamp="2024-01-01", total_equity=10000.0)
        point2 = SimpleNamespace(timestamp="2024-01-02", total_equity=10000.0 * (1 + ret / 100))
        return SimpleNamespace(
            config=config,
            equity_curve=[point1, point2],
            trades=[FakeTrade(cost)],
            rejections=[],
            average_exposure_percent=50.0,
            initial_cash=10000.0,
            ret=ret,
        )

    def summary(result, matched, annual, tickers):
        ret = result.ret
        return {
            "ending_equity": round(10000 * (1 + ret / 100), 4),
            "total_return_amount": round(ret * 100, 4),
            "total_return_percent": ret,
            "maximum_drawdown_percent": 3.0,
            "completed_trades": 1,
            "winning_trades": 1,
            "losing_trades": 0,
            "win_rate_percent": 100.0,
            "profit_factor": 2.0,
            "total_fees": result.trades[0].cost * 2,
            "average_exposure_percent": 50.0,
            "rejected_signals": 0,
            "matched_benchmark_return_percent": 5.0,
            "excess_return_vs_matched_percent": ret - 5.0,
        }

    monkeypatch.setattr(stress, "run_portfolio_backtest", runner)
    monkeypatch.setattr(stress, "run_matched_benchmark", lambda *args, **kwargs: ({"total_return_percent": 5.0}, pd.DataFrame()))
    monkeypatch.setattr(stress, "annual_returns_from_equity", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(stress, "ticker_contributions", lambda result: pd.DataFrame([{"ticker": "AAPL", "net_pnl": result.trades[0].to_dict()["net_pnl"]}]))
    monkeypatch.setattr(stress, "build_variant_summary", summary)
    official = []
    baseline_ret = 10.0 - (0.0005 * 1000 + 5.0 / 100)
    for index in range(13):
        official.append(
            {
                "window_id": f"W{index + 1:02d}",
                "model": "FIXED_BASELINE",
                "ending_equity": round(10000 * (1 + baseline_ret / 100), 4),
                "total_return_amount": round(baseline_ret * 100, 4),
                "total_return_percent": baseline_ret,
                "maximum_drawdown_percent": 3.0,
                "completed_trades": 1,
                "winning_trades": 1,
                "losing_trades": 0,
                "win_rate_percent": 100.0,
                "profit_factor": 2.0,
                "total_fees": (0.0005 * 1000 + 5.0 / 100) * 2,
                "average_exposure_percent": 50.0,
                "rejected_signals": 0,
                "matched_benchmark_return_percent": 5.0,
                "excess_return_vs_matched_percent": baseline_ret - 5.0,
            }
        )
    return pd.DataFrame(official)


def fake_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    official = install_fake_engine(monkeypatch)
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.txt"
    source.write_text("locked\n", encoding="utf-8")
    return run_execution_slippage_stress(
        snapshot=fake_snapshot(),
        official_baseline_test_runs=official,
        source_files={"source": source},
        source_hashes={"source": file_hash(source)},
        source_verification={
            "verified": True,
            "code_hash_verification": True,
            "official_source": True,
        },
        save_authorization=_STRICT_SOURCE_AUTHORIZATION,
        execution_slippage_stress_stamp="STRESS",
    )


def test_fixed_grid_runner_emits_all_runs_without_selecting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    bundle = fake_bundle(monkeypatch, tmp_path)
    assert len(bundle["test_runs"]) == 208
    assert len(bundle["aggregate"]) == 16
    assert bundle["summary"]["scenario_count"] == 16
    assert bundle["summary"]["selected_scenario"] is None
    assert bundle["summary"]["baseline_replay_passed"] is True
    assert bundle["summary"]["baseline_change_authorized"] is False
    assert bool(bundle["screen"]["passed"].all()) is True


def test_save_is_provenanced_non_overwriting_and_rechecks_sources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    bundle = fake_bundle(monkeypatch, tmp_path)
    paths = save_execution_slippage_stress(bundle, tmp_path / "out")
    assert paths["json"].is_file()
    assert paths["provenance"].is_file()
    assert paths["trade_pairs"].is_file()
    with pytest.raises(FileExistsError):
        save_execution_slippage_stress(bundle, tmp_path / "out")
    changed = fake_bundle(monkeypatch, tmp_path / "second")
    Path(changed["source_files"]["source"]).write_text("mutated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Source changed before save"):
        save_execution_slippage_stress(changed, tmp_path / "changed")


def test_main_no_save_never_calls_saver(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    comparison = pd.DataFrame(
        [
            {"scenario_id": BASELINE_SCENARIO_ID, "compounded_return_percent": 10.0, "maximum_drawdown_percent": 5.0, "average_profit_factor": 2.0, "positive_window_count": 8},
            {"scenario_id": PRIMARY_STRESS_SCENARIO_ID, "compounded_return_percent": 8.0, "maximum_drawdown_percent": 6.0, "average_profit_factor": 1.5, "positive_window_count": 7},
        ]
    )
    bundle = {
        "comparison": comparison,
        "summary": {"window_count": 13, "test_run_count": 208, "all_primary_gates_passed": True},
    }
    monkeypatch.setattr(stress, "load_verified_source", lambda **kwargs: {})
    monkeypatch.setattr(stress, "run_execution_slippage_stress", lambda **kwargs: bundle)
    monkeypatch.setattr(stress, "save_execution_slippage_stress", lambda *args, **kwargs: pytest.fail("save must not run"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stress",
            "--snapshot-directory", "snapshot",
            "--holding-path-directory", "holding",
            "--holding-path-stamp", APPROVED_MFE_MAE_HOLDING_PATH_STAMP,
            "--stop-walk-forward-directory", "stop",
            "--stop-walk-forward-stamp", APPROVED_STOP_WALK_FORWARD_STAMP,
            "--no-save",
        ],
    )
    stress.main()
    output = capsys.readouterr().out
    assert "Output artifacts saved: 0" in output
    assert "Scenario selected: NONE" in output


@pytest.mark.skipif(
    not SNAPSHOT_PATH.exists() or not HOLDING_DIRECTORY.exists() or not STOP_DIRECTORY.exists(),
    reason="Official frozen source artifacts are not installed.",
)
def test_official_source_integration_verifies_full_lineage_without_running_grid():
    source = load_verified_source(
        snapshot_directory=SNAPSHOT_PATH,
        holding_path_directory=HOLDING_DIRECTORY,
        holding_path_stamp=APPROVED_MFE_MAE_HOLDING_PATH_STAMP,
        stop_walk_forward_directory=STOP_DIRECTORY,
        stop_walk_forward_stamp=APPROVED_STOP_WALK_FORWARD_STAMP,
        project_root=PROJECT_ROOT,
    )
    assert source["source_verification"]["official_source"] is True
    assert source["source_verification"]["mfe_mae_holding_path_stamp"] == APPROVED_MFE_MAE_HOLDING_PATH_STAMP
    baseline = source["official_baseline_test_runs"]
    assert len(baseline.loc[baseline["model"] == "FIXED_BASELINE"]) == 13
    assert source["snapshot"]["manifest"]["snapshot_id"] == APPROVED_SNAPSHOT_ID
    # The repointing's core promise: mfe_mae's real 256-trade population,
    # loaded and hash-verified, with the 12 non-file-backed holding-path
    # datasets (11 tables + screen) derived in memory, and exactly
    # `path_rows` flagged as the one remaining, honestly-unimplemented key.
    assert len(source["holding_path_trades"]) == 256
    assert source["holding_path_not_implemented"] == ["path_rows"]
    assert source["holding_path_summary"] == {"strategy_change_authorized": False}
    datasets = source["holding_path_derived_datasets"]
    assert set(datasets) == {
        "asset_classes", "exit_categories", "exit_reasons", "exit_semantics",
        "extreme_orders", "force_close_excluded", "holding_buckets", "outcomes",
        "overall", "screen", "tickers", "windows",
    }
    screen = datasets["screen"]
    implemented = screen.loc[screen["status"] == "IMPLEMENTED"]
    assert len(implemented) == 8
    assert bool(implemented["passed"].all()) is True
    not_implemented = screen.loc[screen["status"] == "NOT_IMPLEMENTED"]
    assert set(not_implemented["check"]) == {"path_row_reconciliation", "open_exit_hlc_excluded"}
    assert bool(not_implemented["passed"].any()) is False, "must never fake a pass for a skipped check"
    not_applicable = screen.loc[screen["status"] == "NOT_APPLICABLE"]
    assert set(not_applicable["check"]) == {"mfe_bounds_ordered", "mae_bounds_ordered"}


# --- Real-data tests for the holding-path derivation layer -----------------
# These load mfe_mae's actual saved trades/statistics CSVs (not synthetic
# fixtures) whenever the reference copy is present, per the task's explicit
# ask for deterministic verification against real mfe_mae data.

_REAL_MFE_MAE_DATA_AVAILABLE = REAL_MFE_MAE_TRADES_PATH.exists() and REAL_MFE_MAE_STATISTICS_PATH.exists()
_real_mfe_mae_data_required = pytest.mark.skipif(
    not _REAL_MFE_MAE_DATA_AVAILABLE,
    reason="Real mfe_mae holding-path trades/statistics CSVs are not installed.",
)


@pytest.fixture(scope="module")
def real_mfe_mae_trades() -> pd.DataFrame:
    return pd.read_csv(REAL_MFE_MAE_TRADES_PATH)


@pytest.fixture(scope="module")
def real_mfe_mae_statistics() -> pd.DataFrame:
    return pd.read_csv(REAL_MFE_MAE_STATISTICS_PATH)


@pytest.fixture(scope="module")
def real_holding_path_datasets(real_mfe_mae_trades, real_mfe_mae_statistics) -> dict:
    return _derive_holding_path_datasets(
        mfe_mae_trades=real_mfe_mae_trades,
        mfe_mae_statistics=real_mfe_mae_statistics,
        stamp=APPROVED_MFE_MAE_HOLDING_PATH_STAMP,
        official_source=True,
    )


def test_derived_exit_semantic_matches_original_three_way_mapping():
    assert _derived_exit_semantic("EXIT_SIGNAL_NEXT_OPEN") == "OPEN_EXIT"
    assert _derived_exit_semantic("GAP_STOP_LOSS") == "OPEN_EXIT"
    assert _derived_exit_semantic("STOP_LOSS") == "INTRABAR_STOP_BOUNDED"
    assert _derived_exit_semantic("FORCE_CLOSE_END") == "CLOSE_EXIT"
    with pytest.raises(ValueError, match="Unsupported exit_reason"):
        _derived_exit_semantic("SOMETHING_ELSE")


def test_derived_holding_bucket_matches_original_thresholds():
    assert _derived_holding_bucket(0) == "0"
    assert _derived_holding_bucket(1) == "1_2"
    assert _derived_holding_bucket(2) == "1_2"
    assert _derived_holding_bucket(3) == "3_5"
    assert _derived_holding_bucket(5) == "3_5"
    assert _derived_holding_bucket(6) == "6_10"
    assert _derived_holding_bucket(10) == "6_10"
    assert _derived_holding_bucket(11) == "11_20"
    assert _derived_holding_bucket(20) == "11_20"
    assert _derived_holding_bucket(21) == "21_40"
    assert _derived_holding_bucket(40) == "21_40"
    assert _derived_holding_bucket(41) == "41_PLUS"
    assert _derived_holding_bucket(1000) == "41_PLUS"
    with pytest.raises(ValueError):
        _derived_holding_bucket(-1)


def test_derived_extreme_order_orders_by_offset_and_handles_missing():
    assert _derived_extreme_order(2, 5) == "MFE_BEFORE_MAE"
    assert _derived_extreme_order(5, 2) == "MAE_BEFORE_MFE"
    assert _derived_extreme_order(3, 3) == "AMBIGUOUS_SAME_BAR"
    assert _derived_extreme_order(None, 3) is None
    assert _derived_extreme_order(3, None) is None
    assert _derived_extreme_order(float("nan"), 3) is None


@_real_mfe_mae_data_required
def test_real_mfe_mae_trades_have_the_official_256_trade_population(real_mfe_mae_trades):
    assert len(real_mfe_mae_trades) == 256
    assert real_mfe_mae_trades["window_id"].nunique() == 13


@_real_mfe_mae_data_required
def test_real_derived_screen_is_eight_implemented_two_not_implemented_two_not_applicable(
    real_holding_path_datasets,
):
    screen = real_holding_path_datasets["screen"]
    assert len(screen) == 12
    implemented = screen.loc[screen["status"] == "IMPLEMENTED"]
    assert len(implemented) == 8
    assert bool(implemented["passed"].all()) is True
    assert set(screen.loc[screen["status"] == "NOT_IMPLEMENTED", "check"]) == {
        "path_row_reconciliation", "open_exit_hlc_excluded",
    }
    assert bool(screen.loc[screen["status"] == "NOT_IMPLEMENTED", "passed"].any()) is False
    assert set(screen.loc[screen["status"] == "NOT_APPLICABLE", "check"]) == {
        "mfe_bounds_ordered", "mae_bounds_ordered",
    }


@_real_mfe_mae_data_required
@pytest.mark.parametrize(
    "key,expected_group_type,both_populations",
    [
        ("asset_classes", "ASSET_CLASS", True),
        ("tickers", "TICKER", True),
        ("windows", "WINDOW_ID", True),
        ("exit_reasons", "EXIT_REASON", True),
        ("exit_categories", "EXIT_CATEGORY", True),
        ("outcomes", "OUTCOME_CLASS", True),
        ("overall", "ALL_TRADES", True),
        ("force_close_excluded", "ALL_TRADES", False),
        ("holding_buckets", "HOLDING_BUCKET", True),
        ("exit_semantics", "EXIT_SEMANTIC", True),
        ("extreme_orders", "CENSORED_EXTREME_ORDER", True),
    ],
)
def test_real_derived_table_has_correct_group_type_and_population_coverage(
    real_holding_path_datasets, key, expected_group_type, both_populations,
):
    frame = real_holding_path_datasets[key]
    assert not frame.empty
    assert set(frame["group_type"]) == {expected_group_type}
    populations = set(frame["population"])
    if both_populations:
        assert populations == {
            "PRIMARY_ALL_COMPLETED_BASELINE_TRADES", "SENSITIVITY_EXCLUDING_FORCE_CLOSE_END",
        }
    else:
        assert populations == {"SENSITIVITY_EXCLUDING_FORCE_CLOSE_END"}


@_real_mfe_mae_data_required
def test_real_exit_semantics_table_covers_all_256_trades_across_three_categories(
    real_mfe_mae_trades, real_holding_path_datasets,
):
    primary = real_holding_path_datasets["exit_semantics"]
    primary = primary.loc[
        (primary["population"] == "PRIMARY_ALL_COMPLETED_BASELINE_TRADES")
        & (primary["metric"] == primary["metric"].iloc[0])
    ]
    assert set(primary["group_value"]) == {"OPEN_EXIT", "INTRABAR_STOP_BOUNDED", "CLOSE_EXIT"}
    assert int(primary["population_trade_count"].sum()) == len(real_mfe_mae_trades)


@_real_mfe_mae_data_required
def test_real_extreme_orders_excludes_trades_with_no_fully_held_bar(
    real_mfe_mae_trades, real_holding_path_datasets,
):
    unavailable = int((real_mfe_mae_trades["full_held_bar_available"] == False).sum())  # noqa: E712
    extreme = real_holding_path_datasets["extreme_orders"]
    primary = extreme.loc[
        (extreme["population"] == "PRIMARY_ALL_COMPLETED_BASELINE_TRADES")
        & (extreme["metric"] == extreme["metric"].iloc[0])
    ]
    assert int(primary["population_trade_count"].sum()) == len(real_mfe_mae_trades) - unavailable


@_real_mfe_mae_data_required
def test_real_force_close_excluded_matches_sensitivity_population_size(
    real_mfe_mae_trades, real_holding_path_datasets,
):
    sensitivity_count = int((real_mfe_mae_trades["exit_reason"] != "FORCE_CLOSE_END").sum())
    force_close_excluded = real_holding_path_datasets["force_close_excluded"]
    assert int(force_close_excluded["population_trade_count"].iloc[0]) == sensitivity_count


@_real_mfe_mae_data_required
def test_repointed_provenance_result_keys_match_mfe_maes_real_three_files():
    assert EXPECTED_MFE_MAE_RESULT_KEYS == {"json", "statistics", "trades"}
    assert NOT_IMPLEMENTED_HOLDING_KEYS == frozenset({"path_rows"})
