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
    _derive_path_rows,
    _derived_exit_semantic,
    _derived_extreme_order,
    _derived_holding_bucket,
    _path_rows_for_trade,
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
from src.backtest.run_research_data_snapshot import load_snapshot


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
    # loaded and hash-verified, with the 13 non-file-backed holding-path
    # datasets (12 tables + screen) derived -- 12 in memory from mfe_mae's
    # own data, and `path_rows` reconstructed from the independently
    # loaded snapshot -- and NOTHING left flagged as unimplemented.
    assert len(source["holding_path_trades"]) == 256
    assert source["holding_path_not_implemented"] == []
    assert source["holding_path_summary"] == {"strategy_change_authorized": False}
    datasets = source["holding_path_derived_datasets"]
    assert set(datasets) == {
        "asset_classes", "exit_categories", "exit_reasons", "exit_semantics",
        "extreme_orders", "force_close_excluded", "holding_buckets", "outcomes",
        "overall", "path_rows", "screen", "tickers", "windows",
    }
    path_rows = datasets["path_rows"]
    assert not path_rows.empty
    expected_path_rows = int((source["holding_path_trades"]["holding_period_ticker_bars"] + 1).sum())
    assert len(path_rows) == expected_path_rows
    screen = datasets["screen"]
    implemented = screen.loc[screen["status"] == "IMPLEMENTED"]
    assert len(implemented) == 10
    assert bool(implemented["passed"].all()) is True
    assert set(implemented["check"]).issuperset({"path_row_reconciliation", "open_exit_hlc_excluded"})
    not_applicable = screen.loc[screen["status"] == "NOT_APPLICABLE"]
    assert set(not_applicable["check"]) == {"mfe_bounds_ordered", "mae_bounds_ordered"}
    assert len(screen) == 12, "12 total checks: 10 implemented + 2 structurally not applicable"


# --- Real-data tests for the holding-path derivation layer -----------------
# These load mfe_mae's actual saved trades/statistics CSVs (not synthetic
# fixtures) whenever the reference copy is present, per the task's explicit
# ask for deterministic verification against real mfe_mae data.

_REAL_MFE_MAE_DATA_AVAILABLE = (
    REAL_MFE_MAE_TRADES_PATH.exists()
    and REAL_MFE_MAE_STATISTICS_PATH.exists()
    and SNAPSHOT_PATH.exists()
)
_real_mfe_mae_data_required = pytest.mark.skipif(
    not _REAL_MFE_MAE_DATA_AVAILABLE,
    reason="Real mfe_mae holding-path trades/statistics CSVs or the snapshot are not installed.",
)


@pytest.fixture(scope="module")
def real_mfe_mae_trades() -> pd.DataFrame:
    return pd.read_csv(REAL_MFE_MAE_TRADES_PATH)


@pytest.fixture(scope="module")
def real_mfe_mae_statistics() -> pd.DataFrame:
    return pd.read_csv(REAL_MFE_MAE_STATISTICS_PATH)


@pytest.fixture(scope="module")
def real_snapshot() -> dict:
    return load_snapshot(SNAPSHOT_PATH, verify_code=True, project_root=PROJECT_ROOT)


@pytest.fixture(scope="module")
def real_holding_path_datasets(real_mfe_mae_trades, real_mfe_mae_statistics, real_snapshot) -> dict:
    return _derive_holding_path_datasets(
        mfe_mae_trades=real_mfe_mae_trades,
        mfe_mae_statistics=real_mfe_mae_statistics,
        stamp=APPROVED_MFE_MAE_HOLDING_PATH_STAMP,
        official_source=True,
        data_by_ticker=real_snapshot["data_by_ticker"],
        windows=real_snapshot["windows"],
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


# --- Synthetic unit tests for path_rows bar-slicing (no real data needed) --

def _bars(rows: dict[str, tuple[float, float, float, float]]) -> pd.DataFrame:
    """rows: {iso_timestamp: (open, high, low, close)}, in the given order."""
    index = pd.DatetimeIndex([pd.Timestamp(key) for key in rows])
    values = list(rows.values())
    return pd.DataFrame(
        {
            "Open": [v[0] for v in values],
            "High": [v[1] for v in values],
            "Low": [v[2] for v in values],
            "Close": [v[3] for v in values],
        },
        index=index,
    )


def _trade(
    *,
    trade_id: str = "T1",
    entry_timestamp: str,
    exit_timestamp: str,
    exit_reason: str,
    exit_semantic: str,
    holding_period_ticker_bars: int,
    ticker: str = "AAPL",
    window_id: str = "W1",
) -> dict:
    return {
        "trade_id": trade_id,
        "entry_timestamp": entry_timestamp,
        "exit_timestamp": exit_timestamp,
        "exit_reason": exit_reason,
        "exit_semantic": exit_semantic,
        "holding_period_ticker_bars": holding_period_ticker_bars,
        "ticker": ticker,
        "window_id": window_id,
    }


def test_path_rows_single_bar_stop_loss_trade_excludes_its_only_bar_hlc():
    """A stop touched on the SAME bar as entry (holding_period_ticker_bars=0):
    exactly one row, flagged as both entry and exit, and -- because the
    engine cannot know whether High or Low came first relative to the
    stop touch -- high_low_close_included must be False even though it's
    also the entry bar."""
    window_data = _bars({"2026-01-05": (100.0, 105.0, 94.0, 96.0)})
    trade = _trade(
        entry_timestamp="2026-01-05", exit_timestamp="2026-01-05",
        exit_reason="STOP_LOSS", exit_semantic="INTRABAR_STOP_BOUNDED",
        holding_period_ticker_bars=0,
    )

    rows = _path_rows_for_trade(trade, window_data=window_data, stamp="S1")

    assert len(rows) == 1
    row = rows[0]
    assert row["bar_role"] == "ENTRY_EXIT_BAR"
    assert row["is_entry_bar"] is True
    assert row["is_effective_exit_bar"] is True
    assert row["high_low_close_included"] is False
    assert row["open_included"] is True
    assert row["intrabar_order_uncertain"] is True
    assert row["raw_open"] == 100.0 and row["raw_high"] == 105.0


def test_path_rows_force_close_includes_full_exit_bar_hlc():
    """FORCE_CLOSE_END: the exit price IS that bar's own Close -- no
    post-exit price action to hide, so the exit bar's High/Low/Close
    must be fully included, unlike every other exit type."""
    window_data = _bars(
        {
            "2026-01-05": (100.0, 102.0, 99.0, 101.0),
            "2026-01-06": (101.0, 103.0, 100.0, 102.0),
            "2026-01-07": (102.0, 104.0, 101.0, 103.0),
        }
    )
    trade = _trade(
        entry_timestamp="2026-01-05", exit_timestamp="2026-01-07",
        exit_reason="FORCE_CLOSE_END", exit_semantic="CLOSE_EXIT",
        holding_period_ticker_bars=2,
    )

    rows = _path_rows_for_trade(trade, window_data=window_data, stamp="S1")

    assert len(rows) == 3
    assert [r["bar_role"] for r in rows] == ["ENTRY_BAR", "HOLDING_BAR", "EXIT_BAR"]
    assert [r["high_low_close_included"] for r in rows] == [True, True, True]
    assert rows[-1]["is_effective_exit_bar"] is True
    assert rows[-1]["raw_close"] == 103.0


def test_path_rows_open_exit_gap_scenario_excludes_exit_bar_hlc():
    """GAP_STOP_LOSS / EXIT_SIGNAL_NEXT_OPEN (OPEN_EXIT): the fill is the
    exit bar's Open; that bar's High/Low/Close happen AFTER the position
    was already closed and must never be treated as observed."""
    window_data = _bars(
        {
            "2026-01-05": (100.0, 101.0, 99.0, 100.5),
            "2026-01-06": (90.0, 92.0, 85.0, 87.0),  # the gap-down exit bar
        }
    )
    trade = _trade(
        entry_timestamp="2026-01-05", exit_timestamp="2026-01-06",
        exit_reason="GAP_STOP_LOSS", exit_semantic="OPEN_EXIT",
        holding_period_ticker_bars=1,
    )

    rows = _path_rows_for_trade(trade, window_data=window_data, stamp="S1")

    assert len(rows) == 2
    entry_row, exit_row = rows
    assert entry_row["high_low_close_included"] is True
    assert exit_row["bar_role"] == "EXIT_BAR"
    assert exit_row["high_low_close_included"] is False
    assert exit_row["open_included"] is True
    assert exit_row["raw_open"] == 90.0
    assert exit_row["intrabar_order_uncertain"] is False


def test_path_rows_reconciliation_mismatch_raises():
    window_data = _bars(
        {
            "2026-01-05": (100.0, 101.0, 99.0, 100.5),
            "2026-01-06": (90.0, 92.0, 85.0, 87.0),
        }
    )
    trade = _trade(
        entry_timestamp="2026-01-05", exit_timestamp="2026-01-06",
        exit_reason="GAP_STOP_LOSS", exit_semantic="OPEN_EXIT",
        holding_period_ticker_bars=5,  # wrong -- actual is 1
    )
    with pytest.raises(ValueError, match="path_rows reconciliation failed"):
        _path_rows_for_trade(trade, window_data=window_data, stamp="S1")


def test_path_rows_never_reads_bars_after_the_trades_own_exit():
    """Look-ahead check: window_data extends 20 bars past this trade's
    exit. Not only must no returned row's timestamp exceed the exit --
    corrupting every post-exit bar to an extreme sentinel value must not
    change the output at all, proving those values are never read."""
    rows_dict = {
        "2026-01-05": (100.0, 101.0, 99.0, 100.5),
        "2026-01-06": (101.0, 103.0, 100.0, 102.0),
        "2026-01-07": (90.0, 92.0, 85.0, 87.0),  # gap-down exit bar
    }
    for offset in range(20):
        timestamp = pd.Timestamp("2026-01-08") + pd.Timedelta(days=offset)
        rows_dict[timestamp.date().isoformat()] = (50.0, 55.0, 45.0, 48.0)
    window_data = _bars(rows_dict)
    trade = _trade(
        entry_timestamp="2026-01-05", exit_timestamp="2026-01-07",
        exit_reason="GAP_STOP_LOSS", exit_semantic="OPEN_EXIT",
        holding_period_ticker_bars=2,
    )

    baseline = _path_rows_for_trade(trade, window_data=window_data, stamp="S1")
    assert len(baseline) == 3
    exit_timestamp = pd.Timestamp("2026-01-07")
    assert all(row["ticker_timestamp"] <= exit_timestamp for row in baseline)

    corrupted = window_data.copy(deep=True)
    future_mask = corrupted.index > exit_timestamp
    assert future_mask.sum() == 20
    corrupted.loc[future_mask, ["Open", "High", "Low", "Close"]] = 999_999.0

    replayed = _path_rows_for_trade(trade, window_data=corrupted, stamp="S1")
    assert replayed == baseline, "corrupting post-exit bars must never change the result"


def test_derive_path_rows_slices_each_trade_to_its_own_window():
    """_derive_path_rows must bound each trade's market data to its OWN
    window ([test_start, test_end_exclusive)) before slicing to
    entry/exit -- proven here by two windows on the same ticker whose
    bars would otherwise overlap and by asserting cross-window bars
    never leak into either trade's path_rows."""
    aapl = _bars(
        {
            "2026-01-05": (100.0, 101.0, 99.0, 100.5),
            "2026-01-06": (100.5, 102.0, 99.5, 101.0),
            "2026-02-10": (200.0, 201.0, 199.0, 200.5),
            "2026-02-11": (200.5, 203.0, 199.5, 202.0),
        }
    )
    trades = pd.DataFrame(
        [
            _trade(
                trade_id="T1", ticker="AAPL", window_id="W1",
                entry_timestamp="2026-01-05", exit_timestamp="2026-01-06",
                exit_reason="FORCE_CLOSE_END", exit_semantic="CLOSE_EXIT",
                holding_period_ticker_bars=1,
            ),
            _trade(
                trade_id="T2", ticker="AAPL", window_id="W2",
                entry_timestamp="2026-02-10", exit_timestamp="2026-02-11",
                exit_reason="FORCE_CLOSE_END", exit_semantic="CLOSE_EXIT",
                holding_period_ticker_bars=1,
            ),
        ]
    )
    windows = pd.DataFrame(
        [
            {
                "window_id": "W1", "train_start": "2025-01-01", "train_end_exclusive": "2026-01-05",
                "test_start": "2026-01-05", "test_end_exclusive": "2026-01-20",
            },
            {
                "window_id": "W2", "train_start": "2025-02-01", "train_end_exclusive": "2026-02-10",
                "test_start": "2026-02-10", "test_end_exclusive": "2026-02-20",
            },
        ]
    )

    path_rows = _derive_path_rows(
        trades, data_by_ticker={"AAPL": aapl}, windows=windows, stamp="S1"
    )

    assert set(path_rows["trade_id"]) == {"T1", "T2"}
    t1_timestamps = set(path_rows.loc[path_rows["trade_id"] == "T1", "ticker_timestamp"])
    t2_timestamps = set(path_rows.loc[path_rows["trade_id"] == "T2", "ticker_timestamp"])
    assert t1_timestamps == {pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")}
    assert t2_timestamps == {pd.Timestamp("2026-02-10"), pd.Timestamp("2026-02-11")}
    assert t1_timestamps.isdisjoint(t2_timestamps)


@_real_mfe_mae_data_required
def test_real_mfe_mae_trades_have_the_official_256_trade_population(real_mfe_mae_trades):
    assert len(real_mfe_mae_trades) == 256
    assert real_mfe_mae_trades["window_id"].nunique() == 13


@_real_mfe_mae_data_required
def test_real_derived_screen_is_ten_implemented_two_not_applicable(
    real_holding_path_datasets,
):
    screen = real_holding_path_datasets["screen"]
    assert len(screen) == 12
    implemented = screen.loc[screen["status"] == "IMPLEMENTED"]
    assert len(implemented) == 10
    assert bool(implemented["passed"].all()) is True
    assert set(implemented["check"]).issuperset(
        {"path_row_reconciliation", "open_exit_hlc_excluded"}
    )
    assert (screen["status"] == "NOT_IMPLEMENTED").sum() == 0
    assert set(screen.loc[screen["status"] == "NOT_APPLICABLE", "check"]) == {
        "mfe_bounds_ordered", "mae_bounds_ordered",
    }


@_real_mfe_mae_data_required
def test_real_path_rows_reconciles_against_mfe_maes_own_holding_period(
    real_mfe_mae_trades, real_holding_path_datasets,
):
    path_rows = real_holding_path_datasets["path_rows"]
    expected = int((real_mfe_mae_trades["holding_period_ticker_bars"] + 1).sum())
    assert len(path_rows) == expected
    assert set(path_rows["trade_id"]) == set(real_mfe_mae_trades["trade_id"])
    # Every trade has exactly one entry bar and one effective-exit bar.
    per_trade = path_rows.groupby("trade_id")
    assert (per_trade["is_entry_bar"].sum() == 1).all()
    assert (per_trade["is_effective_exit_bar"].sum() == 1).all()


@_real_mfe_mae_data_required
def test_real_open_exit_bars_never_carry_high_low_close_in_path_rows(
    real_holding_path_datasets,
):
    path_rows = real_holding_path_datasets["path_rows"]
    open_exit_exit_bars = path_rows.loc[
        path_rows["is_effective_exit_bar"] & (path_rows["exit_semantic"] == "OPEN_EXIT")
    ]
    assert not open_exit_exit_bars.empty
    assert bool(open_exit_exit_bars["high_low_close_included"].eq(False).all())


@_real_mfe_mae_data_required
def test_real_close_exit_bars_do_include_high_low_close_in_path_rows(
    real_holding_path_datasets,
):
    path_rows = real_holding_path_datasets["path_rows"]
    close_exit_exit_bars = path_rows.loc[
        path_rows["is_effective_exit_bar"] & (path_rows["exit_semantic"] == "CLOSE_EXIT")
    ]
    assert not close_exit_exit_bars.empty
    assert bool(close_exit_exit_bars["high_low_close_included"].eq(True).all())


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
    assert NOT_IMPLEMENTED_HOLDING_KEYS == frozenset()
