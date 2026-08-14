from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest

import src.backtest.run_portfolio_holding_path_attribution as holding
from src.backtest.run_portfolio_holding_path_attribution import (
    AMBIGUOUS_SAME_BAR,
    APPROVED_FORWARD_RETURN_STATISTICS_STAMP,
    APPROVED_SNAPSHOT_ID,
    CLOSE_EXIT,
    INTRABAR_STOP_BOUNDED,
    MAE_BEFORE_MFE,
    OPEN_EXIT,
    PRIMARY_POPULATION,
    SENSITIVITY_POPULATION,
    _STRICT_SOURCE_AUTHORIZATION,
    build_aggregation_tables,
    build_holding_path_attribution,
    build_screen,
    enrich_holding_paths,
    exit_semantic,
    holding_bucket,
    holding_path_for_trade,
    load_verified_source,
    save_holding_path_attribution,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = PROJECT_ROOT / "data/backtests/portfolio/research_snapshots" / APPROVED_SNAPSHOT_ID
FORWARD_DIRECTORY = PROJECT_ROOT / "data/backtests/portfolio/forward_return_statistics"


def market_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "Open": [99.0, 104.0, 107.0, 102.0, 103.0],
            "High": [105.0, 110.0, 120.0, 112.0, 114.0],
            "Low": [98.0, 103.0, 90.0, 95.0, 96.0],
            "Close": [104.0, 108.0, 91.0, 111.0, 100.0],
        },
        index=index,
    )


def source_trade(
    data: pd.DataFrame,
    *,
    exit_reason: str = "EXIT_SIGNAL_NEXT_OPEN",
    entry_index: int = 0,
    exit_index: int = 2,
    exit_timestamp: object | None = None,
    source_trade_row: int = 0,
) -> dict[str, object]:
    semantic = exit_semantic(exit_reason)
    raw_exit = (
        data.iloc[exit_index]["Open"]
        if semantic == OPEN_EXIT
        else 95.0
        if semantic == INTRABAR_STOP_BOUNDED
        else data.iloc[exit_index]["Close"]
    )
    return {
        "forward_return_statistics_stamp": "FORWARD",
        "entry_statistics_stamp": "ENTRY",
        "timing_stamp": "TIMING",
        "source_stop_walk_forward_stamp": "SOURCE",
        "snapshot_id": "SNAPSHOT",
        "snapshot_fingerprint": "FINGERPRINT",
        "source_trade_row": source_trade_row,
        "trade_id": f"TRADE:{source_trade_row:06d}",
        "window_id": "W01",
        "model": "FIXED_BASELINE",
        "ticker": "AAPL",
        "asset_class": "EQUITY",
        "entry_timestamp": data.index[entry_index],
        "entry_fill_price": 100.0,
        "entry_notional": 1000.0,
        "quantity": 10.0,
        "initial_stop_percent": 5.0,
        "initial_stop_price": 95.0,
        "initial_risk_amount": 50.0,
        "exit_timestamp": data.index[exit_index] if exit_timestamp is None else exit_timestamp,
        "exit_price": raw_exit * 0.9995,
        "gross_pnl": (raw_exit - 100.0) * 10.0,
        "net_pnl": (raw_exit * 0.9995 - 100.0) * 10.0,
        "return_percent": (raw_exit * 0.9995 / 100.0 - 1) * 100,
        "holding_period_ticker_bars": exit_index - entry_index,
        "exit_reason": exit_reason,
        "exit_category": "STOP_LOSS" if "STOP_LOSS" in exit_reason else "TRAILING_CLOSE",
        "outcome_class": "WINNER" if raw_exit > 100 else "LOSER",
    }


def one_path(**changes: object):
    data = market_frame()
    trade = source_trade(data, **changes)
    record, rows = holding_path_for_trade(
        trade=trade,
        window_data=data,
        holding_path_attribution_stamp="HOLDING",
    )
    return data, trade, record, pd.DataFrame(rows)


def window_frame(data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "window_id": "W01",
                "test_start": data.index.min(),
                "test_end_exclusive": data.index.max() + pd.Timedelta(days=1),
            }
        ]
    )


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exit_semantics_are_closed_and_explicit():
    assert exit_semantic("EXIT_SIGNAL_NEXT_OPEN") == OPEN_EXIT
    assert exit_semantic("GAP_STOP_LOSS") == OPEN_EXIT
    assert exit_semantic("STOP_LOSS") == INTRABAR_STOP_BOUNDED
    assert exit_semantic("FORCE_CLOSE_END") == CLOSE_EXIT
    with pytest.raises(ValueError, match="Unsupported exit_reason"):
        exit_semantic("UNKNOWN")


@pytest.mark.parametrize("reason", ["EXIT_SIGNAL_NEXT_OPEN", "GAP_STOP_LOSS"])
def test_open_exit_includes_open_and_excludes_exit_high_low_close(reason: str):
    data, _, record, rows = one_path(exit_reason=reason)
    exit_row = rows.iloc[-1]
    assert record["exit_semantic"] == OPEN_EXIT
    assert record["censored_mfe_percent"] == pytest.approx(10.0)
    assert record["censored_mae_percent"] == pytest.approx(-2.0)
    assert record["possible_mfe_percent"] == pytest.approx(10.0)
    assert record["possible_mae_percent"] == pytest.approx(-2.0)
    assert record["market_exit_reference_price"] == data.iloc[2]["Open"]
    assert bool(exit_row["open_included"]) is True
    assert bool(exit_row["high_low_close_included"]) is False
    assert pd.isna(exit_row["censored_close_price"])
    assert record["full_observed_ticker_bar_count"] == 2


def test_intrabar_stop_has_certain_stop_touch_and_possible_ohlc_bounds():
    _, _, record, rows = one_path(exit_reason="STOP_LOSS")
    exit_row = rows.iloc[-1]
    assert record["exit_semantic"] == INTRABAR_STOP_BOUNDED
    assert record["censored_mfe_percent"] == pytest.approx(10.0)
    assert record["possible_mfe_percent"] == pytest.approx(20.0)
    assert record["censored_mae_percent"] == pytest.approx(-5.0)
    assert record["possible_mae_percent"] == pytest.approx(-10.0)
    assert record["mfe_bound_width_percent"] == pytest.approx(10.0)
    assert record["mae_bound_width_percent"] == pytest.approx(5.0)
    assert bool(record["mfe_exact"]) is False
    assert bool(record["mae_exact"]) is False
    assert exit_row["known_stop_touch_price"] == 95.0
    assert bool(exit_row["high_low_close_included"]) is False
    assert bool(exit_row["intrabar_order_uncertain"]) is True


def test_entry_bar_stop_anchors_mfe_at_zero_and_preserves_bounds():
    data = market_frame().iloc[[0]].copy()
    data.iloc[0, data.columns.get_loc("Low")] = 90.0
    trade = source_trade(data, exit_reason="STOP_LOSS", entry_index=0, exit_index=0)
    record, rows = holding_path_for_trade(
        trade=trade,
        window_data=data,
        holding_path_attribution_stamp="HOLDING",
    )
    assert record["path_ticker_bar_count"] == 1
    assert record["censored_mfe_percent"] == pytest.approx(0.0)
    assert record["censored_mae_percent"] == pytest.approx(-5.0)
    assert record["possible_mfe_percent"] == pytest.approx(5.0)
    assert record["possible_mae_percent"] == pytest.approx(-10.0)
    assert record["censored_extreme_order"] == AMBIGUOUS_SAME_BAR
    assert rows[0]["bar_role"] == "ENTRY_EXIT_BAR"


def test_force_close_maps_union_calendar_timestamp_to_last_real_ticker_bar():
    data = market_frame().iloc[:5].copy()
    data.index = pd.date_range("2024-01-01", periods=5, freq="B")
    union_saturday = data.index[-1] + pd.Timedelta(days=1)
    trade = source_trade(
        data,
        exit_reason="FORCE_CLOSE_END",
        exit_index=4,
        exit_timestamp=union_saturday,
    )
    record, rows = holding_path_for_trade(
        trade=trade,
        window_data=data,
        holding_path_attribution_stamp="HOLDING",
    )
    assert record["exit_semantic"] == CLOSE_EXIT
    assert record["effective_exit_market_timestamp"] == data.index[-1]
    assert record["market_exit_reference_price"] == data.iloc[-1]["Close"]
    assert bool(rows[-1]["high_low_close_included"]) is True
    assert rows[-1]["censored_close_price"] == data.iloc[-1]["Close"]
    assert record["full_observed_ticker_bar_count"] == 5


def test_censored_extreme_order_and_close_path_metrics_are_deterministic():
    _, _, record, _ = one_path(exit_reason="EXIT_SIGNAL_NEXT_OPEN")
    assert record["censored_extreme_order"] == MAE_BEFORE_MFE
    assert record["ticker_bars_to_censored_mae"] == 0
    assert record["ticker_bars_to_censored_mfe"] == 1
    assert record["maximum_close_excursion_percent"] == pytest.approx(8.0)
    assert record["minimum_close_excursion_percent"] == pytest.approx(0.0)
    assert record["maximum_close_drawdown_percent"] == pytest.approx(0.0)
    assert record["underwater_close_bar_count"] == 0
    assert record["observed_close_bar_count"] == 2


def test_same_bar_extremes_are_never_given_false_ordering():
    data = market_frame()
    data.iloc[0, data.columns.get_loc("High")] = 115.0
    data.iloc[0, data.columns.get_loc("Low")] = 85.0
    trade = source_trade(data, exit_reason="EXIT_SIGNAL_NEXT_OPEN")
    record, _ = holding_path_for_trade(
        trade=trade,
        window_data=data,
        holding_path_attribution_stamp="HOLDING",
    )
    assert record["censored_extreme_order"] == AMBIGUOUS_SAME_BAR


@pytest.mark.parametrize(
    ("bars", "bucket"),
    [(0, "0"), (1, "1_2"), (2, "1_2"), (3, "3_5"), (5, "3_5"), (6, "6_10"),
     (10, "6_10"), (11, "11_20"), (20, "11_20"), (21, "21_40"),
     (40, "21_40"), (41, "41_PLUS")],
)
def test_holding_bucket_boundaries_are_predefined(bars: int, bucket: str):
    assert holding_bucket(bars) == bucket


def test_elapsed_ticker_bar_reconciliation_fails_closed():
    data = market_frame()
    trade = source_trade(data)
    trade["holding_period_ticker_bars"] = 99
    with pytest.raises(ValueError, match="holding reconciliation failed"):
        holding_path_for_trade(
            trade=trade,
            window_data=data,
            holding_path_attribution_stamp="HOLDING",
        )


def test_sell_fill_cannot_be_better_than_engine_market_reference():
    data = market_frame()
    trade = source_trade(data)
    trade["exit_price"] = float(data.iloc[2]["Open"]) + 1.0
    with pytest.raises(ValueError, match="adverse slippage"):
        holding_path_for_trade(
            trade=trade,
            window_data=data,
            holding_path_attribution_stamp="HOLDING",
        )


def test_enrichment_preserves_source_order_and_emits_one_row_per_path_bar():
    data = market_frame()
    first = source_trade(data, exit_reason="STOP_LOSS", source_trade_row=7)
    second = source_trade(
        data,
        exit_reason="EXIT_SIGNAL_NEXT_OPEN",
        exit_index=3,
        source_trade_row=2,
    )
    trades, paths = enrich_holding_paths(
        trades=pd.DataFrame([first, second]),
        data_by_ticker={"AAPL": data},
        windows=window_frame(data),
        holding_path_attribution_stamp="HOLDING",
        expected_trade_count=2,
        expected_window_count=1,
    )
    assert trades["source_trade_row"].tolist() == [7, 2]
    assert len(paths) == int(trades["path_ticker_bar_count"].sum())
    assert paths.groupby("trade_id")["ticker_bar_offset"].apply(list).tolist() == [
        [0, 1, 2, 3],
        [0, 1, 2],
    ]


def synthetic_bundle(tmp_path: Path) -> dict[str, object]:
    data = market_frame()
    source_path = tmp_path / "source.txt"
    source_path.write_text("locked\n", encoding="utf-8")
    trade = source_trade(data, exit_reason="STOP_LOSS")
    return build_holding_path_attribution(
        trades=pd.DataFrame([trade]),
        data_by_ticker={"AAPL": data},
        windows=window_frame(data),
        forward_return_statistics_stamp="FORWARD",
        entry_statistics_stamp="ENTRY",
        timing_stamp="TIMING",
        source_stamp="SOURCE",
        snapshot_id="SNAPSHOT",
        snapshot_fingerprint="FINGERPRINT",
        expected_trade_count=1,
        expected_window_count=1,
        holding_path_attribution_stamp="HOLDING",
        source_files={"source": source_path},
        source_hashes={"source": file_hash(source_path)},
        source_verification={
            "verified": True,
            "code_hash_verification": True,
            "official_source": False,
        },
        save_authorization=_STRICT_SOURCE_AUTHORIZATION,
    )


def test_primary_and_force_close_excluded_statistics_are_separate():
    data = market_frame()
    normal = source_trade(data, exit_reason="STOP_LOSS", source_trade_row=0)
    forced = source_trade(data, exit_reason="FORCE_CLOSE_END", source_trade_row=1)
    trades, _ = enrich_holding_paths(
        trades=pd.DataFrame([normal, forced]),
        data_by_ticker={"AAPL": data},
        windows=window_frame(data),
        holding_path_attribution_stamp="HOLDING",
        expected_trade_count=2,
        expected_window_count=1,
    )
    tables = build_aggregation_tables(trades, stamp="HOLDING")
    primary = tables["overall"].query("population == @PRIMARY_POPULATION")
    sensitivity = tables["force_close_excluded"]
    assert primary["population_trade_count"].eq(2).all()
    assert sensitivity["population_trade_count"].eq(1).all()
    assert set(tables) == {
        "overall", "asset_classes", "tickers", "windows", "exit_reasons",
        "exit_categories", "outcomes", "holding_buckets", "extreme_orders",
        "exit_semantics", "force_close_excluded",
    }


def test_quality_screen_is_diagnostic_and_never_authorizes_strategy_change(tmp_path: Path):
    bundle = synthetic_bundle(tmp_path)
    screen = bundle["screen"]
    assert bool(screen["passed"].all()) is True
    assert bool(screen["authorizes_strategy_change"].any()) is False
    assert bundle["summary"]["strategy_change_authorized"] is False


def test_save_requires_official_strict_source_and_rejects_overwrite(tmp_path: Path):
    bundle = synthetic_bundle(tmp_path)
    with pytest.raises(ValueError, match="official source verification"):
        save_holding_path_attribution(bundle, tmp_path / "out")
    bundle["source_verification"]["official_source"] = True
    paths = save_holding_path_attribution(bundle, tmp_path / "out")
    assert paths["json"].is_file()
    assert paths["provenance"].is_file()
    assert paths["path_rows"].is_file()
    with pytest.raises(FileExistsError):
        save_holding_path_attribution(bundle, tmp_path / "out")


def test_save_rechecks_source_hash_immediately_before_writing(tmp_path: Path):
    bundle = synthetic_bundle(tmp_path)
    bundle["source_verification"]["official_source"] = True
    Path(bundle["source_files"]["source"]).write_text("mutated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Source changed before save"):
        save_holding_path_attribution(bundle, tmp_path / "out")


def test_main_no_save_never_calls_saver(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    bundle = {
        "forward_return_statistics_stamp": "FORWARD",
        "snapshot_id": "SNAPSHOT",
        "summary": {
            "primary_trade_count": 1,
            "window_count": 1,
            "path_row_count": 2,
            "intrabar_stop_bounded_trade_count": 1,
            "force_close_excluded_trade_count": 1,
        },
    }
    monkeypatch.setattr(holding, "load_verified_source", lambda **_: {"source": True})
    monkeypatch.setattr(holding, "run_holding_path_attribution", lambda _: bundle)
    monkeypatch.setattr(
        holding,
        "save_holding_path_attribution",
        lambda *_args, **_kwargs: pytest.fail("saver must not run"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "holding",
            "--snapshot-directory", "snapshot",
            "--forward-return-statistics-directory", "forward",
            "--forward-return-statistics-stamp", "FORWARD",
            "--no-save",
        ],
    )
    holding.main()
    output = capsys.readouterr().out
    assert "Output artifacts saved: 0" in output
    assert "Strategy change authorized: NO" in output


@pytest.mark.skipif(
    not SNAPSHOT_PATH.exists() or not FORWARD_DIRECTORY.exists(),
    reason="Official frozen source artifacts are not installed.",
)
def test_official_source_integration_reconciles_population_and_exit_semantics():
    source = load_verified_source(
        snapshot_directory=SNAPSHOT_PATH,
        forward_return_statistics_directory=FORWARD_DIRECTORY,
        forward_return_statistics_stamp=APPROVED_FORWARD_RETURN_STATISTICS_STAMP,
        project_root=PROJECT_ROOT,
    )
    bundle = holding.run_holding_path_attribution(source)
    trades = bundle["trades"]
    assert len(trades) == 256
    assert trades["window_id"].nunique() == 13
    assert trades["exit_reason"].value_counts().to_dict() == holding.OFFICIAL_EXIT_REASON_COUNTS
    assert bundle["summary"]["force_close_end_trade_count"] == 42
    assert bundle["summary"]["force_close_excluded_trade_count"] == 214
    assert bundle["summary"]["intrabar_stop_bounded_trade_count"] == 114
    assert bool(bundle["screen"]["passed"].all()) is True
    assert int(trades["path_ticker_bar_count"].sum()) == len(bundle["path_rows"])
