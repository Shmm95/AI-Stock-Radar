from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.backtest.run_portfolio_forward_return_statistics as forward_statistics
from src.backtest.run_portfolio_forward_return_statistics import (
    APPROVED_ENTRY_STATISTICS_STAMP,
    APPROVED_SNAPSHOT_ID,
    AVAILABLE,
    EXITED_BEFORE_HORIZON,
    EXITED_ON_HORIZON_TIMESTAMP,
    FORWARD_COLUMNS,
    HORIZONS,
    INSUFFICIENT_BARS,
    MODEL_BASELINE,
    OFFICIAL_AVAILABILITY_COUNTS,
    OFFICIAL_EXPECTATIONS,
    OPEN_THROUGH_HORIZON,
    PRIMARY_POPULATION,
    SENSITIVITY_POPULATION,
    STATISTICS_COLUMNS,
    _STRICT_SOURCE_AUTHORIZATION,
    _resolved_expectations,
    build_aggregation_tables,
    build_forward_return_statistics,
    build_json_payload,
    build_statistics_table,
    enrich_forward_returns,
    forward_fields_for_trade,
    load_verified_source,
    run_forward_return_statistics,
    save_forward_return_statistics,
    terminal_index,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = PROJECT_ROOT / "data/backtests/portfolio/research_snapshots" / APPROVED_SNAPSHOT_ID
ENTRY_DIRECTORY = PROJECT_ROOT / "data/backtests/portfolio/entry_statistics"


def market_frame(*, ticker: str = "AAPL", periods: int = 20) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="D" if ticker.endswith("-USD") else "B")
    close = pd.Series(np.arange(periods, dtype=float) + 100.0, index=index)
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 2.0,
            "Low": close - 3.0,
            "Close": close,
        },
        index=index,
    )


def window_frame(data: pd.DataFrame, window_id: str = "W01") -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "window_id": window_id,
            "test_start": data.index[0],
            "test_end_exclusive": data.index[-1] + pd.Timedelta(days=1),
        }]
    )


def source_trade(
    data: pd.DataFrame,
    *,
    ticker: str = "AAPL",
    entry_index: int = 3,
    exit_index: int | None = None,
    exit_timestamp: pd.Timestamp | None = None,
    exit_reason: str = "EXIT_SIGNAL_NEXT_OPEN",
    model: str = MODEL_BASELINE,
    source_trade_row: int = 0,
) -> dict[str, object]:
    actual_exit_index = entry_index + 3 if exit_index is None else exit_index
    return {
        "entry_statistics_stamp": "ENTRY",
        "timing_stamp": "TIMING",
        "source_stop_walk_forward_stamp": "SOURCE",
        "snapshot_id": "SNAPSHOT",
        "snapshot_fingerprint": "FINGERPRINT",
        "source_trade_row": source_trade_row,
        "trade_id": f"TIMING:{source_trade_row:06d}",
        "window_id": "W01",
        "model": model,
        "ticker": ticker,
        "asset_class": "CRYPTO" if ticker.endswith("-USD") else "EQUITY",
        "entry_timestamp": data.index[entry_index],
        "entry_fill_price": 100.0,
        "entry_year": int(data.index[entry_index].year),
        "entry_gap_bucket": "ZERO_TO_POS_0P5",
        "exit_timestamp": data.index[actual_exit_index] if exit_timestamp is None else exit_timestamp,
        "exit_reason": exit_reason,
        "exit_category": "FORCE_CLOSE_END" if exit_reason == "FORCE_CLOSE_END" else "TREND_EXIT",
        "outcome_class": "WINNER",
        "signal_macd": -99.0,
        "signal_score": 88.0,
    }


def enrich_one(
    *,
    data: pd.DataFrame | None = None,
    trade: dict[str, object] | None = None,
    ticker: str = "AAPL",
) -> pd.DataFrame:
    active_data = data if data is not None else market_frame(ticker=ticker)
    active_trade = trade if trade is not None else source_trade(active_data, ticker=ticker)
    return enrich_forward_returns(
        trades=pd.DataFrame([active_trade]),
        data_by_ticker={ticker: active_data},
        windows=window_frame(active_data),
        forward_return_statistics_stamp="FORWARD",
        expected_trade_count=1,
        expected_window_count=1,
    )


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_verification(
    *,
    entry_statistics_stamp: str = "ENTRY",
    timing_stamp: str = "TIMING",
    source_stamp: str = "SOURCE",
    snapshot_id: str = "SNAPSHOT",
    snapshot_fingerprint: str = "FINGERPRINT",
    code_hash_verification: bool = True,
    official_source: bool = False,
) -> dict[str, object]:
    return {
        "verified": True,
        "code_hash_verification": code_hash_verification,
        "official_source": official_source,
        "entry_statistics_stamp": entry_statistics_stamp,
        "timing_stamp": timing_stamp,
        "source_stop_walk_forward_stamp": source_stamp,
        "snapshot_id": snapshot_id,
        "snapshot_fingerprint": snapshot_fingerprint,
    }


def test_horizons_use_completed_ticker_bars_and_reject_i_plus_horizon():
    data = market_frame()
    row = enrich_one(data=data).iloc[0]
    entry = 3
    assert terminal_index(entry, 1) == entry
    assert terminal_index(entry, 3) == entry + 2
    assert terminal_index(entry, 5) == entry + 4
    assert terminal_index(entry, 10) == entry + 9
    assert row["forward_1_terminal_ticker_timestamp"] == data.index[entry]
    assert row["forward_3_terminal_ticker_timestamp"] == data.index[entry + 2]
    assert row["forward_5_terminal_ticker_timestamp"] == data.index[entry + 4]
    assert row["forward_10_terminal_ticker_timestamp"] == data.index[entry + 9]
    assert row["forward_10_terminal_ticker_timestamp"] != data.index[entry + 10]
    with pytest.raises(ValueError, match="Unsupported horizon"):
        terminal_index(entry, 2)


def test_h1_uses_entry_close_high_low_and_inclusive_formulas():
    data = market_frame()
    trade = source_trade(data, entry_index=3, exit_index=10)
    row = enrich_one(data=data, trade=trade).iloc[0]
    entry_row = data.iloc[3]
    assert bool(row["forward_1_available"]) is True
    assert row["forward_1_availability_reason"] == AVAILABLE
    assert row["forward_1_terminal_close"] == entry_row["Close"]
    assert row["forward_1_maximum_observed_high"] == entry_row["High"]
    assert row["forward_1_minimum_observed_low"] == entry_row["Low"]
    assert row["forward_1_close_to_entry_fill_return_percent"] == pytest.approx(3.0)
    assert row["forward_1_observed_high_excursion_percent"] == pytest.approx(5.0)
    assert row["forward_1_observed_low_excursion_percent"] == pytest.approx(0.0)


def test_h3_h5_h10_exact_return_and_excursion_formulas():
    data = market_frame()
    row = enrich_one(data=data).iloc[0]
    for horizon in HORIZONS:
        terminal = 3 + horizon - 1
        observed = data.iloc[3 : terminal + 1]
        assert row[f"forward_{horizon}_terminal_close"] == data.iloc[terminal]["Close"]
        assert row[f"forward_{horizon}_close_to_entry_fill_return_percent"] == pytest.approx(
            (data.iloc[terminal]["Close"] / 100 - 1) * 100
        )
        assert row[f"forward_{horizon}_maximum_observed_high"] == observed["High"].max()
        assert row[f"forward_{horizon}_minimum_observed_low"] == observed["Low"].min()


def test_ticker_local_equity_weekend_and_seven_day_crypto_calendars():
    equity = market_frame(ticker="AAPL", periods=15)
    crypto = market_frame(ticker="BTC-USD", periods=15)
    equity_row = enrich_one(data=equity).iloc[0]
    crypto_row = enrich_one(data=crypto, ticker="BTC-USD").iloc[0]
    assert equity.index[3].weekday() == 3
    assert equity_row["forward_3_terminal_ticker_timestamp"] == equity.index[5]
    assert equity.index[5] - equity.index[3] >= pd.Timedelta(days=4)
    assert crypto_row["forward_3_terminal_ticker_timestamp"] == crypto.index[5]
    assert crypto.index[5] - crypto.index[3] == pd.Timedelta(days=2)


def test_no_cross_window_lookahead_and_unavailable_horizons_remain_rows():
    data = market_frame(periods=10)
    windows = pd.DataFrame(
        [{"window_id": "W01", "test_start": data.index[0], "test_end_exclusive": data.index[6]}]
    )
    trade = source_trade(data, entry_index=3)
    result = enrich_forward_returns(
        trades=pd.DataFrame([trade]),
        data_by_ticker={"AAPL": data},
        windows=windows,
        forward_return_statistics_stamp="FORWARD",
        expected_trade_count=1,
        expected_window_count=1,
    ).iloc[0]
    assert bool(result["forward_3_available"]) is True
    assert bool(result["forward_5_available"]) is False
    assert result["forward_5_availability_reason"] == INSUFFICIENT_BARS
    assert pd.isna(result["forward_5_terminal_close"])
    assert bool(result["forward_10_available"]) is False


def test_exit_relations_include_same_day_gap_stop_and_post_exit_outcomes():
    data = market_frame()
    same_day = source_trade(data, entry_index=3, exit_index=3, exit_reason="STOP_LOSS")
    row = enrich_one(data=data, trade=same_day).iloc[0]
    assert row["forward_1_exit_relation"] == EXITED_ON_HORIZON_TIMESTAMP
    assert pd.isna(row["forward_1_trade_active_through_horizon"])
    assert row["forward_3_exit_relation"] == EXITED_BEFORE_HORIZON
    assert bool(row["forward_3_trade_active_through_horizon"]) is False
    assert not pd.isna(row["forward_3_terminal_close"])

    open_trade = source_trade(data, entry_index=3, exit_index=15)
    open_row = enrich_one(data=data, trade=open_trade).iloc[0]
    assert open_row["forward_10_exit_relation"] == OPEN_THROUGH_HORIZON
    assert bool(open_row["forward_10_trade_active_through_horizon"]) is True


def test_force_close_equity_on_crypto_only_timestamp_never_fabricates_bar():
    data = market_frame(periods=12)
    entry_index = 3
    crypto_only_exit = data.index[4] + pd.Timedelta(days=1)
    assert crypto_only_exit.weekday() >= 5
    assert crypto_only_exit not in data.index
    trade = source_trade(
        data,
        entry_index=entry_index,
        exit_timestamp=crypto_only_exit,
        exit_reason="FORCE_CLOSE_END",
    )
    row = enrich_one(data=data, trade=trade).iloc[0]
    assert bool(row["forward_5_available"]) is True
    assert row["forward_5_terminal_ticker_timestamp"] == data.index[entry_index + 4]
    assert row["forward_5_exit_relation"] == EXITED_BEFORE_HORIZON
    tables = build_aggregation_tables(pd.DataFrame([row]), forward_return_statistics_stamp="FORWARD")
    primary = tables["overall"].query("population == @PRIMARY_POPULATION")
    sensitivity = tables["force_close_excluded"]
    assert primary["population_trade_count"].eq(1).all()
    assert sensitivity["population_trade_count"].eq(0).all()


@pytest.mark.parametrize("column,value", [
    ("Open", None), ("High", float("nan")), ("Low", 0.0), ("Close", -1.0),
    ("Close", float("inf")),
])
def test_invalid_ohlc_values_fail_closed(column: str, value: object):
    data = market_frame()
    data.loc[data.index[3], column] = value
    with pytest.raises(ValueError, match=f"Invalid {column}"):
        enrich_one(data=data)


@pytest.mark.parametrize(
    ("name", "changes"),
    [
        ("High >= Low", {"High": 99.0}),
        ("High >= Open", {"High": 102.0}),
        ("High >= Close", {"High": 102.75}),
        ("Low <= Open", {"Low": 102.75}),
        ("Low <= Close", {"Open": 104.0, "Low": 103.5}),
    ],
)
def test_invalid_ohlc_cross_field_invariants_fail_closed(name: str, changes: dict[str, float]):
    data = market_frame()
    timestamp = data.index[3]
    for column, value in changes.items():
        data.loc[timestamp, column] = value
    with pytest.raises(ValueError, match=name):
        enrich_one(data=data)


def test_equal_boundary_ohlc_and_entry_fill_above_high_are_accepted():
    data = market_frame()
    timestamp = data.index[3]
    data.loc[timestamp, ["Open", "High", "Low", "Close"]] = 100.0
    trade = source_trade(data, entry_index=3)
    trade["entry_fill_price"] = 200.0
    output = enrich_one(data=data, trade=trade)
    assert len(output) == 1
    assert output.iloc[0]["forward_1_maximum_observed_high"] == 100.0


def test_duplicate_ticker_timestamps_are_rejected():
    data = market_frame()
    duplicate = pd.concat([data, data.iloc[[3]]]).sort_index()
    with pytest.raises(ValueError, match="duplicate timestamps"):
        enrich_one(data=duplicate)


def test_future_outcomes_do_not_mutate_entry_fields_or_source_inputs():
    data = market_frame()
    trade = source_trade(data)
    original_data = data.copy(deep=True)
    original_trade = dict(trade)
    before = enrich_one(data=data, trade=trade).iloc[0]
    changed = data.copy(deep=True)
    changed.loc[changed.index > data.index[3], ["Open", "High", "Low", "Close"]] = 500.0
    after = enrich_one(data=changed, trade=trade).iloc[0]
    for column in trade:
        assert before[column] == after[column]
    assert before["forward_3_terminal_close"] != after["forward_3_terminal_close"]
    pd.testing.assert_frame_equal(data, original_data)
    assert trade == original_trade


def test_macd_is_preserved_diagnostic_context_and_does_not_filter():
    data = market_frame()
    trade = source_trade(data)
    trade["signal_macd"] = -1_000_000.0
    row = enrich_one(data=data, trade=trade).iloc[0]
    assert len(row) > 0
    assert row["signal_macd"] == -1_000_000.0


def test_non_baseline_rows_are_rejected_and_source_order_is_preserved():
    data = market_frame()
    baseline_one = source_trade(data, source_trade_row=2)
    alternate = source_trade(data, model="FIXED_MAX_RETURN", source_trade_row=1)
    baseline_two = source_trade(data, source_trade_row=4)
    trades = pd.DataFrame([baseline_one, alternate, baseline_two])
    with pytest.raises(ValueError, match="FIXED_BASELINE"):
        enrich_forward_returns(
            trades=trades,
            data_by_ticker={"AAPL": data},
            windows=window_frame(data),
            forward_return_statistics_stamp="FORWARD",
        )
    valid = pd.DataFrame([baseline_one, baseline_two])
    result = enrich_forward_returns(
        trades=valid,
        data_by_ticker={"AAPL": data},
        windows=window_frame(data),
        forward_return_statistics_stamp="FORWARD",
        expected_trade_count=2,
        expected_window_count=1,
    )
    assert result["source_trade_row"].tolist() == [2, 4]
    bad = valid.copy()
    bad.loc[0, "ticker"] = "SPY"
    with pytest.raises(ValueError, match="controlled basket"):
        enrich_forward_returns(
            trades=bad,
            data_by_ticker={"SPY": data},
            windows=window_frame(data),
            forward_return_statistics_stamp="FORWARD",
        )


def test_statistics_use_ddof_zero_linear_percentiles_and_both_populations():
    data = market_frame()
    normal = enrich_one(data=data).iloc[0].copy()
    forced = enrich_one(data=data).iloc[0].copy()
    normal["source_trade_row"] = 0
    normal["trade_id"] = "TIMING:000000"
    forced["source_trade_row"] = 1
    forced["trade_id"] = "TIMING:000001"
    forced["exit_reason"] = "FORCE_CLOSE_END"
    frame = pd.DataFrame([normal, forced])
    frame.loc[:, "forward_1_close_to_entry_fill_return_percent"] = [0.0, 10.0]
    table = build_statistics_table(
        frame,
        forward_return_statistics_stamp="FORWARD",
        population=PRIMARY_POPULATION,
        group_type="ALL_TRADES",
    )
    record = table.query("horizon_ticker_bars == 1 and metric == 'forward_1_close_to_entry_fill_return_percent'").iloc[0]
    assert record["population_standard_deviation"] == pytest.approx(5.0)
    assert record["p25"] == pytest.approx(2.5)
    assert record["p75"] == pytest.approx(7.5)
    tables = build_aggregation_tables(frame, forward_return_statistics_stamp="FORWARD")
    assert set(tables["overall"]["population"]) == {PRIMARY_POPULATION, SENSITIVITY_POPULATION}
    assert set(tables["windows"]["group_type"]) == {"WINDOW_ID"}
    assert set(tables["entry_gap_buckets"]["group_type"]) == {"ENTRY_GAP_BUCKET"}
    available = tables["availability"]
    assert "HORIZON_AVAILABILITY_REASON" in set(available["group_type"])


def test_aggregation_reconciles_available_and_exit_relation_groups():
    data = market_frame(periods=12)
    full = enrich_one(data=data).iloc[0]
    short = enrich_one(data=data.iloc[:5], trade=source_trade(data.iloc[:5], exit_index=4)).iloc[0]
    short["source_trade_row"] = 1
    short["trade_id"] = "TIMING:000001"
    frame = pd.DataFrame([full, short])
    tables = build_aggregation_tables(frame, forward_return_statistics_stamp="FORWARD")
    availability = tables["availability"].query(
        "population == @PRIMARY_POPULATION and horizon_ticker_bars == 10 and metric == 'forward_10_terminal_close'"
    )
    assert availability["population_trade_count"].sum() == 2
    relations = tables["exit_relations"].query(
        "population == @PRIMARY_POPULATION and horizon_ticker_bars == 1 and metric == 'forward_1_terminal_close'"
    )
    assert relations["population_trade_count"].sum() == 2


def test_output_schema_json_and_hashed_provenance(tmp_path: Path):
    data = market_frame()
    source_file = tmp_path / "source.csv"
    source_file.write_text("value\n1\n", encoding="utf-8")
    bundle = build_forward_return_statistics(
        trades=pd.DataFrame([source_trade(data)]),
        data_by_ticker={"AAPL": data},
        windows=window_frame(data),
        entry_statistics_stamp="ENTRY",
        timing_stamp="TIMING",
        source_stamp="SOURCE",
        snapshot_id="SNAPSHOT",
        snapshot_fingerprint="FINGERPRINT",
        expected_trade_count=1,
        expected_window_count=1,
        forward_return_statistics_stamp="FORWARD",
        source_files={"source": source_file},
        source_hashes={"source": file_hash(source_file)},
        source_verification=source_verification(),
        save_authorization=_STRICT_SOURCE_AUTHORIZATION,
    )
    paths = save_forward_return_statistics(bundle, tmp_path / "output")
    assert tuple(bundle["trades"].columns) == FORWARD_COLUMNS
    assert tuple(bundle["aggregations"]["overall"].columns) == STATISTICS_COLUMNS
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    assert provenance["source_files"]["source"]["sha256"] == file_hash(source_file)
    assert set(provenance["result_files"]) == set(paths).difference({"provenance"})
    assert all(file_hash(Path(meta["path"])) == meta["sha256"] for meta in provenance["result_files"].values())
    payload = build_json_payload(bundle)
    for key in ("field_definitions", "formulas", "population_definition", "source_lineage", "aggregation_results", "limitations"):
        assert key in payload


def test_save_rejects_source_mutation(tmp_path: Path):
    data = market_frame()
    source_file = tmp_path / "source.csv"
    source_file.write_text("value\n1\n", encoding="utf-8")
    bundle = build_forward_return_statistics(
        trades=pd.DataFrame([source_trade(data)]), data_by_ticker={"AAPL": data}, windows=window_frame(data),
        entry_statistics_stamp="ENTRY", timing_stamp="TIMING", source_stamp="SOURCE", snapshot_id="SNAPSHOT",
        snapshot_fingerprint="FINGERPRINT", forward_return_statistics_stamp="FORWARD", source_files={"source": source_file},
        source_hashes={"source": file_hash(source_file)}, source_verification=source_verification(),
        save_authorization=_STRICT_SOURCE_AUTHORIZATION,
    )
    source_file.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Source changed"):
        save_forward_return_statistics(bundle, tmp_path / "output")


def test_saving_requires_nonempty_verified_consumed_source_metadata(tmp_path: Path):
    data = market_frame()
    unsourced = build_forward_return_statistics(
        trades=pd.DataFrame([source_trade(data)]), data_by_ticker={"AAPL": data}, windows=window_frame(data),
        entry_statistics_stamp="ENTRY", timing_stamp="TIMING", source_stamp="SOURCE", snapshot_id="SNAPSHOT",
        snapshot_fingerprint="FINGERPRINT", forward_return_statistics_stamp="FORWARD",
    )
    with pytest.raises(ValueError, match="metadata is missing"):
        save_forward_return_statistics(unsourced, tmp_path / "none")

    empty = build_forward_return_statistics(
        trades=pd.DataFrame([source_trade(data)]), data_by_ticker={"AAPL": data}, windows=window_frame(data),
        entry_statistics_stamp="ENTRY", timing_stamp="TIMING", source_stamp="SOURCE", snapshot_id="SNAPSHOT",
        snapshot_fingerprint="FINGERPRINT", forward_return_statistics_stamp="FORWARD",
        source_files={}, source_hashes={}, source_verification=source_verification(),
        save_authorization=_STRICT_SOURCE_AUTHORIZATION,
    )
    with pytest.raises(ValueError, match="mapping is empty"):
        save_forward_return_statistics(empty, tmp_path / "empty")
    assert not (tmp_path / "none").exists()
    assert not (tmp_path / "empty").exists()


def test_saving_rejects_missing_source_hash_path_and_weakened_verification(tmp_path: Path):
    data = market_frame()
    source_file = tmp_path / "source.csv"
    source_file.write_text("value\n1\n", encoding="utf-8")
    common = {
        "trades": pd.DataFrame([source_trade(data)]), "data_by_ticker": {"AAPL": data}, "windows": window_frame(data),
        "entry_statistics_stamp": "ENTRY", "timing_stamp": "TIMING", "source_stamp": "SOURCE", "snapshot_id": "SNAPSHOT",
        "snapshot_fingerprint": "FINGERPRINT", "forward_return_statistics_stamp": "FORWARD",
        "source_files": {"source": source_file}, "source_verification": source_verification(),
        "save_authorization": _STRICT_SOURCE_AUTHORIZATION,
    }
    missing_hash = build_forward_return_statistics(**common, source_hashes={"source": None})
    with pytest.raises(ValueError, match="recorded source hash is missing"):
        save_forward_return_statistics(missing_hash, tmp_path / "missing-hash")

    missing_path = build_forward_return_statistics(
        **{**common, "source_files": {"source": tmp_path / "missing.csv"}}, source_hashes={"source": "a" * 64}
    )
    with pytest.raises(ValueError, match="source path is missing"):
        save_forward_return_statistics(missing_path, tmp_path / "missing-path")

    weakened = build_forward_return_statistics(
        **{**common, "source_verification": source_verification(code_hash_verification=False)},
        source_hashes={"source": file_hash(source_file)},
    )
    with pytest.raises(ValueError, match="incomplete or weakened"):
        save_forward_return_statistics(weakened, tmp_path / "weakened")


def test_saving_rejects_verified_source_lineage_disagreement(tmp_path: Path):
    data = market_frame()
    source_file = tmp_path / "source.csv"
    source_file.write_text("value\n1\n", encoding="utf-8")
    bundle = build_forward_return_statistics(
        trades=pd.DataFrame([source_trade(data)]), data_by_ticker={"AAPL": data}, windows=window_frame(data),
        entry_statistics_stamp="ENTRY", timing_stamp="TIMING", source_stamp="SOURCE", snapshot_id="SNAPSHOT",
        snapshot_fingerprint="FINGERPRINT", forward_return_statistics_stamp="FORWARD",
        source_files={"source": source_file}, source_hashes={"source": file_hash(source_file)},
        source_verification=source_verification(timing_stamp="OTHER"),
        save_authorization=_STRICT_SOURCE_AUTHORIZATION,
    )
    with pytest.raises(ValueError, match="disagrees"):
        save_forward_return_statistics(bundle, tmp_path / "disagreement")


def test_direct_synthetic_bundle_with_complete_metadata_is_not_saveable(tmp_path: Path):
    data = market_frame()
    source_file = tmp_path / "source.csv"
    source_file.write_text("value\n1\n", encoding="utf-8")
    bundle = build_forward_return_statistics(
        trades=pd.DataFrame([source_trade(data)]), data_by_ticker={"AAPL": data}, windows=window_frame(data),
        entry_statistics_stamp="ENTRY", timing_stamp="TIMING", source_stamp="SOURCE", snapshot_id="SNAPSHOT",
        snapshot_fingerprint="FINGERPRINT", forward_return_statistics_stamp="FORWARD",
        source_files={"source": source_file}, source_hashes={"source": file_hash(source_file)},
        source_verification=source_verification(),
    )
    with pytest.raises(ValueError, match="strict verified provenance"):
        save_forward_return_statistics(bundle, tmp_path / "synthetic")


def test_alternate_expectations_require_one_complete_bundle():
    with pytest.raises(ValueError, match="complete expectation bundle"):
        _resolved_expectations({"entry_statistics_stamp": "OTHER"})
    alternate = dict(OFFICIAL_EXPECTATIONS)
    alternate.update(
        {
            "entry_statistics_stamp": "ALTERNATE_ENTRY",
            "timing_stamp": "ALTERNATE_TIMING",
            "source_stop_walk_forward_stamp": "ALTERNATE_SOURCE",
            "snapshot_id": "ALTERNATE_SNAPSHOT",
            "expected_trade_count": 1,
            "expected_window_count": 1,
            "controlled_tickers": ("AAPL",),
        }
    )
    assert _resolved_expectations(alternate) == _resolved_expectations(dict(alternate))


def test_cli_no_save_never_calls_saver_or_creates_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    output_directory = tmp_path / "forward_returns"
    bundle = {
        "entry_statistics_stamp": "ENTRY", "timing_stamp": "TIMING", "snapshot_id": "SNAPSHOT",
        "summary": {
            "primary_trade_count": 1, "window_count": 1, "force_close_end_trade_count": 0,
            "force_close_excluded_trade_count": 1,
            "horizon_availability_counts": {str(h): {AVAILABLE: 1, INSUFFICIENT_BARS: 0} for h in HORIZONS},
        },
    }
    calls: list[str] = []
    loader_arguments: dict[str, object] = {}
    monkeypatch.setattr(
        forward_statistics,
        "load_verified_source",
        lambda **kwargs: loader_arguments.update(kwargs) or {"verified": True},
    )
    monkeypatch.setattr(forward_statistics, "run_forward_return_statistics", lambda _: bundle)
    monkeypatch.setattr(forward_statistics, "save_forward_return_statistics", lambda *_: calls.append("save"))
    monkeypatch.setattr(sys, "argv", [
        "forward-returns", "--snapshot-directory", str(tmp_path / "snapshot"),
        "--entry-statistics-directory", str(tmp_path / "entry"), "--entry-statistics-stamp", "ENTRY",
        "--output-directory", str(output_directory), "--no-save",
    ])
    forward_statistics.main()
    assert calls == []
    assert not output_directory.exists()
    assert "verify_code" not in loader_arguments
    assert "expectations" not in loader_arguments
    assert "Output artifacts saved: 0 (--no-save)" in capsys.readouterr().out


def test_official_source_provenance_population_and_windows_reconcile():
    if not SNAPSHOT_PATH.exists() or not ENTRY_DIRECTORY.exists():
        pytest.skip("Approved untracked research artifacts are unavailable.")
    source = load_verified_source(
        snapshot_directory=SNAPSHOT_PATH,
        entry_statistics_directory=ENTRY_DIRECTORY,
        entry_statistics_stamp=APPROVED_ENTRY_STATISTICS_STAMP,
        project_root=PROJECT_ROOT,
        verify_code=True,
    )
    bundle = run_forward_return_statistics(source)
    assert len(bundle["trades"]) == 256
    assert bundle["trades"]["window_id"].nunique() == 13
    assert bundle["trades"]["model"].eq(MODEL_BASELINE).all()
    assert set(bundle["trades"]["ticker"]).issubset({"AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA", "BTC-USD", "ETH-USD"})
    with pytest.raises(ValueError, match="requires Entry Statistics stamp"):
        load_verified_source(
            snapshot_directory=SNAPSHOT_PATH,
            entry_statistics_directory=ENTRY_DIRECTORY,
            entry_statistics_stamp="WRONG",
            project_root=PROJECT_ROOT,
        )
    assert bundle["summary"]["horizon_availability_counts"] == {
        str(horizon): {AVAILABLE: expected, INSUFFICIENT_BARS: 256 - expected}
        for horizon, expected in OFFICIAL_AVAILABILITY_COUNTS.items()
    }


def test_synthetic_source_is_not_forced_to_official_availability_counts():
    data = market_frame(periods=6)
    bundle = build_forward_return_statistics(
        trades=pd.DataFrame([source_trade(data, entry_index=3, exit_index=4)]),
        data_by_ticker={"AAPL": data}, windows=window_frame(data),
        entry_statistics_stamp="SYNTHETIC", timing_stamp="SYNTHETIC", source_stamp="SYNTHETIC",
        snapshot_id="SYNTHETIC", snapshot_fingerprint="SYNTHETIC", forward_return_statistics_stamp="FORWARD",
        expected_trade_count=1, expected_window_count=1,
    )
    assert bundle["summary"]["horizon_availability_counts"]["10"][AVAILABLE] == 0
