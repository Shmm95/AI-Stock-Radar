from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.backtest.run_portfolio_entry_statistics as entry_statistics
from src.backtest.run_portfolio_entry_statistics import (
    APPROVED_SNAPSHOT_ID,
    APPROVED_SOURCE_STAMP,
    APPROVED_TIMING_STAMP,
    ASSET_CLASS_ORDER,
    CONTROLLED_TICKERS,
    EXPECTED_TRADE_COUNT,
    EXPECTED_WINDOW_COUNT,
    GAP_BUCKETS,
    MODEL_BASELINE,
    STATISTICS_COLUMNS,
    TRADE_COLUMNS,
    _entry_score,
    build_aggregation_tables,
    build_entry_statistics,
    build_json_payload,
    build_statistics_table,
    calculate_entry_timing_fields,
    causal_signal_features,
    classify_gap_direction,
    classify_outcome,
    enrich_completed_trades,
    entry_gap_bucket,
    load_verified_source,
    reconstruct_initial_stop,
    run_entry_statistics,
    save_entry_statistics,
    verify_provenance_result_files,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = (
    PROJECT_ROOT
    / "data/backtests/portfolio/research_snapshots"
    / APPROVED_SNAPSHOT_ID
)
TIMING_DIRECTORY = PROJECT_ROOT / "data/backtests/portfolio/trade_timing_attribution"
SOURCE_DIRECTORY = PROJECT_ROOT / "data/backtests/portfolio/stop_walk_forward"


def market_frame(
    *,
    start: str = "2024-01-01",
    periods: int = 40,
    ticker: str = "AAPL",
    signal_position: int = 20,
    missing_macd: bool = False,
) -> pd.DataFrame:
    frequency = "D" if ticker.endswith("-USD") else "B"
    index = pd.date_range(start, periods=periods, freq=frequency)
    close = pd.Series(
        [100.0 + position for position in range(periods)], index=index
    )
    frame = pd.DataFrame(
        {
            "Open": close,
            "High": close + 2.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 1_000.0,
            "EMA20": close - 1.0,
            "EMA50": close - 3.0,
            "RSI14": 40.0,
            "MACD": 1.0,
            "RegimeAllowed": True,
        },
        index=index,
    )
    frame.loc[index[signal_position], "RSI14"] = 57.5
    frame.loc[index[signal_position + 1], "Open"] = (
        float(frame.loc[index[signal_position], "Close"]) * 1.01
    )
    if missing_macd:
        frame.loc[index[signal_position], "MACD"] = float("nan")
    return frame


def window_frame(data: pd.DataFrame, window_id: str = "W01") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "window_id": window_id,
                "test_start": data.index[0],
                "test_end_exclusive": data.index[-1] + pd.Timedelta(days=1),
            }
        ]
    )


def source_trade(
    data: pd.DataFrame,
    *,
    ticker: str = "AAPL",
    window_id: str = "W01",
    quantity: float = 10.0,
    signal_position: int = 20,
    holding_bars: int = 2,
    exit_reason: str = "EXIT_SIGNAL_NEXT_OPEN",
    exit_category: str = "TREND_EXIT",
    model: str = MODEL_BASELINE,
) -> dict[str, object]:
    entry_position = signal_position + 1
    exit_position = entry_position + holding_bars
    entry_open = float(data.iloc[entry_position]["Open"])
    entry_price = round(entry_open * 1.0005, 8)
    entry_fee = round(max(entry_price * quantity * 0.0005, 1.0), 2)
    exit_price = round(entry_price * 1.02, 8)
    exit_fee = round(max(exit_price * quantity * 0.0005, 1.0), 2)
    gross = round((exit_price - entry_price) * quantity, 2)
    net = round((exit_price - entry_price) * quantity - entry_fee - exit_fee, 2)
    return {
        "window_id": window_id,
        "model": model,
        "stock_stop_loss_percent": 5.0,
        "crypto_stop_loss_percent": 5.0,
        "ticker": ticker,
        "asset_class": "CRYPTO" if ticker.endswith("-USD") else "EQUITY",
        "entry_timestamp": data.index[entry_position],
        "exit_timestamp": data.index[exit_position],
        "entry_portfolio_bar_index": entry_position,
        "exit_portfolio_bar_index": exit_position,
        "quantity": quantity,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "entry_fee": entry_fee,
        "exit_fee": exit_fee,
        "total_fees": round(entry_fee + exit_fee, 2),
        "gross_pnl": gross,
        "net_pnl": net,
        "return_percent": round(net / (entry_price * quantity) * 100, 4),
        "holding_period_bars": holding_bars,
        "exit_reason": exit_reason,
        "signal_score": _entry_score(data.iloc[signal_position]),
        "signal_reason": "TREND_RSI newly valid.",
        "exit_category": exit_category,
    }


def enrich_one(
    *,
    data: pd.DataFrame | None = None,
    trade: dict[str, object] | None = None,
    ticker: str = "AAPL",
    missing_macd: bool = False,
) -> pd.DataFrame:
    active_data = (
        data
        if data is not None
        else market_frame(ticker=ticker, missing_macd=missing_macd)
    )
    active_trade = trade if trade is not None else source_trade(active_data, ticker=ticker)
    return enrich_completed_trades(
        trades=pd.DataFrame([active_trade]),
        data_by_ticker={ticker: active_data},
        windows=window_frame(active_data),
        timing_stamp="TIMING",
        source_stamp="SOURCE",
        snapshot_id="SNAPSHOT",
        snapshot_fingerprint="FINGERPRINT",
        entry_statistics_stamp="ENTRY",
        expected_population=1,
        expected_windows=1,
    )


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_provenance_hash_mismatch_is_rejected(tmp_path: Path):
    result = tmp_path / "portfolio_trade_timing_trades_STAMP.csv"
    result.write_text("value\n1\n", encoding="utf-8")
    provenance = {
        "result_files": {
            "trades": {"path": str(result), "sha256": "not-the-hash"}
        }
    }
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_provenance_result_files(
            directory=tmp_path,
            provenance=provenance,
            prefix="portfolio_trade_timing_",
            stamp="STAMP",
        )


def test_provenance_filename_mismatch_is_rejected(tmp_path: Path):
    wrong = tmp_path / "wrong.csv"
    wrong.write_text("value\n1\n", encoding="utf-8")
    provenance = {
        "result_files": {
            "trades": {"path": str(wrong), "sha256": file_hash(wrong)}
        }
    }
    with pytest.raises(ValueError, match="filename"):
        verify_provenance_result_files(
            directory=tmp_path,
            provenance=provenance,
            prefix="portfolio_trade_timing_",
            stamp="STAMP",
        )


def test_signal_row_is_predecessor_and_entry_is_next_available_open():
    data = market_frame()
    output = enrich_one(data=data)
    row = output.iloc[0]
    assert row["signal_timestamp"] == data.index[20]
    assert row["entry_timestamp"] == data.index[21]
    assert row["signal_close"] == data.iloc[20]["Close"]
    assert row["entry_open"] == data.iloc[21]["Open"]


def test_weekend_and_missing_sessions_use_ticker_local_observations():
    data = market_frame(start="2024-01-01")
    assert data.index[20].weekday() == 0
    data = data.drop(index=data.index[19])
    data.loc[:, "RSI14"] = 40.0
    data.loc[data.index[20], "RSI14"] = 57.5
    data.loc[data.index[21], "Open"] = float(data.iloc[20]["Close"]) * 1.01
    trade = source_trade(data, signal_position=20)
    output = enrich_one(data=data, trade=trade)
    assert output.iloc[0]["signal_timestamp"] == data.index[20]
    assert output.iloc[0]["entry_timestamp"] == data.index[21]
    assert data.index[21] - data.index[20] >= pd.Timedelta(days=1)


def test_future_data_mutation_does_not_change_causal_features():
    data = market_frame()
    before = causal_signal_features(data, 20)
    changed = data.copy()
    changed.iloc[21:, changed.columns.get_loc("High")] = 1_000_000.0
    changed.iloc[21:, changed.columns.get_loc("Close")] = 1_000_000.0
    changed.iloc[21:, changed.columns.get_loc("MACD")] = -1_000_000.0
    assert causal_signal_features(changed, 20) == before


def test_newly_valid_entry_is_required():
    data = market_frame()
    data.loc[data.index[19], "RSI14"] = 57.5
    with pytest.raises(ValueError, match="not newly valid"):
        enrich_one(data=data)


def test_negative_macd_is_diagnostic_only():
    data = market_frame()
    data.loc[data.index[20], "MACD"] = -5.0
    output = enrich_one(data=data)
    assert len(output) == 1
    assert output.iloc[0]["signal_macd"] == -5.0
    assert output.iloc[0]["macd_percent"] < 0


def test_regime_allowed_missing_fails_closed_with_identifying_error():
    data = market_frame().drop(columns="RegimeAllowed")
    with pytest.raises(
        ValueError,
        match="RegimeAllowed for ticker AAPL.*field is missing",
    ):
        enrich_one(data=data)


@pytest.mark.parametrize(
    "invalid_value",
    [None, float("nan"), "false", "true", 0, 1],
    ids=["none", "nan", "string-false", "string-true", "integer-zero", "integer-one"],
)
def test_non_boolean_regime_allowed_values_fail_closed(invalid_value: object):
    data = market_frame()
    data["RegimeAllowed"] = data["RegimeAllowed"].astype(object)
    data.loc[data.index[20], "RegimeAllowed"] = invalid_value
    data.loc[data.index[20], "MACD"] = -5.0
    with pytest.raises(
        ValueError,
        match="RegimeAllowed for ticker AAPL.*invalid value",
    ):
        enrich_one(data=data)


@pytest.mark.parametrize(
    "regime_value",
    [True, np.bool_(True)],
    ids=["python-bool", "numpy-bool"],
)
def test_real_boolean_regime_allowed_values_are_accepted(regime_value: object):
    data = market_frame()
    data["RegimeAllowed"] = data["RegimeAllowed"].astype(object)
    data.loc[data.index[20], "RegimeAllowed"] = regime_value
    data.loc[data.index[20], "MACD"] = -5.0
    output = enrich_one(data=data)
    assert bool(output.iloc[0]["signal_regime_allowed"]) is True
    assert output.iloc[0]["signal_macd"] == -5.0


def test_all_approved_entry_timing_formulas():
    values = calculate_entry_timing_fields(
        signal_close=100.0,
        entry_open=99.0,
        entry_fill_price=99.0495,
        quantity=10.0,
        entry_fee=1.0,
    )
    assert values["raw_entry_gap_percent"] == pytest.approx(-1.0)
    assert values["raw_entry_gap_amount"] == pytest.approx(-10.0)
    assert values["entry_slippage_per_unit"] == pytest.approx(0.0495)
    assert values["entry_slippage_percent"] == pytest.approx(0.05)
    assert values["entry_slippage_amount"] == pytest.approx(0.495)
    assert values["entry_commission_percent"] == pytest.approx(1 / 990.495 * 100)
    assert values["entry_transaction_cost_amount"] == pytest.approx(1.495)
    assert values["entry_transaction_cost_percent"] == pytest.approx(1.495 / 990 * 100)
    assert values["signed_entry_timing_effect_amount"] == pytest.approx(-8.505)
    assert values["signed_entry_timing_effect_percent"] == pytest.approx(-8.505 / 1000 * 100)
    assert values["adverse_entry_timing_cost_amount"] == 0
    assert values["favorable_entry_timing_benefit_amount"] == pytest.approx(8.505)


def test_minimum_and_proportional_entry_fee_cases_are_validated():
    minimum_data = market_frame()
    minimum = enrich_one(data=minimum_data)
    assert minimum.iloc[0]["entry_fee"] == 1.0

    proportional_data = market_frame()
    trade = source_trade(proportional_data, quantity=100.0)
    proportional = enrich_one(data=proportional_data, trade=trade)
    assert proportional.iloc[0]["entry_fee"] > 1.0
    damaged = dict(trade)
    damaged["entry_fee"] = float(trade["entry_fee"]) + 0.01
    with pytest.raises(ValueError, match="Entry fee mismatch"):
        enrich_one(data=proportional_data, trade=damaged)


def test_stop_reconstruction_and_baseline_validation():
    stop = reconstruct_initial_stop(
        entry_fill_price=100.0, quantity=10.0, stop_percent=5.0
    )
    assert stop == {
        "initial_stop_percent": 5.0,
        "initial_stop_price": 95.0,
        "initial_stop_distance": 5.0,
        "initial_risk_amount": 50.0,
    }
    data = market_frame()
    trade = source_trade(data)
    trade["stock_stop_loss_percent"] = 4.0
    with pytest.raises(ValueError, match="5% stock and crypto stops"):
        enrich_one(data=data, trade=trade)


@pytest.mark.parametrize(
    ("net_pnl", "expected"),
    [(1.0, "WINNER"), (-1.0, "LOSER"), (0.0, "FLAT")],
)
def test_outcome_classification(net_pnl: float, expected: str):
    assert classify_outcome(net_pnl) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1.0, "ADVERSE"), (-1.0, "FAVORABLE"), (0.0, "FLAT")],
)
def test_gap_direction(value: float, expected: str):
    assert classify_gap_direction(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-2.000001, "LT_NEG_2"),
        (-2.0, "NEG_2_TO_NEG_1"),
        (-1.0, "NEG_1_TO_NEG_0P5"),
        (-0.5, "NEG_0P5_TO_0"),
        (0.0, "ZERO_TO_POS_0P5"),
        (0.5, "POS_0P5_TO_POS_1"),
        (1.0, "POS_1_TO_POS_2"),
        (2.0, "GE_POS_2"),
    ],
)
def test_fixed_gap_bucket_boundaries(value: float, expected: str):
    assert entry_gap_bucket(value) == expected


def test_zero_bar_holding_period_and_forced_close_sensitivity():
    data = market_frame()
    trade = source_trade(
        data,
        holding_bars=0,
        exit_reason="FORCE_CLOSE_END",
        exit_category="FORCE_CLOSE_END",
    )
    output = enrich_one(data=data, trade=trade)
    assert output.iloc[0]["holding_period_portfolio_bars"] == 0
    assert output.iloc[0]["holding_period_ticker_bars"] == 0
    tables = build_aggregation_tables(output, entry_statistics_stamp="ENTRY")
    sensitivity = tables["force_close_excluded"]
    assert sensitivity["population_trade_count"].eq(0).all()


def test_holding_period_definitions_across_weekend():
    data = market_frame()
    signal_position = 23
    data.loc[:, "RSI14"] = 40.0
    data.loc[data.index[signal_position], "RSI14"] = 57.5
    data.loc[data.index[signal_position + 1], "Open"] = (
        float(data.iloc[signal_position]["Close"]) * 1.01
    )
    trade = source_trade(data, signal_position=signal_position, holding_bars=1)
    output = enrich_one(data=data, trade=trade)
    row = output.iloc[0]
    assert row["holding_period_portfolio_bars"] == 1
    assert row["holding_period_ticker_bars"] == 1
    assert row["holding_period_calendar_days"] >= 1


def test_equity_force_close_on_crypto_only_date_uses_last_equity_bar():
    data = market_frame()
    signal_position = 22
    data.loc[:, "RSI14"] = 40.0
    data.loc[data.index[signal_position], "RSI14"] = 57.5
    data.loc[data.index[signal_position + 1], "Open"] = (
        float(data.iloc[signal_position]["Close"]) * 1.01
    )
    trade = source_trade(
        data,
        signal_position=signal_position,
        holding_bars=1,
        exit_reason="FORCE_CLOSE_END",
        exit_category="FORCE_CLOSE_END",
    )
    last_equity_timestamp = data.index[signal_position + 2]
    crypto_only_exit_timestamp = last_equity_timestamp + pd.Timedelta(days=1)
    assert last_equity_timestamp.weekday() == 4
    assert crypto_only_exit_timestamp not in data.index
    trade["exit_timestamp"] = crypto_only_exit_timestamp
    trade["exit_portfolio_bar_index"] = int(trade["entry_portfolio_bar_index"]) + 2
    trade["holding_period_bars"] = 2

    output = enrich_one(data=data, trade=trade)
    row = output.iloc[0]
    assert len(output) == 1
    assert row["holding_period_portfolio_bars"] == 2
    assert row["holding_period_ticker_bars"] == 1
    assert row["exit_timestamp"] == crypto_only_exit_timestamp

    future_changed = data.copy()
    future_changed.loc[
        future_changed.index > last_equity_timestamp,
        ["Open", "High", "Low", "Close"],
    ] = 1_000_000.0
    repeated = enrich_one(data=future_changed, trade=trade)
    assert repeated.iloc[0]["holding_period_ticker_bars"] == 1
    assert repeated.iloc[0]["exit_price"] == row["exit_price"]

    tables = build_aggregation_tables(output, entry_statistics_stamp="ENTRY")
    primary = tables["overall"].query("metric == 'raw_entry_gap_percent'").iloc[0]
    sensitivity = tables["force_close_excluded"].query(
        "metric == 'raw_entry_gap_percent'"
    ).iloc[0]
    assert primary["population_trade_count"] == 1
    assert sensitivity["population_trade_count"] == 0


def test_population_statistics_use_ddof_zero_and_linear_percentiles():
    frame = pd.DataFrame({"return_percent": [0.0, 10.0]})
    result = build_statistics_table(
        frame,
        entry_statistics_stamp="ENTRY",
        population="PRIMARY",
        group_type="ALL_TRADES",
        metrics=("return_percent",),
    ).iloc[0]
    assert result["population_standard_deviation"] == pytest.approx(5.0)
    assert result["p25"] == pytest.approx(2.5)
    assert result["p75"] == pytest.approx(7.5)
    assert result["flat_count"] == 1
    assert result["positive_count"] == 1
    assert result["flat_frequency"] == pytest.approx(0.5)


def test_missing_macd_diagnostics_are_preserved_without_trade_removal():
    data = market_frame(missing_macd=True)
    output = enrich_one(data=data, missing_macd=True)
    assert len(output) == 1
    assert pd.isna(output.iloc[0]["signal_macd"])
    assert pd.isna(output.iloc[0]["macd_percent"])
    table = build_statistics_table(
        output,
        entry_statistics_stamp="ENTRY",
        population="PRIMARY",
        group_type="ALL_TRADES",
        metrics=("macd_percent",),
    ).iloc[0]
    assert table["count"] == 0
    assert table["missing_count"] == 1


def test_baseline_filter_preserves_source_order_and_stable_identifiers():
    aapl = market_frame(ticker="AAPL")
    msft = market_frame(ticker="MSFT")
    rows = [
        source_trade(aapl, model="FIXED_MAX_RETURN"),
        source_trade(aapl, ticker="AAPL"),
        source_trade(msft, ticker="MSFT", model="FIXED_LOW_DRAWDOWN"),
        source_trade(msft, ticker="MSFT"),
    ]
    output = enrich_completed_trades(
        trades=pd.DataFrame(rows),
        data_by_ticker={"AAPL": aapl, "MSFT": msft},
        windows=window_frame(aapl),
        timing_stamp="TIMING",
        source_stamp="SOURCE",
        snapshot_id="SNAPSHOT",
        snapshot_fingerprint="FINGERPRINT",
        entry_statistics_stamp="ENTRY",
        expected_population=2,
        expected_windows=1,
    )
    assert output["source_trade_row"].tolist() == [1, 3]
    assert output["trade_id"].tolist() == ["TIMING:000001", "TIMING:000003"]
    assert output["model"].eq(MODEL_BASELINE).all()


def test_enrichment_does_not_mutate_source_frames():
    data = market_frame()
    trades = pd.DataFrame([source_trade(data)])
    windows = window_frame(data)
    original_data = data.copy(deep=True)
    original_trades = trades.copy(deep=True)
    original_windows = windows.copy(deep=True)
    enrich_completed_trades(
        trades=trades,
        data_by_ticker={"AAPL": data},
        windows=windows,
        timing_stamp="TIMING",
        source_stamp="SOURCE",
        snapshot_id="SNAPSHOT",
        snapshot_fingerprint="FINGERPRINT",
        entry_statistics_stamp="ENTRY",
        expected_population=1,
        expected_windows=1,
    )
    pd.testing.assert_frame_equal(data, original_data)
    pd.testing.assert_frame_equal(trades, original_trades)
    pd.testing.assert_frame_equal(windows, original_windows)


def test_all_aggregation_populations_reconcile():
    aapl = market_frame(ticker="AAPL")
    btc = market_frame(ticker="BTC-USD")
    normal = enrich_one(data=aapl)
    forced_trade = source_trade(
        btc,
        ticker="BTC-USD",
        exit_reason="FORCE_CLOSE_END",
        exit_category="FORCE_CLOSE_END",
    )
    forced = enrich_one(data=btc, trade=forced_trade, ticker="BTC-USD")
    trades = pd.concat([normal, forced], ignore_index=True)
    trades.loc[1, "source_trade_row"] = 1
    trades.loc[1, "trade_id"] = "TIMING:000001"
    tables = build_aggregation_tables(trades, entry_statistics_stamp="ENTRY")
    metric = "raw_entry_gap_percent"
    assert tables["overall"].query("metric == @metric").iloc[0]["population_trade_count"] == 2
    for name in (
        "outcomes",
        "asset_classes",
        "tickers",
        "calendar_years",
        "entry_gap_buckets",
        "exit_reasons",
        "exit_categories",
    ):
        rows = tables[name].query("metric == @metric")
        assert int(rows["population_trade_count"].sum()) == 2
    sensitivity = tables["force_close_excluded"].query("metric == @metric").iloc[0]
    assert sensitivity["population_trade_count"] == 1
    assert set(ASSET_CLASS_ORDER).issubset(
        set(tables["asset_classes"]["group_value"])
    )
    assert set(GAP_BUCKETS).issubset(set(tables["entry_gap_buckets"]["group_value"]))


def test_output_schema_and_hashed_provenance(tmp_path: Path):
    data = market_frame()
    source_file = tmp_path / "source.csv"
    source_file.write_text("value\n1\n", encoding="utf-8")
    bundle = build_entry_statistics(
        trades=pd.DataFrame([source_trade(data)]),
        data_by_ticker={"AAPL": data},
        windows=window_frame(data),
        timing_stamp="TIMING",
        source_stamp="SOURCE",
        snapshot_id="SNAPSHOT",
        snapshot_fingerprint="FINGERPRINT",
        expected_trade_count=1,
        expected_window_count=1,
        entry_statistics_stamp="ENTRY",
        source_files={"source": source_file},
    )
    paths = save_entry_statistics(bundle, tmp_path / "output")
    assert tuple(bundle["trades"].columns) == TRADE_COLUMNS
    assert tuple(bundle["aggregations"]["overall"].columns) == STATISTICS_COLUMNS
    assert all(path.exists() for path in paths.values())
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    assert provenance["source_files"]["source"]["sha256"] == file_hash(source_file)
    assert set(provenance["result_files"]) == set(paths).difference({"provenance"})
    for metadata in provenance["result_files"].values():
        path = Path(metadata["path"])
        assert metadata["sha256"] == file_hash(path)
    payload = build_json_payload(bundle)
    for key in (
        "field_definitions",
        "formulas",
        "population_definition",
        "methodological_choices",
        "limitations",
        "source_lineage",
        "aggregation_results",
    ):
        assert key in payload


def test_save_rejects_source_mutation(tmp_path: Path):
    data = market_frame()
    source_file = tmp_path / "source.csv"
    source_file.write_text("value\n1\n", encoding="utf-8")
    bundle = build_entry_statistics(
        trades=pd.DataFrame([source_trade(data)]),
        data_by_ticker={"AAPL": data},
        windows=window_frame(data),
        timing_stamp="TIMING",
        source_stamp="SOURCE",
        snapshot_id="SNAPSHOT",
        snapshot_fingerprint="FINGERPRINT",
        entry_statistics_stamp="ENTRY",
        source_files={"source": source_file},
    )
    source_file.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Source changed"):
        save_entry_statistics(bundle, tmp_path / "output")


def test_cli_no_save_does_not_call_saver_or_create_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    output_directory = tmp_path / "entry_statistics"
    bundle = {
        "timing_stamp": "TIMING",
        "source_stamp": "SOURCE",
        "snapshot_id": "SNAPSHOT",
        "summary": {
            "primary_trade_count": 1,
            "window_count": 1,
            "outcome_counts": {"WINNER": 1, "LOSER": 0, "FLAT": 0},
            "asset_class_counts": {"EQUITY": 1, "CRYPTO": 0},
            "gap_direction_counts": {"ADVERSE": 1, "FAVORABLE": 0, "FLAT": 0},
            "mean_raw_entry_gap_percent": 0.0,
            "median_raw_entry_gap_percent": 0.0,
            "mean_signed_entry_timing_effect_amount": 0.0,
            "total_entry_transaction_cost_amount": 0.0,
            "total_adverse_entry_timing_cost_amount": 0.0,
            "total_favorable_entry_timing_benefit_amount": 0.0,
            "force_close_end_trade_count": 0,
            "force_close_excluded_trade_count": 1,
        },
    }
    save_calls: list[object] = []

    monkeypatch.setattr(
        entry_statistics,
        "load_verified_source",
        lambda **_: {"verified": True},
    )
    monkeypatch.setattr(entry_statistics, "run_entry_statistics", lambda _: bundle)
    monkeypatch.setattr(
        entry_statistics,
        "save_entry_statistics",
        lambda *_: save_calls.append("called"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "entry-statistics",
            "--snapshot-directory",
            str(tmp_path / "snapshot"),
            "--timing-directory",
            str(tmp_path / "timing"),
            "--timing-stamp",
            "TIMING",
            "--output-directory",
            str(output_directory),
            "--no-save",
        ],
    )

    entry_statistics.main()

    assert save_calls == []
    assert not output_directory.exists()
    assert "Output artifacts saved: 0 (--no-save)" in capsys.readouterr().out


def test_official_approved_population_and_provenance_reconcile():
    if not SNAPSHOT_PATH.exists():
        pytest.skip("Approved untracked research artifacts are unavailable.")
    source = load_verified_source(
        snapshot_directory=SNAPSHOT_PATH,
        timing_directory=TIMING_DIRECTORY,
        timing_stamp=APPROVED_TIMING_STAMP,
        source_directory=SOURCE_DIRECTORY,
        project_root=PROJECT_ROOT,
        verify_code=True,
    )
    bundle = run_entry_statistics(source)
    trades = bundle["trades"]
    assert len(trades) == EXPECTED_TRADE_COUNT
    assert trades["window_id"].nunique() == EXPECTED_WINDOW_COUNT
    assert trades["model"].eq(MODEL_BASELINE).all()
    assert set(trades["ticker"]).issubset(CONTROLLED_TICKERS)
    assert bundle["summary"]["primary_trade_count"] == 256
    assert bundle["summary"]["window_count"] == 13
