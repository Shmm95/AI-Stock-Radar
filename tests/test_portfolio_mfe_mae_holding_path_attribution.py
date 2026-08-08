from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.backtest.run_portfolio_mfe_mae_holding_path_attribution as attribution
from src.backtest.run_portfolio_entry_statistics import CONTROLLED_TICKERS, MODEL_BASELINE
from src.backtest.run_portfolio_mfe_mae_holding_path_attribution import (
    APPROVED_FORWARD_RETURN_STATISTICS_STAMP,
    APPROVED_SNAPSHOT_ID,
    OFFICIAL_EXIT_PHASE_COUNTS,
    OFFICIAL_FULL_HELD_AVAILABILITY_COUNTS,
    EXIT_BAR_HIGH_LOW_POST_EXIT_UNKNOWN,
    FORCE_CLOSE_LOCAL_CLOSE,
    FULL_HELD_AVAILABLE,
    FULL_HELD_UNAVAILABLE,
    HIGH_LOW_ORDER_UNKNOWN,
    INTRABAR_INITIAL_STOP,
    INTRABAR_PARTIAL_AMBIGUOUS,
    OPEN_EXIT_EXCLUDED,
    OPEN_GAP_STOP,
    OPEN_PENDING_EXIT,
    PRIMARY_POPULATION,
    SENSITIVITY_POPULATION,
    STATISTICS_COLUMNS,
    TRADE_COLUMNS,
    build_aggregation_table,
    build_json_payload,
    build_mfe_mae_holding_path_attribution,
    build_statistics_table,
    enrich_holding_paths,
    load_verified_source,
    run_mfe_mae_holding_path_attribution,
    save_mfe_mae_holding_path_attribution,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = (
    PROJECT_ROOT
    / "data/backtests/portfolio/research_snapshots"
    / APPROVED_SNAPSHOT_ID
)
FORWARD_DIRECTORY = PROJECT_ROOT / "data/backtests/portfolio/forward_return_statistics"


def sell_fill(raw_price: float, slippage_bps: float = 5.0) -> float:
    return round(raw_price * (1 - slippage_bps / 10_000), 8)


def market_frame(
    *, ticker: str = "AAPL", periods: int = 12, start: str = "2024-01-01"
) -> pd.DataFrame:
    frequency = "D" if ticker.endswith("-USD") else "B"
    index = pd.date_range(start, periods=periods, freq=frequency)
    base = np.arange(periods, dtype=float) + 100.0
    return pd.DataFrame(
        {"Open": base, "High": base + 3.0, "Low": base - 2.0, "Close": base + 1.0},
        index=index,
    )


def window_frame(
    data: pd.DataFrame,
    *,
    window_id: str = "W01",
    end_exclusive: pd.Timestamp | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "window_id": window_id,
                "test_start": data.index.min(),
                "test_end_exclusive": (
                    end_exclusive
                    if end_exclusive is not None
                    else data.index.max() + pd.Timedelta(days=1)
                ),
            }
        ]
    )


def source_trade(
    data: pd.DataFrame,
    *,
    ticker: str = "AAPL",
    entry_index: int = 1,
    exit_index: int = 4,
    exit_timestamp: pd.Timestamp | None = None,
    exit_reason: str = "EXIT_SIGNAL_NEXT_OPEN",
    entry_fill: float = 100.0,
    quantity: float = 10.0,
    source_trade_row: int = 0,
    model: str = MODEL_BASELINE,
) -> dict[str, object]:
    effective_exit = data.index[exit_index]
    recorded_exit = effective_exit if exit_timestamp is None else exit_timestamp
    initial_stop = round(entry_fill * 0.95, 8)
    if exit_reason in {"EXIT_SIGNAL_NEXT_OPEN", "GAP_STOP_LOSS"}:
        raw_exit = float(data.iloc[exit_index]["Open"])
    elif exit_reason == "STOP_LOSS":
        raw_exit = initial_stop
    elif exit_reason == "FORCE_CLOSE_END":
        raw_exit = float(data.iloc[exit_index]["Close"])
    else:
        raw_exit = float(data.iloc[exit_index]["Open"])
    exit_fill = sell_fill(raw_exit)
    gross_pnl = (exit_fill - entry_fill) * quantity
    return {
        "forward_return_statistics_stamp": "FORWARD",
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
        "entry_fill_price": entry_fill,
        "entry_year": int(data.index[entry_index].year),
        "entry_gap_bucket": "ZERO_TO_POS_0P5",
        "quantity": quantity,
        "initial_stop_price": initial_stop,
        "exit_timestamp": recorded_exit,
        "exit_price": exit_fill,
        "gross_pnl": gross_pnl,
        "net_pnl": gross_pnl - 2.0,
        "return_percent": (gross_pnl - 2.0) / (entry_fill * quantity) * 100,
        "holding_period_portfolio_bars": max(exit_index - entry_index, 0),
        "exit_reason": exit_reason,
        "exit_category": (
            "FORCE_CLOSE_END"
            if exit_reason == "FORCE_CLOSE_END"
            else "STOP_LOSS" if "STOP_LOSS" in exit_reason else "TREND_EXIT"
        ),
        "outcome_class": "WINNER" if gross_pnl - 2.0 > 0 else "LOSER",
        "signal_macd": -999.0,
        "signal_score": 75.0,
    }


def enrich_one(
    *, data: pd.DataFrame, trade: dict[str, object] | None = None, ticker: str = "AAPL"
) -> pd.DataFrame:
    active_trade = trade if trade is not None else source_trade(data, ticker=ticker)
    return enrich_holding_paths(
        trades=pd.DataFrame([active_trade]),
        data_by_ticker={ticker: data},
        windows=window_frame(data),
        mfe_mae_holding_path_stamp="PATH",
        expected_trade_count=1,
        expected_window_count=1,
        expected_force_close_count=(
            1 if active_trade["exit_reason"] == "FORCE_CLOSE_END" else 0
        ),
        controlled_tickers=(ticker,),
    )


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verification(
    *, code_hash_verification: bool = True, official_source: bool = False
) -> dict[str, object]:
    return {
        "verified": True,
        "code_hash_verification": code_hash_verification,
        "official_source": official_source,
        "forward_provenance_verified": True,
        "forward_result_hashes_verified": True,
        "forward_source_hashes_verified": True,
        "snapshot_hashes_verified": True,
        "snapshot_code_hashes_verified": code_hash_verification,
        "forward_return_statistics_stamp": "FORWARD",
        "entry_statistics_stamp": "ENTRY",
        "timing_stamp": "TIMING",
        "source_stop_walk_forward_stamp": "SOURCE",
        "snapshot_id": "SNAPSHOT",
        "snapshot_fingerprint": "FINGERPRINT",
    }


def synthetic_bundle(
    tmp_path: Path, *, source_verification: dict[str, object] | None = None
) -> tuple[dict[str, object], Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    data = market_frame()
    source = tmp_path / "source.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    bundle = build_mfe_mae_holding_path_attribution(
        trades=pd.DataFrame([source_trade(data)]),
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
        expected_force_close_count=0,
        controlled_tickers=("AAPL",),
        mfe_mae_holding_path_stamp="PATH",
        source_files={"source": source},
        source_hashes={"source": file_hash(source)},
        source_verification=(
            verification() if source_verification is None else source_verification
        ),
    )
    return bundle, source


def strict_official_bundle(
    *, snapshot_directory: Path = SNAPSHOT_PATH,
) -> tuple[dict[str, object], dict[str, object]]:
    if not snapshot_directory.exists() or not FORWARD_DIRECTORY.exists():
        pytest.skip("Approved untracked research artifacts are unavailable.")
    source = load_verified_source(
        snapshot_directory=snapshot_directory,
        forward_return_statistics_directory=FORWARD_DIRECTORY,
        forward_return_statistics_stamp=APPROVED_FORWARD_RETURN_STATISTICS_STAMP,
        project_root=PROJECT_ROOT,
        verify_code=True,
    )
    return source, run_mfe_mae_holding_path_attribution(source)


def test_entry_bar_is_fully_held_and_mfe_mae_formulas_use_entry_fill():
    data = market_frame()
    trade = source_trade(data, entry_index=1, exit_index=4, entry_fill=101.0, quantity=2.0)
    row = enrich_one(data=data, trade=trade).iloc[0]
    held = data.iloc[1:4]
    maximum = float(held["High"].max())
    minimum = float(held["Low"].min())
    assert row["full_held_ticker_bar_count"] == 3
    assert row["observed_fully_held_mfe_percent"] == pytest.approx(
        max((maximum / 101.0 - 1) * 100, 0)
    )
    assert row["observed_fully_held_mfe_amount"] == pytest.approx(
        max((maximum - 101.0) * 2.0, 0)
    )
    assert row["observed_fully_held_mae_percent"] == pytest.approx(
        max((1 - minimum / 101.0) * 100, 0)
    )
    assert row["observed_fully_held_mae_amount"] == pytest.approx(
        max((101.0 - minimum) * 2.0, 0)
    )
    assert row["minimum_observed_fully_held_return_percent"] <= 0
    assert row["maximum_observed_fully_held_return_percent"] >= 0
    assert row["observed_fully_held_mfe_timestamp"] == data.index[3]
    assert row["observed_fully_held_mfe_offset_from_entry"] == 2
    assert row["observed_fully_held_mfe_completed_ticker_bar_number"] == 3


def test_next_open_exit_excludes_exit_bar_path_but_records_open_and_fill():
    data = market_frame()
    trade = source_trade(data, exit_index=4, exit_reason="EXIT_SIGNAL_NEXT_OPEN")
    before = enrich_one(data=data, trade=trade).iloc[0]
    changed = data.copy(deep=True)
    changed.iloc[4, changed.columns.get_loc("High")] = 10_000.0
    changed.iloc[4, changed.columns.get_loc("Low")] = 0.1
    changed.iloc[4, changed.columns.get_loc("Close")] = 9_000.0
    after = enrich_one(data=changed, trade=trade).iloc[0]
    assert before["exit_execution_phase"] == OPEN_PENDING_EXIT
    assert before["exit_bar_treatment"] == OPEN_EXIT_EXCLUDED
    assert before["exit_open"] == data.iloc[4]["Open"]
    assert before["simulated_exit_fill_return_percent"] == pytest.approx(
        (trade["exit_price"] / trade["entry_fill_price"] - 1) * 100
    )
    for column in (
        "observed_fully_held_mfe_percent",
        "observed_fully_held_mae_percent",
        "maximum_fully_held_close_return_percent",
        "minimum_fully_held_close_return_percent",
    ):
        assert before[column] == after[column]


def test_gap_stop_excludes_exit_bar_and_separates_actual_open_from_fill():
    data = market_frame()
    trade = source_trade(data, exit_index=4, exit_reason="GAP_STOP_LOSS")
    row = enrich_one(data=data, trade=trade).iloc[0]
    assert row["exit_execution_phase"] == OPEN_GAP_STOP
    assert row["exit_bar_treatment"] == OPEN_EXIT_EXCLUDED
    assert row["exit_open"] == data.iloc[4]["Open"]
    assert row["exit_price"] == sell_fill(data.iloc[4]["Open"])
    assert row["full_held_ticker_bar_count"] == 3
    assert pd.isna(row["partial_exit_bar_high"])


def test_same_day_intrabar_stop_has_empty_f_and_missing_not_zero_metrics():
    data = market_frame()
    trade = source_trade(data, entry_index=2, exit_index=2, exit_reason="STOP_LOSS")
    row = enrich_one(data=data, trade=trade).iloc[0]
    assert row["exit_execution_phase"] == INTRABAR_INITIAL_STOP
    assert row["exit_bar_treatment"] == INTRABAR_PARTIAL_AMBIGUOUS
    assert row["intrabar_order_ambiguity"] == EXIT_BAR_HIGH_LOW_POST_EXIT_UNKNOWN
    assert row["full_held_ticker_bar_count"] == 0
    assert bool(row["full_held_bar_available"]) is False
    assert row["full_held_bar_availability_reason"] == FULL_HELD_UNAVAILABLE
    for column in (
        "observed_fully_held_mfe_percent",
        "observed_fully_held_mae_percent",
        "minimum_observed_fully_held_return_percent",
        "maximum_fully_held_close_return_percent",
        "observed_fully_held_mfe_timestamp",
        "bars_to_price_breakeven",
    ):
        assert pd.isna(row[column])
    assert bool(row["partial_exit_bar_observation_available"]) is True
    assert bool(row["partial_exit_bar_contains_unknown_post_exit_price_action"]) is True


def test_later_intrabar_stop_exports_ambiguous_bar_without_extrema_contamination():
    data = market_frame()
    trade = source_trade(data, entry_index=1, exit_index=4, exit_reason="STOP_LOSS")
    changed = data.copy(deep=True)
    changed.iloc[4] = [104.0, 10_000.0, 0.1, 9_000.0]
    row = enrich_one(data=changed, trade=trade).iloc[0]
    assert row["full_held_ticker_bar_count"] == 3
    assert row["observed_fully_held_maximum_high"] == data.iloc[1:4]["High"].max()
    assert row["observed_fully_held_minimum_low"] == data.iloc[1:4]["Low"].min()
    assert row["partial_exit_bar_high"] == 10_000.0
    assert row["partial_exit_bar_low"] == 0.1
    assert row["partial_exit_bar_close"] == 9_000.0
    assert row["partial_exit_bar_high_return_percent"] == pytest.approx(9_900.0)
    assert row["intrabar_stop_price"] == trade["initial_stop_price"]


@pytest.mark.parametrize(
    "reason", ["INTRABAR_TRAILING_STOP", "TRAILING_STOP_INTRABAR", "UNKNOWN_EXIT"]
)
def test_unsupported_exit_semantics_fail_closed(reason: str):
    data = market_frame()
    trade = source_trade(data, exit_reason=reason)
    with pytest.raises(ValueError, match="Unsupported"):
        enrich_one(data=data, trade=trade)


def test_force_close_includes_final_local_bar_through_close():
    data = market_frame()
    trade = source_trade(data, entry_index=1, exit_index=4, exit_reason="FORCE_CLOSE_END")
    row = enrich_one(data=data, trade=trade).iloc[0]
    assert row["exit_execution_phase"] == FORCE_CLOSE_LOCAL_CLOSE
    assert row["exit_bar_treatment"] == FORCE_CLOSE_LOCAL_CLOSE
    assert row["effective_local_exit_timestamp"] == data.index[4]
    assert row["full_held_ticker_bar_count"] == 4
    assert row["maximum_fully_held_close"] == data.iloc[1:5]["Close"].max()
    assert row["intrabar_order_ambiguity"] == HIGH_LOW_ORDER_UNKNOWN


def test_equity_force_close_on_crypto_only_date_uses_prior_local_bar_without_fabrication():
    data = market_frame(periods=8)
    friday = data.index[4]
    assert friday.weekday() == 4
    sunday = friday + pd.Timedelta(days=2)
    trade = source_trade(
        data,
        entry_index=1,
        exit_index=4,
        exit_timestamp=sunday,
        exit_reason="FORCE_CLOSE_END",
    )
    row = enrich_one(data=data, trade=trade).iloc[0]
    assert sunday not in data.index
    assert row["source_exit_timestamp"] == sunday
    assert row["effective_local_exit_timestamp"] == friday
    assert data.index.min() <= row["effective_local_exit_timestamp"] < data.index.max() + pd.Timedelta(days=1)
    assert row["full_held_ticker_bar_count"] == 4
    assert row["exit_price"] == sell_fill(data.loc[friday, "Close"])
    assert not (data.index > sunday).any() or row["effective_local_exit_timestamp"] <= sunday
    tables = build_aggregation_table(pd.DataFrame([row]), stamp="PATH")
    primary = tables.query("population == @PRIMARY_POPULATION and group_type == 'ALL_TRADES'")
    sensitivity = tables.query("population == @SENSITIVITY_POPULATION and group_type == 'ALL_TRADES'")
    assert primary["population_trade_count"].eq(1).all()
    assert sensitivity["population_trade_count"].eq(0).all()


@pytest.mark.parametrize("boundary", ["equal_end", "after_end", "before_start"])
def test_force_close_source_timestamp_must_remain_inside_test_window(boundary: str):
    data = market_frame(periods=8)
    start = data.index[0]
    end = data.index[6]
    timestamp = {
        "equal_end": end,
        "after_end": end + pd.Timedelta(days=1),
        "before_start": start - pd.Timedelta(days=1),
    }[boundary]
    trade = source_trade(
        data,
        entry_index=1,
        exit_index=4,
        exit_timestamp=timestamp,
        exit_reason="FORCE_CLOSE_END",
    )
    with pytest.raises(ValueError, match="Exit timestamp outside test window"):
        enrich_holding_paths(
            trades=pd.DataFrame([trade]),
            data_by_ticker={"AAPL": data},
            windows=window_frame(data, end_exclusive=end),
            mfe_mae_holding_path_stamp="PATH",
            expected_trade_count=1,
            expected_window_count=1,
            expected_force_close_count=1,
            controlled_tickers=("AAPL",),
        )


def test_equity_and_crypto_use_ticker_local_calendars():
    equity = market_frame(ticker="AAPL", periods=10)
    crypto = market_frame(ticker="BTC-USD", periods=10)
    equity_trade = source_trade(equity, ticker="AAPL", entry_index=3, exit_index=6)
    crypto_trade = source_trade(crypto, ticker="BTC-USD", entry_index=3, exit_index=6)
    equity_row = enrich_one(data=equity, trade=equity_trade).iloc[0]
    crypto_row = enrich_one(data=crypto, trade=crypto_trade, ticker="BTC-USD").iloc[0]
    assert equity_row["full_held_ticker_bar_count"] == 3
    assert crypto_row["full_held_ticker_bar_count"] == 3
    assert equity.index[5] - equity.index[3] > crypto.index[5] - crypto.index[3]


def test_window_boundary_blocks_cross_window_exit_and_future_path():
    data = market_frame(periods=10)
    trade = source_trade(data, entry_index=1, exit_index=6)
    windows = window_frame(data, end_exclusive=data.index[5])
    with pytest.raises(ValueError, match="Exit timestamp outside test window"):
        enrich_holding_paths(
            trades=pd.DataFrame([trade]),
            data_by_ticker={"AAPL": data},
            windows=windows,
            mfe_mae_holding_path_stamp="PATH",
            expected_trade_count=1,
            expected_window_count=1,
            expected_force_close_count=0,
            controlled_tickers=("AAPL",),
        )


def test_repeated_extrema_use_first_occurrence_and_offsets_are_explicit():
    data = market_frame()
    data.loc[data.index[1], "High"] = 110.0
    data.loc[data.index[2], "High"] = 110.0
    data.loc[data.index[1], "Low"] = 90.0
    data.loc[data.index[2], "Low"] = 90.0
    trade = source_trade(data, entry_index=1, exit_index=4)
    row = enrich_one(data=data, trade=trade).iloc[0]
    assert row["observed_fully_held_mfe_timestamp"] == data.index[1]
    assert row["observed_fully_held_mae_timestamp"] == data.index[1]
    assert row["observed_fully_held_mfe_offset_from_entry"] == 0
    assert row["observed_fully_held_mfe_completed_ticker_bar_number"] == 1


def test_close_path_first_conditions_counts_frequencies_and_price_breakeven():
    data = market_frame()
    entry_fill = 100.0
    data.loc[data.index[1:5], "Close"] = [100.0, 99.0, 101.0, 98.0]
    data.loc[data.index[1:5], "High"] = [102.0, 102.0, 103.0, 105.0]
    data.loc[data.index[1:5], "Low"] = [98.0, 97.0, 99.0, 96.0]
    trade = source_trade(data, entry_index=1, exit_index=5, entry_fill=entry_fill)
    row = enrich_one(data=data, trade=trade).iloc[0]
    assert row["maximum_fully_held_close"] == 101.0
    assert row["minimum_fully_held_close"] == 98.0
    assert row["first_fully_held_close_equal_entry_timestamp"] == data.index[1]
    assert row["first_fully_held_close_below_entry_timestamp"] == data.index[2]
    assert row["first_fully_held_close_above_entry_timestamp"] == data.index[3]
    assert row["first_fully_held_close_at_or_above_entry_timestamp"] == data.index[1]
    assert row["bars_to_price_breakeven"] == 1
    assert row["fully_held_close_above_entry_count"] == 1
    assert row["fully_held_close_below_entry_count"] == 2
    assert row["fully_held_close_at_entry_count"] == 1
    assert row["fully_held_close_above_entry_frequency"] == pytest.approx(0.25)
    assert row["fully_held_close_below_entry_frequency"] == pytest.approx(0.5)
    assert row["fully_held_close_at_entry_frequency"] == pytest.approx(0.25)
    assert (
        row["fully_held_close_above_entry_count"]
        + row["fully_held_close_below_entry_count"]
        + row["fully_held_close_at_entry_count"]
        == row["full_held_ticker_bar_count"]
    )


def test_price_breakeven_uses_entry_fill_without_commission_and_false_when_absent():
    data = market_frame()
    data.loc[data.index[1:4], "Close"] = [99.0, 99.5, 99.999999999]
    data.loc[data.index[1:4], "Low"] = [98.0, 98.0, 98.0]
    trade = source_trade(data, entry_index=1, exit_index=4, entry_fill=100.0)
    trade["entry_fee"] = 1000.0
    row = enrich_one(data=data, trade=trade).iloc[0]
    assert bool(row["price_breakeven_available"]) is False
    assert pd.isna(row["price_breakeven_timestamp"])
    assert pd.isna(row["bars_to_price_breakeven"])
    assert bool(row["first_fully_held_close_equal_entry_available"]) is False


def test_entry_fill_above_bar_high_is_allowed_and_mfe_is_zero():
    data = market_frame()
    trade = source_trade(data, entry_index=1, exit_index=4, entry_fill=200.0)
    row = enrich_one(data=data, trade=trade).iloc[0]
    assert row["observed_fully_held_mfe_percent"] == 0.0
    assert row["observed_fully_held_mfe_amount"] == 0.0
    assert row["observed_fully_held_mae_percent"] > 0


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("Open", None),
        ("High", np.nan),
        ("Low", 0.0),
        ("Close", -1.0),
        ("Close", np.inf),
    ],
)
def test_malformed_missing_nonfinite_and_nonpositive_ohlc_fail_closed(
    column: str, value: object
):
    data = market_frame()
    data.loc[data.index[2], column] = value
    with pytest.raises(ValueError, match=f"Invalid {column}"):
        enrich_one(data=data)


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("High >= Low", {"High": 1.0}),
        ("High >= Open", {"High": 100.5}),
        ("High >= Close", {"High": 102.5, "Close": 103.0}),
        ("Low <= Open", {"Low": 102.5}),
        ("Low <= Close", {"Open": 104.0, "Low": 103.5, "Close": 103.0}),
    ],
)
def test_cross_field_inconsistent_ohlc_fail_closed(name: str, values: dict[str, float]):
    data = market_frame()
    for column, value in values.items():
        data.loc[data.index[2], column] = value
    with pytest.raises(ValueError, match=name):
        enrich_one(data=data)


def test_duplicate_rows_fail_and_unsorted_timezone_rows_normalize_deterministically():
    data = market_frame()
    duplicate = pd.concat([data, data.iloc[[2]]])
    with pytest.raises(ValueError, match="duplicate timestamps"):
        enrich_one(data=duplicate)

    timezone_data = data.copy()
    timezone_data.index = timezone_data.index.tz_localize("UTC")
    unsorted = timezone_data.iloc[::-1]
    trade = source_trade(data)
    normalized = enrich_one(data=unsorted, trade=trade).iloc[0]
    ordered = enrich_one(data=data, trade=trade).iloc[0]
    for column in (
        "observed_fully_held_mfe_percent",
        "observed_fully_held_mae_percent",
        "effective_local_exit_timestamp",
    ):
        assert normalized[column] == ordered[column]


def test_source_data_and_trade_fields_are_immutable_and_future_rows_cannot_change_population():
    data = market_frame()
    trade = source_trade(data, exit_index=4)
    original_data = data.copy(deep=True)
    original_trade = dict(trade)
    before = enrich_one(data=data, trade=trade).iloc[0]
    changed = data.copy(deep=True)
    changed.loc[changed.index > data.index[4], ["Open", "High", "Low", "Close"]] = 500.0
    after = enrich_one(data=changed, trade=trade).iloc[0]
    for column in trade:
        assert before[column] == after[column]
    for column in (
        "observed_fully_held_mfe_percent",
        "observed_fully_held_mae_percent",
        "maximum_fully_held_close_return_percent",
    ):
        assert before[column] == after[column]
    pd.testing.assert_frame_equal(data, original_data)
    assert trade == original_trade


def test_macd_is_preserved_diagnostic_only_and_cannot_affect_population():
    data = market_frame()
    first = source_trade(data)
    second = source_trade(data)
    first["signal_macd"] = -1e12
    second["signal_macd"] = 1e12
    left = enrich_one(data=data, trade=first).iloc[0]
    right = enrich_one(data=data, trade=second).iloc[0]
    assert left["signal_macd"] == -1e12
    assert right["signal_macd"] == 1e12
    for column in (
        "full_held_ticker_bar_count",
        "observed_fully_held_mfe_percent",
        "observed_fully_held_mae_percent",
    ):
        assert left[column] == right[column]


def test_baseline_only_order_and_stable_identifiers_are_enforced_and_preserved():
    data = market_frame()
    first = source_trade(data, source_trade_row=7)
    second = source_trade(data, source_trade_row=3)
    invalid = source_trade(data, source_trade_row=9, model="FIXED_MAX_RETURN")
    with pytest.raises(ValueError, match="FIXED_BASELINE"):
        enrich_holding_paths(
            trades=pd.DataFrame([first, invalid]), data_by_ticker={"AAPL": data},
            windows=window_frame(data), mfe_mae_holding_path_stamp="PATH",
            controlled_tickers=("AAPL",),
        )
    output = enrich_holding_paths(
        trades=pd.DataFrame([first, second]), data_by_ticker={"AAPL": data},
        windows=window_frame(data), mfe_mae_holding_path_stamp="PATH",
        expected_trade_count=2, expected_window_count=1, expected_force_close_count=0,
        controlled_tickers=("AAPL",),
    )
    assert output["source_trade_row"].tolist() == [7, 3]
    assert output["trade_id"].tolist() == ["TIMING:000007", "TIMING:000003"]
    assert output["source_forward_trade_row"].tolist() == [0, 1]
    assert output["holding_path_trade_id"].is_unique


@pytest.mark.parametrize("value", [4, 4.0])
def test_integral_source_row_values_are_accepted_without_truncation(value: object):
    assert attribution._validated_nonnegative_integer(value, "source_trade_row") == 4


@pytest.mark.parametrize("value", [4.2, np.nan, np.inf, "4 trailing", -1, None])
def test_invalid_source_row_values_fail_closed(value: object):
    with pytest.raises(ValueError, match="source_trade_row"):
        attribution._validated_nonnegative_integer(value, "source_trade_row")


def test_statistics_use_ddof_zero_linear_percentiles_and_missing_counts():
    data = market_frame()
    normal = enrich_one(data=data).iloc[0].copy()
    empty_trade = source_trade(data, entry_index=2, exit_index=2, exit_reason="STOP_LOSS")
    empty = enrich_one(data=data, trade=empty_trade).iloc[0].copy()
    normal["observed_fully_held_mfe_percent"] = 0.0
    empty["observed_fully_held_mfe_percent"] = np.nan
    normal["realized_net_return_percent"] = -10.0
    empty["realized_net_return_percent"] = 10.0
    frame = pd.DataFrame([normal, empty])
    table = build_statistics_table(
        frame, stamp="PATH", population=PRIMARY_POPULATION, group_type="ALL_TRADES"
    )
    mfe = table.query("metric == 'observed_fully_held_mfe_percent'").iloc[0]
    signed = table.query("metric == 'realized_net_return_percent'").iloc[0]
    assert mfe["count"] == 1 and mfe["missing_count"] == 1
    assert pd.isna(mfe["negative_count"])
    assert mfe["metric_semantics"] == "NONNEGATIVE_EXCURSION_MAGNITUDE"
    assert signed["population_standard_deviation"] == pytest.approx(10.0)
    assert signed["p25"] == pytest.approx(-5.0)
    assert signed["p75"] == pytest.approx(5.0)
    assert signed["negative_count"] == 1
    assert signed["positive_count"] == 1


def test_clamped_maximum_and_mae_metric_semantics_are_explicit():
    data = market_frame()
    row = enrich_one(data=data).iloc[0]
    table = build_statistics_table(
        pd.DataFrame([row]),
        stamp="PATH",
        population=PRIMARY_POPULATION,
        group_type="ALL_TRADES",
    )
    maximum = table.query(
        "metric == 'maximum_observed_fully_held_return_percent'"
    ).iloc[0]
    mae = table.query("metric == 'observed_fully_held_mae_percent'").iloc[0]
    minimum = table.query(
        "metric == 'minimum_observed_fully_held_return_percent'"
    ).iloc[0]
    assert maximum["metric_semantics"] == "NONNEGATIVE_FAVORABLE_DIRECTIONAL_MAXIMUM"
    assert pd.isna(maximum["negative_count"])
    assert mae["metric_semantics"] == "NONNEGATIVE_EXCURSION_MAGNITUDE"
    assert minimum["metric_semantics"] == "SIGNED_DIRECTIONAL_VALUE"


def test_all_aggregations_reconcile_primary_and_force_close_sensitivity():
    data = market_frame()
    normal = enrich_one(data=data).iloc[0].copy()
    force_trade = source_trade(
        data, exit_index=4, exit_reason="FORCE_CLOSE_END", source_trade_row=1
    )
    forced = enrich_one(data=data, trade=force_trade).iloc[0].copy()
    frame = pd.DataFrame([normal, forced])
    table = build_aggregation_table(frame, stamp="PATH")
    assert tuple(table.columns) == STATISTICS_COLUMNS
    assert set(table["population"]) == {PRIMARY_POPULATION, SENSITIVITY_POPULATION}
    required = {
        "ALL_TRADES", "OUTCOME_CLASS", "ASSET_CLASS", "TICKER", "WINDOW_ID",
        "ENTRY_YEAR", "ENTRY_GAP_BUCKET", "EXIT_REASON", "EXIT_CATEGORY",
        "EXIT_EXECUTION_PHASE", "EXIT_BAR_TREATMENT", "FULL_HELD_BAR_AVAILABILITY",
    }
    assert required.issubset(set(table["group_type"]))
    primary = table.query(
        "population == @PRIMARY_POPULATION and group_type == 'ALL_TRADES'"
    )
    sensitivity = table.query(
        "population == @SENSITIVITY_POPULATION and group_type == 'ALL_TRADES'"
    )
    assert primary["population_trade_count"].eq(2).all()
    assert sensitivity["population_trade_count"].eq(1).all()


def test_output_schema_is_one_wide_trade_table_and_one_long_statistics_table(
    tmp_path: Path,
):
    bundle, _source = synthetic_bundle(tmp_path)
    assert tuple(bundle["trades"].columns) == TRADE_COLUMNS
    assert tuple(bundle["statistics"].columns) == STATISTICS_COLUMNS
    payload = build_json_payload(bundle)
    for key in (
        "field_definitions", "formulas", "population_definition", "source_lineage",
        "aggregation_results", "limitations",
    ):
        assert key in payload
    assert "capture_efficiency" not in json.dumps(payload).lower()
    assert "per_bar_path" not in json.dumps(payload).lower()


def test_direct_builder_has_no_save_authorization_and_synthetic_bundle_cannot_save(
    tmp_path: Path,
):
    signature = inspect.signature(build_mfe_mae_holding_path_attribution)
    assert "save_authorization" not in signature.parameters
    assert not hasattr(attribution, "_STRICT_SOURCE_AUTHORIZATION")
    bundle, _ = synthetic_bundle(tmp_path)
    with pytest.raises(ValueError, match="strict official loader"):
        save_mfe_mae_holding_path_attribution(bundle, tmp_path / "synthetic")
    assert not (tmp_path / "synthetic").exists()


def test_synthetic_official_looking_metadata_cannot_activate_official_identity(
    tmp_path: Path,
):
    official_looking = verification(official_source=True)
    official_looking.update(
        {
            "forward_return_statistics_stamp": attribution.APPROVED_FORWARD_RETURN_STATISTICS_STAMP,
            "entry_statistics_stamp": attribution.APPROVED_ENTRY_STATISTICS_STAMP,
            "timing_stamp": attribution.APPROVED_TIMING_STAMP,
            "source_stop_walk_forward_stamp": attribution.APPROVED_SOURCE_STAMP,
            "snapshot_id": attribution.APPROVED_SNAPSHOT_ID,
            "snapshot_fingerprint": attribution.APPROVED_SNAPSHOT_FINGERPRINT,
        }
    )
    bundle, _ = synthetic_bundle(
        tmp_path, source_verification=official_looking
    )
    assert bundle["source_verification"]["official_source"] is False
    assert bundle["summary"]["primary_trade_count"] == 1
    assert bundle["summary"]["window_count"] == 1
    with pytest.raises(ValueError, match="strict official loader"):
        save_mfe_mae_holding_path_attribution(
            bundle, tmp_path / "official-looking-synthetic"
        )
    assert not (tmp_path / "official-looking-synthetic").exists()


def test_alternate_and_weakened_bundles_cannot_save(tmp_path: Path):
    bundle, _ = synthetic_bundle(tmp_path)
    bundle["forward_return_statistics_stamp"] = "ALTERNATE_FORWARD"
    bundle["source_verification"]["forward_return_statistics_stamp"] = "ALTERNATE_FORWARD"
    with pytest.raises(ValueError, match="strict official loader"):
        save_mfe_mae_holding_path_attribution(bundle, tmp_path / "alternate")
    weakened, _ = synthetic_bundle(tmp_path / "weakened-source")
    weakened["source_verification"] = verification(code_hash_verification=False)
    with pytest.raises(ValueError, match="strict official loader"):
        save_mfe_mae_holding_path_attribution(weakened, tmp_path / "weakened")


def test_missing_consumed_source_metadata_hash_and_path_fail_closed(tmp_path: Path):
    bundle, source = synthetic_bundle(tmp_path)
    bundle["source_files"] = {}
    bundle["source_hashes"] = {}
    with pytest.raises(ValueError, match="metadata is missing"):
        save_mfe_mae_holding_path_attribution(bundle, tmp_path / "empty")

    bundle, source = synthetic_bundle(tmp_path / "missing-hash")
    bundle["source_hashes"] = {}
    with pytest.raises(ValueError, match="exactly one recorded hash"):
        save_mfe_mae_holding_path_attribution(bundle, tmp_path / "no-hash")

    bundle, source = synthetic_bundle(tmp_path / "blank-hash")
    bundle["source_hashes"]["source"] = ""
    with pytest.raises(ValueError, match="recorded source hash is missing"):
        save_mfe_mae_holding_path_attribution(bundle, tmp_path / "blank-hash-output")

    bundle, source = synthetic_bundle(tmp_path / "missing-path")
    source.unlink()
    with pytest.raises(ValueError, match="source path is missing"):
        save_mfe_mae_holding_path_attribution(bundle, tmp_path / "no-path")


def test_forward_provenance_lineage_and_result_hash_mutations_fail_closed(tmp_path: Path):
    if not SNAPSHOT_PATH.exists() or not FORWARD_DIRECTORY.exists():
        pytest.skip("Approved untracked research artifacts are unavailable.")
    lineage_directory = tmp_path / "lineage"
    shutil.copytree(FORWARD_DIRECTORY, lineage_directory)
    provenance_path = lineage_directory / (
        f"portfolio_forward_return_statistics_provenance_"
        f"{APPROVED_FORWARD_RETURN_STATISTICS_STAMP}.json"
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["timing_stamp"] = "MUTATED"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance timing_stamp mismatch"):
        load_verified_source(
            snapshot_directory=SNAPSHOT_PATH,
            forward_return_statistics_directory=lineage_directory,
            forward_return_statistics_stamp=APPROVED_FORWARD_RETURN_STATISTICS_STAMP,
            project_root=PROJECT_ROOT,
        )

    result_directory = tmp_path / "result"
    shutil.copytree(FORWARD_DIRECTORY, result_directory)
    trades_path = result_directory / (
        f"portfolio_forward_return_statistics_trades_"
        f"{APPROVED_FORWARD_RETURN_STATISTICS_STAMP}.csv"
    )
    trades_path.write_text(trades_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="result hash mismatch"):
        load_verified_source(
            snapshot_directory=SNAPSHOT_PATH,
            forward_return_statistics_directory=result_directory,
            forward_return_statistics_stamp=APPROVED_FORWARD_RETURN_STATISTICS_STAMP,
            project_root=PROJECT_ROOT,
        )


def test_declared_upstream_source_hash_and_missing_path_fail_closed(tmp_path: Path):
    source = tmp_path / "source.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    provenance = {"source_files": {"source": {"path": str(source), "sha256": file_hash(source)}}}
    assert attribution._verify_declared_source_files(
        provenance, project_root=PROJECT_ROOT
    ) == {"source": source}
    bad_hash = json.loads(json.dumps(provenance))
    bad_hash["source_files"]["source"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source hash mismatch"):
        attribution._verify_declared_source_files(bad_hash, project_root=PROJECT_ROOT)
    missing_path = json.loads(json.dumps(provenance))
    missing_path["source_files"]["source"]["path"] = str(tmp_path / "missing.csv")
    with pytest.raises(ValueError, match="declared source is missing"):
        attribution._verify_declared_source_files(missing_path, project_root=PROJECT_ROOT)


@pytest.mark.parametrize(
    "artifact",
    ["config.json", "windows.csv", "market/AAPL.csv"],
)
def test_snapshot_artifact_hash_mutations_fail_closed(tmp_path: Path, artifact: str):
    if not SNAPSHOT_PATH.exists():
        pytest.skip("Approved untracked research snapshot is unavailable.")
    snapshot = tmp_path / "snapshot"
    shutil.copytree(SNAPSHOT_PATH, snapshot)
    target = snapshot / artifact
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Snapshot hash mismatch"):
        attribution._load_verified_snapshot(
            snapshot, verify_code=False, project_root=PROJECT_ROOT
        )


def test_snapshot_manifest_fingerprint_mutation_fails_closed(tmp_path: Path):
    if not SNAPSHOT_PATH.exists():
        pytest.skip("Approved untracked research snapshot is unavailable.")
    snapshot = tmp_path / "snapshot"
    shutil.copytree(SNAPSHOT_PATH, snapshot)
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["period"] = "mutated-period"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Snapshot fingerprint mismatch"):
        attribution._load_verified_snapshot(
            snapshot, verify_code=False, project_root=PROJECT_ROOT
        )


def test_strict_official_saver_revalidates_sources_and_hashes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _source, bundle = strict_official_bundle()
    original_loader = attribution.load_verified_source
    calls: list[str] = []

    def recording_loader(**kwargs):
        calls.append(str(kwargs["forward_return_statistics_stamp"]))
        return original_loader(**kwargs)

    monkeypatch.setattr(attribution, "load_verified_source", recording_loader)
    paths = save_mfe_mae_holding_path_attribution(bundle, tmp_path / "official-output")
    assert calls == [APPROVED_FORWARD_RETURN_STATISTICS_STAMP]
    assert set(paths) == {"trades", "statistics", "json", "provenance"}
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    assert set(provenance["result_files"]) == {"trades", "statistics", "json"}
    for metadata in provenance["result_files"].values():
        assert file_hash(Path(metadata["path"])) == metadata["sha256"]


def test_strict_saver_rejects_every_persisted_result_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _source, bundle = strict_official_bundle()
    baseline_trades = bundle["trades"].copy(deep=True)
    baseline_statistics = bundle["statistics"].copy(deep=True)
    baseline_summary = copy.deepcopy(bundle["summary"])

    def reset_bundle() -> None:
        bundle["trades"] = baseline_trades.copy(deep=True)
        bundle["statistics"] = baseline_statistics.copy(deep=True)
        bundle["summary"] = copy.deepcopy(baseline_summary)

    def reject(label: str, mutation) -> None:
        reset_bundle()
        mutation()
        output = tmp_path / label
        with pytest.raises(ValueError):
            save_mfe_mae_holding_path_attribution(bundle, output)
        assert not output.exists()

    def bump_trade(column: str) -> None:
        index = bundle["trades"][column].first_valid_index()
        assert index is not None
        bundle["trades"].loc[index, column] = (
            float(bundle["trades"].loc[index, column]) + 1.0
        )

    reject("mutated-mfe", lambda: bump_trade("observed_fully_held_mfe_percent"))
    reject("mutated-mae", lambda: bump_trade("observed_fully_held_mae_percent"))
    reject("mutated-close", lambda: bump_trade("maximum_fully_held_close_return_percent"))

    def mutate_exit_classification() -> None:
        current = bundle["trades"].loc[0, "exit_execution_phase"]
        replacement = OPEN_GAP_STOP if current != OPEN_GAP_STOP else OPEN_PENDING_EXIT
        bundle["trades"].loc[0, "exit_execution_phase"] = replacement

    reject("mutated-exit-classification", mutate_exit_classification)

    def mutate_availability() -> None:
        current = bundle["trades"].loc[0, "full_held_bar_availability_reason"]
        replacement = FULL_HELD_UNAVAILABLE if current == FULL_HELD_AVAILABLE else FULL_HELD_AVAILABLE
        bundle["trades"].loc[0, "full_held_bar_availability_reason"] = replacement

    reject("mutated-availability", mutate_availability)

    def mutate_statistics_mean() -> None:
        index = bundle["statistics"]["mean"].first_valid_index()
        assert index is not None
        bundle["statistics"].loc[index, "mean"] = (
            float(bundle["statistics"].loc[index, "mean"]) + 1.0
        )

    reject("mutated-statistics-mean", mutate_statistics_mean)

    def mutate_statistics_count() -> None:
        index = bundle["statistics"].index[0]
        bundle["statistics"].loc[index, "count"] = (
            int(bundle["statistics"].loc[index, "count"]) + 1
        )

    reject("mutated-statistics-count", mutate_statistics_count)
    reject(
        "mutated-summary",
        lambda: bundle["summary"].__setitem__("primary_trade_count", 255),
    )
    reject(
        "reordered-trades",
        lambda: bundle.__setitem__(
            "trades", bundle["trades"].iloc[::-1].reset_index(drop=True)
        ),
    )
    reject(
        "removed-trade",
        lambda: bundle.__setitem__(
            "trades", bundle["trades"].iloc[:-1].reset_index(drop=True)
        ),
    )
    reject(
        "added-trade",
        lambda: bundle.__setitem__(
            "trades",
            pd.concat(
                [bundle["trades"], bundle["trades"].iloc[[0]]], ignore_index=True
            ),
        ),
    )
    reject(
        "removed-statistics-row",
        lambda: bundle.__setitem__(
            "statistics", bundle["statistics"].iloc[:-1].reset_index(drop=True)
        ),
    )
    reject(
        "added-statistics-row",
        lambda: bundle.__setitem__(
            "statistics",
            pd.concat(
                [bundle["statistics"], bundle["statistics"].iloc[[0]]],
                ignore_index=True,
            ),
        ),
    )

    original_formulas = attribution._formulas
    with monkeypatch.context() as patch:
        patch.setattr(
            attribution,
            "_formulas",
            lambda: {
                **original_formulas(),
                "observed_fully_held_mfe_percent": "MUTATED FORMULA",
            },
        )
        reject("mutated-formula-metadata", lambda: None)
    reset_bundle()


def test_strict_saver_rejects_result_source_lineage_mismatch(tmp_path: Path):
    _source, bundle = strict_official_bundle()
    bundle["timing_stamp"] = "MUTATED"
    with pytest.raises(ValueError, match="approved official timing_stamp"):
        save_mfe_mae_holding_path_attribution(bundle, tmp_path / "lineage-mismatch")


def test_alternate_expectations_and_weakened_loader_results_remain_unsaveable(tmp_path: Path):
    if not SNAPSHOT_PATH.exists() or not FORWARD_DIRECTORY.exists():
        pytest.skip("Approved untracked research artifacts are unavailable.")
    alternate = dict(attribution.OFFICIAL_EXPECTATIONS)
    alternate["controlled_tickers"] = tuple(reversed(CONTROLLED_TICKERS))
    alternate_source = load_verified_source(
        snapshot_directory=SNAPSHOT_PATH,
        forward_return_statistics_directory=FORWARD_DIRECTORY,
        forward_return_statistics_stamp=APPROVED_FORWARD_RETURN_STATISTICS_STAMP,
        project_root=PROJECT_ROOT,
        expectations=alternate,
    )
    assert alternate_source["source_verification"]["official_source"] is False
    alternate_bundle = run_mfe_mae_holding_path_attribution(alternate_source)
    with pytest.raises(ValueError, match="strict official loader"):
        save_mfe_mae_holding_path_attribution(alternate_bundle, tmp_path / "alternate-loader")

    weakened_source = load_verified_source(
        snapshot_directory=SNAPSHOT_PATH,
        forward_return_statistics_directory=FORWARD_DIRECTORY,
        forward_return_statistics_stamp=APPROVED_FORWARD_RETURN_STATISTICS_STAMP,
        project_root=PROJECT_ROOT,
        verify_code=False,
    )
    assert weakened_source["source_verification"]["official_source"] is False
    weakened_bundle = run_mfe_mae_holding_path_attribution(weakened_source)
    with pytest.raises(ValueError, match="strict official loader"):
        save_mfe_mae_holding_path_attribution(weakened_bundle, tmp_path / "weakened-loader")


def test_source_mutation_immediately_before_official_save_fails_closed(tmp_path: Path):
    if not SNAPSHOT_PATH.exists() or not FORWARD_DIRECTORY.exists():
        pytest.skip("Approved untracked research artifacts are unavailable.")
    snapshot_copy = tmp_path / "snapshot"
    shutil.copytree(SNAPSHOT_PATH, snapshot_copy)
    source, bundle = strict_official_bundle(snapshot_directory=snapshot_copy)
    config_path = Path(source["source_files"]["snapshot_config"])
    config_path.write_text(config_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Snapshot hash mismatch|Source changed"):
        save_mfe_mae_holding_path_attribution(bundle, tmp_path / "mutated-source")


def test_cli_no_save_returns_before_saver_and_any_output_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    output_directory = tmp_path / "holding_path"
    bundle = {
        "forward_return_statistics_stamp": "FORWARD",
        "entry_statistics_stamp": "ENTRY",
        "timing_stamp": "TIMING",
        "snapshot_id": "SNAPSHOT",
        "summary": {
            "primary_trade_count": 1,
            "window_count": 1,
            "force_close_end_trade_count": 0,
            "force_close_excluded_trade_count": 1,
            "exit_execution_phase_counts": {OPEN_PENDING_EXIT: 1},
            "full_held_bar_availability_counts": {FULL_HELD_AVAILABLE: 1},
        },
    }
    saver_calls: list[str] = []
    loader_arguments: dict[str, object] = {}
    monkeypatch.setattr(
        attribution,
        "load_verified_source",
        lambda **kwargs: loader_arguments.update(kwargs) or {"verified": True},
    )
    monkeypatch.setattr(attribution, "run_mfe_mae_holding_path_attribution", lambda _: bundle)
    monkeypatch.setattr(
        attribution,
        "save_mfe_mae_holding_path_attribution",
        lambda *_: saver_calls.append("save"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "holding-path",
            "--snapshot-directory", str(tmp_path / "snapshot"),
            "--forward-return-statistics-directory", str(tmp_path / "forward"),
            "--forward-return-statistics-stamp", "FORWARD",
            "--output-directory", str(output_directory),
            "--no-save",
        ],
    )
    attribution.main()
    assert saver_calls == []
    assert not output_directory.exists()
    assert "verify_code" not in loader_arguments
    assert "expectations" not in loader_arguments
    assert not list(tmp_path.rglob("*.csv"))
    assert not list(tmp_path.rglob("*.json"))
    assert "Output artifacts saved: 0 (--no-save)" in capsys.readouterr().out


def test_clean_process_import_does_not_load_engine_modules():
    script = """
import sys
import src.backtest.run_portfolio_mfe_mae_holding_path_attribution
blocked = [
    name for name in (
        'src.backtest.portfolio_backtest_engine',
        'src.backtest.run_portfolio_backtest',
    )
    if name in sys.modules
]
assert not blocked, f'Forbidden transitive imports: {blocked}'
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_analyzer_has_no_engine_replay_per_bar_or_capture_efficiency_logic():
    source = Path(attribution.__file__).read_text(encoding="utf-8").lower()
    assert "run_portfolio_backtest(" not in source
    assert "per_bar_path" not in source
    assert "capture_efficiency" not in source


def test_official_lineage_hashes_population_basket_windows_and_identifiers_reconcile():
    if not SNAPSHOT_PATH.exists() or not FORWARD_DIRECTORY.exists():
        pytest.skip("Approved untracked research artifacts are unavailable.")
    source = load_verified_source(
        snapshot_directory=SNAPSHOT_PATH,
        forward_return_statistics_directory=FORWARD_DIRECTORY,
        forward_return_statistics_stamp=APPROVED_FORWARD_RETURN_STATISTICS_STAMP,
        project_root=PROJECT_ROOT,
        verify_code=True,
    )
    bundle = run_mfe_mae_holding_path_attribution(source)
    trades = bundle["trades"]
    assert len(trades) == 256
    assert trades["window_id"].nunique() == 13
    assert trades["model"].eq(MODEL_BASELINE).all()
    assert set(trades["ticker"]) == set(CONTROLLED_TICKERS)
    source_rows = trades["source_trade_row"].astype(int).tolist()
    assert source_rows == sorted(source_rows)
    assert trades["trade_id"].tolist() == [
        f"20260802_085312:{row:06d}" for row in source_rows
    ]
    assert bundle["summary"]["force_close_end_trade_count"] == 42
    assert bundle["summary"]["force_close_excluded_trade_count"] == 214
    assert bundle["summary"]["exit_execution_phase_counts"] == OFFICIAL_EXIT_PHASE_COUNTS
    assert (
        bundle["summary"]["full_held_bar_availability_counts"]
        == OFFICIAL_FULL_HELD_AVAILABILITY_COUNTS
    )
    assert source["source_verification"]["official_source"] is True
    assert source["source_verification"]["verified"] is True
    assert source["source_verification"]["code_hash_verification"] is True
    assert any(
        label.startswith("forward_return_statistics_source:")
        for label in source["source_files"]
    )
    changed_phase = trades.copy(deep=True)
    changed_phase.loc[changed_phase.index[0], "exit_execution_phase"] = OPEN_GAP_STOP
    with pytest.raises(ValueError, match="exit-phase distribution mismatch"):
        attribution._validate_official_distributions(
            attribution._population_summary(changed_phase)
        )
    changed_availability = trades.copy(deep=True)
    changed_availability.loc[
        changed_availability.index[0], "full_held_bar_availability_reason"
    ] = FULL_HELD_UNAVAILABLE
    with pytest.raises(ValueError, match="availability distribution mismatch"):
        attribution._validate_official_distributions(
            attribution._population_summary(changed_availability)
        )
    with pytest.raises(ValueError, match="requires Forward Return Statistics stamp"):
        load_verified_source(
            snapshot_directory=SNAPSHOT_PATH,
            forward_return_statistics_directory=FORWARD_DIRECTORY,
            forward_return_statistics_stamp="WRONG",
            project_root=PROJECT_ROOT,
        )


def test_strict_official_analysis_identity_enforces_distribution_locks(
    monkeypatch: pytest.MonkeyPatch,
):
    if not SNAPSHOT_PATH.exists() or not FORWARD_DIRECTORY.exists():
        pytest.skip("Approved untracked research artifacts are unavailable.")
    source = load_verified_source(
        snapshot_directory=SNAPSHOT_PATH,
        forward_return_statistics_directory=FORWARD_DIRECTORY,
        forward_return_statistics_stamp=APPROVED_FORWARD_RETURN_STATISTICS_STAMP,
        project_root=PROJECT_ROOT,
        verify_code=True,
    )
    original_builder = attribution._build_bundle_from_source

    def phase_mutation(source_value, **kwargs):
        bundle = original_builder(source_value, **kwargs)
        current = bundle["trades"].loc[0, "exit_execution_phase"]
        bundle["trades"].loc[0, "exit_execution_phase"] = (
            OPEN_GAP_STOP if current != OPEN_GAP_STOP else OPEN_PENDING_EXIT
        )
        bundle["summary"] = attribution._population_summary(bundle["trades"])
        return bundle

    with monkeypatch.context() as patch:
        patch.setattr(attribution, "_build_bundle_from_source", phase_mutation)
        with pytest.raises(ValueError, match="exit-phase distribution mismatch"):
            run_mfe_mae_holding_path_attribution(source)

    def availability_mutation(source_value, **kwargs):
        bundle = original_builder(source_value, **kwargs)
        current = bundle["trades"].loc[0, "full_held_bar_availability_reason"]
        bundle["trades"].loc[0, "full_held_bar_availability_reason"] = (
            FULL_HELD_UNAVAILABLE if current == FULL_HELD_AVAILABLE else FULL_HELD_AVAILABLE
        )
        bundle["summary"] = attribution._population_summary(bundle["trades"])
        return bundle

    with monkeypatch.context() as patch:
        patch.setattr(attribution, "_build_bundle_from_source", availability_mutation)
        with pytest.raises(ValueError, match="availability distribution mismatch"):
            run_mfe_mae_holding_path_attribution(source)


def test_official_distribution_mismatches_fail_closed():
    correct = {
        "exit_execution_phase_counts": dict(OFFICIAL_EXIT_PHASE_COUNTS),
        "full_held_bar_availability_counts": dict(OFFICIAL_FULL_HELD_AVAILABILITY_COUNTS),
    }
    attribution._validate_official_distributions(correct)
    bad_phase = {**correct, "exit_execution_phase_counts": dict(OFFICIAL_EXIT_PHASE_COUNTS)}
    bad_phase["exit_execution_phase_counts"][OPEN_PENDING_EXIT] -= 1
    bad_phase["exit_execution_phase_counts"][OPEN_GAP_STOP] += 1
    with pytest.raises(ValueError, match="exit-phase distribution mismatch"):
        attribution._validate_official_distributions(bad_phase)
    bad_availability = {
        **correct,
        "full_held_bar_availability_counts": dict(OFFICIAL_FULL_HELD_AVAILABILITY_COUNTS),
    }
    bad_availability["full_held_bar_availability_counts"][FULL_HELD_AVAILABLE] -= 1
    bad_availability["full_held_bar_availability_counts"][FULL_HELD_UNAVAILABLE] += 1
    with pytest.raises(ValueError, match="availability distribution mismatch"):
        attribution._validate_official_distributions(bad_availability)
