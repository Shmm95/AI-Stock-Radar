from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.portfolio_backtest_models import PortfolioBacktestConfig
from src.backtest.run_research_data_snapshot import (
    compare_replay, create_snapshot_from_data, load_snapshot, replay_gate,
    run_snapshot_timing, run_snapshot_walk_forward, sha256_json, verify_snapshot,
)


def _frame():
    index=pd.date_range("2020-01-01","2023-12-29",freq="B");x=np.arange(len(index),dtype=float)
    close=100+x*.03+np.sin(x/9)*3
    return pd.DataFrame({"Open":close*.999,"High":close*1.01,"Low":close*.99,
        "Close":close,"EMA20":close*.99,"EMA50":close*.98,
        "RSI14":np.where(x%18==0,40,56),"RegimeAllowed":True},index=index)


def _snapshot(tmp_path):
    config=PortfolioBacktestConfig(initial_cash=10000,maximum_position_percent=100,
        maximum_crypto_allocation_percent=100,commission_rate=0,minimum_fee=0,slippage_bps=0)
    return create_snapshot_from_data(data_by_ticker={"AAA":_frame()},config=config,
        output_root=tmp_path,project_root=tmp_path,train_months=12,test_months=6,step_months=12)


def test_json_hash_is_order_independent():
    assert sha256_json({"a":1,"b":2})==sha256_json({"b":2,"a":1})


def test_snapshot_round_trip_and_verification(tmp_path):
    path=_snapshot(tmp_path);result=verify_snapshot(path)
    assert result["passed"]
    loaded=load_snapshot(path)
    assert loaded["manifest"]["windows"]["count"]==3
    assert len(loaded["data_by_ticker"]["AAA"])==len(_frame())
    assert loaded["data_by_ticker"]["AAA"]["RegimeAllowed"].dtype==bool


def test_snapshot_tamper_is_detected(tmp_path):
    path=_snapshot(tmp_path);market=next((path/"market").glob("*.csv"))
    market.write_text(market.read_text()+"\n",encoding="utf-8")
    assert not verify_snapshot(path)["passed"]
    with pytest.raises(ValueError,match="verification failed"):load_snapshot(path)


def _runs(value=10.0):
    return pd.DataFrame([{"window_id":"W01","model":"M","variant_id":"V",
        "selected_variant_id":"V","total_return_percent":value,
        "maximum_drawdown_percent":5.0,"profit_factor":1.5,"completed_trades":3}])


def test_replay_comparison_passes_inside_tolerance():
    result=compare_replay(_runs(),_runs(10.005),tolerance_percent=.01)
    assert bool(result.iloc[0].passed)


def test_replay_comparison_fails_on_metric_or_variant_change():
    replay=_runs(10.02);replay.loc[0,"variant_id"]="OTHER"
    result=compare_replay(_runs(),replay,tolerance_percent=.01)
    assert not bool(result.iloc[0].passed)
    assert not bool(result.iloc[0].variant_match)


def test_snapshot_walk_forward_writes_provenance(tmp_path):
    path=_snapshot(tmp_path);output=tmp_path/"results"
    result=run_snapshot_walk_forward(snapshot_path=path,output_directory=output,
        stock_stops=(3.5,5,7.5),crypto_stops=(3.5,5,7.5))
    assert result["paths"]["provenance"].exists()
    assert result["paths"]["json"].exists()
    assert len(result["bundle"]["aggregate"])==5


def test_end_to_end_replay_gate_passes(tmp_path):
    path=_snapshot(tmp_path);output=tmp_path/"results"
    reference=run_snapshot_walk_forward(snapshot_path=path,output_directory=output)
    gate=replay_gate(snapshot_path=path,reference_directory=output,
        stamp=reference["stamp"],tolerance_percent=.01)
    assert gate["passed"]
    assert gate["checks"].passed.all()


def test_snapshot_timing_uses_same_provenance(tmp_path):
    path=_snapshot(tmp_path);output=tmp_path/"results"
    reference=run_snapshot_walk_forward(snapshot_path=path,output_directory=output)
    timing=run_snapshot_timing(snapshot_path=path,reference_directory=output,
        stamp=reference["stamp"],models=("FIXED_BASELINE",),horizons=(5,),
        output_directory=tmp_path/"timing")
    assert timing["paths"]["provenance"].exists()
    assert timing["result"]["summary"].iloc[0].maximum_source_return_difference_percent==pytest.approx(0)
