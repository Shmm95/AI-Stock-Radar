from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.backtest.portfolio_backtest_engine import run_portfolio_backtest
from src.backtest.portfolio_backtest_models import PortfolioBacktestConfig
from src.backtest.run_portfolio_trade_timing_attribution import (
    MODEL_BASELINE, classify_common_regime, common_benchmark, exit_category,
    first_recovery_bars, forward_metrics, infer_exit_category, normalize_horizons,
    run_trade_timing_attribution,
)


def _frame(start="2021-01-01",end="2022-12-30"):
    index=pd.date_range(start,end,freq="B");x=np.arange(len(index),dtype=float)
    close=100+x*.05+np.sin(x/8)*3
    return pd.DataFrame({"Open":close*.999,"High":close*1.012,"Low":close*.988,
        "Close":close,"EMA20":close*.99,"EMA50":close*.98,
        "RSI14":np.where(x%18==0,40,56),"RegimeAllowed":True},index=index)


def test_horizon_normalization():
    assert normalize_horizons([20,5,5])==(5,20)
    with pytest.raises(ValueError):normalize_horizons([0])


def test_forward_metrics_and_recovery():
    frame=pd.DataFrame({"Close":[100,105,110,95]},index=pd.date_range("2020-01-01",periods=4))
    metrics=forward_metrics(frame,"2020-01-01",100,(1,2))
    assert metrics["forward_return_2_bars_percent"]==pytest.approx(10)
    assert metrics["max_forward_return_2_bars_percent"]==pytest.approx(10)
    assert first_recovery_bars(frame,"2020-01-01",108,3)==2


def test_exit_categories():
    assert exit_category("STOP_LOSS")=="STOP_LOSS"
    assert exit_category("EXIT_SIGNAL_NEXT_OPEN","EMA20 below EMA50")=="TREND_EXIT"
    assert exit_category("EXIT_SIGNAL_NEXT_OPEN","highest-Close trailing level")=="TRAILING_CLOSE"


def test_infers_trend_exit_from_previous_bar():
    frame=_frame("2021-01-01","2021-02-01")
    frame.loc[frame.index[-2],"EMA20"]=90
    frame.loc[frame.index[-2],"EMA50"]=100
    class Trade:
        exit_reason="EXIT_SIGNAL_NEXT_OPEN"
        entry_timestamp=str(frame.index[0])
        exit_timestamp=str(frame.index[-1])
        entry_price=float(frame.iloc[0].Open)
    assert infer_exit_category(Trade(),frame,7.5)=="TREND_EXIT"


def test_common_benchmark_is_model_independent():
    one=pd.DataFrame({"Open":[100,100],"Close":[100,110]},index=pd.date_range("2020-01-01",periods=2))
    two=pd.DataFrame({"Open":[100,100],"Close":[100,90]},index=pd.date_range("2020-01-01",periods=2))
    summary,rows=common_benchmark({"AAA":one,"BBB":two},window_id="W01",
        test_start="2020-01-01",test_end_exclusive="2020-01-03")
    assert summary["equal_weight_full_exposure_return_percent"]==pytest.approx(0)
    assert summary["common_regime"]=="SIDEWAYS"
    assert len(rows)==2


def test_regime_validation():
    assert classify_common_regime(12)=="BULL"
    assert classify_common_regime(-12)=="BEAR"
    with pytest.raises(ValueError):classify_common_regime(0,bull=-10,bear=10)


def test_integration_replays_one_window():
    data={"AAA":_frame()};config=PortfolioBacktestConfig(initial_cash=10000,
        maximum_position_percent=100,maximum_crypto_allocation_percent=100,
        commission_rate=0,minimum_fee=0,slippage_bps=0)
    test=data["AAA"].loc[(data["AAA"].index>=pd.Timestamp("2022-01-01")) &
                         (data["AAA"].index<pd.Timestamp("2022-07-01"))]
    expected=run_portfolio_backtest(data_by_ticker={"AAA":test},config=config,include_benchmark=False)
    source={"stamp":"X","payload":{"base_config":config.to_dict()},
        "windows":pd.DataFrame([{"window_id":"W01","test_start":"2022-01-01",
            "test_end_exclusive":"2022-07-01","selected_stock_stop_loss_percent":5,
            "selected_crypto_stop_loss_percent":5}]),
        "test_runs":pd.DataFrame([{"window_id":"W01","model":MODEL_BASELINE,
            "total_return_percent":expected.total_return_percent}])}
    result=run_trade_timing_attribution(source=source,data_by_ticker=data,
        models=(MODEL_BASELINE,),horizons=(5,))
    assert len(result["window_models"])==1
    assert result["window_models"].iloc[0].source_return_difference_percent==pytest.approx(0)
    assert len(result["ticker_windows"])==1
    assert set(("summary","trades","exit_summary","rejections","rejection_summary")).issubset(result)
