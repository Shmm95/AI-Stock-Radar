from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.run_portfolio_benchmark_gap_attribution import (
    aggregate_assets, aggregate_tickers, analyze_benchmark_gap,
    build_window_attribution, classify_regime, compound_returns,
    discover_stamp, validate_source_bundle,
)


def _bundle():
    aggregate=pd.DataFrame([{"model":"MODEL_A","window_count":2,"initial_cash":10000,
        "compounded_return_percent":4.5,"cagr_percent":2.2,"maximum_drawdown_percent":5,
        "return_drawdown_ratio":.9,"matched_benchmark_compounded_return_percent":8,
        "average_profit_factor":1.5,"average_exposure_percent":50,
        "positive_window_count":1,"matched_benchmark_beat_count":1,"total_trades":3}])
    runs=pd.DataFrame([
        {"window_id":"W01","model":"MODEL_A","variant_id":"V1","test_start":"2020-01-01",
         "test_end_exclusive":"2020-07-01","initial_cash":10000,"total_return_percent":10,
         "matched_benchmark_return_percent":20,"excess_return_vs_matched_percent":-10,
         "maximum_drawdown_percent":4,"matched_benchmark_drawdown_percent":6,
         "profit_factor":2,"average_exposure_percent":60,"completed_trades":2,"total_fees":4},
        {"window_id":"W02","model":"MODEL_A","variant_id":"V1","test_start":"2020-07-01",
         "test_end_exclusive":"2021-01-01","initial_cash":10000,"total_return_percent":-5,
         "matched_benchmark_return_percent":-10,"excess_return_vs_matched_percent":5,
         "maximum_drawdown_percent":5,"matched_benchmark_drawdown_percent":12,
         "profit_factor":1,"average_exposure_percent":40,"completed_trades":1,"total_fees":2}])
    tickers=pd.DataFrame([
        {"window_id":"W01","model":"MODEL_A","ticker":"AAA","asset_class":"EQUITY",
         "completed_trades":2,"gross_profit":200,"gross_loss":50,"net_pnl":150,"total_fees":4},
        {"window_id":"W02","model":"MODEL_A","ticker":"AAA","asset_class":"EQUITY",
         "completed_trades":1,"gross_profit":0,"gross_loss":50,"net_pnl":-50,"total_fees":2}])
    rejections=pd.DataFrame([{"window_id":"W01","model":"MODEL_A",
                              "reason_code":"MAX_OPEN_POSITIONS","count":3}])
    return {"aggregate":aggregate,"windows":pd.DataFrame([{"window_id":"W01",
        "selected_variant_id":"V1"},{"window_id":"W02","selected_variant_id":"V1"}]),
        "test_runs":runs,"tickers":tickers,"rejections":rejections,
        "equity":pd.DataFrame(),"stamp":"20200101_000000","source_paths":{}}


def test_regime_classification_and_validation():
    assert classify_regime(5)=="BULL"
    assert classify_regime(-5)=="BEAR"
    assert classify_regime(1)=="SIDEWAYS"
    with pytest.raises(ValueError):classify_regime(0,bullish_threshold=-1,bearish_threshold=1)


def test_compound_returns():
    assert compound_returns([10,-5])==pytest.approx(4.5)


def test_exact_wealth_gap_bridge():
    frame=build_window_attribution(_bundle()["test_runs"])
    assert frame.iloc[-1].cumulative_strategy_wealth==pytest.approx(10450)
    assert frame.iloc[-1].cumulative_benchmark_wealth==pytest.approx(10800)
    assert frame.wealth_gap_contribution_amount.sum()==pytest.approx(-350)


def test_ticker_and_asset_aggregation():
    tickers=aggregate_tickers(_bundle()["tickers"])
    assert tickers.iloc[0].completed_trades==3
    assert tickers.iloc[0].net_pnl==pytest.approx(100)
    assets=aggregate_assets(tickers)
    assert assets.iloc[0].net_pnl_share_percent==pytest.approx(100)


def test_analysis_builds_all_reports():
    bundle=_bundle();validate_source_bundle(bundle)
    result=analyze_benchmark_gap(bundle)
    assert result["summary"].iloc[0].matched_benchmark_gap_percent==pytest.approx(-3.5)
    assert result["summary"].iloc[0].benchmark_beat_rate_percent==pytest.approx(50)
    assert set(("summary","windows","regimes","tickers","assets","tails","rejections")).issubset(result)


def test_discover_stamp_uses_latest(tmp_path):
    (tmp_path/"portfolio_stop_walk_forward_aggregate_20200101_000000.csv").write_text("x\n")
    (tmp_path/"portfolio_stop_walk_forward_aggregate_20210101_000000.csv").write_text("x\n")
    assert discover_stamp(tmp_path)=="20210101_000000"
