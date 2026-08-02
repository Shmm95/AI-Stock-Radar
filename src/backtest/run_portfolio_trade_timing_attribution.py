"""Trade-event attribution for portfolio stop walk-forward results.

The runner replays selected unseen test windows with the original portfolio
engine and records diagnostics that summary exports cannot identify: return
before first entry, return after exits, stop recovery, rejected-signal forward
returns, no-entry ticker windows, and one common full-exposure market regime.

Forward-event metrics are diagnostics, not additive counterfactual PnL. A
rejected signal may repeat and capital cannot necessarily fund every event.
Historical research only; this module cannot place broker orders.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from src.backtest.portfolio_backtest_engine import run_portfolio_backtest
from src.backtest.portfolio_backtest_models import PortfolioBacktestConfig
from src.backtest.run_portfolio_backtest import prepare_portfolio_data
from src.backtest.run_portfolio_position_ablation import DEFAULT_TICKERS
from src.backtest.run_portfolio_walk_forward import slice_prepared_data

DEFAULT_SOURCE_DIRECTORY = Path("data/backtests/portfolio/stop_walk_forward")
DEFAULT_OUTPUT_DIRECTORY = Path("data/backtests/portfolio/trade_timing_attribution")
DEFAULT_HORIZONS = (5, 10, 20)
MODEL_DYNAMIC = "DYNAMIC_SELECTED"
MODEL_BASELINE = "FIXED_BASELINE"
MODEL_MAX_RETURN = "FIXED_MAX_RETURN"
MODEL_LOW_DRAWDOWN = "FIXED_LOW_DRAWDOWN"
MODEL_RANK_WINNER = "FIXED_RANK_WINNER"
DEFAULT_MODELS = (MODEL_BASELINE, MODEL_MAX_RETURN, MODEL_LOW_DRAWDOWN, MODEL_DYNAMIC)
FIXED_PROFILES = {
    MODEL_BASELINE: (5.0, 5.0), MODEL_MAX_RETURN: (3.5, 3.5),
    MODEL_LOW_DRAWDOWN: (7.5, 3.5), MODEL_RANK_WINNER: (5.0, 3.5),
}


def normalize_horizons(values: Iterable[int]) -> tuple[int, ...]:
    result = tuple(sorted({int(value) for value in values}))
    if not result or any(value <= 0 for value in result):
        raise ValueError("horizons must contain positive integers.")
    return result


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy(); output.index = pd.to_datetime(output.index)
    if output.index.tz is not None: output.index = output.index.tz_convert(None)
    return output.sort_index().loc[lambda x: ~x.index.duplicated(keep="last")]


def classify_common_regime(return_percent: float, *, bull: float = 10.0,
                           bear: float = -10.0) -> str:
    if bear >= bull: raise ValueError("bear threshold must be below bull threshold.")
    if return_percent >= bull: return "BULL"
    if return_percent <= bear: return "BEAR"
    return "SIDEWAYS"


def _position(data: pd.DataFrame, timestamp: Any) -> int | None:
    index = data.index
    location = int(index.searchsorted(pd.Timestamp(timestamp), side="left"))
    return location if location < len(index) else None


def forward_metrics(data: pd.DataFrame, timestamp: Any, reference_price: float,
                    horizons: Sequence[int] = DEFAULT_HORIZONS) -> dict[str, Any]:
    """Calculate close-to-reference forward and maximum-close returns."""
    data = _normalize_frame(data); position = _position(data, timestamp)
    output: dict[str, Any] = {}
    for horizon in normalize_horizons(horizons):
        end = None if position is None else position + horizon
        if position is None or end >= len(data) or reference_price <= 0:
            output[f"forward_return_{horizon}_bars_percent"] = None
            output[f"max_forward_return_{horizon}_bars_percent"] = None
            continue
        close = float(data.iloc[end]["Close"])
        future = data.iloc[position + 1:end + 1]
        maximum_close = float(future["Close"].max()) if not future.empty else close
        output[f"forward_return_{horizon}_bars_percent"] = round(
            (close/reference_price-1)*100, 4)
        output[f"max_forward_return_{horizon}_bars_percent"] = round(
            (maximum_close/reference_price-1)*100, 4)
    return output


def first_recovery_bars(data: pd.DataFrame, timestamp: Any, target_price: float,
                        maximum_horizon: int) -> int | None:
    data = _normalize_frame(data); position = _position(data, timestamp)
    if position is None: return None
    future = data.iloc[position+1:position+maximum_horizon+1]
    matches = future.index[future["Close"].astype(float) >= target_price]
    if len(matches) == 0: return None
    return int(data.index.get_loc(matches[0]) - position)


def exit_category(exit_reason: str, signal_reason: str = "") -> str:
    reason = str(exit_reason).upper(); signal = str(signal_reason).upper()
    if "STOP_LOSS" in reason: return "STOP_LOSS"
    if "FORCE_CLOSE" in reason: return "FORCE_CLOSE_END"
    trend, trailing = "EMA20 BELOW EMA50" in signal, "TRAILING" in signal
    if trend and trailing: return "TREND_AND_TRAILING"
    if trailing: return "TRAILING_CLOSE"
    if trend: return "TREND_EXIT"
    return reason or "UNKNOWN"


def infer_exit_category(trade: Any, data: pd.DataFrame,
                        trailing_percent: float) -> str:
    """Reconstruct the close-based exit trigger from causal market bars."""
    direct = exit_category(trade.exit_reason, "")
    if direct != "EXIT_SIGNAL_NEXT_OPEN":
        return direct
    frame = _normalize_frame(data)
    entry_position = _position(frame, trade.entry_timestamp)
    exit_position = _position(frame, trade.exit_timestamp)
    if entry_position is None or exit_position is None or exit_position <= 0:
        return "EXIT_SIGNAL_NEXT_OPEN"
    signal_position = exit_position - 1
    if signal_position < entry_position:
        return "EXIT_SIGNAL_NEXT_OPEN"
    signal_row = frame.iloc[signal_position]
    closes = frame.iloc[entry_position:signal_position + 1]["Close"].astype(float)
    highest_close = max(float(trade.entry_price), float(closes.max()))
    trend = float(signal_row["EMA20"]) < float(signal_row["EMA50"])
    trailing_level = highest_close * (1 - float(trailing_percent) / 100)
    trailing = float(signal_row["Close"]) <= trailing_level
    if trend and trailing: return "TREND_AND_TRAILING"
    if trailing: return "TRAILING_CLOSE"
    if trend: return "TREND_EXIT"
    return "EXIT_SIGNAL_UNRESOLVED"


def common_benchmark(data_by_ticker: dict[str, pd.DataFrame], *, window_id: str,
                     test_start: Any, test_end_exclusive: Any,
                     bull_threshold: float = 10.0,
                     bear_threshold: float = -10.0) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ticker_rows=[]
    for ticker, raw in sorted(data_by_ticker.items()):
        data=_normalize_frame(raw)
        if data.empty: continue
        start_open=float(data.iloc[0]["Open"]); end_close=float(data.iloc[-1]["Close"])
        value=(end_close/start_open-1)*100 if start_open>0 else 0.0
        ticker_rows.append({"window_id":window_id,"ticker":ticker,
            "asset_class":"CRYPTO" if ticker.endswith(("-USD","-EUR","-GBP")) else "EQUITY",
            "test_start":pd.Timestamp(test_start),"test_end_exclusive":pd.Timestamp(test_end_exclusive),
            "start_open":round(start_open,8),"end_close":round(end_close,8),
            "full_window_return_percent":round(value,4)})
    if not ticker_rows: raise ValueError(f"No benchmark ticker data for {window_id}.")
    value=sum(row["full_window_return_percent"] for row in ticker_rows)/len(ticker_rows)
    summary={"window_id":window_id,"test_start":pd.Timestamp(test_start),
        "test_end_exclusive":pd.Timestamp(test_end_exclusive),"ticker_count":len(ticker_rows),
        "equal_weight_full_exposure_return_percent":round(value,4),
        "common_regime":classify_common_regime(value,bull=bull_threshold,bear=bear_threshold)}
    return summary,ticker_rows


def _profile_for(model: str, window: dict[str, Any]) -> tuple[float, float]:
    if model == MODEL_DYNAMIC:
        return float(window["selected_stock_stop_loss_percent"]), float(window["selected_crypto_stop_loss_percent"])
    if model not in FIXED_PROFILES: raise ValueError(f"Unknown model: {model}")
    return FIXED_PROFILES[model]


def load_source(source_directory: Path, stamp: str) -> dict[str, Any]:
    source_directory=Path(source_directory)
    json_path=source_directory/f"portfolio_stop_walk_forward_{stamp}.json"
    windows_path=source_directory/f"portfolio_stop_walk_forward_windows_{stamp}.csv"
    runs_path=source_directory/f"portfolio_stop_walk_forward_test_runs_{stamp}.csv"
    for path in (json_path,windows_path,runs_path):
        if not path.exists(): raise FileNotFoundError(path)
    payload=json.loads(json_path.read_text(encoding="utf-8"))
    return {"stamp":stamp,"payload":payload,"windows":pd.read_csv(windows_path),
            "test_runs":pd.read_csv(runs_path),"paths":{"json":json_path,"windows":windows_path,"test_runs":runs_path}}


def _config_from_source(source: dict[str, Any]) -> PortfolioBacktestConfig:
    values=source["payload"].get("base_config")
    if not isinstance(values,dict): raise ValueError("Source JSON has no base_config.")
    config=PortfolioBacktestConfig(**values);config.validate();return config


def _trade_record(trade: Any, *, data: pd.DataFrame, window_id: str, model: str,
                  stock_stop: float, crypto_stop: float, horizons: Sequence[int]) -> dict[str, Any]:
    row={"window_id":window_id,"model":model,"stock_stop_loss_percent":stock_stop,
         "crypto_stop_loss_percent":crypto_stop,**trade.to_dict()}
    row["exit_category"]=infer_exit_category(trade,data,7.5)
    row.update(forward_metrics(data,trade.exit_timestamp,trade.exit_price,horizons))
    maximum=max(horizons)
    recovery=first_recovery_bars(data,trade.exit_timestamp,trade.entry_price,maximum)
    row[f"recovered_entry_within_{maximum}_bars"]=recovery is not None
    row["first_entry_recovery_bars"]=recovery
    return row


def _rejection_record(item: Any, *, data: pd.DataFrame, window_id: str, model: str,
                      stock_stop: float, crypto_stop: float,
                      horizons: Sequence[int]) -> dict[str, Any]:
    position=_position(data,item.timestamp)
    reference=float(data.iloc[position]["Open"]) if position is not None else 0.0
    row={"window_id":window_id,"model":model,"stock_stop_loss_percent":stock_stop,
         "crypto_stop_loss_percent":crypto_stop,"reference_open":round(reference,8),**item.to_dict()}
    row.update(forward_metrics(data,item.timestamp,reference,horizons));return row


def _ticker_window_records(*, result: Any, data_by_ticker: dict[str,pd.DataFrame],
                           window_id: str, model: str, benchmark_tickers: list[dict[str,Any]]) -> list[dict[str,Any]]:
    trades=defaultdict(list);rejections=Counter()
    for trade in result.trades: trades[trade.ticker].append(trade)
    for item in result.rejections: rejections[item.ticker]+=1
    benchmark={row["ticker"]:row for row in benchmark_tickers};records=[]
    for ticker,data in sorted(data_by_ticker.items()):
        frame=_normalize_frame(data); ticker_trades=sorted(trades[ticker],key=lambda x:x.entry_timestamp)
        first_open=float(frame.iloc[0]["Open"]);first_entry=ticker_trades[0] if ticker_trades else None
        if first_entry:
            position=_position(frame,first_entry.entry_timestamp)
            delay=position
            pre_return=(first_entry.entry_price/first_open-1)*100 if first_open>0 else 0.0
        else:
            delay=None;pre_return=None
        records.append({"window_id":window_id,"model":model,"ticker":ticker,
            "asset_class":benchmark[ticker]["asset_class"],"entered":bool(ticker_trades),
            "first_entry_timestamp":None if first_entry is None else first_entry.entry_timestamp,
            "first_entry_delay_ticker_bars":delay,"pre_first_entry_return_percent":None if pre_return is None else round(pre_return,4),
            "full_window_return_percent":benchmark[ticker]["full_window_return_percent"],
            "completed_trades":len(ticker_trades),"net_pnl":round(sum(x.net_pnl for x in ticker_trades),2),
            "rejection_count":int(rejections[ticker])})
    return records


def _finite_average(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:return 0.0
    values=pd.to_numeric(frame[column],errors="coerce").dropna()
    values=values[values.map(lambda value:isfinite(float(value)))]
    return round(float(values.mean()),4) if len(values) else 0.0


def build_exit_summary(trades: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    records=[]
    if trades.empty:return pd.DataFrame()
    maximum=max(horizons)
    for (model,category),group in trades.groupby(["model","exit_category"],sort=True):
        row={"model":model,"exit_category":category,"trade_count":len(group),
             "net_pnl":round(float(group.net_pnl.sum()),2),"average_trade_return_percent":_finite_average(group,"return_percent"),
             f"entry_recovery_rate_{maximum}_bars_percent":round(float(group[f"recovered_entry_within_{maximum}_bars"].mean()*100),4)}
        for horizon in horizons:
            row[f"average_post_exit_return_{horizon}_bars_percent"]=_finite_average(group,f"forward_return_{horizon}_bars_percent")
            row[f"average_max_post_exit_return_{horizon}_bars_percent"]=_finite_average(group,f"max_forward_return_{horizon}_bars_percent")
        records.append(row)
    return pd.DataFrame(records)


def build_rejection_summary(rejections: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    records=[]
    if rejections.empty:return pd.DataFrame()
    for (model,reason),group in rejections.groupby(["model","reason_code"],sort=True):
        row={"model":model,"reason_code":reason,"event_count":len(group)}
        for horizon in horizons:
            column=f"forward_return_{horizon}_bars_percent";maximum=f"max_forward_return_{horizon}_bars_percent"
            values=pd.to_numeric(group[column],errors="coerce")
            row[f"average_rejected_return_{horizon}_bars_percent"]=_finite_average(group,column)
            row[f"positive_rejected_return_rate_{horizon}_bars_percent"]=round(float((values.dropna()>0).mean()*100),4) if values.notna().any() else 0.0
            row[f"average_rejected_max_return_{horizon}_bars_percent"]=_finite_average(group,maximum)
        records.append(row)
    return pd.DataFrame(records)


def build_model_summary(window_models: pd.DataFrame, trades: pd.DataFrame,
                        rejections: pd.DataFrame, ticker_windows: pd.DataFrame,
                        horizons: Sequence[int]) -> pd.DataFrame:
    records=[];maximum=max(horizons)
    for model,windows in window_models.groupby("model",sort=False):
        model_trades=trades.loc[trades.model==model] if not trades.empty else trades
        model_rejections=rejections.loc[rejections.model==model] if not rejections.empty else rejections
        ticker=ticker_windows.loc[ticker_windows.model==model]
        returns=windows.rerun_return_percent.astype(float)
        compound=1.0
        for value in returns:compound*=1+value/100
        stop=model_trades.loc[model_trades.exit_category=="STOP_LOSS"] if not model_trades.empty else model_trades
        records.append({"model":model,"window_count":len(windows),
            "rerun_compounded_return_percent":round((compound-1)*100,4),
            "maximum_source_return_difference_percent":round(float(windows.source_return_difference_percent.abs().max()),6),
            "trade_count":len(model_trades),"net_pnl_sum_across_reset_windows":round(float(model_trades.net_pnl.sum()),2) if not model_trades.empty else 0.0,
            "rejection_event_count":len(model_rejections),"no_entry_ticker_window_count":int((~ticker.entered.astype(bool)).sum()),
            "average_first_entry_delay_bars":_finite_average(ticker,"first_entry_delay_ticker_bars"),
            "average_pre_first_entry_return_percent":_finite_average(ticker,"pre_first_entry_return_percent"),
            f"stop_entry_recovery_rate_{maximum}_bars_percent":round(float(stop[f"recovered_entry_within_{maximum}_bars"].mean()*100),4) if not stop.empty else 0.0,
            f"average_post_exit_return_{maximum}_bars_percent":_finite_average(model_trades,f"forward_return_{maximum}_bars_percent"),
            f"average_rejected_return_{maximum}_bars_percent":_finite_average(model_rejections,f"forward_return_{maximum}_bars_percent")})
    return pd.DataFrame(records)


def run_trade_timing_attribution(*, source: dict[str,Any], data_by_ticker: dict[str,pd.DataFrame],
        models: Sequence[str]=DEFAULT_MODELS, horizons: Sequence[int]=DEFAULT_HORIZONS,
        bull_threshold: float=10.0, bear_threshold: float=-10.0) -> dict[str,Any]:
    horizons=normalize_horizons(horizons);config=_config_from_source(source)
    source_runs=source["test_runs"].set_index(["window_id","model"])
    trade_rows=[];rejection_rows=[];ticker_window_rows=[];window_model_rows=[]
    common_rows=[];common_ticker_rows=[]
    for window in source["windows"].to_dict("records"):
        window_id=str(window["window_id"]);start=pd.Timestamp(window["test_start"]);end=pd.Timestamp(window["test_end_exclusive"])
        test_data=slice_prepared_data(data_by_ticker,start=start,end_exclusive=end)
        common,tickers=common_benchmark(test_data,window_id=window_id,test_start=start,
            test_end_exclusive=end,bull_threshold=bull_threshold,bear_threshold=bear_threshold)
        common_rows.append(common);common_ticker_rows.extend(tickers)
        for model in models:
            stock_stop,crypto_stop=_profile_for(model,window)
            result=run_portfolio_backtest(data_by_ticker=test_data,
                config=replace(config,stock_stop_loss_percent=stock_stop,crypto_stop_loss_percent=crypto_stop),
                include_benchmark=False)
            source_return=float(source_runs.loc[(window_id,model),"total_return_percent"]) if (window_id,model) in source_runs.index else float("nan")
            window_model_rows.append({**common,"model":model,"stock_stop_loss_percent":stock_stop,
                "crypto_stop_loss_percent":crypto_stop,"rerun_return_percent":result.total_return_percent,
                "source_return_percent":source_return,"source_return_difference_percent":round(result.total_return_percent-source_return,6),
                "maximum_drawdown_percent":result.maximum_drawdown_percent,"average_exposure_percent":result.average_exposure_percent,
                "completed_trades":result.completed_trades,"rejection_events":len(result.rejections)})
            for trade in result.trades:
                trade_rows.append(_trade_record(trade,data=test_data[trade.ticker],window_id=window_id,
                    model=model,stock_stop=stock_stop,crypto_stop=crypto_stop,horizons=horizons))
            for item in result.rejections:
                rejection_rows.append(_rejection_record(item,data=test_data[item.ticker],window_id=window_id,
                    model=model,stock_stop=stock_stop,crypto_stop=crypto_stop,horizons=horizons))
            ticker_window_rows.extend(_ticker_window_records(result=result,data_by_ticker=test_data,
                window_id=window_id,model=model,benchmark_tickers=tickers))
    frames={"window_models":pd.DataFrame(window_model_rows),"trades":pd.DataFrame(trade_rows),
        "rejections":pd.DataFrame(rejection_rows),"ticker_windows":pd.DataFrame(ticker_window_rows),
        "common_benchmark":pd.DataFrame(common_rows),"common_benchmark_tickers":pd.DataFrame(common_ticker_rows)}
    frames["exit_summary"]=build_exit_summary(frames["trades"],horizons)
    frames["rejection_summary"]=build_rejection_summary(frames["rejections"],horizons)
    frames["summary"]=build_model_summary(frames["window_models"],frames["trades"],frames["rejections"],frames["ticker_windows"],horizons)
    frames.update({"source_stamp":source["stamp"],"horizons":list(horizons),"models":list(models),
        "limitations":["Forward event returns are non-additive diagnostics, not executable counterfactual PnL.",
        "Repeated rejection events can refer to overlapping opportunities.",
        "Common benchmark is raw equal-weight full exposure and is used for regime labels, not execution comparison."]})
    return frames


def _safe(value:Any)->Any:
    if isinstance(value,(pd.Timestamp,datetime)):return value.isoformat()
    if isinstance(value,float) and not isfinite(value):return None
    if isinstance(value,dict):return {str(k):_safe(v) for k,v in value.items()}
    if isinstance(value,list):return [_safe(v) for v in value]
    return value


def save_trade_timing(result:dict[str,Any],output_directory:Path=DEFAULT_OUTPUT_DIRECTORY):
    output_directory.mkdir(parents=True,exist_ok=True);stamp=datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    names=("summary","window_models","common_benchmark","common_benchmark_tickers","trades","exit_summary","rejections","rejection_summary","ticker_windows")
    paths={name:output_directory/f"portfolio_trade_timing_{name}_{stamp}.csv" for name in names}
    paths["json"]=output_directory/f"portfolio_trade_timing_{stamp}.json"
    payload={"created_at":datetime.now(UTC).isoformat(),"source_stamp":result["source_stamp"],
        "method":"Replay-based trade timing and opportunity diagnostics","horizons":result["horizons"],
        "models":result["models"],"limitations":result["limitations"],
        **{name:result[name].to_dict("records") for name in names}}
    paths["json"].write_text(json.dumps(_safe(payload),indent=2)+"\n",encoding="utf-8")
    for name in names:result[name].to_csv(paths[name],index=False)
    return paths


def main():
    parser=argparse.ArgumentParser(description="Replay stop walk-forward windows for timing attribution")
    parser.add_argument("tickers",nargs="*");parser.add_argument("--source-directory",type=Path,default=DEFAULT_SOURCE_DIRECTORY)
    parser.add_argument("--stamp",required=True);parser.add_argument("--models",nargs="+",default=list(DEFAULT_MODELS))
    parser.add_argument("--horizons",nargs="+",type=int,default=list(DEFAULT_HORIZONS))
    parser.add_argument("--period",default="10y");parser.add_argument("--regime-period",default="max")
    parser.add_argument("--bull-threshold",type=float,default=10.0);parser.add_argument("--bear-threshold",type=float,default=-10.0)
    parser.add_argument("--no-crypto-regime",action="store_true");parser.add_argument("--output-directory",type=Path,default=DEFAULT_OUTPUT_DIRECTORY)
    args=parser.parse_args();source=load_source(args.source_directory,args.stamp)
    data=prepare_portfolio_data(tickers=args.tickers or list(DEFAULT_TICKERS),period=args.period,
        regime_period=args.regime_period,use_crypto_regime=not args.no_crypto_regime)
    result=run_trade_timing_attribution(source=source,data_by_ticker=data,models=args.models,horizons=args.horizons,
        bull_threshold=args.bull_threshold,bear_threshold=args.bear_threshold)
    print(result["summary"].to_string(index=False))
    for name,path in save_trade_timing(result,args.output_directory).items():print(name,path.resolve())


if __name__=="__main__":main()
