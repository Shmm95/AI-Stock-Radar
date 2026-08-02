"""Attribute portfolio walk-forward performance gaps to observable diagnostics.

This module consumes an existing Portfolio Stop Walk-Forward result set. It
does not re-run a strategy, download market data, or place orders. Attribution
is limited to facts identifiable from the exported aggregate, window, test-run,
ticker, rejection, and equity reports; causal entry/exit attribution requires
trade and per-ticker benchmark logs and is explicitly marked as unavailable.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

DEFAULT_INPUT_DIRECTORY = Path("data/backtests/portfolio/stop_walk_forward")
DEFAULT_OUTPUT_DIRECTORY = Path("data/backtests/portfolio/benchmark_gap_attribution")
SOURCE_PREFIX = "portfolio_stop_walk_forward"


def discover_stamp(input_directory: Path, stamp: str | None = None) -> str:
    """Return the requested or most recent complete result-set stamp."""
    input_directory = Path(input_directory)
    if stamp:
        candidate = str(stamp).strip()
        if not candidate:
            raise ValueError("stamp cannot be blank.")
        return candidate
    prefix = f"{SOURCE_PREFIX}_aggregate_"
    candidates = sorted(
        path.stem[len(prefix):]
        for path in input_directory.glob(f"{prefix}*.csv")
        if path.stem.startswith(prefix)
    )
    if not candidates:
        raise FileNotFoundError(f"No aggregate result found in {input_directory}.")
    return candidates[-1]


def source_paths(input_directory: Path, stamp: str) -> dict[str, Path]:
    names = ("aggregate", "windows", "test_runs", "tickers", "rejections", "equity")
    return {name: Path(input_directory) / f"{SOURCE_PREFIX}_{name}_{stamp}.csv"
            for name in names}


def load_source_bundle(input_directory: Path, stamp: str | None = None) -> dict[str, Any]:
    resolved_stamp = discover_stamp(input_directory, stamp)
    paths = source_paths(input_directory, resolved_stamp)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Incomplete result set: " + ", ".join(missing))
    bundle = {name: pd.read_csv(path) for name, path in paths.items()}
    bundle.update({"stamp": resolved_stamp, "source_paths": paths})
    validate_source_bundle(bundle)
    return bundle


def _require(frame: pd.DataFrame, label: str, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def validate_source_bundle(bundle: dict[str, Any]) -> None:
    _require(bundle["aggregate"], "aggregate", ("model", "initial_cash",
        "compounded_return_percent", "maximum_drawdown_percent",
        "return_drawdown_ratio", "matched_benchmark_compounded_return_percent",
        "average_profit_factor", "average_exposure_percent"))
    _require(bundle["test_runs"], "test_runs", ("window_id", "model", "variant_id",
        "test_start", "test_end_exclusive", "total_return_percent",
        "matched_benchmark_return_percent", "excess_return_vs_matched_percent",
        "maximum_drawdown_percent", "matched_benchmark_drawdown_percent",
        "profit_factor", "average_exposure_percent", "completed_trades", "total_fees"))
    _require(bundle["tickers"], "tickers", ("window_id", "model", "ticker",
        "asset_class", "completed_trades", "gross_profit", "gross_loss", "net_pnl",
        "total_fees"))
    _require(bundle["rejections"], "rejections", ("window_id", "model", "reason_code", "count"))
    runs = bundle["test_runs"]
    if runs.empty or runs.duplicated(["window_id", "model"]).any():
        raise ValueError("test_runs must contain exactly one row per window/model.")
    aggregate_models = set(bundle["aggregate"]["model"].astype(str))
    run_models = set(runs["model"].astype(str))
    if aggregate_models != run_models:
        raise ValueError("aggregate and test_runs model sets differ.")


def classify_regime(benchmark_return: float, *, bullish_threshold: float = 5.0,
                    bearish_threshold: float = -5.0) -> str:
    if bearish_threshold >= bullish_threshold:
        raise ValueError("bearish_threshold must be below bullish_threshold.")
    if benchmark_return >= bullish_threshold:
        return "BULL"
    if benchmark_return <= bearish_threshold:
        return "BEAR"
    return "SIDEWAYS"


def compound_returns(values: Iterable[float]) -> float:
    wealth = 1.0
    for value in values:
        wealth *= 1.0 + float(value) / 100.0
    return round((wealth - 1.0) * 100.0, 6)


def _finite_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    values = values[values.map(lambda value: isfinite(float(value)) if pd.notna(value) else False)]
    return round(float(values.mean()), 4) if len(values) else 0.0


def _correlation(left: pd.Series, right: pd.Series) -> float | None:
    left = pd.to_numeric(left, errors="coerce")
    right = pd.to_numeric(right, errors="coerce")
    valid = left.notna() & right.notna()
    if valid.sum() < 2 or left[valid].nunique() < 2 or right[valid].nunique() < 2:
        return None
    value = float(left[valid].corr(right[valid]))
    return round(value, 4) if isfinite(value) else None


def build_window_attribution(test_runs: pd.DataFrame, *, bullish_threshold: float = 5.0,
                             bearish_threshold: float = -5.0) -> pd.DataFrame:
    """Build per-window diagnostics and an exact chronological wealth-gap bridge."""
    output: list[dict[str, Any]] = []
    for model, group in test_runs.groupby("model", sort=False):
        rows = group.copy()
        rows["test_start"] = pd.to_datetime(rows["test_start"])
        rows = rows.sort_values(["test_start", "window_id"], kind="mergesort")
        strategy_wealth = benchmark_wealth = float(rows.iloc[0].get("initial_cash", 10_000.0))
        initial_cash = strategy_wealth
        for row in rows.to_dict("records"):
            strategy_return = float(row["total_return_percent"])
            benchmark_return = float(row["matched_benchmark_return_percent"])
            strategy_open, benchmark_open = strategy_wealth, benchmark_wealth
            strategy_gain = strategy_open * strategy_return / 100.0
            benchmark_gain = benchmark_open * benchmark_return / 100.0
            strategy_wealth += strategy_gain
            benchmark_wealth += benchmark_gain
            gap = strategy_return - benchmark_return
            record = dict(row)
            record.update({
                "strategy_return_percent": round(strategy_return, 4),
                "benchmark_return_percent": round(benchmark_return, 4),
                "matched_gap_percent": round(gap, 4),
                "regime": classify_regime(benchmark_return,
                    bullish_threshold=bullish_threshold, bearish_threshold=bearish_threshold),
                "benchmark_beaten": gap > 1e-9,
                "idle_cash_percent_proxy": round(max(100.0-float(row["average_exposure_percent"]),0.0),4),
                "upside_capture_percent": (round(strategy_return/benchmark_return*100,4)
                    if benchmark_return > 0 else None),
                "downside_capture_percent": (round(strategy_return/benchmark_return*100,4)
                    if benchmark_return < 0 else None),
                "strategy_opening_wealth": round(strategy_open, 6),
                "benchmark_opening_wealth": round(benchmark_open, 6),
                "strategy_window_gain": round(strategy_gain, 6),
                "benchmark_window_gain": round(benchmark_gain, 6),
                "wealth_gap_contribution_amount": round(strategy_gain-benchmark_gain, 6),
                "wealth_gap_contribution_percent_initial_cash": round(
                    (strategy_gain-benchmark_gain)/initial_cash*100, 6),
                "cumulative_strategy_wealth": round(strategy_wealth, 6),
                "cumulative_benchmark_wealth": round(benchmark_wealth, 6),
                "cumulative_wealth_gap_amount": round(strategy_wealth-benchmark_wealth, 6),
            })
            output.append(record)
    return pd.DataFrame(output)


def aggregate_tickers(tickers: pd.DataFrame) -> pd.DataFrame:
    grouped = tickers.groupby(["model", "ticker", "asset_class"], as_index=False).agg(
        completed_trades=("completed_trades", "sum"), gross_profit=("gross_profit", "sum"),
        gross_loss=("gross_loss", "sum"), net_pnl=("net_pnl", "sum"), total_fees=("total_fees", "sum"))
    grouped["profit_factor"] = grouped.apply(lambda row:
        (float("inf") if row.gross_profit > 0 else 0.0) if row.gross_loss <= 0
        else row.gross_profit/row.gross_loss, axis=1)
    grouped["model_positive_pnl"] = grouped.groupby("model")["net_pnl"].transform(
        lambda values: values[values > 0].sum())
    grouped["positive_pnl_share_percent"] = grouped.apply(lambda row:
        row.net_pnl/row.model_positive_pnl*100 if row.net_pnl > 0 and row.model_positive_pnl > 0 else 0.0, axis=1)
    return grouped.drop(columns=["model_positive_pnl"]).sort_values(
        ["model", "net_pnl"], ascending=[True, False]).reset_index(drop=True)


def aggregate_assets(ticker_summary: pd.DataFrame) -> pd.DataFrame:
    grouped = ticker_summary.groupby(["model", "asset_class"], as_index=False).agg(
        completed_trades=("completed_trades", "sum"), gross_profit=("gross_profit", "sum"),
        gross_loss=("gross_loss", "sum"), net_pnl=("net_pnl", "sum"), total_fees=("total_fees", "sum"))
    grouped["profit_factor"] = grouped.apply(lambda row:
        (float("inf") if row.gross_profit > 0 else 0.0) if row.gross_loss <= 0
        else row.gross_profit/row.gross_loss, axis=1)
    grouped["model_net_pnl"] = grouped.groupby("model")["net_pnl"].transform("sum")
    grouped["net_pnl_share_percent"] = grouped.apply(lambda row:
        row.net_pnl/row.model_net_pnl*100 if row.model_net_pnl else 0.0, axis=1)
    return grouped.drop(columns=["model_net_pnl"])


def aggregate_regimes(windows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (model, regime), group in windows.groupby(["model", "regime"], sort=True):
        records.append({"model":model, "regime":regime, "window_count":len(group),
            "strategy_compounded_return_percent":compound_returns(group.strategy_return_percent),
            "benchmark_compounded_return_percent":compound_returns(group.benchmark_return_percent),
            "average_matched_gap_percent":round(float(group.matched_gap_percent.mean()),4),
            "median_matched_gap_percent":round(float(group.matched_gap_percent.median()),4),
            "benchmark_beat_count":int(group.benchmark_beaten.sum()),
            "average_exposure_percent":round(float(group.average_exposure_percent.mean()),4),
            "average_profit_factor":_finite_mean(group.profit_factor),
            "average_strategy_drawdown_percent":round(float(group.maximum_drawdown_percent.mean()),4),
            "average_benchmark_drawdown_percent":round(float(group.matched_benchmark_drawdown_percent.mean()),4)})
    return pd.DataFrame(records)


def aggregate_rejections(rejections: pd.DataFrame) -> pd.DataFrame:
    return (rejections.groupby(["model","reason_code"],as_index=False)["count"].sum()
            .sort_values(["model","count"],ascending=[True,False]).reset_index(drop=True))


def build_tail_report(windows: pd.DataFrame, count: int = 3) -> pd.DataFrame:
    frames=[]
    for model,group in windows.groupby("model",sort=False):
        frame=group.nsmallest(count,"wealth_gap_contribution_amount").copy()
        frame.insert(2,"tail_rank",range(1,len(frame)+1));frames.append(frame)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()


def build_model_summary(aggregate: pd.DataFrame, windows: pd.DataFrame,
                        tickers: pd.DataFrame) -> pd.DataFrame:
    records=[]
    for row in aggregate.to_dict("records"):
        model=str(row["model"]); group=windows.loc[windows.model==model]
        ticker=tickers.loc[tickers.model==model]
        positive=ticker.loc[ticker.net_pnl>0].sort_values("net_pnl",ascending=False)
        positive_total=float(positive.net_pnl.sum())
        total_gap=float(row["compounded_return_percent"])-float(row["matched_benchmark_compounded_return_percent"])
        bull=group.loc[group.regime=="BULL"]; bear=group.loc[group.regime=="BEAR"]
        records.append({**row,
            "matched_benchmark_gap_percent":round(total_gap,4),
            "benchmark_beat_rate_percent":round(float(group.benchmark_beaten.mean()*100),4),
            "average_window_matched_gap_percent":round(float(group.matched_gap_percent.mean()),4),
            "median_window_matched_gap_percent":round(float(group.matched_gap_percent.median()),4),
            "worst_window_matched_gap_percent":round(float(group.matched_gap_percent.min()),4),
            "best_window_matched_gap_percent":round(float(group.matched_gap_percent.max()),4),
            "bull_upside_capture_percent":_finite_mean(bull.upside_capture_percent),
            "bear_downside_capture_percent":_finite_mean(bear.downside_capture_percent),
            "exposure_gap_correlation":_correlation(group.average_exposure_percent,group.matched_gap_percent),
            "average_idle_cash_percent_proxy":round(100-float(row["average_exposure_percent"]),4),
            "profitable_ticker_count":int((ticker.net_pnl>0).sum()),
            "losing_ticker_count":int((ticker.net_pnl<0).sum()),
            "largest_positive_ticker":None if positive.empty else str(positive.iloc[0].ticker),
            "largest_positive_pnl_share_percent":round(
                float(positive.iloc[0].net_pnl)/positive_total*100,4) if positive_total else 0.0,
            "top_3_positive_pnl_share_percent":round(
                float(positive.head(3).net_pnl.sum())/positive_total*100,4) if positive_total else 0.0,
            "exact_final_wealth_gap_amount":round(float(group.iloc[-1].cumulative_wealth_gap_amount),4),
            "exact_final_wealth_gap_percent_initial_cash":round(
                float(group.iloc[-1].cumulative_wealth_gap_amount)/float(row["initial_cash"])*100,4)})
    return pd.DataFrame(records)


def analyze_benchmark_gap(bundle: dict[str, Any], *, bullish_threshold: float = 5.0,
                          bearish_threshold: float = -5.0, tail_count: int = 3) -> dict[str, Any]:
    windows=build_window_attribution(bundle["test_runs"],bullish_threshold=bullish_threshold,
                                     bearish_threshold=bearish_threshold)
    tickers=aggregate_tickers(bundle["tickers"]);assets=aggregate_assets(tickers)
    return {"summary":build_model_summary(bundle["aggregate"],windows,tickers),
        "windows":windows,"regimes":aggregate_regimes(windows),"tickers":tickers,
        "assets":assets,"tails":build_tail_report(windows,tail_count),
        "rejections":aggregate_rejections(bundle["rejections"]),
        "selection_counts":dict(Counter(bundle["windows"].get("selected_variant_id",pd.Series(dtype=str)).astype(str))),
        "source_stamp":bundle["stamp"],"bullish_threshold":bullish_threshold,
        "bearish_threshold":bearish_threshold,
        "limitations":[
            "Matched benchmark already uses each model's average exposure; its gap is not a cash-allocation gap.",
            "Idle cash is diagnostic only and cannot be added to matched-benchmark gap.",
            "Current exports do not identify late-entry versus early-exit cost.",
            "Ticker PnL is strategy-side contribution, not ticker-level benchmark alpha.",
            "True timing attribution requires trade logs and per-ticker benchmark curves."]}


def _safe(value: Any) -> Any:
    if isinstance(value,(pd.Timestamp,datetime)):return value.isoformat()
    if isinstance(value,float) and not isfinite(value):return None
    if isinstance(value,dict):return {str(k):_safe(v) for k,v in value.items()}
    if isinstance(value,list):return [_safe(v) for v in value]
    return value


def save_attribution(result: dict[str, Any], output_directory: Path = DEFAULT_OUTPUT_DIRECTORY):
    output_directory.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    frames=("summary","windows","regimes","tickers","assets","tails","rejections")
    paths={name:output_directory/f"portfolio_benchmark_gap_{name}_{stamp}.csv" for name in frames}
    paths["json"]=output_directory/f"portfolio_benchmark_gap_{stamp}.json"
    payload={"created_at":datetime.now(UTC).isoformat(),"source_stamp":result["source_stamp"],
        "method":"Observable attribution from stop walk-forward exports",
        "thresholds":{"bullish":result["bullish_threshold"],"bearish":result["bearish_threshold"]},
        "selection_counts":result["selection_counts"],"limitations":result["limitations"],
        **{name:result[name].to_dict("records") for name in frames}}
    paths["json"].write_text(json.dumps(_safe(payload),indent=2)+"\n",encoding="utf-8")
    for name in frames:result[name].to_csv(paths[name],index=False)
    return paths


def main():
    parser=argparse.ArgumentParser(description="Attribute stop walk-forward benchmark gaps")
    parser.add_argument("--input-directory",type=Path,default=DEFAULT_INPUT_DIRECTORY)
    parser.add_argument("--stamp")
    parser.add_argument("--bullish-threshold",type=float,default=5.0)
    parser.add_argument("--bearish-threshold",type=float,default=-5.0)
    parser.add_argument("--tail-count",type=int,default=3)
    parser.add_argument("--output-directory",type=Path,default=DEFAULT_OUTPUT_DIRECTORY)
    args=parser.parse_args()
    if args.tail_count<1:raise ValueError("tail-count must be positive.")
    bundle=load_source_bundle(args.input_directory,args.stamp)
    result=analyze_benchmark_gap(bundle,bullish_threshold=args.bullish_threshold,
        bearish_threshold=args.bearish_threshold,tail_count=args.tail_count)
    print(result["summary"].to_string(index=False))
    for name,path in save_attribution(result,args.output_directory).items():print(name,path.resolve())


if __name__=="__main__":main()
