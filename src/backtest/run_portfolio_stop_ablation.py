"""Full-period portfolio stop-loss ablation.

Tests stock and crypto initial-stop distances on identical prepared data while
holding the validated V1 portfolio controls fixed. Historical research only;
this module cannot place broker orders.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.backtest.portfolio_backtest_engine import run_portfolio_backtest
from src.backtest.portfolio_backtest_models import PortfolioBacktestConfig
from src.backtest.run_portfolio_position_ablation import (
    DEFAULT_REGIME_PERIOD, DEFAULT_TICKERS, annual_returns_from_equity,
    build_variant_summary, rejection_counts, run_matched_benchmark,
    ticker_contributions,
)

DEFAULT_STOPS = (3.5, 5.0, 7.5)
DEFAULT_BASELINE_STOP = 5.0
DEFAULT_OUTPUT_DIRECTORY = Path("data/backtests/portfolio/stop_ablation")


def normalize_stops(values: Iterable[float]) -> tuple[float, ...]:
    result = tuple(sorted({round(float(value), 8) for value in values}))
    if not result or any(value <= 0 or value > 100 for value in result):
        raise ValueError("stop values must be non-empty and in (0, 100].")
    return result


def _token(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def stop_variant_id(stock_stop: float, crypto_stop: float) -> str:
    return f"S{_token(stock_stop)}_C{_token(crypto_stop)}"


def build_stop_grid(stock_stops: Iterable[float], crypto_stops: Iterable[float]):
    return tuple(
        {
            "variant_id": stop_variant_id(stock, crypto),
            "stock_stop_loss_percent": stock,
            "crypto_stop_loss_percent": crypto,
        }
        for stock in normalize_stops(stock_stops)
        for crypto in normalize_stops(crypto_stops)
    )


def build_stop_config(base_config: PortfolioBacktestConfig, stock_stop: float,
                      crypto_stop: float) -> PortfolioBacktestConfig:
    config = replace(base_config, stock_stop_loss_percent=float(stock_stop),
                     crypto_stop_loss_percent=float(crypto_stop))
    config.validate()
    return config


def rank_stop_candidates(summary: pd.DataFrame,
                         baseline_stop: float = DEFAULT_BASELINE_STOP) -> pd.DataFrame:
    required = {"variant_id", "stock_stop_loss_percent",
                "crypto_stop_loss_percent", "return_drawdown_ratio",
                "excess_return_vs_matched_percent", "profit_factor",
                "total_return_percent", "maximum_drawdown_percent"}
    missing = required.difference(summary.columns)
    if missing or summary.empty:
        raise ValueError(f"Invalid stop summary; missing={sorted(missing)}")
    ranked = summary.copy().reset_index(drop=True)
    rules = (("return_drawdown_ratio", False),
             ("excess_return_vs_matched_percent", False),
             ("profit_factor", False), ("total_return_percent", False),
             ("maximum_drawdown_percent", True))
    rank_cols = []
    for column, ascending in rules:
        name = f"rank_{column}"
        values = pd.to_numeric(ranked[column], errors="coerce")
        values = values.fillna(float("inf") if ascending else float("-inf"))
        ranked[name] = values.rank(method="min", ascending=ascending)
        rank_cols.append(name)
    ranked["selection_rank_sum"] = ranked[rank_cols].sum(axis=1)
    ranked["distance_to_baseline"] = (
        (ranked.stock_stop_loss_percent - baseline_stop).abs()
        + (ranked.crypto_stop_loss_percent - baseline_stop).abs()
    )
    ranked = ranked.sort_values(
        ["selection_rank_sum", "return_drawdown_ratio",
         "excess_return_vs_matched_percent", "profit_factor",
         "maximum_drawdown_percent", "total_return_percent",
         "distance_to_baseline", "stock_stop_loss_percent",
         "crypto_stop_loss_percent"],
        ascending=[True, False, False, False, True, False, True, True, True],
        kind="mergesort").reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    ranked["selected_candidate"] = ranked["rank"].eq(1)
    return ranked


def run_portfolio_stop_ablation(*, data_by_ticker: dict[str, pd.DataFrame],
        base_config: PortfolioBacktestConfig,
        stock_stops: Iterable[float] = DEFAULT_STOPS,
        crypto_stops: Iterable[float] = DEFAULT_STOPS,
        baseline_stop: float = DEFAULT_BASELINE_STOP) -> dict[str, Any]:
    if not data_by_ticker:
        raise ValueError("data_by_ticker cannot be empty.")
    grid = build_stop_grid(stock_stops, crypto_stops)
    summaries, ticker_frames, annual_frames, rejection_rows, equity_frames = [], [], [], [], []
    raw_results = {}
    for sequence, variant in enumerate(grid, 1):
        variant_id = variant["variant_id"]
        stock_stop = float(variant["stock_stop_loss_percent"])
        crypto_stop = float(variant["crypto_stop_loss_percent"])
        print(f"[{sequence}/{len(grid)}] {variant_id}")
        config = build_stop_config(base_config, stock_stop, crypto_stop)
        result = run_portfolio_backtest(data_by_ticker=data_by_ticker,
                                        config=config, include_benchmark=False)
        matched, _ = run_matched_benchmark(
            data_by_ticker, initial_cash=config.initial_cash,
            exposure_percent=max(result.average_exposure_percent, 0.0001),
            config=config, label=f"MATCHED_{variant_id}")
        annual = annual_returns_from_equity(
            result.equity_curve, initial_cash=result.initial_cash
        )
        tickers = ticker_contributions(result)
        summary = build_variant_summary(result, matched, annual, tickers)
        summary.update(variant)
        summaries.append(summary)
        raw_results[variant_id] = result.to_dict()
        for frame, target in ((tickers, ticker_frames), (annual, annual_frames)):
            if not frame.empty:
                frame = frame.copy(); frame.insert(0, "variant_id", variant_id)
                frame.insert(1, "stock_stop_loss_percent", stock_stop)
                frame.insert(2, "crypto_stop_loss_percent", crypto_stop)
                target.append(frame)
        for reason, count in sorted(rejection_counts(result).items()):
            rejection_rows.append({**variant, "reason_code": reason, "count": count})
        equity = pd.DataFrame([point.to_dict() for point in result.equity_curve])
        if not equity.empty:
            equity.insert(0, "variant_id", variant_id)
            equity.insert(1, "stock_stop_loss_percent", stock_stop)
            equity.insert(2, "crypto_stop_loss_percent", crypto_stop)
            equity_frames.append(equity)
    concat = lambda frames: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return {"summary": rank_stop_candidates(pd.DataFrame(summaries), baseline_stop),
            "tickers": concat(ticker_frames), "annual": concat(annual_frames),
            "rejections": pd.DataFrame(rejection_rows), "equity": concat(equity_frames),
            "raw_results": raw_results, "base_config": base_config.to_dict(),
            "stop_grid": list(grid), "baseline_stop": baseline_stop}


def _safe(value: Any) -> Any:
    if isinstance(value, float) and not isfinite(value): return None
    if isinstance(value, dict): return {k: _safe(v) for k, v in value.items()}
    if isinstance(value, list): return [_safe(v) for v in value]
    return value


def save_portfolio_stop_ablation(bundle: dict[str, Any], output_directory: Path = DEFAULT_OUTPUT_DIRECTORY):
    output_directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    paths = {name: output_directory / f"portfolio_stop_ablation_{name}_{stamp}.csv"
             for name in ("summary", "tickers", "annual", "rejections", "equity")}
    paths["json"] = output_directory / f"portfolio_stop_ablation_{stamp}.json"
    payload = {"created_at": datetime.now(UTC).isoformat(),
               "method": "Full-period stock/crypto initial-stop ablation; exploratory only.",
               "base_config": bundle["base_config"], "baseline_stop": bundle["baseline_stop"],
               "stop_grid": bundle["stop_grid"], "summary": bundle["summary"].to_dict("records"),
               "runs": bundle["raw_results"]}
    paths["json"].write_text(json.dumps(_safe(payload), indent=2) + "\n", encoding="utf-8")
    for name in ("summary", "tickers", "annual", "rejections", "equity"):
        bundle[name].to_csv(paths[name], index=False)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tickers", nargs="*")
    parser.add_argument("--stock-stops", nargs="+", type=float, default=list(DEFAULT_STOPS))
    parser.add_argument("--crypto-stops", nargs="+", type=float, default=list(DEFAULT_STOPS))
    parser.add_argument("--period", default="5y"); parser.add_argument("--regime-period", default=DEFAULT_REGIME_PERIOD)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    config = PortfolioBacktestConfig(risk_per_trade_percent=1.0,
        maximum_total_open_risk_percent=4.0, maximum_open_positions=4,
        stock_stop_loss_percent=5.0, crypto_stop_loss_percent=5.0)
    from src.backtest.run_portfolio_backtest import prepare_portfolio_data
    data = prepare_portfolio_data(tickers=args.tickers or list(DEFAULT_TICKERS),
                                  period=args.period, regime_period=args.regime_period)
    bundle = run_portfolio_stop_ablation(data_by_ticker=data, base_config=config,
        stock_stops=args.stock_stops, crypto_stops=args.crypto_stops)
    print(bundle["summary"].to_string(index=False))
    if not args.no_save:
        for name, path in save_portfolio_stop_ablation(bundle).items(): print(name, path.resolve())


if __name__ == "__main__": main()
