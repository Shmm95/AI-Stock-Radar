"""Orchestrates the Phase-1 mean-reversion prototype research: fetch ->
quality-check -> compute signals -> backtest each component in
isolation and combined -> print + save results.

Read-only research. Never touches data/live/, the frozen engine, or
CONTROLLED_TICKERS/LIVE_CONTROLLED_TICKERS. Run by hand:

    .venv/bin/python -m src.research.mean_reversion_prototype.run_research
"""

from __future__ import annotations

import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

from src.research.mean_reversion_prototype.backtest import run_backtest, summarize_trades
from src.research.mean_reversion_prototype.data_loader import CANDIDATE_SYMBOLS, assess_quality, fetch_hourly_bars
from src.research.mean_reversion_prototype.signals import SignalParameters, compute_signals

OUTPUT_DIRECTORY = Path("data/research")

_CONFIGS = [
    ("RSI_ONLY", dict(use_rsi_signal=True, use_vwap_signal=False, use_volume_weighting=True)),
    ("VWAP_ONLY", dict(use_rsi_signal=False, use_vwap_signal=True, use_volume_weighting=True)),
    ("COMBINED_OR", dict(use_rsi_signal=True, use_vwap_signal=True, use_volume_weighting=True)),
    ("COMBINED_OR_NO_VOLUME_WEIGHT", dict(use_rsi_signal=True, use_vwap_signal=True, use_volume_weighting=False)),
]


def main() -> None:
    params = SignalParameters()
    quality_reports = {}
    results = {}

    for symbol in CANDIDATE_SYMBOLS:
        print(f"Fetching {symbol}...", flush=True)
        raw = fetch_hourly_bars(symbol)
        clean, report = assess_quality(symbol, raw)
        quality_reports[symbol] = report._asdict()
        signal_frame = compute_signals(clean, params)

        results[symbol] = {}
        history_days = (clean.index.max() - clean.index.min()).days
        for name, kwargs in _CONFIGS:
            trades = run_backtest(signal_frame, params, **kwargs)
            summary = summarize_trades(trades)
            summary["trades_per_day"] = round(summary["trade_count"] / history_days, 3)
            summary["trades_per_week"] = round(7 * summary["trade_count"] / history_days, 2)
            results[symbol][name] = summary
        print(f"  {symbol}: {results[symbol]['COMBINED_OR']}", flush=True)

    print("\n=== AGGREGATE ACROSS TICKERS ===")
    aggregate = {}
    for name, _ in _CONFIGS:
        trade_counts = [results[s][name]["trade_count"] for s in results]
        pfs = [results[s][name]["profit_factor"] for s in results if isinstance(results[s][name]["profit_factor"], (int, float))]
        win_rates = [results[s][name]["win_rate_percent"] for s in results if results[s][name]["win_rate_percent"] is not None]
        aggregate[name] = {
            "total_trades": sum(trade_counts),
            "mean_profit_factor": round(statistics.mean(pfs), 3) if pfs else None,
            "mean_win_rate_percent": round(statistics.mean(win_rates), 1) if win_rates else None,
        }
        print(f"{name}: {aggregate[name]}")

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / f"mean_reversion_prototype_v1_{stamp}.json"
    output_path.write_text(
        json.dumps(
            {
                "status": "RESEARCH_CANDIDATE_NOT_OFFICIAL",
                "generated_at": datetime.now(UTC).isoformat(),
                "parameters": params.__dict__,
                "data_quality": quality_reports,
                "results_by_ticker": results,
                "aggregate": aggregate,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
