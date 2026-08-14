"""Phase 2: disciplined in-sample/out-of-sample parameter search for the
V1 combined signal. The overfitting risk here is real (5-ticker
universe, ~2.6-5.6 years of hourly data) -- this module exists
specifically to prevent "found the best combo, here it is" reporting,
which on a grid this size would just be curve-fitting to noise.

Design:
- ONE global, chronological split date across all 5 tickers (not a
  per-ticker 70% mark) -- derived from BTC/USD's own full range (the
  longest, most complete history), applied uniformly so the
  out-of-sample window is one real, shared stretch of calendar time,
  not a patchwork of different ranges per ticker.
- Signals/indicators are computed over each ticker's FULL continuous
  series for every parameter combination, then the resulting TRADES are
  split by entry time relative to the split date. This is correct, not
  a shortcut: every indicator here is strictly backward-looking, so
  computing it over the full series never lets a fixed, already-chosen
  parameter combination "see" the future -- it only avoids an
  artificial cold-start bias at the OOS window's own start (which would
  happen if indicators were reset at the split date instead).
- The grid search NEVER looks at out-of-sample trades when selecting
  the top combinations. OOS is touched exactly once, at the very end,
  to re-score whatever the in-sample search already chose.
"""

from __future__ import annotations

import itertools
from dataclasses import replace
from typing import Any

import pandas as pd

from src.research.mean_reversion_prototype.backtest import run_backtest
from src.research.mean_reversion_prototype.signals import SignalParameters, compute_signals

RSI_ENTRY_GRID = (5.0, 10.0, 15.0, 20.0)
RSI_EXIT_GRID = (60.0, 70.0, 80.0)
TIMEOUT_BARS_GRID = (12, 24, 48)
VWAP_DEVIATION_GRID = (1.5, 2.0, 2.5, 3.0)
SMA_PERIOD_GRID = (50, 100, 200)


def full_grid() -> list[SignalParameters]:
    combos = itertools.product(
        RSI_ENTRY_GRID, RSI_EXIT_GRID, TIMEOUT_BARS_GRID, VWAP_DEVIATION_GRID, SMA_PERIOD_GRID
    )
    base = SignalParameters()
    return [
        replace(
            base,
            rsi_entry_threshold=rsi_entry,
            rsi_exit_threshold=rsi_exit,
            max_holding_bars=timeout,
            vwap_deviation_entry=vwap_dev,
            trend_sma_period=sma_period,
        )
        for rsi_entry, rsi_exit, timeout, vwap_dev, sma_period in combos
    ]


def compute_split_date(reference_frame: pd.DataFrame, in_sample_fraction: float = 0.70) -> pd.Timestamp:
    start, end = reference_frame.index.min(), reference_frame.index.max()
    return start + (end - start) * in_sample_fraction


def _pool_summary(trade_lists: list[list]) -> dict[str, Any]:
    all_trades = [t for trades in trade_lists for t in trades]
    if not all_trades:
        return {"trade_count": 0, "profit_factor": None, "win_rate_percent": None}
    scaled = [t.net_pnl_percent * t.position_multiplier for t in all_trades]
    wins = [v for v in scaled if v > 0]
    losses = [v for v in scaled if v <= 0]
    gross_profit, gross_loss = sum(wins), sum(losses)
    pf = (
        round(gross_profit / abs(gross_loss), 3) if gross_loss != 0
        else (float("inf") if gross_profit > 0 else None)
    )
    return {
        "trade_count": len(all_trades),
        "profit_factor": pf if pf != float("inf") else "inf",
        "win_rate_percent": round(100 * len(wins) / len(all_trades), 1),
        "net_pnl_percent_total": round(sum(scaled), 2),
    }


def evaluate_combo(
    signal_frames: dict[str, pd.DataFrame], params: SignalParameters, split_date: pd.Timestamp
) -> dict[str, Any]:
    """Runs the combined-OR, volume-weighted backtest for one parameter
    combination across every ticker's full series, then splits the
    resulting trades by entry time -- IS trades entered before
    `split_date`, OOS trades at or after it. OOS is computed here but
    the caller (the grid search) must not act on it until selection is
    already final."""
    in_sample_trades, out_of_sample_trades = [], []
    for symbol, frame in signal_frames.items():
        trades = run_backtest(frame, params, use_rsi_signal=True, use_vwap_signal=True, use_volume_weighting=True)
        in_sample_trades.append([t for t in trades if t.entry_time < split_date])
        out_of_sample_trades.append([t for t in trades if t.entry_time >= split_date])

    return {
        "params": {
            "rsi_entry_threshold": params.rsi_entry_threshold,
            "rsi_exit_threshold": params.rsi_exit_threshold,
            "max_holding_bars": params.max_holding_bars,
            "vwap_deviation_entry": params.vwap_deviation_entry,
            "trend_sma_period": params.trend_sma_period,
        },
        "in_sample": _pool_summary(in_sample_trades),
        "out_of_sample": _pool_summary(out_of_sample_trades),
    }


def run_grid_search(
    raw_frames: dict[str, pd.DataFrame], reference_symbol: str = "BTC/USD"
) -> tuple[list[dict[str, Any]], pd.Timestamp]:
    """Computes signals once per parameter combination (over the FULL
    series, see module docstring), evaluates in-sample and out-of-sample
    pooled results for every combination in `full_grid()`, and returns
    everything -- selection/reporting discipline (only trusting
    in-sample for ranking) is the caller's responsibility, not enforced
    by this function's return value alone."""
    split_date = compute_split_date(raw_frames[reference_symbol])

    grid = full_grid()
    all_results = []
    for index, params in enumerate(grid, 1):
        signal_frames = {symbol: compute_signals(frame, params) for symbol, frame in raw_frames.items()}
        all_results.append(evaluate_combo(signal_frames, params, split_date))
        if index % 25 == 0 or index == len(grid):
            print(f"  [{index}/{len(grid)}] combinations done", flush=True)

    return all_results, split_date
