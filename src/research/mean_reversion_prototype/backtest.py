"""A small, standalone, long-only backtest for this prototype's signals.

Deliberately NOT `src.backtest.portfolio_backtest_engine` -- that engine
is frozen/protected and built for the daily EMA20/EMA50/RSI14 strategy's
own event ordering (pending-order queues, next-Open execution, shared
portfolio cash). This prototype needs none of that: one signal, one
ticker at a time, next-bar-Open entry, and a same-loop exit check. A
from-scratch loop keeps that scope honest rather than bending an
unrelated engine to fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.research.mean_reversion_prototype.signals import SignalParameters

# Simple, fixed assumptions -- not tuned to this data, kept deliberately
# plain so the component comparison in the report isn't confounded by a
# second set of free parameters.
BASE_POSITION_NOTIONAL = 1000.0
STOP_LOSS_PERCENT = 3.0  # see the report for the rationale (hourly crypto
# range is smaller per-bar than daily equity, but this is a trade meant
# to resolve within ~24 bars, not a multi-day hold -- 3% is inside the
# typical multi-bar adverse excursion without being so tight it turns
# ordinary noise into a stop-out).
ROUND_TRIP_COST_PERCENT = 0.10  # 0.05% commission per side, no separate
# slippage line item -- a simple, symmetric assumption, not the frozen
# daily system's own (unrelated) fee schedule.


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    exit_reason: str
    holding_bars: int
    triggered_by: str  # "RSI", "VWAP", or "BOTH"
    volume_strong: bool
    position_multiplier: float
    net_pnl_percent: float


def _entry_trigger_label(row: pd.Series) -> str:
    if row["rsi_signal"] and row["vwap_signal"]:
        return "BOTH"
    if row["rsi_signal"]:
        return "RSI"
    return "VWAP"


def run_backtest(
    frame: pd.DataFrame,
    params: SignalParameters,
    *,
    use_rsi_signal: bool = True,
    use_vwap_signal: bool = True,
    use_volume_weighting: bool = True,
) -> list[Trade]:
    """One ticker, long-only, one open position at a time.

    `use_rsi_signal`/`use_vwap_signal` isolate each entry trigger for the
    component comparison; both True reproduces `combined_signal`.
    `use_volume_weighting=False` sizes every trade at 1.0x, to isolate
    the sizing effect on PF from the entry-trigger effect on frequency.

    Entry: at the NEXT bar's Open after a signal bar's Close (no
    lookahead -- mirrors the daily system's own next-available-Open
    convention). Exit: RSI back to `rsi_exit_threshold`, OR price back
    at/above the rolling VWAP, OR `max_holding_bars` reached, OR the
    stop-loss -- whichever coincides first, evaluated at each
    subsequent bar's Close (stop-loss additionally checked intrabar via
    that bar's Low). Exiting on ANY reversion signal regardless of
    which trigger opened the trade is deliberate: once in the trade, any
    sign the mean-reversion thesis has played out is a legitimate exit.
    """
    if not use_rsi_signal and not use_vwap_signal:
        raise ValueError("At least one of use_rsi_signal/use_vwap_signal must be True.")

    entry_signal = pd.Series(False, index=frame.index)
    if use_rsi_signal:
        entry_signal = entry_signal | frame["rsi_signal"]
    if use_vwap_signal:
        entry_signal = entry_signal | frame["vwap_signal"]

    trades: list[Trade] = []
    in_position = False
    entry_index = None
    entry_price = None
    stop_price = None
    triggered_by = None
    volume_strong = None
    position_multiplier = None

    rows = frame.reset_index()
    n = len(rows)

    for i in range(n):
        if in_position:
            row = rows.iloc[i]
            holding_bars = i - entry_index
            hit_stop = row["low"] <= stop_price
            rsi_exit = row["rsi2"] >= params.rsi_exit_threshold
            vwap_exit = (
                params.vwap_exit_at_or_above_mean
                and pd.notna(row["rolling_vwap"])
                and row["close"] >= row["rolling_vwap"]
            )
            timed_out = holding_bars >= params.max_holding_bars

            exit_reason = None
            exit_price = None
            if hit_stop:
                exit_reason, exit_price = "STOP_LOSS", stop_price
            elif rsi_exit:
                exit_reason, exit_price = "RSI_EXIT", row["open"] if i + 1 >= n else rows.iloc[i + 1]["open"]
            elif vwap_exit:
                exit_reason, exit_price = "VWAP_EXIT", row["open"] if i + 1 >= n else rows.iloc[i + 1]["open"]
            elif timed_out:
                exit_reason, exit_price = "TIME_EXIT", row["open"] if i + 1 >= n else rows.iloc[i + 1]["open"]

            if exit_reason == "STOP_LOSS":
                # Stop triggers intrabar -- exits at the stop price itself,
                # this same bar, not a future Open (a resting stop would
                # have filled during this bar in reality).
                gross_pct = (exit_price / entry_price - 1.0) * 100.0
                net_pct = gross_pct - ROUND_TRIP_COST_PERCENT
                trades.append(Trade(
                    entry_time=rows.iloc[entry_index]["timestamp"], exit_time=row["timestamp"],
                    entry_price=entry_price, exit_price=exit_price, exit_reason=exit_reason,
                    holding_bars=holding_bars, triggered_by=triggered_by,
                    volume_strong=volume_strong, position_multiplier=position_multiplier,
                    net_pnl_percent=net_pct,
                ))
                in_position = False
            elif exit_reason is not None and i + 1 < n:
                next_row = rows.iloc[i + 1]
                gross_pct = (next_row["open"] / entry_price - 1.0) * 100.0
                net_pct = gross_pct - ROUND_TRIP_COST_PERCENT
                trades.append(Trade(
                    entry_time=rows.iloc[entry_index]["timestamp"], exit_time=next_row["timestamp"],
                    entry_price=entry_price, exit_price=next_row["open"], exit_reason=exit_reason,
                    holding_bars=holding_bars + 1, triggered_by=triggered_by,
                    volume_strong=volume_strong, position_multiplier=position_multiplier,
                    net_pnl_percent=net_pct,
                ))
                in_position = False
            continue

        if entry_signal.iloc[i] and i + 1 < n:
            row = rows.iloc[i]
            next_row = rows.iloc[i + 1]
            in_position = True
            entry_index = i + 1
            entry_price = next_row["open"]
            stop_price = entry_price * (1 - STOP_LOSS_PERCENT / 100.0)
            triggered_by = _entry_trigger_label(row)
            volume_strong = bool(row["volume_strong"]) if pd.notna(row["volume_strong"]) else False
            position_multiplier = (
                1.0 if (not use_volume_weighting or volume_strong)
                else params.weak_volume_size_multiplier
            )

    return trades


def summarize_trades(trades: list[Trade]) -> dict:
    if not trades:
        return {
            "trade_count": 0, "win_rate_percent": None, "profit_factor": None,
            "avg_holding_bars": None, "net_pnl_percent_total": 0.0,
        }
    wins = [t.net_pnl_percent * t.position_multiplier for t in trades if t.net_pnl_percent > 0]
    losses = [t.net_pnl_percent * t.position_multiplier for t in trades if t.net_pnl_percent <= 0]
    gross_profit = sum(wins)
    gross_loss = sum(losses)
    profit_factor = (
        round(gross_profit / abs(gross_loss), 3) if gross_loss != 0
        else (float("inf") if gross_profit > 0 else None)
    )
    return {
        "trade_count": len(trades),
        "win_rate_percent": round(100 * len(wins) / len(trades), 1),
        "profit_factor": profit_factor if profit_factor != float("inf") else "inf",
        "avg_holding_bars": round(sum(t.holding_bars for t in trades) / len(trades), 1),
        "net_pnl_percent_total": round(sum(t.net_pnl_percent * t.position_multiplier for t in trades), 2),
        "rsi_triggered": sum(1 for t in trades if t.triggered_by == "RSI"),
        "vwap_triggered": sum(1 for t in trades if t.triggered_by == "VWAP"),
        "both_triggered": sum(1 for t in trades if t.triggered_by == "BOTH"),
        "volume_strong_count": sum(1 for t in trades if t.volume_strong),
        "volume_weak_count": sum(1 for t in trades if not t.volume_strong),
    }
