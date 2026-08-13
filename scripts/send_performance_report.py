"""Send a short, once-daily Telegram summary of generate_performance_report.py's
real P&L numbers.

Integration point, deliberately NOT the 4x/day send_status_update.py and
NOT run_daily_decision.py:
- send_status_update.py runs four times a day so a quiet day still shows a
  heartbeat. Real P&L only changes when an order actually fills -- showing
  it four times a day when nothing filled would just be repeated noise, not
  new information. Once a day (recommended: shortly after the daily
  decision cron, once that day's fills, if any, are settled) is enough.
- run_daily_decision.py is the protected daily decision path; financial
  reporting is a different concern (reads Alpaca's own fill history, not
  local state) and generate_performance_report.py's own module docstring
  already documents that it deliberately never imports run_daily_decision.py
  or order_submission's write paths. Keeping this as its own script
  preserves that separation instead of bolting reporting onto the decision
  script.

This script does NOT change generate_performance_report.py's own
calculation logic -- it imports and calls its existing, already-reliable
functions (fetch_filled_orders, reconstruct_round_trips, compute_metrics)
and only adds a short-message Telegram rendering on top.

Recommended cron line (NOT added automatically -- a separate, manual
deploy step):
    15 21 * * * cd /root/AI-Stock-Radar && /root/AI-Stock-Radar/.venv/bin/python scripts/send_performance_report.py >> /root/AI-Stock-Radar/logs/performance_report.log 2>&1
(run once daily, right after the 21:15 UTC daily-decision cron.)
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_performance_report import (
    ClosedTrade,
    compute_metrics,
    fetch_filled_orders,
    reconstruct_round_trips,
)
from src.live import order_submission
from src.notify.telegram_notifier import send_telegram_message


def _notify_safe(text: str) -> bool:
    """Same fail-safe contract as run_daily_decision.py's own _notify_safe
    (and send_status_update.py's, and crypto_stop_monitor.py's): never
    raises, and reports whether delivery could be confirmed, so a silent
    notifier failure is never indistinguishable from "nothing to report."
    This is exactly the class of bug this project hit and fixed before
    (a notification path that swallowed its own delivery failure) --
    deliberately not repeating it here.
    """
    try:
        return bool(send_telegram_message(text))
    except Exception:
        return False


def build_summary_text(closed_trades: list[ClosedTrade], open_lots: list) -> str:
    """Build the short Telegram summary. Never raises -- pure formatting
    over already-computed values; the caller guards the actual data
    fetch/computation separately."""
    now = datetime.now(UTC)
    lines = [f"AI-Stock-Radar performance summary -- {now.date().isoformat()}"]

    if not closed_trades and not open_lots:
        lines.append("Henüz gerçekleşen işlem yok.")
        return "\n".join(lines)

    if not closed_trades:
        lines.append("Henüz kapanmış (P&L hesaplanabilir) işlem yok.")
    else:
        cumulative_pnl = sum(trade.pnl for trade in closed_trades)
        deployed_capital = sum(trade.entry_price * trade.quantity for trade in closed_trades)
        pnl_percent = (cumulative_pnl / deployed_capital * 100) if deployed_capital else None
        wins = sum(1 for trade in closed_trades if trade.pnl > 0)
        win_rate = round(100 * wins / len(closed_trades), 1)

        percent_text = f" ({pnl_percent:+.2f}%)" if pnl_percent is not None else ""
        lines.append(f"Total P&L: ${cumulative_pnl:,.2f}{percent_text}")
        lines.append(f"Closed trades: {len(closed_trades)}, win rate: {win_rate}%")

        metrics = compute_metrics(closed_trades)
        for asset_class in sorted(metrics):
            summary = metrics[asset_class]
            lines.append(
                f"  {asset_class}: ${summary['cumulative_pnl']:,.2f} "
                f"({summary['total_trades']} trades, {summary['win_rate_percent']}% win rate)"
            )

    if open_lots:
        lines.append(f"Açık pozisyon (P&L henüz hesaplanmadı): {len(open_lots)}")

    return "\n".join(lines)


def main() -> None:
    try:
        client = order_submission.get_trading_client()
        orders = fetch_filled_orders(client)
        closed_trades, open_lots = reconstruct_round_trips(orders)
        text = build_summary_text(closed_trades, open_lots)
    except Exception as error:
        # Mirrors run_daily_decision.py's main(): never let a failure here
        # go unreported. A short, safe (no credentials/tokens) error
        # message is still sent, and the run still fails loudly afterward.
        text = (
            f"AI-Stock-Radar performance summary FAILED at {datetime.now(UTC).isoformat()}\n"
            f"{type(error).__name__}: {error}"
        )
        print(text)
        notified = _notify_safe(text)
        if not notified:
            print(
                "WARNING: performance-summary FAILURE Telegram notification could not be "
                "confirmed sent.",
                file=sys.stderr,
            )
        raise

    print(text)
    notified = _notify_safe(text)
    if not notified:
        print(
            "WARNING: performance-summary Telegram notification could not be confirmed sent.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
