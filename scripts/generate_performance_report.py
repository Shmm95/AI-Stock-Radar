"""Read-only Alpaca paper-account performance report.

Reads REAL order history from Alpaca -- never submits, modifies, or
cancels anything. `src/live/order_submission.py`, `run_daily_decision.py`,
and `crypto_stop_monitor.py` are never imported here; this script only
calls `TradingClient.get_orders` (a read-only endpoint) via
`order_submission.get_trading_client()` for the paper-only client
builder.

Round-trip reconstruction: this system is long-only (no shorts), so a
SELL always closes a previously-opened BUY, never opens a new short
position. Filled orders are grouped by symbol, sorted chronologically,
and FIFO-matched: each SELL consumes quantity from the oldest still-open
BUY lot(s) for that symbol first. A BUY lot left over at the end (not
fully consumed by a later SELL) is a still-open position -- reported
separately, with no P&L (there is nothing to compare its entry price
against yet).

P&L is computed directly from Alpaca's own recorded `filled_qty` /
`filled_avg_price` on both sides of a matched pair -- not from local
bookkeeping (`position_state.json`), which is exactly why this avoids
the fee-in-kind quantity drift found in earlier crypto work: Alpaca's
own fill records are already ground truth, nothing to reconcile.

Operational health: scans `data/live/decisions/*.json` (the daily
runner's own decision logs) for any `needs_manual_review` entries and
lists them, so a recurring problem is visible without opening each
log file by hand.

Output: printed to stdout for immediate viewing, AND (unless
`--no-save` is passed) written to
`data/live/reports/performance_report_<stamp>.md` -- matching this
project's existing convention of a persistent, timestamped history
under `data/live/` (see `data/live/decisions/`), so past reports stay
comparable over time rather than only ever being visible in a
terminal that already scrolled away.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpaca.common.enums import Sort
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

from src.live import order_submission

DEFAULT_DECISION_LOG_DIRECTORY = Path("data/live/decisions")
DEFAULT_REPORT_DIRECTORY = Path("data/live/reports")
DEFAULT_ORDER_LOOKBACK_LIMIT = 500
_QUANTITY_EPSILON = 1e-9
# scripts/run_heartbeat_test.py's fixed-$15-notional BTC/USD round trips --
# a pipeline exerciser, never a strategy signal. Every real order it
# submits is tagged with this client_order_id prefix specifically so
# this report can exclude them here: real P&L/win-rate/trade-count must
# never include a trade that was never a trading decision.
HEARTBEAT_CLIENT_ORDER_ID_PREFIX = "HEARTBEAT_"


def _enum_value(value: Any) -> str:
    return str(value.value if hasattr(value, "value") else value)


@dataclass(slots=True)
class ClosedTrade:
    symbol: str
    asset_class: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: str
    exit_time: str
    pnl: float


@dataclass(slots=True)
class OpenLot:
    symbol: str
    asset_class: str
    quantity: float
    entry_price: float
    entry_time: str


def _is_heartbeat_order(order: Any) -> bool:
    client_order_id = getattr(order, "client_order_id", None) or ""
    return str(client_order_id).startswith(HEARTBEAT_CLIENT_ORDER_ID_PREFIX)


def fetch_filled_orders(
    client: TradingClient, *, limit: int = DEFAULT_ORDER_LOOKBACK_LIMIT
) -> list[Any]:
    """Fetch every filled REAL-strategy order, oldest first. Read-only.

    Excludes heartbeat pipeline-test fills (client_order_id prefixed
    `HEARTBEAT_`) -- see `HEARTBEAT_CLIENT_ORDER_ID_PREFIX`. Use
    `count_heartbeat_fills` for a separate, explicitly-labeled count of
    those; they must never be merged into this function's output.
    """
    request = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=limit, direction=Sort.ASC)
    orders = client.get_orders(filter=request)
    return [
        order
        for order in orders
        if _enum_value(order.status) == "filled"
        and order.filled_qty is not None
        and float(order.filled_qty) > 0
        and not _is_heartbeat_order(order)
    ]


def count_heartbeat_fills(
    client: TradingClient, *, limit: int = DEFAULT_ORDER_LOOKBACK_LIMIT
) -> int:
    """Separate, explicitly-labeled count of heartbeat pipeline-test fills.
    Deliberately never merged into `fetch_filled_orders`'s output or any
    real metric -- see that function's docstring."""
    request = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=limit, direction=Sort.ASC)
    orders = client.get_orders(filter=request)
    return sum(
        1
        for order in orders
        if _enum_value(order.status) == "filled"
        and order.filled_qty is not None
        and float(order.filled_qty) > 0
        and _is_heartbeat_order(order)
    )


def reconstruct_round_trips(orders: list[Any]) -> tuple[list[ClosedTrade], list[OpenLot]]:
    """FIFO-match BUY fills to SELL fills per symbol. See module docstring."""
    by_symbol: dict[str, list[Any]] = {}
    for order in orders:
        by_symbol.setdefault(order.symbol, []).append(order)

    closed_trades: list[ClosedTrade] = []
    open_lots: list[OpenLot] = []

    for symbol, symbol_orders in by_symbol.items():
        symbol_orders.sort(key=lambda order: order.filled_at or datetime.min)
        asset_class = _enum_value(symbol_orders[0].asset_class)
        buy_queue: list[dict[str, Any]] = []

        for order in symbol_orders:
            side = _enum_value(order.side)
            quantity = float(order.filled_qty)
            price = float(order.filled_avg_price)
            filled_at = order.filled_at.isoformat() if order.filled_at else ""

            if side == "buy":
                buy_queue.append({"quantity": quantity, "price": price, "time": filled_at})
                continue

            # side == "sell": consume the oldest open BUY lot(s) first.
            remaining = quantity
            while remaining > _QUANTITY_EPSILON and buy_queue:
                lot = buy_queue[0]
                matched_quantity = min(lot["quantity"], remaining)
                closed_trades.append(
                    ClosedTrade(
                        symbol=symbol,
                        asset_class=asset_class,
                        entry_price=lot["price"],
                        exit_price=price,
                        quantity=matched_quantity,
                        entry_time=lot["time"],
                        exit_time=filled_at,
                        pnl=(price - lot["price"]) * matched_quantity,
                    )
                )
                lot["quantity"] -= matched_quantity
                remaining -= matched_quantity
                if lot["quantity"] <= _QUANTITY_EPSILON:
                    buy_queue.pop(0)
            # A `remaining` left over here would mean a SELL with no
            # matching prior BUY -- unexpected for a long-only system.
            # Not modeled as a trade; the quantity is simply not
            # reflected in any closed-trade record.

        for lot in buy_queue:
            if lot["quantity"] > _QUANTITY_EPSILON:
                open_lots.append(
                    OpenLot(
                        symbol=symbol,
                        asset_class=asset_class,
                        quantity=lot["quantity"],
                        entry_price=lot["price"],
                        entry_time=lot["time"],
                    )
                )

    return closed_trades, open_lots


def compute_metrics(closed_trades: list[ClosedTrade]) -> dict[str, dict[str, Any]]:
    by_asset_class: dict[str, list[ClosedTrade]] = {}
    for trade in closed_trades:
        by_asset_class.setdefault(trade.asset_class, []).append(trade)

    metrics: dict[str, dict[str, Any]] = {}
    for asset_class, trades in by_asset_class.items():
        total = len(trades)
        wins = sum(1 for trade in trades if trade.pnl > 0)
        cumulative_pnl = sum(trade.pnl for trade in trades)
        metrics[asset_class] = {
            "total_trades": total,
            "win_rate_percent": round(wins / total * 100, 2) if total else None,
            "average_pnl": round(cumulative_pnl / total, 4) if total else None,
            "cumulative_pnl": round(cumulative_pnl, 4),
        }
    return metrics


def scan_needs_manual_review(decision_log_directory: Path) -> list[dict[str, Any]]:
    """Read-only scan of past decision logs for unresolved review flags."""
    entries: list[dict[str, Any]] = []
    if not decision_log_directory.exists():
        return entries
    for path in sorted(decision_log_directory.glob("decision_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for item in payload.get("needs_manual_review") or []:
            entries.append(
                {
                    "decision_log": path.name,
                    "as_of_bar_timestamp": payload.get("as_of_bar_timestamp"),
                    "ticker": item.get("ticker"),
                    "issue": item.get("issue"),
                }
            )
    return entries


def build_report_text(
    *,
    closed_trades: list[ClosedTrade],
    open_lots: list[OpenLot],
    metrics: dict[str, dict[str, Any]],
    review_entries: list[dict[str, Any]],
    generated_at: str,
    heartbeat_fill_count: int = 0,
) -> str:
    lines = [f"# AI-Stock-Radar Performance Report — {generated_at}", ""]

    if not closed_trades and not open_lots:
        lines.append("No filled orders found yet. Nothing to report.")
        lines.append("")
    else:
        for asset_class in sorted(metrics):
            summary = metrics[asset_class]
            lines.append(f"## {asset_class}")
            if summary["total_trades"]:
                lines.append(f"- Closed trades: {summary['total_trades']}")
                lines.append(f"- Win rate: {summary['win_rate_percent']}%")
                lines.append(f"- Average P&L/trade: ${summary['average_pnl']:,.2f}")
                lines.append(f"- Cumulative P&L: ${summary['cumulative_pnl']:,.2f}")
            else:
                lines.append("- No closed trades yet.")
            lines.append("")

        if closed_trades:
            lines.append("## Closed trade detail")
            for trade in closed_trades:
                lines.append(
                    f"- {trade.symbol}: entry {trade.entry_price:g} @ {trade.entry_time} "
                    f"-> exit {trade.exit_price:g} @ {trade.exit_time}, "
                    f"qty {trade.quantity:g}, P&L ${trade.pnl:,.2f}"
                )
            lines.append("")

        if open_lots:
            lines.append("## Still-open positions (P&L not computed)")
            for lot in open_lots:
                lines.append(
                    f"- {lot.symbol}: qty {lot.quantity:g} @ entry {lot.entry_price:g} "
                    f"({lot.entry_time})"
                )
            lines.append("")

    lines.append(
        f"Pipeline test: {heartbeat_fill_count} heartbeat round-trip fill(s) recorded "
        f"separately (client_order_id prefixed `HEARTBEAT_`), excluded from every "
        f"metric above."
    )
    lines.append("")

    lines.append("## Operational health")
    if not review_entries:
        lines.append("No needs_manual_review entries found in decision logs.")
    else:
        plural = "y" if len(review_entries) == 1 else "ies"
        lines.append(f"{len(review_entries)} needs_manual_review entr{plural} found:")
        for entry in review_entries:
            lines.append(
                f"- [{entry['as_of_bar_timestamp']}] {entry['ticker']}: {entry['issue']} "
                f"(log: {entry['decision_log']})"
            )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only Alpaca paper-account performance report."
    )
    parser.add_argument("--decision-log-directory", type=Path, default=DEFAULT_DECISION_LOG_DIRECTORY)
    parser.add_argument("--report-directory", type=Path, default=DEFAULT_REPORT_DIRECTORY)
    parser.add_argument("--no-save", action="store_true")
    arguments = parser.parse_args()

    client = order_submission.get_trading_client()
    orders = fetch_filled_orders(client)
    heartbeat_fill_count = count_heartbeat_fills(client)
    closed_trades, open_lots = reconstruct_round_trips(orders)
    metrics = compute_metrics(closed_trades)
    review_entries = scan_needs_manual_review(arguments.decision_log_directory)

    report_text = build_report_text(
        closed_trades=closed_trades,
        open_lots=open_lots,
        metrics=metrics,
        review_entries=review_entries,
        generated_at=datetime.now(UTC).isoformat(),
        heartbeat_fill_count=heartbeat_fill_count,
    )
    print(report_text)

    if not arguments.no_save:
        arguments.report_directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        report_path = arguments.report_directory / f"performance_report_{stamp}.md"
        report_path.write_text(report_text + "\n", encoding="utf-8")
        print(f"\nReport saved to: {report_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
