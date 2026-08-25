"""Read-only Telegram "system is alive" status ping.

Separate from every event-based notification the daily runner and crypto
monitor already send (a decision result, a triggered stop) -- this script
exists purely so the owner's phone shows a heartbeat on a fixed schedule
even on a day with zero trading activity, distinguishing "nothing happened"
from "the job never ran." Intended to run four times a day from cron; see
the module's own report for the exact lines (not added to cron here).

Fully read-only, like generate_performance_report.py: no order is placed,
modified, or cancelled, and neither run_daily_decision.py's nor
crypto_stop_monitor.py's own notification logic is touched or imported.
Every external call (Alpaca cash balance, the broker P&L snapshot below,
local position-state file, local decision log) is individually wrapped so
a single failure degrades that one line of the message to "unavailable"
rather than crashing the whole script -- the message that gets sent is the
point, so this must never itself become the silent failure it exists to
catch.

P&L snapshot (added 2026-08-24): a live, broker-truth portfolio summary
via `src.live.read_only_portfolio_snapshot.fetch_portfolio_pnl` -- see
that module's own docstring for what it computes and why. Folded into
this SAME 4x/day cron (no new schedule needed) rather than
`send_performance_report.py`'s separate once-daily cadence, since a
quiet-day heartbeat and an open-position P&L check are naturally the
same audience/timing here.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.portfolio_backtest_engine import _MutablePosition
from src.live import order_submission
from src.live import read_only_portfolio_snapshot as ros
from src.live.account_state import get_live_cash_balance
from src.live.position_state import DEFAULT_STATE_PATH, LiveRunnerState, load_position_state
from src.notify.telegram_notifier import send_telegram_message

DEFAULT_DECISION_LOG_DIRECTORY = Path("data/live/decisions")
# A daily job is expected roughly every 24h; this only flags the run as
# stale in the message, it never changes control flow or exit behavior.
STALE_AFTER_HOURS = 30.0


def _notify_safe(text: str) -> bool:
    """Same fail-safe contract as run_daily_decision.py's own _notify_safe:
    never raises, and reports whether delivery could be confirmed so a
    silent notifier failure is never indistinguishable from "nothing to
    report." Not imported from run_daily_decision.py -- each live script
    in this codebase defines its own local copy (see
    crypto_stop_monitor.py's own _notify_safe) rather than sharing one.
    """
    try:
        return bool(send_telegram_message(text))
    except Exception:
        return False


def _safe_cash_balance() -> tuple[float | None, str | None]:
    try:
        return get_live_cash_balance(), None
    except Exception as error:
        return None, type(error).__name__


def _safe_position_state(state_path: Path) -> tuple[LiveRunnerState | None, str | None]:
    try:
        return load_position_state(state_path), None
    except Exception as error:
        return None, type(error).__name__


def _safe_portfolio_pnl() -> tuple[ros.ReadOnlyPortfolioPnl | None, str | None]:
    """One TradingClient, built here and passed straight through to
    `fetch_portfolio_pnl` -- which itself uses that SAME client for its
    own `get_account`/`get_all_positions`/`get_orders` reads (see that
    module's own docstring). A separate connection from whatever
    `get_live_cash_balance()` above builds for the pre-existing cash
    line -- that line is left untouched (own independent source, own
    existing tests) rather than folded into this new one, matching this
    function's own "one failure degrades only this source" contract."""
    try:
        client = order_submission.get_trading_client()
        return ros.fetch_portfolio_pnl(client), None
    except Exception as error:
        return None, type(error).__name__


def _format_pnl_lines(pnl: ros.ReadOnlyPortfolioPnl) -> list[str]:
    lines = [
        "P&L snapshot (broker, live):",
        f"  Portfolio value: ${pnl.portfolio_value:,.2f}",
        f"  Cash: ${pnl.cash:,.2f}",
        f"  Day P&L: {pnl.day_pnl:+,.2f}",
    ]
    if pnl.positions:
        for position in sorted(pnl.positions, key=lambda item: item.ticker):
            lines.append(
                f"  {position.ticker} ({position.side}): qty {position.quantity:g} "
                f"entry {position.avg_entry_price:,.2f} current {position.current_price:,.2f} "
                f"unrealized {position.broker_unrealized_pnl:+,.2f} "
                f"({position.broker_unrealized_pnl_pct * 100:+.2f}%)"
            )
    else:
        lines.append("  No broker positions open.")
    realized_note = (
        "" if pnl.realized_history_complete
        else " (order history capped at the lookback limit -- may be incomplete)"
    )
    lines.append(f"  Realized closed-trade P&L: {pnl.realized_closed_trade_pnl:+,.2f}{realized_note}")
    lines.append(f"  Broker unrealized P&L: {pnl.broker_unrealized_pnl_total:+,.2f}")
    lines.append(f"  {pnl.combined_strategy_pnl_label}: {pnl.combined_strategy_pnl:+,.2f}")
    return lines


def _latest_decision_log(decision_log_directory: Path) -> Path | None:
    directory = Path(decision_log_directory)
    if not directory.is_dir():
        return None
    candidates = sorted(directory.glob("decision_*.json"))
    return candidates[-1] if candidates else None


def _read_latest_decision(
    decision_log_directory: Path,
) -> tuple[dict | None, Path | None, str | None]:
    """Returns (payload, path, error). A decision log is only ever written
    on a SUCCESSFUL run (run_daily_decision.py writes it before returning;
    an exception there produces no file at all) -- so finding one at all
    already means that run succeeded; a crash is reported separately by
    that script's own failure notification, not by this one.
    """
    path = _latest_decision_log(decision_log_directory)
    if path is None:
        return None, None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload, path, None
    except Exception as error:
        return None, path, type(error).__name__


def _format_positions(positions: dict[str, _MutablePosition]) -> str:
    if not positions:
        return "  Açık pozisyon yok."
    lines = []
    for ticker in sorted(positions):
        position = positions[ticker]
        lines.append(
            f"  {ticker} ({position.asset_class}): qty {position.quantity:g} "
            f"@ entry {position.entry_price:g} ({position.entry_timestamp})"
        )
    return "\n".join(lines)


def _hours_since(iso_timestamp: str, *, now: datetime) -> float | None:
    try:
        then = datetime.fromisoformat(iso_timestamp)
    except (TypeError, ValueError):
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=UTC)
    return (now - then).total_seconds() / 3600.0


def build_status_text(
    *,
    state_path: Path = DEFAULT_STATE_PATH,
    decision_log_directory: Path = DEFAULT_DECISION_LOG_DIRECTORY,
    now: datetime | None = None,
) -> str:
    """Build the full status message. Never raises -- every external call
    is individually guarded above; this function only formats what came
    back (or the fact that it did not)."""
    now = now or datetime.now(UTC)
    lines = [f"AI-Stock-Radar status check -- {now.isoformat()}"]

    cash, cash_error = _safe_cash_balance()
    if cash_error is None:
        lines.append(f"Cash balance: ${cash:,.2f}")
    else:
        lines.append(f"Cash balance: unavailable ({cash_error})")

    pnl, pnl_error = _safe_portfolio_pnl()
    if pnl_error is None:
        lines.extend(_format_pnl_lines(pnl))
    else:
        lines.append(f"P&L snapshot: unavailable ({pnl_error})")

    runner_state, state_error = _safe_position_state(state_path)
    if state_error is None:
        lines.append("Open positions:")
        lines.append(_format_positions(runner_state.positions))
    else:
        lines.append(f"Open positions: unavailable ({state_error})")

    decision, decision_path, decision_error = _read_latest_decision(decision_log_directory)
    if decision_path is None:
        lines.append("Last daily decision run: no decision log found yet.")
    elif decision_error is not None:
        lines.append(
            f"Last daily decision run: log unreadable ({decision_error}) -- {decision_path.name}"
        )
    else:
        generated_at = decision.get("generated_at", "unknown time")
        hours_ago = _hours_since(str(generated_at), now=now)
        staleness = ""
        if hours_ago is not None and hours_ago > STALE_AFTER_HOURS:
            staleness = f" -- STALE, {hours_ago:.1f}h ago, verify the job is still running"
        lines.append(f"Last daily decision run: {generated_at} (succeeded){staleness}")
        review_count = len(decision.get("needs_manual_review") or [])
        lines.append(f"needs_manual_review pending: {review_count}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a read-only, fixed-schedule Telegram status-update ping."
    )
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--decision-log-directory", type=Path, default=DEFAULT_DECISION_LOG_DIRECTORY)
    arguments = parser.parse_args()

    try:
        text = build_status_text(
            state_path=arguments.state_path,
            decision_log_directory=arguments.decision_log_directory,
        )
    except Exception as error:
        # build_status_text guards every individual call already; this is
        # defense in depth only, matching the rest of this codebase's
        # live scripts -- never let a bug here prevent SOME message
        # (even a short "status unavailable" one) from going out.
        text = (
            f"AI-Stock-Radar status check FAILED at {datetime.now(UTC).isoformat()}\n"
            f"{type(error).__name__}: durum sorgulanamadı."
        )

    print(text)
    notified = _notify_safe(text)
    if not notified:
        print(
            "WARNING: status-update Telegram notification could not be confirmed sent.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
