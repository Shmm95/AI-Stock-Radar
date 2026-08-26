"""Dashboard V1 -- builds a `DashboardSnapshot` from live, read-only
sources. Runs under the privileged `ai-dashboard-producer` system user
(see `deploy/systemd/`); the unprivileged `ai-dashboard` user that
serves the HTTP app never runs this module and never sees credentials
or the `.guard` directory.

IMPORT DISCIPLINE (owner's own explicit ban list, 2026-08-25) -- this
module NEVER imports, directly or transitively through anything it
imports:
    order_intent.py, order_intent_reconciliation.py, run_daily_decision.py,
    order_submission.py, crypto_stop_monitor.py, equity_session_orchestrator.py,
    session_replay_journal.py (mutation functions), missed_session_window_handling.py.
Enforced by `tests/test_dashboard_read_only_ast.py`.

The only `src/live/` functions this module calls are the ones the
owner explicitly named as safe:
    read_only_portfolio_snapshot.fetch_portfolio_pnl
    position_state.load_position_state
    session_replay_pass_gate.has_valid_pass_gate_for_today
    equity_session_detection.fetch_expected_equity_sessions (best-effort only)
`session_replay_pass_gate.write_pass_gate` is deliberately never
imported -- this module only ever READS that gate.

THREE THINGS ARE DELIBERATELY RE-IMPLEMENTED HERE IN A FEW LINES,
RATHER THAN IMPORTED, TO KEEP THIS MODULE'S IMPORT GRAPH PROVABLY
CLEAN OF THE BANNED MODULES (never import the whole module just to
reach one harmless constant/helper inside it):
  1. `_build_trading_client()` -- the same 3 lines
     `order_submission.get_trading_client()` has, without importing
     that module (which also holds the real order-submission functions
     the AST gate exists to keep out).
  2. `_guard_directory()` -- the same env-var resolution
     `order_intent.py`'s own `INTENT_DIRECTORY` uses, without importing
     that module.
  3. `STOP_FLAG_PATH`/`FREEZE_FLAG_PATH` -- the exact same repo-relative
     paths `run_daily_decision.py` defines, without importing that
     module.

INTENT-JOURNAL SANITIZATION (owner's "Dashboard Order-Intent
Sanitization Contract V1", 2026-08-25): `build_intent_summary_view`
below reads `.guard/order_intents/*.json` as PLAIN JSON (`json.loads`
on each file, no `order_intent.OrderIntent.from_dict`, no import of
that module at all) and touches ONLY three keys per record: `status`,
`created_at`, `last_reconciled_at` -- `created_at` used ONLY to compute
an age BUCKET, never emitted as an exact timestamp. Every other key in
each record (ticker, client_order_id, quantity, ...) is never read,
never stored in a local variable beyond the raw `dict` from
`json.loads`, and never reaches the returned `IntentSummaryView` --
see that class's own docstring for the complete, exhaustive output
shape. `tests/test_dashboard_snapshot.py`'s leak test proves this by
planting a unique sentinel in every forbidden field and asserting none
of them appear anywhere in the serialized snapshot.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from alpaca.trading.client import TradingClient

from src.data.alpaca_market_data import _require_credentials
from src.dashboard.models import (
    AGE_BUCKET_5M_TO_1H,
    AGE_BUCKET_GT_1H,
    AGE_BUCKET_LT_5M,
    AGE_BUCKET_NONE,
    COLLECTION_STATUS_ERROR,
    COLLECTION_STATUS_OK,
    COLLECTION_STATUS_PARTIAL,
    INTENT_SUMMARY_STATUS_ATTENTION,
    INTENT_SUMMARY_STATUS_IN_FLIGHT,
    INTENT_SUMMARY_STATUS_OK,
    INTENT_SUMMARY_STATUS_UNKNOWN,
    SCHEMA_VERSION,
    DashboardAccountView,
    DashboardError,
    DashboardPnlView,
    DashboardPositionView,
    DashboardSnapshot,
    DashboardSystemView,
    DashboardTradeView,
    IntentSummaryView,
)
import scripts.generate_performance_report as generate_performance_report
from src.live import equity_session_detection
from src.live import position_state as ps
from src.live import read_only_portfolio_snapshot as ros
from src.live import session_replay_pass_gate

# Mirrors run_daily_decision.py's own module-level constants exactly --
# never imported from there. See module docstring, item 3.
STOP_FLAG_PATH = Path("data/live/STOP")
FREEZE_FLAG_PATH = Path("data/live/FREEZE")

DEFAULT_DECISION_LOG_DIRECTORY = Path("data/live/decisions")
# Matches scripts/send_status_update.py's own staleness threshold --
# duplicated, not imported (that script is not designed to be imported
# as a library, and this keeps this module's own import graph minimal).
DECISION_STALE_AFTER_HOURS = 30.0

_NON_TERMINAL_STATUSES = ("PREPARED", "SUBMITTING", "BROKER_ACKNOWLEDGED", "COMMITTED", "UNCERTAIN")
_KNOWN_STATUSES = frozenset(_NON_TERMINAL_STATUSES) | {"TERMINAL"}
_INTENT_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_STALE_NON_TERMINAL_AFTER_SECONDS = 3600.0  # 1 hour -- see build_intent_summary_view's own docstring


def _mask_account_number(account_number: str | None) -> str | None:
    """Last 4 characters only -- deliberately re-implemented here
    rather than importing `broker_reconciliation._mask_account_number`,
    to keep this module's import graph independent of that module too
    (not banned, but not needed either)."""
    if not account_number:
        return None
    return f"...{account_number[-4:]}"


def _build_trading_client() -> TradingClient:
    """See module docstring, item 1."""
    api_key, secret_key = _require_credentials()
    return TradingClient(api_key=api_key, secret_key=secret_key, paper=True)


def _guard_directory() -> Path:
    """See module docstring, item 2."""
    return Path(os.environ.get("AI_STOCK_RADAR_GUARD_DIR", str(Path.home() / ".ai_stock_radar_guard")))


def _enum_value(value) -> str:
    return str(value.value if hasattr(value, "value") else value)


def build_pnl_and_account_and_positions(
    client: TradingClient,
) -> tuple[DashboardAccountView | None, list[DashboardPositionView], DashboardPnlView | None, str | None, str | None]:
    """One `fetch_portfolio_pnl` call feeds account/positions/pnl all
    at once -- the P&L computation itself is entirely reused, never
    reimplemented (per the owner's own explicit instruction). Also
    makes one direct `get_account()` call for the two extra account
    fields (`buying_power`, masked account number) that
    `fetch_portfolio_pnl` does not itself expose.

    Returns (account_view, position_views, pnl_view, account_suffix_masked, error_type)."""
    try:
        pnl = ros.fetch_portfolio_pnl(client)
        account = client.get_account()
    except Exception as error:
        return None, [], None, None, type(error).__name__

    account_view = DashboardAccountView(
        cash=str(pnl.cash),
        equity=str(pnl.portfolio_value),
        portfolio_value=str(pnl.portfolio_value),
        buying_power=str(account.buying_power) if account.buying_power is not None else "0",
        last_equity=str(pnl.portfolio_value - pnl.day_pnl),
        day_pnl=str(pnl.day_pnl),
    )
    position_views = [
        DashboardPositionView(
            symbol=position.ticker,
            asset_class=position.asset_class,
            side=position.side,
            quantity=str(position.quantity),
            available_quantity=str(position.quantity),  # see note below
            avg_entry_price=str(position.avg_entry_price),
            current_price=str(position.current_price),
            cost_basis=str(position.avg_entry_price * abs(position.quantity)),
            market_value=str(position.market_value),
            unrealized_pnl_usd=str(position.broker_unrealized_pnl),
            unrealized_pnl_percent=str(position.broker_unrealized_pnl_pct),
        )
        for position in pnl.positions
    ]
    pnl_view = DashboardPnlView(
        realized_closed_trade_pnl_usd=str(pnl.realized_closed_trade_pnl),
        unrealized_pnl_usd=str(pnl.broker_unrealized_pnl_total),
        combined_strategy_pnl_usd=str(pnl.combined_strategy_pnl),
        realized_history_complete=pnl.realized_history_complete,
        combined_strategy_pnl_label=pnl.combined_strategy_pnl_label,
    )
    return account_view, position_views, pnl_view, _mask_account_number(account.account_number), None


# NOTE on `available_quantity`: `read_only_portfolio_snapshot.ReadOnlyPositionPnl`
# does not carry Alpaca's own `qty_available` field (it wasn't part of
# that module's own spec). Using the full `quantity` here is a
# deliberate, documented simplification for V1 -- revisit if a real
# case (e.g. a partially-collateralized short) makes this misleading.


def build_recent_trades(client: TradingClient, *, limit: int = 20) -> tuple[list[DashboardTradeView], str | None]:
    """Reuses the exact same `generate_performance_report.py` functions
    `read_only_portfolio_snapshot.py` already reuses -- never a second,
    divergent implementation of FIFO trade reconstruction."""
    try:
        filled_orders = generate_performance_report.fetch_filled_orders(client)
        closed_trades, _open_lots = generate_performance_report.reconstruct_round_trips(filled_orders)
    except Exception as error:
        return [], type(error).__name__

    most_recent = sorted(closed_trades, key=lambda trade: trade.exit_time, reverse=True)[:limit]
    return [
        DashboardTradeView(
            symbol=trade.symbol, asset_class=trade.asset_class, entry_price=str(trade.entry_price),
            exit_price=str(trade.exit_price), quantity=str(trade.quantity), entry_time=trade.entry_time,
            exit_time=trade.exit_time, pnl_usd=str(trade.pnl),
        )
        for trade in most_recent
    ], None


def _age_bucket(created_at: str, *, now: datetime) -> str | None:
    """`None` return means "could not parse" -- the caller treats that
    as a parse error, never as `AGE_BUCKET_NONE` (which specifically
    means "no non-terminal intents exist at all")."""
    try:
        parsed = datetime.strptime(created_at, _INTENT_TIMESTAMP_FORMAT).replace(tzinfo=UTC)
    except (TypeError, ValueError):
        return None
    age_seconds = (now - parsed).total_seconds()
    if age_seconds < 300:
        return AGE_BUCKET_LT_5M
    if age_seconds < 3600:
        return AGE_BUCKET_5M_TO_1H
    return AGE_BUCKET_GT_1H


_AGE_BUCKET_ORDER = {AGE_BUCKET_NONE: 0, AGE_BUCKET_LT_5M: 1, AGE_BUCKET_5M_TO_1H: 2, AGE_BUCKET_GT_1H: 3}


def build_intent_summary_view(*, now: datetime | None = None) -> IntentSummaryView:
    """Reads `.guard/order_intents/*.json` as PLAIN JSON -- see module
    docstring's "INTENT-JOURNAL SANITIZATION" section. Never raises:
    every failure mode (missing directory, unreadable directory,
    corrupt JSON, unknown status) is captured in the returned view's
    own `summary_status`/`parse_error_count`/`source_available` fields,
    per the owner's own fail-closed rules -- an error here must never
    look like "zero intents, everything fine."

    `stale_non_terminal_count`: a non-terminal intent's own `created_at`
    age is >= 1 hour (`AGE_BUCKET_GT_1H`) -- this codebase's Phase 1.5
    stray-intent recovery normally resolves a leftover intent within
    seconds of a run starting, so an hour-plus is a real, worth-flagging
    anomaly, not routine in-flight latency. (No sharper threshold was
    specified; this is a documented, adjustable choice, not a
    contractual number.)
    """
    now = now or datetime.now(UTC)
    directory = _guard_directory() / "order_intents"

    if not directory.exists():
        # A directory that has simply never been created yet (no
        # intent has ever been written) is order_intent.py's own
        # documented "zero intents" state (`list_intents()` returns
        # `[]`) -- legitimately OK, not a read failure.
        return IntentSummaryView(
            summary_status=INTENT_SUMMARY_STATUS_OK, non_terminal_count=0, prepared_count=0, submitting_count=0,
            broker_acknowledged_count=0, committed_count=0, uncertain_count=0, stale_non_terminal_count=0,
            oldest_non_terminal_age_bucket=AGE_BUCKET_NONE, parse_error_count=0, source_available=True,
            checked_at_utc=now.isoformat(),
        )

    try:
        record_paths = sorted(directory.glob("*.json"))
    except OSError:
        return IntentSummaryView(
            summary_status=INTENT_SUMMARY_STATUS_UNKNOWN, non_terminal_count=0, prepared_count=0,
            submitting_count=0, broker_acknowledged_count=0, committed_count=0, uncertain_count=0,
            stale_non_terminal_count=0, oldest_non_terminal_age_bucket=AGE_BUCKET_NONE, parse_error_count=1,
            source_available=False, checked_at_utc=now.isoformat(),
        )

    counts = {status: 0 for status in _NON_TERMINAL_STATUSES}
    stale_non_terminal_count = 0
    oldest_bucket = AGE_BUCKET_NONE
    parse_error_count = 0

    for path in record_paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            status = record.get("status")
            created_at = record.get("created_at")
        except (OSError, json.JSONDecodeError):
            parse_error_count += 1
            continue
        if status not in _KNOWN_STATUSES:
            parse_error_count += 1
            continue
        if status == "TERMINAL":
            continue

        counts[status] += 1
        bucket = _age_bucket(created_at, now=now) if created_at is not None else None
        if bucket is None:
            parse_error_count += 1
            continue
        if bucket == AGE_BUCKET_GT_1H:
            stale_non_terminal_count += 1
        if _AGE_BUCKET_ORDER[bucket] > _AGE_BUCKET_ORDER[oldest_bucket]:
            oldest_bucket = bucket

    non_terminal_count = sum(counts.values())
    if parse_error_count > 0:
        summary_status = INTENT_SUMMARY_STATUS_UNKNOWN
    elif counts["UNCERTAIN"] > 0:
        summary_status = INTENT_SUMMARY_STATUS_ATTENTION
    elif non_terminal_count > 0:
        summary_status = INTENT_SUMMARY_STATUS_IN_FLIGHT
    else:
        summary_status = INTENT_SUMMARY_STATUS_OK

    return IntentSummaryView(
        summary_status=summary_status,
        non_terminal_count=non_terminal_count,
        prepared_count=counts["PREPARED"],
        submitting_count=counts["SUBMITTING"],
        broker_acknowledged_count=counts["BROKER_ACKNOWLEDGED"],
        committed_count=counts["COMMITTED"],
        uncertain_count=counts["UNCERTAIN"],
        stale_non_terminal_count=stale_non_terminal_count,
        oldest_non_terminal_age_bucket=oldest_bucket,
        parse_error_count=parse_error_count,
        source_available=True,
        checked_at_utc=now.isoformat(),
    )


def _hours_since(iso_timestamp: str, *, now: datetime) -> float | None:
    try:
        then = datetime.fromisoformat(iso_timestamp)
    except (TypeError, ValueError):
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=UTC)
    return (now - then).total_seconds() / 3600.0


def _latest_decision_log(decision_log_directory: Path) -> dict | None:
    directory = Path(decision_log_directory)
    if not directory.is_dir():
        return None
    candidates = sorted(directory.glob("decision_*.json"))
    if not candidates:
        return None
    try:
        return json.loads(candidates[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _sessions_behind(client: TradingClient | None, *, last_processed_equity_session_date: str | None) -> int | None:
    if client is None or last_processed_equity_session_date is None:
        return None
    try:
        expected = equity_session_detection.fetch_expected_equity_sessions(
            client, since_date=last_processed_equity_session_date,
        )
    except Exception:
        return None
    return len(expected)


def build_system_view(
    *,
    state_path: Path,
    guard_path: Path,
    decision_log_directory: Path = DEFAULT_DECISION_LOG_DIRECTORY,
    pass_gate_directory: Path = session_replay_pass_gate.DEFAULT_PASS_GATE_DIRECTORY,
    account: object | None = None,  # the raw Alpaca TradeAccount, for status/restrictions only
    broker_position_symbols: frozenset[str] = frozenset(),
    client: TradingClient | None = None,
    now: datetime | None = None,
) -> tuple[DashboardSystemView | None, str | None]:
    now = now or datetime.now(UTC)
    try:
        state = ps.load_position_state(state_path, guard_path=guard_path)
    except Exception as error:
        return None, type(error).__name__

    local_symbols = frozenset(state.positions.keys())
    symbol_diff = sorted(local_symbols.symmetric_difference(broker_position_symbols))

    decision = _latest_decision_log(decision_log_directory)
    last_decision_generated_at = None
    last_decision_stale = None
    needs_manual_review_count = None
    if decision is not None:
        last_decision_generated_at = decision.get("generated_at")
        if last_decision_generated_at:
            hours_ago = _hours_since(str(last_decision_generated_at), now=now)
            last_decision_stale = hours_ago is not None and hours_ago > DECISION_STALE_AFTER_HOURS
        needs_manual_review_count = len(decision.get("needs_manual_review") or [])

    broker_restrictions: list[str] = []
    broker_account_status = None
    if account is not None:
        broker_account_status = _enum_value(account.status) if getattr(account, "status", None) is not None else None
        if getattr(account, "trading_blocked", False):
            broker_restrictions.append("TRADING_BLOCKED")
        if getattr(account, "transfers_blocked", False):
            broker_restrictions.append("TRANSFERS_BLOCKED")
        if getattr(account, "account_blocked", False):
            broker_restrictions.append("ACCOUNT_BLOCKED")

    replay_gate_present = session_replay_pass_gate.has_valid_pass_gate_for_today(pass_gate_directory)
    settled_sessions_behind = _sessions_behind(
        client, last_processed_equity_session_date=state.last_processed_equity_session_date,
    )

    system_view = DashboardSystemView(
        stop_present=STOP_FLAG_PATH.exists(),
        freeze_present=FREEZE_FLAG_PATH.exists(),
        broker_account_status=broker_account_status,
        broker_restrictions=broker_restrictions,
        local_position_count=len(local_symbols),
        broker_position_count=len(broker_position_symbols),
        local_broker_symbol_diff=symbol_diff,
        last_decision_generated_at=last_decision_generated_at,
        last_decision_stale=last_decision_stale,
        needs_manual_review_count=needs_manual_review_count,
        replay_gate_present_today=replay_gate_present,
        last_processed_equity_session_date=state.last_processed_equity_session_date,
        settled_sessions_behind=settled_sessions_behind,
        intent_summary=build_intent_summary_view(now=now),
    )
    return system_view, None


def build_snapshot(
    *,
    state_path: Path,
    guard_path: Path,
    decision_log_directory: Path = DEFAULT_DECISION_LOG_DIRECTORY,
    pass_gate_directory: Path = session_replay_pass_gate.DEFAULT_PASS_GATE_DIRECTORY,
) -> DashboardSnapshot:
    """The one entry point `scripts/generate_dashboard_snapshot.py`
    calls. Builds ONE `TradingClient` (see `_build_trading_client`) and
    reuses it for every broker-facing section."""
    now = datetime.now(UTC)
    errors: list[DashboardError] = []

    try:
        client = _build_trading_client()
        client_error = None
    except Exception as error:
        client = None
        client_error = type(error).__name__

    if client_error is not None:
        errors.append(DashboardError(section="broker_client", error_type=client_error))
        account_view, position_views, pnl_view, account_suffix, pnl_error = None, [], None, None, client_error
        raw_account, trades, trades_error = None, [], None
    else:
        account_view, position_views, pnl_view, account_suffix, pnl_error = build_pnl_and_account_and_positions(client)
        if pnl_error is not None:
            errors.append(DashboardError(section="pnl", error_type=pnl_error))
        try:
            raw_account = client.get_account()
        except Exception as error:
            raw_account = None
            errors.append(DashboardError(section="account_status", error_type=type(error).__name__))
        trades, trades_error = build_recent_trades(client)
        if trades_error is not None:
            errors.append(DashboardError(section="recent_trades", error_type=trades_error))

    broker_position_symbols = frozenset(position.symbol for position in position_views)
    system_view, system_error = build_system_view(
        state_path=state_path, guard_path=guard_path, decision_log_directory=decision_log_directory,
        pass_gate_directory=pass_gate_directory, account=raw_account, broker_position_symbols=broker_position_symbols,
        client=client if client_error is None else None, now=now,
    )
    if system_error is not None:
        errors.append(DashboardError(section="system", error_type=system_error))

    # collection_status: OK only if every section succeeded; PARTIAL if
    # the core (P&L/account) came back but something else didn't; the
    # CALLER (scripts/generate_dashboard_snapshot.py) is responsible for
    # deciding a catastrophic pnl_error means "do not write this
    # snapshot at all" -- this function always returns a best-effort
    # object, never raises.
    if not errors:
        collection_status = COLLECTION_STATUS_OK
    elif pnl_error is not None and client_error is not None:
        collection_status = COLLECTION_STATUS_ERROR
    else:
        collection_status = COLLECTION_STATUS_PARTIAL

    return DashboardSnapshot(
        schema_version=SCHEMA_VERSION,
        generated_at_utc=now.isoformat(),
        collection_status=collection_status,
        source_account_suffix_masked=account_suffix,
        account=account_view,
        positions=position_views,
        pnl=pnl_view,
        recent_trades=trades,
        system=system_view,
        errors=errors,
    )
