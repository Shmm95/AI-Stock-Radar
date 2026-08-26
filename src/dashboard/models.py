"""Dashboard V1 -- JSON-serializable snapshot data models
(`dashboard_snapshot_v1.json`, schema at
`config/dashboard_snapshot_schema_v1.json`).

Every numeric money/quantity field is a plain `str` (decimal string
representation) -- exact precision preserved, no float rounding, no
`Decimal` in the JSON output. The P&L math itself is never redone
here; `snapshot_builder.py` converts from `read_only_portfolio_snapshot`'s
`Decimal`-based result into this shape.

INTENT-SUMMARY SANITIZATION CONTRACT V1 (owner, 2026-08-25) --
`IntentSummaryView` below is the ENTIRE allowed output shape for
anything derived from the order-intent journal. No other field from an
`order_intent.py` journal record may ever reach this snapshot, the HTTP
response, or any dashboard log line -- see `snapshot_builder.py`'s own
`build_intent_summary_view` for the enforcement and
`tests/test_dashboard_snapshot.py`'s leak test for the proof.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = 1

COLLECTION_STATUS_OK = "OK"
COLLECTION_STATUS_PARTIAL = "PARTIAL"
COLLECTION_STATUS_ERROR = "ERROR"

INTENT_SUMMARY_STATUS_OK = "OK"
INTENT_SUMMARY_STATUS_IN_FLIGHT = "IN_FLIGHT"
INTENT_SUMMARY_STATUS_ATTENTION = "ATTENTION"
INTENT_SUMMARY_STATUS_UNKNOWN = "UNKNOWN"

AGE_BUCKET_NONE = "NONE"
AGE_BUCKET_LT_5M = "LT_5M"
AGE_BUCKET_5M_TO_1H = "5M_TO_1H"
AGE_BUCKET_GT_1H = "GT_1H"

COMBINED_STRATEGY_PNL_LABEL = "Strategy fill-based P&L (realized + broker unrealized)"


@dataclass(slots=True)
class DashboardAccountView:
    cash: str
    equity: str
    portfolio_value: str
    buying_power: str
    last_equity: str
    day_pnl: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cash": self.cash, "equity": self.equity, "portfolio_value": self.portfolio_value,
            "buying_power": self.buying_power, "last_equity": self.last_equity, "day_pnl": self.day_pnl,
        }


@dataclass(slots=True)
class DashboardPositionView:
    symbol: str
    asset_class: str
    side: str
    quantity: str
    available_quantity: str
    avg_entry_price: str
    current_price: str
    cost_basis: str
    market_value: str
    unrealized_pnl_usd: str
    unrealized_pnl_percent: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "asset_class": self.asset_class, "side": self.side,
            "quantity": self.quantity, "available_quantity": self.available_quantity,
            "avg_entry_price": self.avg_entry_price, "current_price": self.current_price,
            "cost_basis": self.cost_basis, "market_value": self.market_value,
            "unrealized_pnl_usd": self.unrealized_pnl_usd, "unrealized_pnl_percent": self.unrealized_pnl_percent,
        }


@dataclass(slots=True)
class DashboardPnlView:
    realized_closed_trade_pnl_usd: str
    unrealized_pnl_usd: str
    combined_strategy_pnl_usd: str
    realized_history_complete: bool
    combined_strategy_pnl_label: str = field(default=COMBINED_STRATEGY_PNL_LABEL)

    def to_dict(self) -> dict[str, Any]:
        return {
            "realized_closed_trade_pnl_usd": self.realized_closed_trade_pnl_usd,
            "unrealized_pnl_usd": self.unrealized_pnl_usd,
            "combined_strategy_pnl_usd": self.combined_strategy_pnl_usd,
            "realized_history_complete": self.realized_history_complete,
            "combined_strategy_pnl_label": self.combined_strategy_pnl_label,
        }


@dataclass(slots=True)
class DashboardTradeView:
    symbol: str
    asset_class: str
    entry_price: str
    exit_price: str
    quantity: str
    entry_time: str
    exit_time: str
    pnl_usd: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "asset_class": self.asset_class, "entry_price": self.entry_price,
            "exit_price": self.exit_price, "quantity": self.quantity, "entry_time": self.entry_time,
            "exit_time": self.exit_time, "pnl_usd": self.pnl_usd,
        }


@dataclass(slots=True)
class IntentSummaryView:
    """THE ENTIRE allowed output shape for order-intent journal data --
    see module docstring's sanitization contract. No other field is
    ever added to this class."""

    summary_status: str  # OK | IN_FLIGHT | ATTENTION | UNKNOWN
    non_terminal_count: int
    prepared_count: int
    submitting_count: int
    broker_acknowledged_count: int
    committed_count: int
    uncertain_count: int
    stale_non_terminal_count: int
    oldest_non_terminal_age_bucket: str  # NONE | LT_5M | 5M_TO_1H | GT_1H
    parse_error_count: int
    source_available: bool
    checked_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary_status": self.summary_status,
            "non_terminal_count": self.non_terminal_count,
            "prepared_count": self.prepared_count,
            "submitting_count": self.submitting_count,
            "broker_acknowledged_count": self.broker_acknowledged_count,
            "committed_count": self.committed_count,
            "uncertain_count": self.uncertain_count,
            "stale_non_terminal_count": self.stale_non_terminal_count,
            "oldest_non_terminal_age_bucket": self.oldest_non_terminal_age_bucket,
            "parse_error_count": self.parse_error_count,
            "source_available": self.source_available,
            "checked_at_utc": self.checked_at_utc,
        }


@dataclass(slots=True)
class DashboardSystemView:
    stop_present: bool
    freeze_present: bool
    broker_account_status: str | None
    broker_restrictions: list[str]
    local_position_count: int
    broker_position_count: int
    local_broker_symbol_diff: list[str]
    last_decision_generated_at: str | None
    last_decision_stale: bool | None
    needs_manual_review_count: int | None
    replay_gate_present_today: bool
    last_processed_equity_session_date: str | None
    settled_sessions_behind: int | None
    intent_summary: IntentSummaryView

    def to_dict(self) -> dict[str, Any]:
        return {
            "stop_present": self.stop_present,
            "freeze_present": self.freeze_present,
            "broker_account_status": self.broker_account_status,
            "broker_restrictions": list(self.broker_restrictions),
            "local_position_count": self.local_position_count,
            "broker_position_count": self.broker_position_count,
            "local_broker_symbol_diff": list(self.local_broker_symbol_diff),
            "last_decision_generated_at": self.last_decision_generated_at,
            "last_decision_stale": self.last_decision_stale,
            "needs_manual_review_count": self.needs_manual_review_count,
            "replay_gate_present_today": self.replay_gate_present_today,
            "last_processed_equity_session_date": self.last_processed_equity_session_date,
            "settled_sessions_behind": self.settled_sessions_behind,
            "intent_summary": self.intent_summary.to_dict(),
        }


@dataclass(slots=True)
class DashboardError:
    section: str
    error_type: str

    def to_dict(self) -> dict[str, Any]:
        return {"section": self.section, "error_type": self.error_type}


# Static, hardcoded -- documents which read-only source fed each
# section. Never dynamic, never includes a hostname/path/credential.
SOURCE_PROVENANCE = {
    "account": "alpaca_trading_api:get_account",
    "positions": "alpaca_trading_api:get_all_positions",
    "recent_trades": "alpaca_trading_api:get_orders(closed)",
    "pnl": "read_only_portfolio_snapshot.fetch_portfolio_pnl",
    "session_cursor": "local_position_state_json",
    "replay_gate": "local_pass_gate_file",
    "decision_log": "local_decision_log_json",
    "intent_summary": "local_order_intent_journal_sanitized",
}


@dataclass(slots=True)
class DashboardSnapshot:
    schema_version: int
    generated_at_utc: str
    collection_status: str  # OK | PARTIAL | ERROR
    source_account_suffix_masked: str | None
    account: DashboardAccountView | None
    positions: list[DashboardPositionView]
    pnl: DashboardPnlView | None
    recent_trades: list[DashboardTradeView]
    system: DashboardSystemView | None
    errors: list[DashboardError] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at_utc": self.generated_at_utc,
            "collection_status": self.collection_status,
            "source_account_suffix_masked": self.source_account_suffix_masked,
            "account": self.account.to_dict() if self.account is not None else None,
            "positions": [position.to_dict() for position in self.positions],
            "pnl": self.pnl.to_dict() if self.pnl is not None else None,
            "recent_trades": [trade.to_dict() for trade in self.recent_trades],
            "system": self.system.to_dict() if self.system is not None else None,
            "source_provenance": dict(SOURCE_PROVENANCE),
            "errors": [error.to_dict() for error in self.errors],
        }
