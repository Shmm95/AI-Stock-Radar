"""Tests for src/dashboard/models.py + snapshot_builder.py.

Two concerns: (1) the snapshot builder produces a correct,
schema-valid `DashboardSnapshot` from fake broker/local sources, and
(2) THE LEAK TEST -- the order-intent sanitization contract (owner's
own "Dashboard Order-Intent Sanitization Contract V1", 2026-08-25) is
actually enforced, not just documented. No real network access, no
real `.env`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import jsonschema
import pytest

from src.dashboard import models
from src.dashboard import snapshot_builder as sb
from src.live import position_state as ps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA = json.loads((PROJECT_ROOT / "config" / "dashboard_snapshot_schema_v1.json").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _isolated_guard_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_STOCK_RADAR_GUARD_DIR", str(tmp_path / "guard"))


class _FakeAccount:
    def __init__(
        self, *, cash="100000.00", equity="102500.00", last_equity="100000.00", buying_power="200000.00",
        account_number="PA3HONFDTEST", status="ACTIVE", trading_blocked=False, transfers_blocked=False,
        account_blocked=False,
    ):
        self.cash = cash
        self.equity = equity
        self.last_equity = last_equity
        self.buying_power = buying_power
        self.account_number = account_number
        self.status = status
        self.trading_blocked = trading_blocked
        self.transfers_blocked = transfers_blocked
        self.account_blocked = account_blocked


class _FakePosition:
    def __init__(
        self, *, symbol, side="long", qty, avg_entry_price, current_price, market_value,
        unrealized_pl, unrealized_plpc, asset_class="us_equity",
    ):
        self.symbol = symbol
        self.side = side
        self.qty = qty
        self.avg_entry_price = avg_entry_price
        self.current_price = current_price
        self.market_value = market_value
        self.unrealized_pl = unrealized_pl
        self.unrealized_plpc = unrealized_plpc
        self.asset_class = asset_class


class _FakeOrder:
    def __init__(
        self, *, symbol, side, status, filled_qty, filled_avg_price, filled_at,
        asset_class="us_equity", client_order_id="AAPL-ENTRY-1",
    ):
        self.symbol = symbol
        self.side = side
        self.status = status
        self.filled_qty = filled_qty
        self.filled_avg_price = filled_avg_price
        self.filled_at = filled_at
        self.asset_class = asset_class
        self.client_order_id = client_order_id


class _FakeClient:
    def __init__(self, *, account, positions, orders):
        self._account = account
        self._positions = positions
        self._orders = orders

    def get_account(self):
        return self._account

    def get_all_positions(self):
        return list(self._positions)

    def get_orders(self, *, filter=None):
        return list(self._orders)


def _at(day: int) -> datetime:
    return datetime(2026, 8, day, 20, 0, 0, tzinfo=UTC)


def test_build_snapshot_end_to_end_is_schema_valid(tmp_path, monkeypatch):
    client = _FakeClient(
        account=_FakeAccount(),
        positions=[
            _FakePosition(
                symbol="AAPL", qty="10", avg_entry_price="150.00", current_price="160.00",
                market_value="1600.00", unrealized_pl="100.00", unrealized_plpc="0.0667",
            ),
        ],
        orders=[
            _FakeOrder(symbol="MSFT", side="buy", status="filled", filled_qty="5", filled_avg_price="300.00", filled_at=_at(1)),
            _FakeOrder(symbol="MSFT", side="sell", status="filled", filled_qty="5", filled_avg_price="310.00", filled_at=_at(2)),
        ],
    )
    monkeypatch.setattr(sb, "_build_trading_client", lambda: client)

    snapshot = sb.build_snapshot(
        state_path=tmp_path / "position_state.json", guard_path=tmp_path / "guard.json",
        decision_log_directory=tmp_path / "decisions", pass_gate_directory=tmp_path / "pass_gate",
    )
    payload = snapshot.to_dict()

    jsonschema.validate(payload, SCHEMA)
    assert payload["collection_status"] == models.COLLECTION_STATUS_OK
    assert payload["account"]["portfolio_value"] == "102500.00"
    assert payload["positions"][0]["symbol"] == "AAPL"
    assert payload["pnl"]["combined_strategy_pnl_label"] == "Strategy fill-based P&L (realized + broker unrealized)"
    assert payload["recent_trades"][0]["symbol"] == "MSFT"
    assert payload["system"]["intent_summary"]["summary_status"] == "OK"
    assert payload["source_account_suffix_masked"] == "...TEST"
    assert payload["errors"] == []


def test_build_snapshot_is_partial_when_a_secondary_section_fails(tmp_path, monkeypatch):
    client = _FakeClient(account=_FakeAccount(), positions=[], orders=[])
    monkeypatch.setattr(sb, "_build_trading_client", lambda: client)

    def _broken_system_view(**kwargs):
        return None, "SimulatedSystemFailure"

    monkeypatch.setattr(sb, "build_system_view", _broken_system_view)

    snapshot = sb.build_snapshot(
        state_path=tmp_path / "position_state.json", guard_path=tmp_path / "guard.json",
    )
    assert snapshot.collection_status == models.COLLECTION_STATUS_PARTIAL
    assert snapshot.account is not None  # core still present
    assert any(e.section == "system" for e in snapshot.errors)


def test_build_snapshot_is_error_and_never_written_when_core_client_construction_fails(tmp_path, monkeypatch):
    def _broken_client():
        raise RuntimeError("ALPACA_API_KEY not set")

    monkeypatch.setattr(sb, "_build_trading_client", _broken_client)

    snapshot = sb.build_snapshot(
        state_path=tmp_path / "position_state.json", guard_path=tmp_path / "guard.json",
    )
    assert snapshot.collection_status == models.COLLECTION_STATUS_ERROR
    assert snapshot.account is None
    assert "ALPACA_API_KEY not set" not in json.dumps(snapshot.to_dict()), "must never leak the raw exception message"


def test_session_cursor_two_indicators_are_independent(tmp_path, monkeypatch):
    """Owner's own explicit requirement: replay-gate presence and
    cursor freshness must never be merged into one field."""
    client = _FakeClient(account=_FakeAccount(), positions=[], orders=[])
    monkeypatch.setattr(sb, "_build_trading_client", lambda: client)

    state_path = tmp_path / "position_state.json"
    state = ps.LiveRunnerState()
    state.last_processed_equity_session_date = "2026-08-20"
    ps.save_position_state(state, state_path, guard_path=tmp_path / "guard.json")

    snapshot = sb.build_snapshot(
        state_path=state_path, guard_path=tmp_path / "guard.json", pass_gate_directory=tmp_path / "pass_gate",
    )
    system = snapshot.system
    assert system.replay_gate_present_today is False  # no gate file written in this test
    assert system.last_processed_equity_session_date == "2026-08-20"  # present regardless of the gate


# --- IntentSummaryView status derivation -----------------------------------


def _write_intent(directory: Path, name: str, payload: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_intent_summary_ok_when_directory_never_created(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_STOCK_RADAR_GUARD_DIR", str(tmp_path / "guard-never-created"))
    view = sb.build_intent_summary_view()
    assert view.summary_status == models.INTENT_SUMMARY_STATUS_OK
    assert view.source_available is True
    assert view.non_terminal_count == 0


def test_intent_summary_in_flight_for_a_non_terminal_prepared_intent(tmp_path, monkeypatch):
    guard = tmp_path / "guard"
    monkeypatch.setenv("AI_STOCK_RADAR_GUARD_DIR", str(guard))
    now = datetime.now(UTC)
    _write_intent(guard / "order_intents", "a", {"status": "PREPARED", "created_at": now.strftime("%Y-%m-%dT%H:%M:%SZ")})

    view = sb.build_intent_summary_view(now=now)
    assert view.summary_status == models.INTENT_SUMMARY_STATUS_IN_FLIGHT
    assert view.non_terminal_count == 1
    assert view.prepared_count == 1
    assert view.oldest_non_terminal_age_bucket == models.AGE_BUCKET_LT_5M


def test_intent_summary_attention_when_any_uncertain(tmp_path, monkeypatch):
    guard = tmp_path / "guard"
    monkeypatch.setenv("AI_STOCK_RADAR_GUARD_DIR", str(guard))
    now = datetime.now(UTC)
    _write_intent(guard / "order_intents", "a", {"status": "PREPARED", "created_at": now.strftime("%Y-%m-%dT%H:%M:%SZ")})
    _write_intent(guard / "order_intents", "b", {"status": "UNCERTAIN", "created_at": now.strftime("%Y-%m-%dT%H:%M:%SZ")})

    view = sb.build_intent_summary_view(now=now)
    assert view.summary_status == models.INTENT_SUMMARY_STATUS_ATTENTION
    assert view.uncertain_count == 1


def test_intent_summary_stale_bucket_and_count_for_an_old_intent(tmp_path, monkeypatch):
    guard = tmp_path / "guard"
    monkeypatch.setenv("AI_STOCK_RADAR_GUARD_DIR", str(guard))
    now = datetime.now(UTC)
    old = now - timedelta(hours=2)
    _write_intent(guard / "order_intents", "a", {"status": "SUBMITTING", "created_at": old.strftime("%Y-%m-%dT%H:%M:%SZ")})

    view = sb.build_intent_summary_view(now=now)
    assert view.oldest_non_terminal_age_bucket == models.AGE_BUCKET_GT_1H
    assert view.stale_non_terminal_count == 1
    assert view.summary_status == models.INTENT_SUMMARY_STATUS_IN_FLIGHT  # stale, but not UNCERTAIN


def test_intent_summary_terminal_intents_are_excluded_entirely(tmp_path, monkeypatch):
    guard = tmp_path / "guard"
    monkeypatch.setenv("AI_STOCK_RADAR_GUARD_DIR", str(guard))
    now = datetime.now(UTC)
    _write_intent(guard / "order_intents", "a", {"status": "TERMINAL", "created_at": now.strftime("%Y-%m-%dT%H:%M:%SZ")})

    view = sb.build_intent_summary_view(now=now)
    assert view.summary_status == models.INTENT_SUMMARY_STATUS_OK
    assert view.non_terminal_count == 0


def test_intent_summary_unknown_on_corrupt_json_never_shown_as_zero(tmp_path, monkeypatch):
    guard = tmp_path / "guard"
    monkeypatch.setenv("AI_STOCK_RADAR_GUARD_DIR", str(guard))
    (guard / "order_intents").mkdir(parents=True)
    (guard / "order_intents" / "bad.json").write_text("{ not json", encoding="utf-8")

    view = sb.build_intent_summary_view()
    assert view.summary_status == models.INTENT_SUMMARY_STATUS_UNKNOWN
    assert view.parse_error_count > 0


def test_intent_summary_unknown_on_an_unrecognized_status_value(tmp_path, monkeypatch):
    guard = tmp_path / "guard"
    monkeypatch.setenv("AI_STOCK_RADAR_GUARD_DIR", str(guard))
    now = datetime.now(UTC)
    _write_intent(guard / "order_intents", "a", {"status": "SOME_FUTURE_STATUS", "created_at": now.strftime("%Y-%m-%dT%H:%M:%SZ")})

    view = sb.build_intent_summary_view(now=now)
    assert view.summary_status == models.INTENT_SUMMARY_STATUS_UNKNOWN
    assert view.parse_error_count == 1


def test_intent_summary_unknown_when_directory_exists_but_is_unreadable(tmp_path, monkeypatch):
    guard = tmp_path / "guard"
    intents_dir = guard / "order_intents"
    intents_dir.mkdir(parents=True)
    monkeypatch.setenv("AI_STOCK_RADAR_GUARD_DIR", str(guard))
    monkeypatch.setattr(Path, "glob", lambda self, pattern: (_ for _ in ()).throw(OSError("permission denied")))

    view = sb.build_intent_summary_view()
    assert view.summary_status == models.INTENT_SUMMARY_STATUS_UNKNOWN
    assert view.source_available is False
    assert view.parse_error_count > 0


# --- THE LEAK TEST -----------------------------------------------------


_FORBIDDEN_INTENT_FIELDS_WITH_SENTINELS = {
    "intent_id": "SENTINEL-INTENT-ID-7f3a",
    "client_order_id": "SENTINEL-CLIENT-ORDER-ID-9c1e",
    "broker_order_id": "SENTINEL-BROKER-ORDER-ID-2b6d",
    "operation_id": "SENTINEL-OPERATION-ID-4e8f",
    "target_broker_order_id": "SENTINEL-TARGET-BROKER-ORDER-ID-1a2b",
    "parent_intent_id": "SENTINEL-PARENT-INTENT-ID-3c4d",
    "account_identity": "SENTINEL-ACCOUNT-IDENTITY-5e6f",
    "ticker": "SENTINEL-TICKER-XYZQ",
    "side": "SENTINEL-SIDE-VALUE",
    "order_type": "SENTINEL-ORDER-TYPE-VALUE",
    "action_kind": "SENTINEL-ACTION-KIND-VALUE",
    "source_signal_timestamp": "SENTINEL-SOURCE-SIGNAL-TS-2020-01-01",
    "quantity": "SENTINEL-QUANTITY-99999",
    "notional": "SENTINEL-NOTIONAL-88888",
    "stop_price": "SENTINEL-STOP-PRICE-77777",
    "broker_status": "SENTINEL-BROKER-STATUS-VALUE",
    "attempt_count": "SENTINEL-ATTEMPT-COUNT-66666",
    "pre_state_hash": "SENTINEL-PRE-STATE-HASH-abcdef",
    "last_error": "SENTINEL-LAST-ERROR-MESSAGE-TEXT",
}


def test_intent_journal_sanitization_leak_test(tmp_path, monkeypatch):
    """THE required leak test (owner's own spec): plant a unique
    sentinel in every field that must NEVER reach the snapshot, then
    assert none of them appear ANYWHERE in the fully serialized output
    -- not in intent_summary, not in errors[], not anywhere."""
    guard = tmp_path / "guard"
    monkeypatch.setenv("AI_STOCK_RADAR_GUARD_DIR", str(guard))
    now = datetime.now(UTC)

    record = {
        "status": "BROKER_ACKNOWLEDGED",
        "created_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_reconciled_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        **_FORBIDDEN_INTENT_FIELDS_WITH_SENTINELS,
    }
    _write_intent(guard / "order_intents", "leaky-record", record)

    view = sb.build_intent_summary_view(now=now)
    serialized = json.dumps(view.to_dict())

    for field_name, sentinel in _FORBIDDEN_INTENT_FIELDS_WITH_SENTINELS.items():
        assert sentinel not in serialized, f"forbidden field {field_name!r}'s sentinel leaked into the snapshot"

    # Recursive key-denylist: none of the forbidden KEY NAMES may
    # appear at any depth of the output structure either.
    def _all_keys(value) -> set[str]:
        keys: set[str] = set()
        if isinstance(value, dict):
            for key, sub_value in value.items():
                keys.add(key)
                keys |= _all_keys(sub_value)
        elif isinstance(value, list):
            for item in value:
                keys |= _all_keys(item)
        return keys

    present_keys = _all_keys(view.to_dict())
    forbidden_keys = set(_FORBIDDEN_INTENT_FIELDS_WITH_SENTINELS.keys())
    assert not (present_keys & forbidden_keys), f"forbidden key(s) present: {present_keys & forbidden_keys}"

    # And the exact allowed shape -- nothing more.
    allowed_keys = {
        "summary_status", "non_terminal_count", "prepared_count", "submitting_count",
        "broker_acknowledged_count", "committed_count", "uncertain_count", "stale_non_terminal_count",
        "oldest_non_terminal_age_bucket", "parse_error_count", "source_available", "checked_at_utc",
    }
    assert set(view.to_dict().keys()) == allowed_keys


def test_intent_journal_sanitization_leak_test_via_full_snapshot(tmp_path, monkeypatch):
    """Same leak test, but through the REAL end-to-end `build_snapshot`
    entry point -- proves the sentinel doesn't leak through any OTHER
    section either (errors[], source_provenance, etc)."""
    guard = tmp_path / "guard"
    monkeypatch.setenv("AI_STOCK_RADAR_GUARD_DIR", str(guard))
    now = datetime.now(UTC)
    record = {
        "status": "PREPARED",
        "created_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        **_FORBIDDEN_INTENT_FIELDS_WITH_SENTINELS,
    }
    _write_intent(guard / "order_intents", "leaky-record", record)

    client = _FakeClient(account=_FakeAccount(), positions=[], orders=[])
    monkeypatch.setattr(sb, "_build_trading_client", lambda: client)

    snapshot = sb.build_snapshot(
        state_path=tmp_path / "position_state.json", guard_path=tmp_path / "guard.json",
        pass_gate_directory=tmp_path / "pass_gate",
    )
    serialized = json.dumps(snapshot.to_dict())
    for field_name, sentinel in _FORBIDDEN_INTENT_FIELDS_WITH_SENTINELS.items():
        assert sentinel not in serialized, f"forbidden field {field_name!r} leaked through build_snapshot"

    jsonschema.validate(snapshot.to_dict(), SCHEMA)  # additionalProperties=false at every level
