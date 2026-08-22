"""Tests for scripts/reconcile_stale_equity_positions.py -- the CEG-class
stale-position cleanup script.

Deterministic, network-free: a minimal `FakeClient` stands in for
`TradingClient.get_order_by_id`, same fake-object discipline as
`test_broker_reconciliation.py`. Every ambiguous-case test proves the
fail-closed contract: ANY single field mismatch refuses the
correction, never guesses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import src.live.position_state as ps
from scripts.reconcile_stale_equity_positions import diagnose_and_clean_stale_position
from src.backtest.portfolio_backtest_engine import _MutablePosition


class FakeOrder:
    def __init__(self, *, id: str, symbol: str, side: str, status: str, filled_qty, filled_avg_price="270.0") -> None:
        self.id = id
        self.symbol = symbol
        self.side = side
        self.status = status
        self.filled_qty = filled_qty
        self.filled_avg_price = filled_avg_price


class FakeClient:
    def __init__(self, orders_by_id: dict[str, FakeOrder]) -> None:
        self._orders_by_id = orders_by_id

    def get_order_by_id(self, order_id: str):
        return self._orders_by_id[str(order_id)]


def _position(ticker: str, quantity: float = 71.0) -> _MutablePosition:
    return _MutablePosition(
        ticker=ticker, asset_class="EQUITY", entry_timestamp="2026-08-14T00:00:00Z",
        entry_portfolio_bar_index=1, quantity=quantity, entry_price=279.8748675, entry_fee=0.0,
        stop_loss_price=265.88, highest_close=280.0, trailing_close_percent=7.5,
        initial_risk_amount=100.0, signal_score=0.0, signal_reason="test",
    )


def _state_with_stale_position(ticker: str = "CEG", quantity: float = 71.0) -> ps.LiveRunnerState:
    state = ps.LiveRunnerState()
    state.positions[ticker] = _position(ticker, quantity)
    state.equity_stop_orders[ticker] = "stop-order-1"
    state.submitted_actions[f"{ticker}|PROTECTIVE_STOP|2026-08-14"] = {
        "order_id": "stop-order-1", "status": "new", "kind": "PROTECTIVE_STOP",
    }
    return state


@pytest.fixture
def isolated_state_path(tmp_path: Path) -> Path:
    return tmp_path / "position_state.json"


@pytest.fixture
def guard_path(tmp_path: Path) -> Path:
    return tmp_path / "high_water_mark.json"


def test_no_action_when_ticker_not_in_local_positions(isolated_state_path, guard_path):
    ps.save_position_state(ps.LiveRunnerState(), isolated_state_path, guard_path=guard_path)
    client = FakeClient({})
    result = diagnose_and_clean_stale_position("CEG", isolated_state_path, client, guard_path=guard_path)
    assert result["status"] == "NO_ACTION"


def test_ambiguous_when_no_tracked_stop_order(isolated_state_path, guard_path):
    state = ps.LiveRunnerState()
    state.positions["CEG"] = _position("CEG")
    ps.save_position_state(state, isolated_state_path, guard_path=guard_path)
    client = FakeClient({})
    result = diagnose_and_clean_stale_position("CEG", isolated_state_path, client, guard_path=guard_path)
    assert result["status"] == "AMBIGUOUS"
    assert "no tracked equity_stop_orders" in result["reason"]


def test_ambiguous_when_stop_order_not_filled(isolated_state_path, guard_path):
    state = _state_with_stale_position()
    ps.save_position_state(state, isolated_state_path, guard_path=guard_path)
    client = FakeClient({"stop-order-1": FakeOrder(id="stop-order-1", symbol="CEG", side="sell", status="new", filled_qty=None)})
    result = diagnose_and_clean_stale_position("CEG", isolated_state_path, client, guard_path=guard_path)
    assert result["status"] == "AMBIGUOUS"
    assert "not 'filled'" in result["reason"]


def test_ambiguous_when_quantity_mismatches(isolated_state_path, guard_path):
    state = _state_with_stale_position(quantity=71.0)
    ps.save_position_state(state, isolated_state_path, guard_path=guard_path)
    client = FakeClient({"stop-order-1": FakeOrder(id="stop-order-1", symbol="CEG", side="sell", status="filled", filled_qty="70.0")})
    result = diagnose_and_clean_stale_position("CEG", isolated_state_path, client, guard_path=guard_path)
    assert result["status"] == "AMBIGUOUS"
    assert "filled_qty" in result["reason"]


def test_ambiguous_when_symbol_mismatches(isolated_state_path, guard_path):
    state = _state_with_stale_position()
    ps.save_position_state(state, isolated_state_path, guard_path=guard_path)
    client = FakeClient({"stop-order-1": FakeOrder(id="stop-order-1", symbol="WRONG", side="sell", status="filled", filled_qty="71.0")})
    result = diagnose_and_clean_stale_position("CEG", isolated_state_path, client, guard_path=guard_path)
    assert result["status"] == "AMBIGUOUS"
    assert "symbol" in result["reason"]


def test_ambiguous_when_side_is_not_sell(isolated_state_path, guard_path):
    state = _state_with_stale_position()
    ps.save_position_state(state, isolated_state_path, guard_path=guard_path)
    client = FakeClient({"stop-order-1": FakeOrder(id="stop-order-1", symbol="CEG", side="buy", status="filled", filled_qty="71.0")})
    result = diagnose_and_clean_stale_position("CEG", isolated_state_path, client, guard_path=guard_path)
    assert result["status"] == "AMBIGUOUS"
    assert "side" in result["reason"]


def test_confirmed_stale_dry_run_makes_no_changes(isolated_state_path, guard_path):
    state = _state_with_stale_position()
    ps.save_position_state(state, isolated_state_path, guard_path=guard_path)
    client = FakeClient({"stop-order-1": FakeOrder(id="stop-order-1", symbol="CEG", side="sell", status="filled", filled_qty="71.0", filled_avg_price="265.58")})

    result = diagnose_and_clean_stale_position("CEG", isolated_state_path, client, guard_path=guard_path, apply=False)
    assert result["status"] == "CONFIRMED_STALE"
    assert result["broker_fill_price"] == "265.58"
    assert "DRY-RUN" in result["note"]

    reloaded = ps.load_position_state(isolated_state_path, guard_path=guard_path)
    assert "CEG" in reloaded.positions  # untouched -- dry-run made no changes
    assert "CEG" in reloaded.equity_stop_orders


def test_confirmed_stale_apply_removes_position_and_stop_and_persists(isolated_state_path, guard_path):
    state = _state_with_stale_position()
    ps.save_position_state(state, isolated_state_path, guard_path=guard_path)
    client = FakeClient({"stop-order-1": FakeOrder(id="stop-order-1", symbol="CEG", side="sell", status="filled", filled_qty="71.0")})

    result = diagnose_and_clean_stale_position("CEG", isolated_state_path, client, guard_path=guard_path, apply=True)
    assert result["status"] == "CONFIRMED_STALE"
    assert "applied" in result["note"].lower()

    reloaded = ps.load_position_state(isolated_state_path, guard_path=guard_path)
    assert "CEG" not in reloaded.positions
    assert "CEG" not in reloaded.equity_stop_orders
    stop_key = "CEG|PROTECTIVE_STOP|2026-08-14"
    assert reloaded.submitted_actions[stop_key]["status"] == "filled"


def test_apply_never_touches_other_tickers_or_cash_field():
    """cash_balance_usd is never a field on LiveRunnerState at all --
    confirmed structurally: this script cannot touch what does not
    exist to touch (see module docstring's own explanation of why)."""
    assert not hasattr(ps.LiveRunnerState(), "cash_balance_usd")


def test_apply_leaves_unrelated_positions_untouched(isolated_state_path, guard_path):
    state = _state_with_stale_position()
    state.positions["ABBV"] = _position("ABBV", quantity=10.0)
    ps.save_position_state(state, isolated_state_path, guard_path=guard_path)
    client = FakeClient({"stop-order-1": FakeOrder(id="stop-order-1", symbol="CEG", side="sell", status="filled", filled_qty="71.0")})

    diagnose_and_clean_stale_position("CEG", isolated_state_path, client, guard_path=guard_path, apply=True)

    reloaded = ps.load_position_state(isolated_state_path, guard_path=guard_path)
    assert "ABBV" in reloaded.positions
    assert reloaded.positions["ABBV"].quantity == 10.0
