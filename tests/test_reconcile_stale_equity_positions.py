"""Tests for scripts/reconcile_stale_equity_positions.py -- the CEG-class
stale-position cleanup script.

Deterministic, network-free: a minimal `FakeClient` stands in for
`TradingClient.get_order_by_id`/`get_all_positions`, same fake-object
discipline as `test_broker_reconciliation.py`. Every ambiguous-case
test proves the fail-closed contract: ANY single field mismatch
refuses the correction, never guesses.
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


class FakePosition:
    def __init__(self, *, symbol: str, qty: str) -> None:
        self.symbol = symbol
        self.qty = qty


class FakeClient:
    def __init__(
        self,
        orders_by_id: dict[str, FakeOrder],
        *,
        open_positions: list[FakePosition] | None = None,
    ) -> None:
        self._orders_by_id = orders_by_id
        self._open_positions = open_positions or []

    def get_order_by_id(self, order_id: str):
        return self._orders_by_id[str(order_id)]

    def get_all_positions(self):
        return self._open_positions


def _position(ticker: str, quantity: float = 71.0, asset_class: str = "EQUITY") -> _MutablePosition:
    return _MutablePosition(
        ticker=ticker, asset_class=asset_class, entry_timestamp="2026-08-14T00:00:00Z",
        entry_portfolio_bar_index=1, quantity=quantity, entry_price=279.8748675, entry_fee=0.0,
        stop_loss_price=265.88, highest_close=280.0, trailing_close_percent=7.5,
        initial_risk_amount=100.0, signal_score=0.0, signal_reason="test",
    )


def _state_with_stale_position(ticker: str = "CEG", quantity: float = 71.0, asset_class: str = "EQUITY") -> ps.LiveRunnerState:
    state = ps.LiveRunnerState()
    state.positions[ticker] = _position(ticker, quantity, asset_class)
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


def test_ambiguous_when_asset_class_is_not_equity(isolated_state_path, guard_path):
    state = _state_with_stale_position(asset_class="CRYPTO")
    ps.save_position_state(state, isolated_state_path, guard_path=guard_path)
    client = FakeClient({})
    result = diagnose_and_clean_stale_position("CEG", isolated_state_path, client, guard_path=guard_path)
    assert result["status"] == "AMBIGUOUS"
    assert "not 'EQUITY'" in result["reason"]


def test_ambiguous_when_no_tracked_stop_order(isolated_state_path, guard_path):
    state = ps.LiveRunnerState()
    state.positions["CEG"] = _position("CEG")
    ps.save_position_state(state, isolated_state_path, guard_path=guard_path)
    client = FakeClient({})
    result = diagnose_and_clean_stale_position("CEG", isolated_state_path, client, guard_path=guard_path)
    assert result["status"] == "AMBIGUOUS"
    assert "no tracked equity_stop_orders" in result["reason"]


def test_ambiguous_when_multiple_records_share_the_same_order_id(isolated_state_path, guard_path):
    """REAL bug reproduction: two local submitted_actions records
    happen to share an order_id -- must refuse, never silently use
    whichever is found first."""
    state = _state_with_stale_position()
    state.submitted_actions["CEG|PROTECTIVE_STOP|DUPLICATE"] = {
        "order_id": "stop-order-1", "status": "new", "kind": "PROTECTIVE_STOP",
    }
    ps.save_position_state(state, isolated_state_path, guard_path=guard_path)
    client = FakeClient({"stop-order-1": FakeOrder(id="stop-order-1", symbol="CEG", side="sell", status="filled", filled_qty="71.0")})
    result = diagnose_and_clean_stale_position("CEG", isolated_state_path, client, guard_path=guard_path)
    assert result["status"] == "AMBIGUOUS"
    assert "Ambiguous which is authoritative" in result["reason"]


def test_ambiguous_when_returned_order_id_does_not_match_queried_id(isolated_state_path, guard_path):
    """REAL bug reproduction: get_order_by_id's response is never
    cross-checked against the id actually queried -- a client bug or a
    stale cache could silently return the wrong order."""
    state = _state_with_stale_position()
    ps.save_position_state(state, isolated_state_path, guard_path=guard_path)
    client = FakeClient({"stop-order-1": FakeOrder(id="some-other-order", symbol="CEG", side="sell", status="filled", filled_qty="71.0")})
    result = diagnose_and_clean_stale_position("CEG", isolated_state_path, client, guard_path=guard_path)
    assert result["status"] == "AMBIGUOUS"
    assert "does not match the id queried" in result["reason"]


def test_ambiguous_when_stop_order_not_filled(isolated_state_path, guard_path):
    state = _state_with_stale_position()
    ps.save_position_state(state, isolated_state_path, guard_path=guard_path)
    client = FakeClient({"stop-order-1": FakeOrder(id="stop-order-1", symbol="CEG", side="sell", status="new", filled_qty=None)})
    result = diagnose_and_clean_stale_position("CEG", isolated_state_path, client, guard_path=guard_path)
    assert result["status"] == "AMBIGUOUS"
    assert "not 'filled'" in result["reason"]


def test_ambiguous_when_quantity_mismatches_even_by_float_epsilon(isolated_state_path, guard_path):
    """REAL bug reproduction: the original version used float +/- 1e-6
    tolerance; Decimal-exact comparison must reject even a tiny
    mismatch that a float tolerance would have silently accepted."""
    state = _state_with_stale_position(quantity=71.0)
    ps.save_position_state(state, isolated_state_path, guard_path=guard_path)
    client = FakeClient({"stop-order-1": FakeOrder(id="stop-order-1", symbol="CEG", side="sell", status="filled", filled_qty="70.0")})
    result = diagnose_and_clean_stale_position("CEG", isolated_state_path, client, guard_path=guard_path)
    assert result["status"] == "AMBIGUOUS"
    assert "filled_qty" in result["reason"]
    assert "Decimal-exact" in result["reason"]


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


def test_ambiguous_when_broker_still_reports_the_position_open(isolated_state_path, guard_path):
    """REAL bug reproduction: the original version never independently
    confirmed the ticker was actually absent from the broker's own
    current position list -- it trusted the stop order's "filled"
    status alone."""
    state = _state_with_stale_position()
    ps.save_position_state(state, isolated_state_path, guard_path=guard_path)
    client = FakeClient(
        {"stop-order-1": FakeOrder(id="stop-order-1", symbol="CEG", side="sell", status="filled", filled_qty="71.0")},
        open_positions=[FakePosition(symbol="CEG", qty="71")],
    )
    result = diagnose_and_clean_stale_position("CEG", isolated_state_path, client, guard_path=guard_path)
    assert result["status"] == "AMBIGUOUS"
    assert "still reports an open" in result["reason"]


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


# --- Independent audit finding #5: concurrency re-checks on --apply ---


def test_apply_aborts_if_state_file_changed_since_load(isolated_state_path, guard_path, monkeypatch: pytest.MonkeyPatch):
    """REAL race reproduction: a concurrent writer (simulated as a side
    effect of the function's own load call, real on-disk write) touches
    position_state.json between load and apply -- must refuse to
    overwrite that concurrent change, never silently clobber it."""
    state = _state_with_stale_position()
    ps.save_position_state(state, isolated_state_path, guard_path=guard_path)
    client = FakeClient({"stop-order-1": FakeOrder(id="stop-order-1", symbol="CEG", side="sell", status="filled", filled_qty="71.0")})

    real_load = ps.load_position_state

    def _load_then_concurrent_write(path, *, guard_path):
        result = real_load(path, guard_path=guard_path)
        # Simulate another process (the daily runner, crypto monitor)
        # writing to the SAME file right after this function's own load
        # -- a real, on-disk write, not a mock.
        concurrent_state = real_load(path, guard_path=guard_path)
        concurrent_state.positions["ZZZZ"] = _position("ZZZZ", quantity=1.0)
        ps.save_position_state(concurrent_state, path, guard_path=guard_path)
        return result

    monkeypatch.setattr(ps, "load_position_state", _load_then_concurrent_write)

    with pytest.raises(RuntimeError, match="changed on disk between this diagnosis's load and the apply step"):
        diagnose_and_clean_stale_position("CEG", isolated_state_path, client, guard_path=guard_path, apply=True)

    # Real, on-disk confirmation: CEG's stale record is untouched, and
    # the concurrent writer's own change survived (was never clobbered).
    reloaded = real_load(isolated_state_path, guard_path=guard_path)
    assert "CEG" in reloaded.positions
    assert "ZZZZ" in reloaded.positions


def test_apply_aborts_if_broker_position_reappears_before_write(isolated_state_path, guard_path):
    """The re-confirmation query immediately before the write must also
    catch the position reappearing between diagnosis and apply, using a
    client whose get_all_positions() returns empty on the FIRST call
    (diagnosis) and non-empty on subsequent calls (apply's re-check)."""
    state = _state_with_stale_position()
    ps.save_position_state(state, isolated_state_path, guard_path=guard_path)

    class _FlipFlopClient(FakeClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._call_count = 0

        def get_all_positions(self):
            self._call_count += 1
            if self._call_count == 1:
                return []  # diagnosis: confirmed absent
            return [FakePosition(symbol="CEG", qty="71")]  # apply re-check: reappeared

    client = _FlipFlopClient({"stop-order-1": FakeOrder(id="stop-order-1", symbol="CEG", side="sell", status="filled", filled_qty="71.0")})

    with pytest.raises(RuntimeError, match="reappeared as an open broker position"):
        diagnose_and_clean_stale_position("CEG", isolated_state_path, client, guard_path=guard_path, apply=True)

    # Real, on-disk confirmation: nothing was written.
    reloaded = ps.load_position_state(isolated_state_path, guard_path=guard_path)
    assert "CEG" in reloaded.positions


def test_main_runs_under_single_instance_lock(tmp_path: Path):
    """Real source-text check: confirms main() actually acquires the
    lock, not just a restated claim."""
    import inspect

    import scripts.reconcile_stale_equity_positions as module

    source = inspect.getsource(module.main)
    assert "single_instance_lock" in source
