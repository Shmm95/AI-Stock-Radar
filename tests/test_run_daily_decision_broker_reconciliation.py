"""Tests for run_daily_decision.py's broker-reconciliation wiring
(added 2026-08-21 after the CEG governance finding -- see the
session's own report: src/live/broker_reconciliation.py's Scenario F
existed and was fully tested, but was wired only into
scripts/run_control_arm_decision.py, never here).

The test this file exists for: broker-closed/local-still-open, against
the LIVE path -- the exact CEG reproduction, not the control-arm's own
already-covered case (tests/test_broker_reconciliation.py exercises
Scenario F in isolation; this file proves it is actually REACHED from
run_daily_decision()'s real call chain, and that it halts BEFORE any
market-data fetch, same "zero API calls beyond reconciliation itself"
discipline already established elsewhere in this codebase).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_daily_decision as runner  # noqa: E402

from src.backtest.portfolio_backtest_engine import _MutablePosition  # noqa: E402
from src.live import broker_reconciliation as br  # noqa: E402
from src.live import position_state as ps  # noqa: E402


class _FakeAccount:
    def __init__(self, suffix: str = "TEST") -> None:
        self.account_number = f"PA3HONFD{suffix}"


class _FakeOrder:
    def __init__(self, *, id: str, client_order_id: str, symbol: str, side: str, status: str) -> None:
        self.id = id
        self.client_order_id = client_order_id
        self.symbol = symbol
        self.side = side
        self.status = status
        self.qty = "1"
        self.filled_qty = "1"
        self.submitted_at = "2026-08-14T00:00:00Z"
        self.updated_at = "2026-08-18T18:38:00Z"


class _FakePosition:
    def __init__(self, *, symbol: str, qty: str) -> None:
        self.symbol = symbol
        self.qty = qty
        self.side = "long"


def _ceg_position() -> _MutablePosition:
    return _MutablePosition(
        ticker="CEG", asset_class="EQUITY", entry_timestamp="2026-08-14T00:00:00Z",
        entry_portfolio_bar_index=1, quantity=71.0, entry_price=279.8748675, entry_fee=0.0,
        stop_loss_price=265.88, highest_close=280.0, trailing_close_percent=7.5,
        initial_risk_amount=100.0, signal_score=0.0, signal_reason="test",
    )


def _write_stale_ceg_state(state_path: Path, guard_path: Path) -> None:
    """Real reproduction of the actual reported state: CEG open
    locally, its protective stop tracked, but (as this test's fake
    client will simulate) no longer genuinely open at the broker."""
    state = ps.LiveRunnerState()
    state.positions["CEG"] = _ceg_position()
    state.equity_stop_orders["CEG"] = "stop-order-1"
    state.submitted_actions["CEG|PROTECTIVE_STOP|2026-08-14"] = {
        "order_id": "stop-order-1",
        "client_order_id": "CEG-PROTECTIVE-STOP-FOR-entry-1",
        "status": "new",
        "kind": "PROTECTIVE_STOP",
    }
    ps.save_position_state(state, state_path, guard_path=guard_path)


class _BrokerClosedFakeClient:
    """Broker shows CEG's stop NOT among open orders and NO open CEG
    position -- exactly what the real broker reported (stop filled,
    position closed there) while local state still thought it was
    open. get_order_by_id is never expected to be called here since
    Scenario F's own invariant check (equity_stop_orders) fires from
    the open-orders/positions snapshot alone, before any Scenario C
    per-order resolution."""

    def get_account(self):
        return _FakeAccount()

    def get_orders(self, *args, **kwargs):
        return []  # the stop order is NOT open at the broker anymore

    def get_all_positions(self):
        return []  # CEG is NOT an open broker position anymore

    def get_order_by_id(self, order_id):
        raise AssertionError(
            f"get_order_by_id({order_id!r}) was called -- Scenario F's own "
            f"invariant check should have raised before any Scenario C "
            f"per-order resolution was ever attempted."
        )


def test_broker_closed_local_still_open_halts_before_any_market_data_fetch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """The real CEG reproduction, against the LIVE path. Real
    prepare_live_market_data is poisoned (raises if called) to prove
    the failure happens BEFORE any market-data fetch -- the exact
    'silent for three consecutive runs' failure mode this wiring exists
    to close, now impossible: this either raises loudly, or the run
    never gets far enough to silently do nothing."""
    state_path = tmp_path / "position_state.json"
    guard_path = tmp_path / "high_water_mark.json"
    _write_stale_ceg_state(state_path, guard_path)

    def _poison_prepare_market_data(tickers):
        raise AssertionError(
            "prepare_live_market_data() was called -- broker reconciliation "
            "should have raised BEFORE any market-data fetch."
        )

    monkeypatch.setattr(runner, "prepare_live_market_data", _poison_prepare_market_data)
    monkeypatch.setattr(
        runner, "get_live_cash_balance",
        lambda client=None: (_ for _ in ()).throw(AssertionError("get_live_cash_balance() called before reconciliation")),
    )
    monkeypatch.setenv("LIVE_ACCOUNT_NUMBER_SUFFIX", "TEST")  # matches _FakeAccount's account

    with pytest.raises(br.StopOrderInvariantViolationError, match="CEG"):
        runner.run_daily_decision(
            state_path=state_path,
            decision_log_directory=tmp_path / "decisions",
            guard_path=guard_path,
            trading_client=_BrokerClosedFakeClient(),
        )

    # Real, on-disk confirmation: the stale record is UNCHANGED --
    # Scenario F raises before any mutation, exactly like every other
    # reconcile() failure mode (see broker_reconciliation.py's own
    # docstring: "on a raise, runner_state has NOT been mutated").
    reloaded = ps.load_position_state(state_path, guard_path=guard_path)
    assert "CEG" in reloaded.positions
    assert "CEG" in reloaded.equity_stop_orders


def test_clean_broker_state_reconciles_without_raising(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Contrast case: the SAME stale-looking local record, but the
    broker DOES still show the stop genuinely open -- must NOT raise,
    proving this isn't a blanket "any open position blocks the run"
    regression."""
    state_path = tmp_path / "position_state.json"
    guard_path = tmp_path / "high_water_mark.json"
    _write_stale_ceg_state(state_path, guard_path)

    class _CleanFakeClient:
        def get_account(self):
            return _FakeAccount()

        def get_orders(self, *args, **kwargs):
            return [_FakeOrder(id="stop-order-1", client_order_id="CEG-PROTECTIVE-STOP-FOR-entry-1", symbol="CEG", side="sell", status="new")]

        def get_all_positions(self):
            return [_FakePosition(symbol="CEG", qty="71")]

    def _fake_prepare(tickers):
        # Reached this time (reconciliation passed) -- raise a distinct,
        # recognizable error so the test can assert reconciliation
        # itself was the thing that succeeded, without needing a full
        # realistic market-data fixture.
        raise RuntimeError("REACHED_MARKET_DATA_FETCH")

    monkeypatch.setattr(runner, "prepare_live_market_data", _fake_prepare)
    monkeypatch.setattr(runner, "get_live_cash_balance", lambda client=None: 100_000.0)
    monkeypatch.setenv("LIVE_ACCOUNT_NUMBER_SUFFIX", "TEST")  # matches _FakeAccount's account

    with pytest.raises(RuntimeError, match="REACHED_MARKET_DATA_FETCH"):
        runner.run_daily_decision(
            state_path=state_path,
            decision_log_directory=tmp_path / "decisions",
            guard_path=guard_path,
            trading_client=_CleanFakeClient(),
        )


# --- Independent audit finding #1: LIVE_ACCOUNT_NUMBER_SUFFIX is ---
# --- mandatory, not an optional skip-and-warn.                    ---


class _PoisonReconciliationClient:
    """Any real broker call here means the mandatory suffix check
    failed to stop the run BEFORE reconcile() was ever reached."""

    def get_account(self):
        raise AssertionError("get_account() was called -- the mandatory suffix check should have raised first.")

    def get_orders(self, *args, **kwargs):
        raise AssertionError("get_orders() was called -- the mandatory suffix check should have raised first.")

    def get_all_positions(self):
        raise AssertionError("get_all_positions() was called -- the mandatory suffix check should have raised first.")


@pytest.mark.parametrize(
    "raw_suffix",
    [
        None,  # unset entirely
        "",  # blank
        "   ",  # whitespace-only
        "AB",  # too short
        "ABCDE",  # too long
    ],
)
def test_missing_or_malformed_live_account_suffix_fails_closed_before_any_broker_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw_suffix: str | None
):
    state_path = tmp_path / "position_state.json"
    guard_path = tmp_path / "high_water_mark.json"
    ps.save_position_state(ps.LiveRunnerState(), state_path, guard_path=guard_path)

    if raw_suffix is None:
        monkeypatch.delenv("LIVE_ACCOUNT_NUMBER_SUFFIX", raising=False)
    else:
        monkeypatch.setenv("LIVE_ACCOUNT_NUMBER_SUFFIX", raw_suffix)

    def _poison_prepare(tickers):
        raise AssertionError("prepare_live_market_data() was called -- the suffix check should have raised first.")

    monkeypatch.setattr(runner, "prepare_live_market_data", _poison_prepare)
    monkeypatch.setattr(
        runner, "get_live_cash_balance",
        lambda client=None: (_ for _ in ()).throw(AssertionError("get_live_cash_balance() called before the suffix check")),
    )

    with pytest.raises(br.AccountIdentityMismatchError):
        runner.run_daily_decision(
            state_path=state_path,
            decision_log_directory=tmp_path / "decisions",
            guard_path=guard_path,
            trading_client=_PoisonReconciliationClient(),
        )


def test_valid_four_character_live_account_suffix_passes_the_mandatory_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Contrast case: a real, valid 4-character suffix must NOT be
    rejected -- not a blanket "always fail" regression."""
    state_path = tmp_path / "position_state.json"
    guard_path = tmp_path / "high_water_mark.json"
    ps.save_position_state(ps.LiveRunnerState(), state_path, guard_path=guard_path)
    monkeypatch.setenv("LIVE_ACCOUNT_NUMBER_SUFFIX", "  TEST  ")  # real value, incidental whitespace -- must be stripped

    class _CleanClient:
        def get_account(self):
            return _FakeAccount()

        def get_orders(self, *args, **kwargs):
            return []

        def get_all_positions(self):
            return []

    def _fake_prepare(tickers):
        raise RuntimeError("REACHED_MARKET_DATA_FETCH")

    monkeypatch.setattr(runner, "prepare_live_market_data", _fake_prepare)
    monkeypatch.setattr(runner, "get_live_cash_balance", lambda client=None: 100_000.0)

    with pytest.raises(RuntimeError, match="REACHED_MARKET_DATA_FETCH"):
        runner.run_daily_decision(
            state_path=state_path,
            decision_log_directory=tmp_path / "decisions",
            guard_path=guard_path,
            trading_client=_CleanClient(),
        )


# --- Independent audit finding #3: orders_enabled=True (real ---
# --- production combination) and Scenario C's early/durable save. ---


def test_orders_enabled_combination_reconciles_and_reaches_market_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """The real production flag combination (--enable-equity-orders)
    was never exercised against this wiring before -- must reconcile
    cleanly and still reach the market-data fetch, same as the
    dry-run case."""
    state_path = tmp_path / "position_state.json"
    guard_path = tmp_path / "high_water_mark.json"
    ps.save_position_state(ps.LiveRunnerState(), state_path, guard_path=guard_path)
    monkeypatch.setenv("LIVE_ACCOUNT_NUMBER_SUFFIX", "TEST")

    class _CleanClient:
        def get_account(self):
            return _FakeAccount()

        def get_orders(self, *args, **kwargs):
            return []

        def get_all_positions(self):
            return []

    def _fake_prepare(tickers):
        raise RuntimeError("REACHED_MARKET_DATA_FETCH")

    monkeypatch.setattr(runner, "prepare_live_market_data", _fake_prepare)
    monkeypatch.setattr(runner, "get_live_cash_balance", lambda client=None: 100_000.0)

    with pytest.raises(RuntimeError, match="REACHED_MARKET_DATA_FETCH"):
        runner.run_daily_decision(
            state_path=state_path,
            decision_log_directory=tmp_path / "decisions",
            guard_path=guard_path,
            trading_client=_CleanClient(),
            enable_equity_orders=True,
        )


def test_scenario_c_resolution_saves_state_before_market_data_fetch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A stale (non-terminal locally, no longer open at the broker)
    order that Scenario C can safely resolve must be durably saved to
    disk BEFORE the market-data fetch -- proves the early-save path
    (reconciliation_result.order_status_updates non-empty ->
    save_position_state) is really reached, not just present in code."""
    state_path = tmp_path / "position_state.json"
    guard_path = tmp_path / "high_water_mark.json"

    state = ps.LiveRunnerState()
    position = _MutablePosition(
        ticker="XYZ", asset_class="EQUITY", entry_timestamp="2026-08-20T00:00:00Z",
        entry_portfolio_bar_index=1, quantity=5.0, entry_price=50.0, entry_fee=0.0,
        stop_loss_price=45.0, highest_close=50.0, trailing_close_percent=7.5,
        initial_risk_amount=10.0, signal_score=0.0, signal_reason="test",
    )
    state.positions["XYZ"] = position
    state.submitted_actions["XYZ|ENTRY_MARKET_BUY|2026-08-20"] = {
        "order_id": "entry-order-1",
        "client_order_id": "XYZ-ENTRY-MARKET-BUY-2026-08-20",
        "status": "new",  # stale -- no longer open at the broker (see _CleanClient below)
        "kind": "ENTRY_MARKET_BUY",
    }
    ps.save_position_state(state, state_path, guard_path=guard_path)

    class _FilledOrder:
        id = "entry-order-1"
        client_order_id = "XYZ-ENTRY-MARKET-BUY-2026-08-20"
        symbol = "XYZ"
        side = "buy"
        status = "filled"
        filled_qty = "5.0"

    class _ScenarioCClient:
        def get_account(self):
            return _FakeAccount()

        def get_orders(self, *args, **kwargs):
            return []  # entry order no longer open -- Scenario C candidate

        def get_all_positions(self):
            return [_FakePosition(symbol="XYZ", qty="5")]

        def get_order_by_id(self, order_id):
            assert order_id == "entry-order-1"
            return _FilledOrder()

    def _fake_prepare(tickers):
        raise RuntimeError("REACHED_MARKET_DATA_FETCH")

    monkeypatch.setattr(runner, "prepare_live_market_data", _fake_prepare)
    monkeypatch.setattr(runner, "get_live_cash_balance", lambda client=None: 100_000.0)
    monkeypatch.setenv("LIVE_ACCOUNT_NUMBER_SUFFIX", "TEST")

    with pytest.raises(RuntimeError, match="REACHED_MARKET_DATA_FETCH"):
        runner.run_daily_decision(
            state_path=state_path,
            decision_log_directory=tmp_path / "decisions",
            guard_path=guard_path,
            trading_client=_ScenarioCClient(),
        )

    # Real, on-disk confirmation: the status correction was saved BEFORE
    # the (poisoned, never-reached) market-data fetch.
    reloaded = ps.load_position_state(state_path, guard_path=guard_path)
    assert reloaded.submitted_actions["XYZ|ENTRY_MARKET_BUY|2026-08-20"]["status"] == "filled"


# --- Independent audit finding #6 (second half): skip_broker_reconciliation ---
# --- really skips the internal reconciliation, for run_control_arm_decision.py's ---
# --- own reuse of this function.                                              ---


def test_skip_broker_reconciliation_true_never_touches_the_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A poison client (any real call raises) proves
    skip_broker_reconciliation=True genuinely skips reconcile() and the
    mandatory suffix check entirely -- no LIVE_ACCOUNT_NUMBER_SUFFIX
    needed either, since the whole block that would read it is skipped."""
    state_path = tmp_path / "position_state.json"
    guard_path = tmp_path / "high_water_mark.json"
    ps.save_position_state(ps.LiveRunnerState(), state_path, guard_path=guard_path)
    monkeypatch.delenv("LIVE_ACCOUNT_NUMBER_SUFFIX", raising=False)

    def _fake_prepare(tickers):
        raise RuntimeError("REACHED_MARKET_DATA_FETCH")

    monkeypatch.setattr(runner, "prepare_live_market_data", _fake_prepare)
    monkeypatch.setattr(runner, "get_live_cash_balance", lambda client=None: 100_000.0)

    with pytest.raises(RuntimeError, match="REACHED_MARKET_DATA_FETCH"):
        runner.run_daily_decision(
            state_path=state_path,
            decision_log_directory=tmp_path / "decisions",
            guard_path=guard_path,
            trading_client=_PoisonReconciliationClient(),
            skip_broker_reconciliation=True,
        )


def test_run_control_arm_decision_reuses_its_own_client_and_skips_duplicate_reconciliation():
    """Real source-text check (same discipline as test_strategy_config.py's
    drift-detection tests): confirms the actual call site, not a
    restated claim -- catches a future edit silently dropping either
    kwarg."""
    import inspect

    import scripts.run_control_arm_decision as carm

    source = inspect.getsource(carm)
    call_start = source.index("result = rdd.run_daily_decision(")
    call_text = source[call_start:call_start + 1200]
    assert "trading_client=reconciliation_client" in call_text
    assert "skip_broker_reconciliation=True" in call_text
