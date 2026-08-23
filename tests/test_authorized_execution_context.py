"""Tests for src/live/authorized_execution_context.py and its wiring into
scripts/run_daily_decision.py + scripts/run_control_arm_decision.py --
independent audit finding "Madde E" (2026-08-22): "Doğrudan runner
bypass'ı hâlâ açık -- scripts/run_daily_decision.py --enable-equity-orders
doğrudan çalıştırılabilir ve wrapper'daki hiçbir reconciliation, identity,
TTL veya journal gate'ini görmez."

SCOPE, DELIBERATE (owner's own explicit decision, 2026-08-22): this closes
the BARE-CALL bypass -- a call to `run_daily_decision(enable_equity_orders=True)`
that skips BOTH run_daily_decision.py's own `main()` and
run_control_arm_decision.py's `_execute()`. It does NOT reject
run_daily_decision.py's own `main()`-driven CLI usage (the real live
cron's production path) -- `main()` self-authorizes after its own
STOP/FREEZE preflight, by deliberate scope decision (see
src.live.authorized_execution_context's own module docstring).

Maps to the 9 closure criteria from the audit's own remediation plan:
  1. Direct-bare-call rejection (main()/wrapper both bypassed).
  2. Programmatic call rejection with no context.
  3. Fake/mimicked context rejection (None, object(), bool, env-var
     string, file-marker Path).
  4. Second check at the side-effect boundary (inner functions, called
     directly, still refuse).
  5. All equity paths covered (entry BUY, exit SELL, protective stop,
     stop cancel).
  6. Zero-progress proof (poison client, unchanged state hash, no
     decision log).
  7. The real, authorized wrapper path still works.
  8. Authorization is not constructed early (source-text proof in
     run_control_arm_decision.py).
  9. Regression -- covered by the full suite, not repeated here.
"""

from __future__ import annotations

import hashlib
import inspect
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_daily_decision as runner  # noqa: E402

import scripts.run_control_arm_decision as carm  # noqa: E402
from src.backtest.portfolio_backtest_engine import _MutablePosition  # noqa: E402
from src.backtest.portfolio_backtest_models import PortfolioTrade  # noqa: E402
from src.live import order_submission  # noqa: E402
from src.live.authorized_execution_context import (  # noqa: E402
    AuthorizedExecutionContext,
    authorize_order_execution,
    require_authorized_execution_context,
)
from src.live.position_state import LiveRunnerState, load_position_state, save_position_state  # noqa: E402


def equity_position(ticker: str = "AAPL", quantity: float = 10.0) -> _MutablePosition:
    return _MutablePosition(
        ticker=ticker, asset_class="EQUITY", entry_timestamp="2026-08-10",
        entry_portfolio_bar_index=5, quantity=quantity, entry_price=190.0, entry_fee=1.0,
        stop_loss_price=180.5, highest_close=190.0, trailing_close_percent=7.5,
        initial_risk_amount=95.0, signal_score=80.0, signal_reason="TEST",
    )


def equity_trade(exit_reason: str = "EXIT_SIGNAL_NEXT_OPEN") -> PortfolioTrade:
    return PortfolioTrade(
        ticker="AAPL", asset_class="EQUITY", entry_timestamp="2026-08-10", exit_timestamp="2026-08-11",
        entry_portfolio_bar_index=5, exit_portfolio_bar_index=6, quantity=10.0, entry_price=190.0,
        exit_price=195.0, entry_fee=1.0, exit_fee=1.0, total_fees=2.0, gross_pnl=50.0, net_pnl=48.0,
        return_percent=2.5, holding_period_bars=1, exit_reason=exit_reason, signal_score=80.0,
        signal_reason="TEST",
    )


class PoisonClient:
    """Raises on ANY attribute access -- proves zero broker calls (criterion #6)."""

    def __getattr__(self, name):
        def _boom(*a, **k):
            raise AssertionError(f"{name}() was called -- authorization check did not fire first")
        return _boom


# --- Criterion #3: fake/mimicked context rejection ---


@pytest.mark.parametrize(
    "fake_context",
    [None, object(), True, False, "authorized", "TRUE", Path("/tmp/authorized_marker")],
    ids=["none", "plain_object", "bool_true", "bool_false", "string", "env_var_style_string", "file_marker_path"],
)
def test_require_authorized_execution_context_rejects_every_fake(fake_context):
    with pytest.raises(RuntimeError, match="AuthorizedExecutionContext"):
        require_authorized_execution_context(fake_context, action_description="test action")


def test_require_authorized_execution_context_accepts_the_real_thing():
    context = authorize_order_execution()
    require_authorized_execution_context(context, action_description="test action")  # must not raise


def test_authorized_execution_context_cannot_be_constructed_directly():
    """The class itself refuses direct construction -- only
    authorize_order_execution()'s own private sentinel satisfies it."""
    with pytest.raises(RuntimeError):
        AuthorizedExecutionContext(object())  # wrong token -- not the module-private sentinel


# --- Criteria #2 and #6: programmatic call rejection + zero-progress proof ---


def test_run_daily_decision_rejects_missing_authorization_with_zero_progress(tmp_path: Path):
    state_path = tmp_path / "position_state.json"
    guard_path = tmp_path / "high_water_mark.json"
    save_position_state(LiveRunnerState(), state_path, guard_path=guard_path)
    state_bytes_before = state_path.read_bytes()
    decisions_dir = tmp_path / "decisions"

    with pytest.raises(RuntimeError, match="AuthorizedExecutionContext"):
        runner.run_daily_decision(
            state_path=state_path,
            decision_log_directory=decisions_dir,
            guard_path=guard_path,
            trading_client=PoisonClient(),
            enable_equity_orders=True,
            # authorization intentionally omitted -- the real bug this closes
        )

    assert state_path.read_bytes() == state_bytes_before, "state file must be byte-for-byte unchanged"
    assert not decisions_dir.exists(), "no decision log may be created"


def test_run_daily_decision_rejects_a_fake_authorization_with_zero_progress(tmp_path: Path):
    state_path = tmp_path / "position_state.json"
    guard_path = tmp_path / "high_water_mark.json"
    save_position_state(LiveRunnerState(), state_path, guard_path=guard_path)
    state_bytes_before = state_path.read_bytes()

    with pytest.raises(RuntimeError, match="AuthorizedExecutionContext"):
        runner.run_daily_decision(
            state_path=state_path,
            decision_log_directory=tmp_path / "decisions",
            guard_path=guard_path,
            trading_client=PoisonClient(),
            enable_crypto_orders=True,
            authorization="i_am_authorized",  # mimicked, not real -- must still be rejected
        )

    assert state_path.read_bytes() == state_bytes_before


def test_run_daily_decision_dry_run_never_requires_authorization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Neither order flag set -- the dry-run path -- must be completely
    unaffected by this task; `authorization=None` (the default) must
    work exactly as before."""
    state_path = tmp_path / "position_state.json"
    guard_path = tmp_path / "high_water_mark.json"
    save_position_state(LiveRunnerState(), state_path, guard_path=guard_path)

    def _fake_prepare(tickers):
        raise RuntimeError("REACHED_MARKET_DATA_FETCH")

    monkeypatch.setattr(runner, "prepare_live_market_data", _fake_prepare)
    monkeypatch.setattr(runner, "get_live_cash_balance", lambda client=None: 100_000.0)

    class _CleanClient:
        def get_account(self):
            class _Account:
                account_number = "TEST0000TEST"
            return _Account()

        def get_orders(self, *a, **k):
            return []

        def get_all_positions(self):
            return []

    monkeypatch.setenv("LIVE_ACCOUNT_NUMBER_SUFFIX", "TEST")
    with pytest.raises(RuntimeError, match="REACHED_MARKET_DATA_FETCH"):
        runner.run_daily_decision(
            state_path=state_path, decision_log_directory=tmp_path / "decisions",
            guard_path=guard_path, trading_client=_CleanClient(),
        )  # no authorization kwarg at all -- must reach past the auth check into the dry-run flow


# --- Criterion #4: inner-function-level check, called directly ---


def test_execute_equity_orders_rejects_direct_call_without_authorization(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(order_submission, "submit_equity_market_order", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("must never reach the broker")
    ))
    state = LiveRunnerState()
    with pytest.raises(RuntimeError, match="AuthorizedExecutionContext"):
        runner._execute_equity_orders(
            client=PoisonClient(), runner_state=state, equity_date="2026-08-14",
            newly_opened={"AAPL": equity_position()}, closed_trades=[],
        )  # authorization omitted


def test_resolve_or_submit_order_rejects_direct_call_without_authorization():
    def fail_if_called():
        raise AssertionError("submit() must never be reached")

    with pytest.raises(RuntimeError, match="AuthorizedExecutionContext"):
        runner._resolve_or_submit_order(
            PoisonClient(), client_order_id="X-ENTRY-1", submit=fail_if_called,
            ticker="AAPL", action_kind="ENTRY_MARKET_BUY",
        )  # authorization omitted -- the shared choke point every real order funnels through


# --- Criterion #5: every real equity path individually rejects without authorization ---


def test_entry_buy_path_rejects_without_authorization(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(order_submission, "get_order_by_client_order_id", lambda *a, **k: None)
    monkeypatch.setattr(
        order_submission, "submit_equity_market_order",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("entry BUY reached the broker")),
    )
    state = LiveRunnerState()
    with pytest.raises(RuntimeError, match="AuthorizedExecutionContext"):
        runner._execute_equity_orders(
            client=PoisonClient(), runner_state=state, equity_date="2026-08-14",
            newly_opened={"AAPL": equity_position()}, closed_trades=[],
        )


def test_signal_exit_sell_path_rejects_without_authorization(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(order_submission, "get_order_by_client_order_id", lambda *a, **k: None)
    monkeypatch.setattr(
        order_submission, "submit_equity_market_order",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("exit SELL reached the broker")),
    )
    state = LiveRunnerState()  # no resting stop -- goes straight to the sell branch
    with pytest.raises(RuntimeError, match="AuthorizedExecutionContext"):
        runner._execute_equity_orders(
            client=PoisonClient(), runner_state=state, equity_date="2026-08-15",
            newly_opened={}, closed_trades=[equity_trade(exit_reason="EXIT_SIGNAL_NEXT_OPEN")],
        )


def test_protective_stop_path_rejects_without_authorization(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(order_submission, "get_order_by_client_order_id", lambda *a, **k: None)
    monkeypatch.setattr(order_submission, "submit_equity_market_order", lambda *a, **k: object())
    monkeypatch.setattr(order_submission, "wait_for_fill_or_timeout", lambda *a, **k: "filled")
    monkeypatch.setattr(
        order_submission, "submit_equity_stop_sell",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("protective stop reached the broker")),
    )
    state = LiveRunnerState()
    with pytest.raises(RuntimeError, match="AuthorizedExecutionContext"):
        runner._execute_equity_orders(
            client=PoisonClient(), runner_state=state, equity_date="2026-08-14",
            newly_opened={"AAPL": equity_position()}, closed_trades=[],
        )


def test_stop_cancel_path_rejects_without_authorization(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        order_submission, "cancel_order_and_confirm",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("stop cancel reached the broker")),
    )
    state = LiveRunnerState(equity_stop_orders={"AAPL": "stop-1"})
    with pytest.raises(RuntimeError, match="AuthorizedExecutionContext"):
        runner._execute_equity_orders(
            client=PoisonClient(), runner_state=state, equity_date="2026-08-15",
            newly_opened={}, closed_trades=[equity_trade(exit_reason="EXIT_SIGNAL_NEXT_OPEN")],
        )


def test_reconciliation_protective_stop_path_rejects_without_authorization(monkeypatch: pytest.MonkeyPatch):
    state = LiveRunnerState(
        portfolio_bar_index=3,
        positions={"AAPL": equity_position()},
        submitted_actions={
            "AAPL|ENTRY_MARKET_BUY|3": {
                "order_id": "entry-1", "status": "accepted", "kind": "ENTRY_MARKET_BUY", "submitted_at": "x",
            }
        },
    )
    monkeypatch.setattr(order_submission, "get_order_status", lambda *a, **k: "filled")
    monkeypatch.setattr(order_submission, "get_order_by_client_order_id", lambda *a, **k: None)
    monkeypatch.setattr(
        order_submission, "submit_equity_stop_sell",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("reconciliation stop reached the broker")),
    )
    with pytest.raises(RuntimeError, match="AuthorizedExecutionContext"):
        runner._reconcile_pending_equity_orders(client=PoisonClient(), runner_state=state)


def test_crypto_entry_buy_path_rejects_without_authorization(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(runner, "_try_acquire_crypto_lock", lambda: True)
    monkeypatch.setattr(runner, "_release_crypto_lock", lambda: None)
    monkeypatch.setattr(order_submission, "get_order_by_client_order_id", lambda *a, **k: None)
    monkeypatch.setattr(
        order_submission, "submit_equity_market_order",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("crypto entry BUY reached the broker")),
    )
    crypto_position = _MutablePosition(
        ticker="BTC-USD", asset_class="CRYPTO", entry_timestamp="2026-08-10",
        entry_portfolio_bar_index=5, quantity=0.01, entry_price=65000.0, entry_fee=1.0,
        stop_loss_price=61750.0, highest_close=65000.0, trailing_close_percent=7.5,
        initial_risk_amount=32.5, signal_score=80.0, signal_reason="TEST",
    )
    state = LiveRunnerState()
    with pytest.raises(RuntimeError, match="AuthorizedExecutionContext"):
        runner._execute_crypto_orders(
            client=PoisonClient(), runner_state=state, crypto_date="2026-08-14",
            newly_opened={"BTC-USD": crypto_position}, closed_trades=[],
        )


def test_crypto_signal_exit_sell_path_rejects_without_authorization(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(runner, "_try_acquire_crypto_lock", lambda: True)
    monkeypatch.setattr(runner, "_release_crypto_lock", lambda: None)
    monkeypatch.setattr(order_submission, "get_order_by_client_order_id", lambda *a, **k: None)
    monkeypatch.setattr(
        order_submission, "submit_equity_market_order",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("crypto exit SELL reached the broker")),
    )
    crypto_trade = PortfolioTrade(
        ticker="BTC-USD", asset_class="CRYPTO", entry_timestamp="2026-08-10", exit_timestamp="2026-08-11",
        entry_portfolio_bar_index=5, exit_portfolio_bar_index=6, quantity=0.01, entry_price=65000.0,
        exit_price=66000.0, entry_fee=1.0, exit_fee=1.0, total_fees=2.0, gross_pnl=10.0, net_pnl=8.0,
        return_percent=0.1, holding_period_bars=1, exit_reason="EXIT_SIGNAL_NEXT_OPEN",
        signal_score=80.0, signal_reason="TEST",
    )
    state = LiveRunnerState()
    with pytest.raises(RuntimeError, match="AuthorizedExecutionContext"):
        runner._execute_crypto_orders(
            client=PoisonClient(), runner_state=state, crypto_date="2026-08-15",
            newly_opened={}, closed_trades=[crypto_trade],
        )


# --- Criterion #7: the real, authorized path still works ---


def test_authorized_call_reaches_the_real_submission_path(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(order_submission, "get_order_by_client_order_id", lambda *a, **k: None)
    monkeypatch.setattr(order_submission, "submit_equity_market_order", lambda *a, **k: _FakeOrder("entry-1"))
    monkeypatch.setattr(order_submission, "wait_for_fill_or_timeout", lambda *a, **k: "filled")
    monkeypatch.setattr(order_submission, "submit_equity_stop_sell", lambda *a, **k: _FakeOrder("stop-1"))

    state = LiveRunnerState()
    actions, review = runner._execute_equity_orders(
        client=PoisonClient(), runner_state=state, equity_date="2026-08-14",
        newly_opened={"AAPL": equity_position()}, closed_trades=[],
        authorization=authorize_order_execution(),
    )
    assert review == []
    assert state.submitted_actions["AAPL|ENTRY_MARKET_BUY|2026-08-14"]["order_id"] == "entry-1"
    assert state.equity_stop_orders["AAPL"] == "stop-1"


class _FakeOrder:
    def __init__(self, order_id: str):
        self.id = order_id


# --- Criterion #8: authorization is not constructed early ---


def test_run_control_arm_decision_authorizes_only_after_write_ahead_evidence_check():
    """Real source-text check (same discipline this codebase already
    uses for its other drift-detection tests): the
    authorize_order_execution() call site in _execute() must appear
    AFTER _verify_write_ahead_evidence_before_broker_call(...), never
    before -- proving the context is not constructed until every
    preflight guard (identity, reconciliation, TTL, session,
    write-ahead evidence) has already passed."""
    source = inspect.getsource(carm._execute)
    verify_index = source.index("_verify_write_ahead_evidence_before_broker_call(")
    authorize_index = source.index("authorize_order_execution()")
    run_call_index = source.index("result = rdd.run_daily_decision(")
    assert verify_index < authorize_index < run_call_index


def test_run_control_arm_decision_passes_authorization_to_run_daily_decision():
    source = inspect.getsource(carm._execute)
    call_start = source.index("result = rdd.run_daily_decision(")
    call_text = source[call_start:call_start + 1400]
    assert "authorization=authorization" in call_text


# --- Criterion #1: main() self-authorizes, after its own preflight ---


def test_main_self_authorizes_after_stop_freeze_preflight():
    """Real source-text check: main()'s own authorize_order_execution()
    call must appear after the FREEZE_FLAG_PATH check and before the
    run_daily_decision() call -- proving main() is a blessed,
    self-authorizing entry point (the live cron's real production path,
    deliberately unchanged), not that the check is missing entirely."""
    source = inspect.getsource(runner.main)
    freeze_index = source.index("freeze = FREEZE_FLAG_PATH.exists()")
    authorize_index = source.index("authorize_order_execution()")
    run_call_index = source.index("result = run_daily_decision(")
    assert freeze_index < authorize_index < run_call_index


def test_main_passes_authorization_to_run_daily_decision():
    source = inspect.getsource(runner.main)
    call_start = source.index("result = run_daily_decision(")
    call_text = source[call_start:call_start + 500]
    assert "authorization=authorization" in call_text
