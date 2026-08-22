"""Tests for the write-ahead "blind" PREPARED intent mechanism that
closes scripts/run_control_arm_decision.py's own "INTENT JOURNAL
ORDERING GAP" module-docstring section: `_write_blind_prepared_intents`
(pre-run) + `_run_intent_protocol`'s correlation/close logic (post-run)
+ `_verify_write_ahead_evidence_before_broker_call` (the real runtime
gate for `_guard_against_premature_order_activation`'s conditional
path).

Deterministic, no network/broker dependency -- every function under
test here operates purely on local state (`LiveRunnerState`,
`order_intent.py`'s own on-disk journal) and a plain `decision` dict
shaped like `rdd.run_daily_decision()`'s real return value. `real
run_daily_decision()` itself is never called (would require full
market-data/engine setup out of scope for this unit-level proof) --
each test instead proves its own function's real, on-disk behavior
directly, which is what actually needs proving here.

`order_intent.INTENT_DIRECTORY` is monkeypatched to a per-test tmp_path
so every test reads/writes REAL files on a REAL filesystem, isolated
from both the real guard directory and from each other.
"""

from __future__ import annotations

import time

import pytest

import scripts.run_control_arm_decision as carm
import src.live.order_intent as order_intent
import src.live.position_state as ps
from src.backtest.portfolio_backtest_engine import _PendingOrder
from src.backtest.portfolio_backtest_models import PortfolioSignal


@pytest.fixture(autouse=True)
def _isolated_intent_directory(tmp_path, monkeypatch):
    intent_dir = tmp_path / "order_intents"
    monkeypatch.setattr(order_intent, "INTENT_DIRECTORY", intent_dir)
    return intent_dir


def _pending_buy(ticker: str, price: float = 50.0) -> _PendingOrder:
    return _PendingOrder(
        signal=PortfolioSignal(timestamp="2026-08-14", ticker=ticker, action="BUY", reference_price=price),
        submitted_portfolio_bar_index=10,
    )


def _pending_exit(ticker: str, price: float = 60.0) -> _PendingOrder:
    return _PendingOrder(
        signal=PortfolioSignal(timestamp="2026-08-14", ticker=ticker, action="EXIT", reference_price=price),
        submitted_portfolio_bar_index=10,
    )


# --- (a) Normal flow: PREPARED intent really written to disk BEFORE ---
# --- rdd.run_daily_decision() would be called.                      ---

def test_a_blind_prepared_intent_written_to_real_disk_before_run(tmp_path):
    runner_state = ps.LiveRunnerState()
    runner_state.pending_buys["MDCP"] = _pending_buy("MDCP")

    before = time.time()
    intents = carm._write_blind_prepared_intents(runner_state, "2026-08-17", pre_state_hash="deadbeef")
    after = time.time()

    assert len(intents) == 1
    intent = intents[0]
    assert intent.status == order_intent.PREPARED
    assert intent.ticker == "MDCP"
    assert intent.action_kind == "ENTRY_MARKET_BUY"
    assert intent.side == "BUY"
    assert intent.quantity is None  # real quantity not known yet -- write-ahead, not post-hoc
    assert intent.notional == 50.0  # best available "intended price" pre-run
    assert intent.pre_state_hash == "deadbeef"

    # REAL file, on a REAL filesystem -- not a mock. Existence + a fresh
    # mtime (written just now, between `before` and `after`) is the
    # actual proof this landed on disk before any decision was computed.
    on_disk_path = order_intent._intent_path(intent.intent_id)
    assert on_disk_path.is_file()
    assert before <= on_disk_path.stat().st_mtime <= after + 0.001

    reloaded = order_intent.load_intent(intent.intent_id)
    assert reloaded.status == order_intent.PREPARED
    assert reloaded.client_order_id == carm.rdd._deterministic_client_order_id("MDCP", "ENTRY_MARKET_BUY", "2026-08-17")


def test_a_blind_prepared_intent_written_for_a_pending_exit_too():
    runner_state = ps.LiveRunnerState()
    runner_state.pending_exits["QQEX"] = _pending_exit("QQEX")

    intents = carm._write_blind_prepared_intents(runner_state, "2026-08-17", pre_state_hash=None)

    assert len(intents) == 1
    intent = intents[0]
    assert intent.status == order_intent.PREPARED
    assert intent.action_kind == "SIGNAL_EXIT_MARKET_SELL"
    assert intent.side == "SELL"
    assert order_intent._intent_path(intent.intent_id).is_file()


# --- (b) Crash simulation: process interrupted right after the      ---
# --- blind write, before rdd.run_daily_decision() is ever called.   ---
# --- The PREPARED intent must survive on disk, and the NEXT run's   ---
# --- own write-ahead step must reconcile with it (idempotent reuse, ---
# --- not a second, divergent intent).                               ---

def test_b_crash_before_run_daily_decision_leaves_prepared_intent_on_disk_and_next_run_reconciles():
    runner_state = ps.LiveRunnerState()
    runner_state.pending_buys["CRSH"] = _pending_buy("CRSH", price=42.0)

    # RUN 1: the write-ahead step completes...
    first_run_intents = carm._write_blind_prepared_intents(runner_state, "2026-08-17", pre_state_hash="hash-1")
    first_intent = first_run_intents[0]
    assert first_intent.status == order_intent.PREPARED

    # ...then the process is killed (simulated: we simply never call
    # rdd.run_daily_decision() or _run_intent_protocol() -- exactly what
    # a real SIGKILL right here would also produce, since nothing further
    # would execute). The PREPARED intent's file is untouched, still on
    # disk, exactly as run 1 left it -- real filesystem check, not a mock.
    on_disk_path = order_intent._intent_path(first_intent.intent_id)
    assert on_disk_path.is_file()
    reloaded_after_crash = order_intent.load_intent(first_intent.intent_id)
    assert reloaded_after_crash.status == order_intent.PREPARED

    # RUN 2 (the next real invocation, e.g. tomorrow's cron firing):
    # CRSH is STILL genuinely pending (a real crash before
    # rdd.run_daily_decision() means it was never consumed), so a fresh
    # position_state load still has it in pending_buys. This run's own
    # _write_blind_prepared_intents call must find run 1's leftover
    # PREPARED intent and REUSE it -- same intent_id, not a new one --
    # rather than creating a second, divergent record for the same
    # candidate.
    second_run_intents = carm._write_blind_prepared_intents(runner_state, "2026-08-17", pre_state_hash="hash-2")
    second_intent = second_run_intents[0]

    assert second_intent.intent_id == first_intent.intent_id  # same on-disk record, real reconciliation
    assert second_intent.status == order_intent.PREPARED

    # Still exactly ONE intent file for this candidate anywhere on disk
    # -- proves run 2 did not fork a second, divergent PREPARED record.
    all_intents_on_disk = order_intent.list_intents()
    assert len(all_intents_on_disk) == 1
    assert all_intents_on_disk[0].intent_id == first_intent.intent_id


def test_b_stray_non_prepared_intent_is_a_fail_closed_conflict_not_silently_duplicated():
    """Contrast case: if a prior run's crash happened LATER (after
    SUBMITTING), the leftover intent is no longer safely resumable by
    this narrow pre-write step -- must fail closed, not silently create
    a second record for the same client_order_id."""
    runner_state = ps.LiveRunnerState()
    runner_state.pending_buys["STLE"] = _pending_buy("STLE")

    intents = carm._write_blind_prepared_intents(runner_state, "2026-08-17", pre_state_hash=None)
    order_intent.transition_intent(intents[0], order_intent.SUBMITTING, increment_attempt=True)

    with pytest.raises(carm.StrayPreparedIntentConflictError, match="STLE"):
        carm._write_blind_prepared_intents(runner_state, "2026-08-17", pre_state_hash=None)


# --- (c) Rejected candidate: cap was full / candidate not actually  ---
# --- attempted this run -- its blind PREPARED intent must close     ---
# --- TERMINAL (attempted_but_not_executed), never left hanging,     ---
# --- never falsely COMMITTED.                                       ---

def test_c_unmaterialized_blind_intent_closes_terminal_attempted_but_not_executed():
    runner_state = ps.LiveRunnerState()
    runner_state.pending_buys["CAPX"] = _pending_buy("CAPX")  # will NOT make it into executed_today below

    blind_intents = carm._write_blind_prepared_intents(runner_state, "2026-08-17", pre_state_hash="h")
    blind_intent = blind_intents[0]
    assert blind_intent.status == order_intent.PREPARED

    # Real decision-dict shape rdd.run_daily_decision() returns -- CAPX
    # is absent from executed_today (e.g. the position cap was already
    # full when the frozen engine reached it, so it was rejected, not
    # opened -- see decision["rejected_today"] in the real engine output).
    decision = {
        "as_of_bar_timestamp_equity": "2026-08-17",
        "executed_today": {"entries": [], "exits": []},
        "rejected_today": [{"ticker": "CAPX", "reason_code": "POSITION_CAP_REACHED", "reason": "6/6 slots full"}],
    }

    result_intents = carm._run_intent_protocol(decision, pre_state_hash="h", blind_intents=blind_intents)

    assert len(result_intents) == 1
    closed = result_intents[0]
    assert closed.intent_id == blind_intent.intent_id  # same record, not a new one
    assert closed.status == order_intent.TERMINAL
    assert "attempted_but_not_executed" in closed.last_error

    # Real, on-disk confirmation -- not just the in-memory return value.
    reloaded = order_intent.load_intent(blind_intent.intent_id)
    assert reloaded.status == order_intent.TERMINAL
    assert "attempted_but_not_executed" in reloaded.last_error

    # A TERMINAL intent has no further valid transition -- the
    # end-of-run COMMITTED loop in _execute() must skip it (guards
    # against a real InvalidTransitionError this task's own review
    # caught and fixed -- see that loop's own comment).
    with pytest.raises(order_intent.InvalidTransitionError):
        order_intent.transition_intent(reloaded, order_intent.COMMITTED)


def test_c_materialized_blind_intent_is_reused_updated_and_committable():
    """Contrast case: the SAME kind of blind intent, but THIS candidate
    DID execute this run -- must be reused (not duplicated), updated
    with the real fill quantity/price, and walked through to
    BROKER_ACKNOWLEDGED (committable), never closed TERMINAL."""
    runner_state = ps.LiveRunnerState()
    runner_state.pending_buys["FILL"] = _pending_buy("FILL", price=19.5)

    blind_intents = carm._write_blind_prepared_intents(runner_state, "2026-08-17", pre_state_hash="h")
    blind_intent = blind_intents[0]
    assert blind_intent.quantity is None  # unknown pre-run

    decision = {
        "as_of_bar_timestamp_equity": "2026-08-17",
        "executed_today": {
            "entries": [
                {"ticker": "FILL", "quantity": 12.0, "fill_price": 19.7, "stop_loss_price": 17.0},
            ],
            "exits": [],
        },
    }

    result_intents = carm._run_intent_protocol(decision, pre_state_hash="h", blind_intents=blind_intents)

    assert len(result_intents) == 1
    materialized = result_intents[0]
    assert materialized.intent_id == blind_intent.intent_id  # reused, not a fresh create_intent()
    assert materialized.status == order_intent.BROKER_ACKNOWLEDGED
    assert materialized.quantity == 12.0  # real number now filled in
    assert materialized.notional == 19.7
    assert materialized.stop_price == 17.0

    committed = order_intent.transition_intent(materialized, order_intent.COMMITTED)
    assert committed.status == order_intent.COMMITTED
    reloaded = order_intent.load_intent(blind_intent.intent_id)
    assert reloaded.status == order_intent.COMMITTED


# --- Guard's new conditional logic ---

def test_guard_still_blocks_by_default_write_ahead_owner_approval_false(monkeypatch):
    monkeypatch.setattr(carm, "_WRITE_AHEAD_JOURNAL_OWNER_APPROVED", False)
    from types import SimpleNamespace
    args = SimpleNamespace(enable_equity_orders=True, enable_crypto_orders=False)
    with pytest.raises(RuntimeError, match="BLOCKED"):
        carm._guard_against_premature_order_activation(args)


def test_guard_passes_early_check_when_owner_approved_flag_is_true(monkeypatch):
    """The early guard alone is NOT sufficient enforcement (see its own
    docstring) -- but proves its logic really is conditional now, not
    hardcoded-unconditional."""
    monkeypatch.setattr(carm, "_WRITE_AHEAD_JOURNAL_OWNER_APPROVED", True)
    from types import SimpleNamespace
    args = SimpleNamespace(enable_equity_orders=True, enable_crypto_orders=False)
    carm._guard_against_premature_order_activation(args)  # must not raise


def test_real_runtime_gate_blocks_when_write_ahead_evidence_is_missing():
    from types import SimpleNamespace
    runner_state = ps.LiveRunnerState()
    runner_state.pending_buys["NOEV"] = _pending_buy("NOEV")  # no _write_blind_prepared_intents call happened
    args = SimpleNamespace(enable_equity_orders=True, enable_crypto_orders=False)
    with pytest.raises(RuntimeError, match="NOEV"):
        carm._verify_write_ahead_evidence_before_broker_call(runner_state, "2026-08-17", args)


def test_real_runtime_gate_passes_when_write_ahead_evidence_genuinely_exists():
    from types import SimpleNamespace
    runner_state = ps.LiveRunnerState()
    runner_state.pending_buys["HASV"] = _pending_buy("HASV")
    carm._write_blind_prepared_intents(runner_state, "2026-08-17", pre_state_hash=None)
    args = SimpleNamespace(enable_equity_orders=True, enable_crypto_orders=False)
    carm._verify_write_ahead_evidence_before_broker_call(runner_state, "2026-08-17", args)  # must not raise


def test_real_runtime_gate_is_a_no_op_when_no_order_flag_is_set():
    """Today's actual production state: neither flag is ever set, so
    this check must never block a normal read-only run regardless of
    write-ahead evidence."""
    from types import SimpleNamespace
    runner_state = ps.LiveRunnerState()
    runner_state.pending_buys["NEVR"] = _pending_buy("NEVR")  # no evidence written
    args = SimpleNamespace(enable_equity_orders=False, enable_crypto_orders=False)
    carm._verify_write_ahead_evidence_before_broker_call(runner_state, "2026-08-17", args)  # must not raise
