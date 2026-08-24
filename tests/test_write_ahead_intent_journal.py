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
    # REAL BUG FOUND AND FIXED (2026-08-22, independent audit finding
    # #2): must use client_order_id_for_action -- the same canonical
    # mapping the real broker-order call sites in run_daily_decision.py
    # use -- not the raw _deterministic_client_order_id with a
    # hand-picked literal (that was the bug: "ENTRY_MARKET_BUY" here
    # vs. "ENTRY-MARKET-BUY" there produced two different ids for the
    # same real ticker+action+date).
    assert reloaded.client_order_id == carm.rdd.client_order_id_for_action("MDCP", "ENTRY_MARKET_BUY", "2026-08-17")


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


def test_b2_a_leftover_terminal_intent_is_ignored_not_raised_on(tmp_path):
    """Reboot-drill finding #1 (2026-08-24), the real deadlock this
    closes: a TERMINAL record for the SAME client_order_id (e.g. Phase
    1.5's own stray-recovery closed a confirmed-404 PREPARED intent
    ABANDONED_NO_SUBMISSION earlier in the SAME run) must be ignored --
    never raised on, never reused/mutated. This step writes a genuinely
    fresh PREPARED intent (its own new intent_id) for this run's own
    live candidate. Confirmed via a real, end-to-end reboot-drill
    reproduction before this fix (the deadlock this closes), not
    assumed."""
    runner_state = ps.LiveRunnerState()
    runner_state.pending_buys["TERM"] = _pending_buy("TERM")

    first_run_intents = carm._write_blind_prepared_intents(runner_state, "2026-08-17", pre_state_hash=None)
    old_intent = order_intent.transition_intent(
        first_run_intents[0], order_intent.TERMINAL,
        last_error="ABANDONED_NO_SUBMISSION: simulated prior-run abandonment",
    )

    second_run_intents = carm._write_blind_prepared_intents(runner_state, "2026-08-17", pre_state_hash=None)

    assert second_run_intents[0].status == order_intent.PREPARED
    assert second_run_intents[0].intent_id != old_intent.intent_id  # a genuinely NEW record, old one untouched
    reloaded_old = order_intent.load_intent(old_intent.intent_id)
    assert reloaded_old.status == order_intent.TERMINAL  # the old TERMINAL record is left exactly as it was
    assert reloaded_old.last_error == "ABANDONED_NO_SUBMISSION: simulated prior-run abandonment"


# --- _stray_intent_matches_live_candidate (reboot-drill finding #1, 2026-08-24) ---


def test_matches_live_candidate_true_for_an_exact_match():
    runner_state = ps.LiveRunnerState()
    runner_state.pending_buys["MTCH"] = _pending_buy("MTCH")
    intent = carm._write_blind_prepared_intents(runner_state, "2026-08-17", pre_state_hash=None)[0]
    assert carm._stray_intent_matches_live_candidate(intent, runner_state) is True


def test_matches_live_candidate_false_when_candidate_no_longer_pending():
    runner_state = ps.LiveRunnerState()
    runner_state.pending_buys["GONE"] = _pending_buy("GONE")
    intent = carm._write_blind_prepared_intents(runner_state, "2026-08-17", pre_state_hash=None)[0]
    del runner_state.pending_buys["GONE"]  # consumed/expired by something else this run
    assert carm._stray_intent_matches_live_candidate(intent, runner_state) is False


def test_matches_live_candidate_false_for_a_different_account_identity():
    runner_state = ps.LiveRunnerState()
    runner_state.pending_buys["ACCT"] = _pending_buy("ACCT")
    intent = carm._write_blind_prepared_intents(runner_state, "2026-08-17", pre_state_hash=None)[0]
    intent.account_identity = "some_other_account"
    assert carm._stray_intent_matches_live_candidate(intent, runner_state) is False


def test_matches_live_candidate_false_for_a_mismatched_side():
    runner_state = ps.LiveRunnerState()
    runner_state.pending_buys["SIDE"] = _pending_buy("SIDE")
    intent = carm._write_blind_prepared_intents(runner_state, "2026-08-17", pre_state_hash=None)[0]
    intent.side = "SELL"  # should be BUY for an ENTRY_MARKET_BUY candidate
    assert carm._stray_intent_matches_live_candidate(intent, runner_state) is False


def test_matches_live_candidate_false_for_a_different_session_date():
    """A stale intent whose own client_order_id was computed for a
    DIFFERENT session must never spuriously match today's candidate --
    the recomputed client_order_id (using the intent's own stored
    source_signal_timestamp) will legitimately differ from what a fresh
    intent for TODAY's session would carry."""
    runner_state = ps.LiveRunnerState()
    runner_state.pending_buys["OLDS"] = _pending_buy("OLDS")
    intent = carm._write_blind_prepared_intents(runner_state, "2026-08-01", pre_state_hash=None)[0]
    # Simulate a stale intent from an EARLIER session's own client_order_id
    # ending up compared against today's ("2026-08-17") differently-computed id.
    intent.client_order_id = "OLDS-ENTRY-MARKET-BUY-2026-07-15"
    assert carm._stray_intent_matches_live_candidate(intent, runner_state) is False


def test_matches_live_candidate_false_for_protective_stop_and_cancel_action_kinds():
    """No "pending candidate" concept applies to reactively-decided
    action kinds -- always False, never even reaches a pending_buys/
    pending_exits lookup."""
    from types import SimpleNamespace

    runner_state = ps.LiveRunnerState()
    stop_intent = SimpleNamespace(action_kind="PROTECTIVE_STOP")
    cancel_intent = SimpleNamespace(action_kind="CANCEL_PROTECTIVE_STOP")
    assert carm._stray_intent_matches_live_candidate(stop_intent, runner_state) is False
    assert carm._stray_intent_matches_live_candidate(cancel_intent, runner_state) is False


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


# --- Independent audit finding #3 (2026-08-22), approach (B):
# _order_intent_hook_for_run + _run_intent_protocol's idempotent handling
# of an intent the hook already advanced DURING run_daily_decision() ---


class _FakeBrokerOrder:
    def __init__(self, order_id: str, status: str = "filled") -> None:
        self.id = order_id
        self.status = status


def test_order_intent_hook_for_run_advances_the_matching_blind_intent():
    """The hook must transition the SAME on-disk intent
    _write_blind_prepared_intents already wrote -- not create a second,
    divergent one -- and record the REAL broker order id/status, not a
    placeholder."""
    runner_state = ps.LiveRunnerState()
    runner_state.pending_buys["CEG"] = _pending_buy("CEG", price=75.0)
    blind_intents = carm._write_blind_prepared_intents(runner_state, "2026-08-18", pre_state_hash="h")
    blind_intent = blind_intents[0]
    client_order_id = blind_intent.client_order_id

    hook = carm._order_intent_hook_for_run(blind_intents)
    hook("SUBMITTING", ticker="CEG", action_kind="ENTRY_MARKET_BUY", client_order_id=client_order_id)

    mid_flight = order_intent.load_intent(blind_intent.intent_id)
    assert mid_flight.status == order_intent.SUBMITTING  # durable evidence exists even if a crash happens right here

    hook(
        "BROKER_ACKNOWLEDGED", ticker="CEG", action_kind="ENTRY_MARKET_BUY",
        client_order_id=client_order_id, order=_FakeBrokerOrder("real-broker-order-123", status="filled"),
    )
    acknowledged = order_intent.load_intent(blind_intent.intent_id)
    assert acknowledged.status == order_intent.BROKER_ACKNOWLEDGED
    assert acknowledged.broker_order_id == "real-broker-order-123"
    assert acknowledged.broker_status == "filled"


def test_order_intent_hook_for_run_is_a_no_op_for_an_unjournaled_action_kind():
    """PROTECTIVE_STOP is never pre-journaled (only pending_buys/pending_exits
    are) -- a hook call for it must be a silent no-op, never an error."""
    hook = carm._order_intent_hook_for_run([])
    hook("SUBMITTING", ticker="CEG", action_kind="PROTECTIVE_STOP", client_order_id="CEG-PROTECTIVE-STOP-FOR-xyz")
    hook(
        "BROKER_ACKNOWLEDGED", ticker="CEG", action_kind="PROTECTIVE_STOP",
        client_order_id="CEG-PROTECTIVE-STOP-FOR-xyz", order=_FakeBrokerOrder("stop-1"),
    )  # must not raise


def test_run_intent_protocol_does_not_re_transition_an_intent_the_hook_already_advanced():
    """REAL BUG THIS WOULD HAVE BEEN WITHOUT THE FIX: if
    _run_intent_protocol blindly re-ran its own PREPARED -> SUBMITTING ->
    BROKER_ACKNOWLEDGED transitions on an intent the real hook already
    advanced to BROKER_ACKNOWLEDGED during run_daily_decision(), it would
    raise InvalidTransitionError (BROKER_ACKNOWLEDGED has no self-
    transition) and the whole run would crash AFTER real orders had
    already been placed -- the worst possible time to fail. Must instead
    recognize the already-advanced intent and reuse it as-is."""
    runner_state = ps.LiveRunnerState()
    runner_state.pending_buys["CEG"] = _pending_buy("CEG", price=75.0)
    blind_intents = carm._write_blind_prepared_intents(runner_state, "2026-08-18", pre_state_hash="h")
    blind_intent = blind_intents[0]

    hook = carm._order_intent_hook_for_run(blind_intents)
    hook("SUBMITTING", ticker="CEG", action_kind="ENTRY_MARKET_BUY", client_order_id=blind_intent.client_order_id)
    hook(
        "BROKER_ACKNOWLEDGED", ticker="CEG", action_kind="ENTRY_MARKET_BUY",
        client_order_id=blind_intent.client_order_id,
        order=_FakeBrokerOrder("real-broker-order-123", status="filled"),
    )

    decision = {
        "as_of_bar_timestamp_equity": "2026-08-18",
        "executed_today": {
            "entries": [
                {"ticker": "CEG", "quantity": 5.0, "fill_price": 75.2, "stop_loss_price": 70.0},
            ],
            "exits": [],
        },
    }
    result_intents = carm._run_intent_protocol(decision, pre_state_hash="h", blind_intents=blind_intents)

    assert len(result_intents) == 1
    result = result_intents[0]
    assert result.intent_id == blind_intent.intent_id
    assert result.status == order_intent.BROKER_ACKNOWLEDGED
    assert result.broker_order_id == "real-broker-order-123"  # the REAL hook-recorded value, not overwritten


def test_run_intent_protocol_routes_unmaterialized_submitting_intent_to_uncertain_not_terminal():
    """independent-audit-round-2 finding #4 (2026-08-23): the abandoned-
    guess loop used to transition straight to TERMINAL without checking
    the blind intent's current status. Before this fix, a blind
    candidate the hook advanced to SUBMITTING (a real broker call was at
    least attempted) but that still does not appear in this run's own
    executed_today (e.g. a crash between SUBMITTING and
    BROKER_ACKNOWLEDGED, or a rejection) would hit
    `transition_intent(blind, TERMINAL)` -- not a valid transition from
    SUBMITTING (`_VALID_TRANSITIONS`) -- and raise InvalidTransitionError
    uncaught, crashing the whole run. Must route to UNCERTAIN instead."""
    runner_state = ps.LiveRunnerState()
    runner_state.pending_buys["CEG"] = _pending_buy("CEG", price=75.0)
    blind_intents = carm._write_blind_prepared_intents(runner_state, "2026-08-18", pre_state_hash="h")
    blind_intent = blind_intents[0]

    hook = carm._order_intent_hook_for_run(blind_intents)
    hook("SUBMITTING", ticker="CEG", action_kind="ENTRY_MARKET_BUY", client_order_id=blind_intent.client_order_id)

    decision = {"as_of_bar_timestamp_equity": "2026-08-18", "executed_today": {"entries": [], "exits": []}}
    result_intents = carm._run_intent_protocol(decision, pre_state_hash="h", blind_intents=blind_intents)

    assert len(result_intents) == 1
    assert result_intents[0].status == order_intent.UNCERTAIN
    reloaded = order_intent.load_intent(blind_intent.intent_id)
    assert reloaded.status == order_intent.UNCERTAIN


def test_run_intent_protocol_routes_unmaterialized_broker_acknowledged_intent_to_uncertain():
    """Same finding #4 fix, the BROKER_ACKNOWLEDGED case: the hook
    completed a real (scaffold) acknowledgment for this candidate, yet
    it still does not appear in executed_today -- genuinely ambiguous,
    never silently claimed as 'never attempted' (TERMINAL), and
    BROKER_ACKNOWLEDGED has no valid transition straight to TERMINAL
    either."""
    runner_state = ps.LiveRunnerState()
    runner_state.pending_buys["CEG"] = _pending_buy("CEG", price=75.0)
    blind_intents = carm._write_blind_prepared_intents(runner_state, "2026-08-18", pre_state_hash="h")
    blind_intent = blind_intents[0]

    hook = carm._order_intent_hook_for_run(blind_intents)
    hook("SUBMITTING", ticker="CEG", action_kind="ENTRY_MARKET_BUY", client_order_id=blind_intent.client_order_id)
    hook(
        "BROKER_ACKNOWLEDGED", ticker="CEG", action_kind="ENTRY_MARKET_BUY",
        client_order_id=blind_intent.client_order_id,
        order=_FakeBrokerOrder("real-broker-order-999", status="rejected"),
    )

    decision = {"as_of_bar_timestamp_equity": "2026-08-18", "executed_today": {"entries": [], "exits": []}}
    result_intents = carm._run_intent_protocol(decision, pre_state_hash="h", blind_intents=blind_intents)

    assert len(result_intents) == 1
    assert result_intents[0].status == order_intent.UNCERTAIN
    assert "not assumed to be" in result_intents[0].last_error.lower() or "ambiguous" in result_intents[0].last_error.lower()


def test_run_intent_protocol_leaves_a_pre_existing_uncertain_blind_intent_untouched():
    """The `else` branch: a blind intent already UNCERTAIN (e.g. from a
    prior partial run) must not be re-transitioned again here."""
    runner_state = ps.LiveRunnerState()
    runner_state.pending_buys["CEG"] = _pending_buy("CEG", price=75.0)
    blind_intents = carm._write_blind_prepared_intents(runner_state, "2026-08-18", pre_state_hash="h")
    blind_intent = blind_intents[0]
    blind_intent = order_intent.transition_intent(blind_intent, order_intent.SUBMITTING)
    blind_intent = order_intent.transition_intent(blind_intent, order_intent.UNCERTAIN, last_error="pre-existing")
    blind_intents[0] = blind_intent

    decision = {"as_of_bar_timestamp_equity": "2026-08-18", "executed_today": {"entries": [], "exits": []}}
    result_intents = carm._run_intent_protocol(decision, pre_state_hash="h", blind_intents=blind_intents)

    assert len(result_intents) == 1
    assert result_intents[0].status == order_intent.UNCERTAIN
    assert result_intents[0].last_error == "pre-existing"  # untouched, not overwritten


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


# --- Task 3 (2026-08-22), item 3/4: PROTECTIVE_STOP/CANCEL_PROTECTIVE_STOP
# just-in-time journaling via _order_intent_hook_for_run ---


def test_protective_stop_hook_creates_prepared_intent_before_submitting_on_disk():
    """REAL GAP CLOSED (Task 3): before this fix, _order_intent_hook_for_run
    silently no-op'd for PROTECTIVE_STOP (no pre-run blind candidate
    exists for it) -- a real stop-placement broker call could crash with
    ZERO durable evidence. Proves the hook now creates a real PREPARED
    intent on disk BEFORE the broker call, exactly the crash-window
    guarantee this whole mechanism exists to provide."""
    hook = carm._order_intent_hook_for_run([])
    hook(
        "SUBMITTING", ticker="CEG", action_kind="PROTECTIVE_STOP",
        client_order_id="CEG-PROTECTIVE-STOP-FOR-entry-order-1",
        side="SELL", order_type="stop", quantity=5.0, stop_price=70.0,
        parent_order_id="entry-order-1",
    )
    intents_on_disk = order_intent.list_intents()
    assert len(intents_on_disk) == 1
    on_disk = intents_on_disk[0]
    assert on_disk.status == order_intent.SUBMITTING  # already advanced past PREPARED by this point
    assert on_disk.ticker == "CEG"
    assert on_disk.action_kind == "PROTECTIVE_STOP"
    assert on_disk.quantity == 5.0
    assert on_disk.stop_price == 70.0
    assert on_disk.client_order_id == "CEG-PROTECTIVE-STOP-FOR-entry-order-1"


def test_protective_stop_hook_acknowledges_with_real_broker_order_id():
    hook = carm._order_intent_hook_for_run([])
    hook(
        "SUBMITTING", ticker="CEG", action_kind="PROTECTIVE_STOP",
        client_order_id="CEG-PROTECTIVE-STOP-FOR-entry-order-1",
        side="SELL", order_type="stop", quantity=5.0, stop_price=70.0,
        parent_order_id="entry-order-1",
    )

    class _FakeStopOrder:
        id = "real-stop-order-1"
        status = "new"

    hook(
        "BROKER_ACKNOWLEDGED", ticker="CEG", action_kind="PROTECTIVE_STOP",
        client_order_id="CEG-PROTECTIVE-STOP-FOR-entry-order-1", order=_FakeStopOrder(),
    )
    on_disk = order_intent.list_intents()[0]
    assert on_disk.status == order_intent.BROKER_ACKNOWLEDGED
    assert on_disk.broker_order_id == "real-stop-order-1"


def test_cancel_protective_stop_hook_creates_prepared_intent_keyed_by_operation_id():
    """CANCEL has no client_order_id -- must be keyed/found by
    operation_id instead (OrderIntent.operation_id/target_broker_order_id,
    see that dataclass's own field comments)."""
    hook = carm._order_intent_hook_for_run([])
    hook(
        "SUBMITTING", ticker="CEG", action_kind="CANCEL_PROTECTIVE_STOP",
        operation_id="CEG-CANCEL-PROTECTIVE-STOP-resting-stop-1",
        side="SELL", order_type="stop", parent_order_id="resting-stop-1",
    )
    on_disk = order_intent.list_intents()
    assert len(on_disk) == 1
    assert on_disk[0].operation_id == "CEG-CANCEL-PROTECTIVE-STOP-resting-stop-1"
    assert on_disk[0].target_broker_order_id == "resting-stop-1"
    assert on_disk[0].client_order_id == ""
    assert on_disk[0].status == order_intent.SUBMITTING


def test_cancel_protective_stop_hook_broker_acknowledged_uses_explicit_broker_status():
    """independent-audit-round-2 finding #3 (2026-08-23), consumer side:
    `_order_intent_hook_for_run`'s own `hook()` must prefer an explicitly
    passed `broker_status` (the only thing a CANCEL caller can supply --
    it has no `Order` object) over trying to derive one from `order`
    (which is `None` for a cancel)."""
    hook = carm._order_intent_hook_for_run([])
    hook(
        "SUBMITTING", ticker="CEG", action_kind="CANCEL_PROTECTIVE_STOP",
        operation_id="CEG-CANCEL-PROTECTIVE-STOP-resting-stop-1",
        side="SELL", order_type="stop", parent_order_id="resting-stop-1",
    )
    hook(
        "BROKER_ACKNOWLEDGED", ticker="CEG", action_kind="CANCEL_PROTECTIVE_STOP",
        operation_id="CEG-CANCEL-PROTECTIVE-STOP-resting-stop-1",
        side="SELL", order_type="stop", parent_order_id="resting-stop-1",
        broker_status="pending_cancel",
    )
    on_disk = order_intent.list_intents()
    assert len(on_disk) == 1
    assert on_disk[0].status == order_intent.BROKER_ACKNOWLEDGED
    assert on_disk[0].broker_status == "pending_cancel"  # not None, the real bug before this fix
    assert on_disk[0].broker_order_id is None  # a cancel confirmation has no order id of its own here


def test_unknown_action_kind_is_fail_closed_not_silently_skipped():
    """REAL FAIL-CLOSED PROOF (Task 3): an action kind this hook does not
    know how to journal must raise, never silently pass through
    unjournaled."""
    hook = carm._order_intent_hook_for_run([])
    with pytest.raises(RuntimeError, match="unrecognized action_kind"):
        hook("SUBMITTING", ticker="XYZ", action_kind="SOME_FUTURE_ACTION_KIND", client_order_id="XYZ-1")


def test_just_in_time_intent_is_idempotently_reused_not_duplicated_on_retry():
    """A crash between SUBMITTING and BROKER_ACKNOWLEDGED, followed by a
    retry within the same run reaching this hook again for the exact
    same client_order_id, must reuse the same on-disk PREPARED/SUBMITTING
    intent -- never create a second, divergent one."""
    hook = carm._order_intent_hook_for_run([])
    hook(
        "SUBMITTING", ticker="CEG", action_kind="PROTECTIVE_STOP",
        client_order_id="CEG-PROTECTIVE-STOP-FOR-entry-order-1",
        side="SELL", order_type="stop", quantity=5.0, stop_price=70.0,
        parent_order_id="entry-order-1",
    )
    assert len(order_intent.list_intents()) == 1
    first_intent_id = order_intent.list_intents()[0].intent_id

    # A SECOND hook instance -- simulating a fresh in-process retry that
    # rebuilt its own just_in_time_by_key dict from scratch -- must find
    # the SAME on-disk intent via find_intent_by_client_order_id, not
    # create a new one.
    second_hook = carm._order_intent_hook_for_run([])
    second_hook(
        "SUBMITTING", ticker="CEG", action_kind="PROTECTIVE_STOP",
        client_order_id="CEG-PROTECTIVE-STOP-FOR-entry-order-1",
        side="SELL", order_type="stop", quantity=5.0, stop_price=70.0,
        parent_order_id="entry-order-1",
    )
    all_intents = order_intent.list_intents()
    assert len(all_intents) == 1
    assert all_intents[0].intent_id == first_intent_id


def test_just_in_time_intents_are_exposed_for_the_finalization_loop():
    """_execute()'s own COMMITTED/TERMINAL finalization loop needs these
    intents alongside _run_intent_protocol's own return value -- exposed
    as a plain function attribute, not silently dropped."""
    hook = carm._order_intent_hook_for_run([])
    hook(
        "SUBMITTING", ticker="CEG", action_kind="PROTECTIVE_STOP",
        client_order_id="CEG-PROTECTIVE-STOP-FOR-entry-order-1",
        side="SELL", order_type="stop", quantity=5.0, stop_price=70.0,
        parent_order_id="entry-order-1",
    )
    assert hasattr(hook, "just_in_time_intents")
    assert len(hook.just_in_time_intents) == 1


def test_finalize_run_intents_commits_then_terminates_broker_acknowledged_intents():
    """Real-behavior replacement (independent-audit-round-2 test-quality
    note, 2026-08-23) for the original source-text-matching version of
    this test: `_execute()`'s own finalization loop was extracted into
    `_finalize_run_intents` specifically so this could call it directly
    and assert genuine, on-disk resulting statuses -- not merely that
    certain tokens appear near each other in `_execute`'s source.
    Closes the real, pre-existing gap where COMMITTED was previously a
    dead end (nothing ever reached TERMINAL for a run's own materialized
    intents, despite order_intent.py's own _VALID_TRANSITIONS always
    having allowed COMMITTED -> TERMINAL)."""
    materialized = order_intent.create_intent(
        client_order_id="AAPL-ENTRY-MARKET-BUY-2026-08-20", account_identity="control_arm_v1", ticker="AAPL",
        side="BUY", order_type="market", action_kind="ENTRY_MARKET_BUY", source_signal_timestamp="2026-08-20T00:00:00Z",
    )
    materialized = order_intent.transition_intent(materialized, order_intent.SUBMITTING)
    materialized = order_intent.transition_intent(
        materialized, order_intent.BROKER_ACKNOWLEDGED, broker_order_id="real-1", broker_status="filled",
    )
    abandoned = order_intent.create_intent(
        client_order_id="MSFT-ENTRY-MARKET-BUY-2026-08-20", account_identity="control_arm_v1", ticker="MSFT",
        side="BUY", order_type="market", action_kind="ENTRY_MARKET_BUY", source_signal_timestamp="2026-08-20T00:00:00Z",
    )
    abandoned = order_intent.transition_intent(abandoned, order_intent.TERMINAL, last_error="attempted_but_not_executed")

    carm._finalize_run_intents([materialized, abandoned])

    reloaded_materialized = order_intent.load_intent(materialized.intent_id)
    assert reloaded_materialized.status == order_intent.TERMINAL
    reloaded_abandoned = order_intent.load_intent(abandoned.intent_id)
    assert reloaded_abandoned.status == order_intent.TERMINAL  # untouched -- was already TERMINAL, never re-transitioned


def test_finalize_run_intents_folds_in_just_in_time_intents_from_the_hook():
    """`_execute()`'s own caller passes `list(intents) +
    list(hook.just_in_time_intents.values())` -- proves
    `_finalize_run_intents` itself has no special-casing that would only
    work for `_run_intent_protocol`'s own ENTRY/EXIT intents, by feeding
    it a PROTECTIVE_STOP-shaped one the hook would have created."""
    stop_intent = order_intent.create_intent(
        client_order_id="CEG-PROTECTIVE-STOP-FOR-entry-order-1", account_identity="control_arm_v1", ticker="CEG",
        side="SELL", order_type="stop", action_kind="PROTECTIVE_STOP", source_signal_timestamp="2026-08-20T00:00:00Z",
    )
    stop_intent = order_intent.transition_intent(stop_intent, order_intent.SUBMITTING)
    stop_intent = order_intent.transition_intent(
        stop_intent, order_intent.BROKER_ACKNOWLEDGED, broker_order_id="stop-real-1", broker_status="new",
    )

    carm._finalize_run_intents([stop_intent])

    reloaded = order_intent.load_intent(stop_intent.intent_id)
    assert reloaded.status == order_intent.TERMINAL
