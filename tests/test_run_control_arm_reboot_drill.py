"""Tests for `scripts/run_control_arm_reboot_drill.py` -- the reboot-drill
harness. Covers the individually-testable UNITS in-process (fake broker
ledger persistence, fake trading client behavior, the barrier
controller's own no-op paths -- never triggering a real SIGSTOP inside
the pytest process itself), plus one real subprocess-based end-to-end
test that actually arms a drill, waits for it to reach a real SIGSTOP,
kills it (simulating a crash), and runs `--phase recover` as a fresh
process -- the same manual sequence already verified by hand while
building this harness, now automated.

CRITICAL SAFETY NOTE FOR THIS TEST FILE ITSELF: `BarrierController.maybe_pause`
calls `os.kill(os.getpid(), signal.SIGSTOP)` when its armed barrier
matches -- calling that in-process, inside THIS pytest run, would freeze
the test runner itself. Every in-process test below only exercises
`maybe_pause` with an armed barrier of `None` or a deliberately
NON-matching name, so the pause branch is structurally unreachable here.
The one test that needs a REAL pause exercises it in a REAL subprocess
instead (`test_arm_then_kill_then_recover_end_to_end`), never in-process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.live import position_state as ps  # noqa: E402

# The drill module, as a MODULE-LEVEL side effect of merely being
# imported (see that file's own "STEP 0" docstring section), (a)
# neutralizes `dotenv.load_dotenv` GLOBALLY and (b) sets/pops several
# `os.environ` keys (`AI_STOCK_RADAR_GUARD_DIR`, `ALPACA_API_KEY`, ...).
# Correct and necessary for the drill's own SUBPROCESS invocations
# (separate Python processes, their own separate module/env state --
# nothing below affects them), but importing this module as a plain
# library -- exactly what this test file does, to reach
# `FakeBrokerLedger`/`FakeTradingClient`/`BarrierController` for
# in-process unit tests -- would otherwise leak those changes to the
# REST of this same pytest process. A real, confirmed regression this
# fix closes (it broke `test_telegram_notifier.py`/
# `test_telegram_network_isolation.py`, which run in the same session
# and depend on the genuine `dotenv.load_dotenv`). Snapshot before
# import, restore immediately after.
_ENV_KEYS_THE_DRILL_MODULE_TOUCHES = (
    "AI_STOCK_RADAR_GUARD_DIR", "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "LIVE_ACCOUNT_NUMBER_SUFFIX",
    "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "HEALTHCHECKS_LIVENESS_URL", "HEALTHCHECKS_OPERATIONAL_URL",
)
_env_snapshot = {key: os.environ.get(key) for key in _ENV_KEYS_THE_DRILL_MODULE_TOUCHES}

import run_control_arm_reboot_drill as drill  # noqa: E402

import dotenv  # noqa: E402

dotenv.load_dotenv = drill._REAL_LOAD_DOTENV
for _key, _value in _env_snapshot.items():
    if _value is None:
        os.environ.pop(_key, None)
    else:
        os.environ[_key] = _value


# --- FakeBrokerLedger --------------------------------------------------


def test_ledger_round_trips_through_disk(tmp_path):
    ledger = drill.FakeBrokerLedger(tmp_path / "ledger.json")
    ledger.orders["order-1"] = {"id": "order-1", "client_order_id": "AA-1", "symbol": "AA", "side": "buy", "status": "filled"}
    ledger.positions["AA"] = {"symbol": "AA", "qty": "10", "side": "long"}
    ledger.save()

    reloaded = drill.FakeBrokerLedger(tmp_path / "ledger.json")
    assert reloaded.orders == ledger.orders
    assert reloaded.positions == ledger.positions
    assert reloaded.account_number == ledger.account_number


def test_ledger_starts_empty_when_no_file_exists(tmp_path):
    ledger = drill.FakeBrokerLedger(tmp_path / "does-not-exist.json")
    assert ledger.orders == {}
    assert ledger.positions == {}


def test_ledger_find_by_client_order_id(tmp_path):
    ledger = drill.FakeBrokerLedger(tmp_path / "ledger.json")
    ledger.orders["order-1"] = {"id": "order-1", "client_order_id": "AA-1", "symbol": "AA", "side": "buy", "status": "filled"}
    assert ledger.find_by_client_order_id("AA-1")["id"] == "order-1"
    assert ledger.find_by_client_order_id("does-not-exist") is None


# --- FakeTradingClient ---------------------------------------------------


def test_fake_client_get_account_and_get_all_positions(tmp_path):
    ledger = drill.FakeBrokerLedger(tmp_path / "ledger.json")
    ledger.positions["AA"] = {"symbol": "AA", "qty": "10", "side": "long"}
    client = drill.FakeTradingClient(ledger, today_iso="2026-08-23")
    assert client.get_account().account_number.endswith(drill._ACCOUNT_SUFFIX)
    positions = client.get_all_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "AA"


def test_fake_client_submit_order_fills_immediately_and_persists(tmp_path):
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    ledger = drill.FakeBrokerLedger(tmp_path / "ledger.json")
    client = drill.FakeTradingClient(ledger, today_iso="2026-08-23")
    request = MarketOrderRequest(
        symbol="AA", qty=10, side=OrderSide.BUY, time_in_force=TimeInForce.DAY, client_order_id="AA-1",
    )
    order = client.submit_order(request)
    assert order.status == "filled"
    assert order.client_order_id == "AA-1"

    reloaded = drill.FakeBrokerLedger(tmp_path / "ledger.json")
    assert reloaded.find_by_client_order_id("AA-1")["status"] == "filled"


def test_fake_client_get_order_by_client_id_raises_404_when_absent(tmp_path):
    from alpaca.common.exceptions import APIError

    ledger = drill.FakeBrokerLedger(tmp_path / "ledger.json")
    client = drill.FakeTradingClient(ledger, today_iso="2026-08-23")
    with pytest.raises(APIError) as exc_info:
        client.get_order_by_client_id("does-not-exist")
    assert exc_info.value.status_code == 404


def test_fake_client_cancel_order_by_id_marks_canceled(tmp_path):
    ledger = drill.FakeBrokerLedger(tmp_path / "ledger.json")
    ledger.orders["stop-1"] = {"id": "stop-1", "client_order_id": "", "symbol": "AA", "side": "sell", "status": "new"}
    client = drill.FakeTradingClient(ledger, today_iso="2026-08-23")
    client.cancel_order_by_id("stop-1")
    assert ledger.orders["stop-1"]["status"] == "canceled"


def test_fake_client_get_orders_accepts_the_real_filter_kwarg(tmp_path):
    """broker_reconciliation._fetch_open_orders calls
    `client.get_orders(filter=GetOrdersRequest(...))` -- a real signature
    mismatch here would only surface at drill-run time, not import time,
    so this is pinned directly."""
    ledger = drill.FakeBrokerLedger(tmp_path / "ledger.json")
    client = drill.FakeTradingClient(ledger, today_iso="2026-08-23")
    assert client.get_orders(filter=object()) == []


def test_fake_client_get_calendar_returns_a_settled_entry_for_the_requested_date():
    from datetime import date

    from alpaca.trading.requests import GetCalendarRequest

    ledger = drill.FakeBrokerLedger(Path("/tmp/does-not-matter-for-this-test.json"))
    client = drill.FakeTradingClient(ledger, today_iso="2026-08-23")
    entries = client.get_calendar(GetCalendarRequest(start=date(2026, 8, 23), end=date(2026, 8, 23)))
    assert len(entries) == 1
    assert entries[0].date == date(2026, 8, 23)


# --- BarrierController (no-op paths only -- see module docstring) -------


def test_barrier_controller_never_pauses_when_no_barrier_is_armed(monkeypatch):
    monkeypatch.setattr(os, "kill", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must never pause")))
    controller = drill.BarrierController(Path("/tmp"), armed_barrier=None, action_kind_filter=None)
    for barrier_name in drill.BARRIER_NAMES:
        controller.maybe_pause(barrier_name)
    assert controller.fired is False


def test_barrier_controller_never_pauses_on_a_non_matching_barrier_name(monkeypatch):
    monkeypatch.setattr(os, "kill", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must never pause")))
    controller = drill.BarrierController(Path("/tmp"), armed_barrier="AFTER_COMMITTED", action_kind_filter=None)
    controller.maybe_pause("AFTER_SUBMITTING")
    assert controller.fired is False


def test_barrier_controller_never_pauses_on_a_non_matching_action_kind(monkeypatch, tmp_path):
    from types import SimpleNamespace

    monkeypatch.setattr(os, "kill", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must never pause")))
    controller = drill.BarrierController(tmp_path, armed_barrier="AFTER_SUBMITTING", action_kind_filter="ENTRY_MARKET_BUY")
    fake_intent = SimpleNamespace(action_kind="CANCEL_PROTECTIVE_STOP", to_dict=lambda: {})
    controller.maybe_pause("AFTER_SUBMITTING", fake_intent)
    assert controller.fired is False


def test_barrier_controller_only_pauses_once(monkeypatch, tmp_path):
    """After firing once, a second matching call must be a no-op --
    confirmed by asserting os.kill is called exactly once."""
    calls = []
    monkeypatch.setattr(os, "kill", lambda *a, **k: calls.append(a))
    controller = drill.BarrierController(tmp_path, armed_barrier="AFTER_STATE_SAVE", action_kind_filter=None)
    controller.maybe_pause("AFTER_STATE_SAVE")
    controller.maybe_pause("AFTER_STATE_SAVE")
    assert len(calls) == 1
    assert (tmp_path / "barrier.json").is_file()


# --- state-construction fixtures ----------------------------------------


def test_build_initial_runner_state_submit_has_fresh_pending_signal_metadata():
    state = drill._build_initial_runner_state(
        action="submit", yesterday_iso="2026-08-22", today_iso="2026-08-23", stop_order_id=None,
    )
    assert "AA" in state.pending_buys
    record = state.pending_signal_metadata["AA|BUY"]
    assert record["status"] == "pending"
    assert record["target_execution_session_date"] == "2026-08-23"


def test_build_initial_runner_state_cancel_has_stop_order_and_submitted_action():
    state = drill._build_initial_runner_state(
        action="cancel", yesterday_iso="2026-08-22", today_iso="2026-08-23", stop_order_id="stop-1",
    )
    assert state.positions["AA"].ticker == "AA"
    assert state.equity_stop_orders["AA"] == "stop-1"
    assert any(v["order_id"] == "stop-1" for v in state.submitted_actions.values())
    assert "AA" in state.pending_exits
    assert state.pending_signal_metadata["AA|EXIT"]["status"] == "pending"


def test_build_initial_runner_state_rejects_unknown_action():
    with pytest.raises(ValueError):
        drill._build_initial_runner_state(action="bogus", yesterday_iso="2026-08-22", today_iso="2026-08-23", stop_order_id=None)


# --- CLI smoke tests (subprocess, no real barrier reached) ---------------


def _run_drill_cli(*args: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "scripts.run_control_arm_reboot_drill", *args],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=timeout,
    )


def test_self_check_reports_fake_client_and_env_isolation(tmp_path):
    result = _run_drill_cli("--case-dir", str(tmp_path), "--self-check")
    report = json.loads(result.stdout)
    assert report["checks"]["fake_client_reachable"]["pass"] is True
    assert report["checks"]["env_not_loaded"]["pass"] is True
    # network_isolation's own pass/fail depends entirely on whether THIS
    # test host has a real network-isolated sandbox around it -- not
    # asserted either way here, see the report's own "note" field.


def test_recover_without_a_prior_arm_raises_a_clear_error(tmp_path):
    result = _run_drill_cli("--case-dir", str(tmp_path), "--phase", "recover", "--broker-outcome", "absent")
    assert result.returncode != 0
    assert "does not exist" in result.stderr or "does not exist" in result.stdout


# --- real subprocess end-to-end: arm -> real SIGSTOP -> kill -> recover --


def _wait_for_process_state(pid: int, target_states: tuple[str, ...], *, timeout: float) -> str:
    deadline = time.time() + timeout
    last_state = ""
    while time.time() < deadline:
        probe = subprocess.run(["ps", "-o", "state=", "-p", str(pid)], capture_output=True, text=True)
        last_state = probe.stdout.strip()
        if last_state and last_state[0] in target_states:
            return last_state
        time.sleep(0.1)
    return last_state


def _arm_wait_for_stop_then_kill(
    case_dir: Path, *, barrier: str, action: str, broker_outcome: str, extra_wait_states: tuple[str, ...] = ("T",),
) -> dict:
    """Shared arm -> real SIGSTOP -> SIGKILL sequence used by both
    end-to-end tests below. Returns the parsed `barrier.json` payload."""
    arm = subprocess.Popen(
        [
            sys.executable, "-m", "scripts.run_control_arm_reboot_drill",
            "--case-dir", str(case_dir), "--phase", "arm",
            "--barrier", barrier, "--action", action, "--broker-outcome", broker_outcome,
        ],
        cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        state = _wait_for_process_state(arm.pid, extra_wait_states, timeout=30.0)
        assert state.startswith("T"), (
            f"drill process never reached a stopped state (last ps state: {state!r}) -- "
            f"barrier.json exists: {(case_dir / 'barrier.json').is_file()}"
        )
        assert (case_dir / "barrier.json").is_file()
        return json.loads((case_dir / "barrier.json").read_text(encoding="utf-8"))
    finally:
        arm.kill()  # SIGKILL -- simulates the real crash/reboot at this exact point
        arm.wait(timeout=10.0)


@pytest.mark.skipif(sys.platform == "win32", reason="SIGSTOP/ps -o state= are POSIX-only")
def test_arm_then_kill_then_recover_end_to_end(tmp_path):
    """The real sequence this whole harness exists for: arm a drill for
    real, wait for the REAL SIGSTOP, kill -9 the stopped process
    (simulating a crash at that exact point), then run --phase recover
    as a genuinely fresh process and confirm RECOVERY ACTUALLY SUCCEEDED
    -- not merely that one stray intent was found (reboot-drill finding
    #1's own acceptance test, 2026-08-24; the ORIGINAL version of this
    test only asserted the "before recovery" snapshot, which does not
    prove recovery itself worked -- it did NOT, before the finding #1
    fix: this exact scenario reproduced a real
    `StrayPreparedIntentConflictError` deadlock, confirmed by hand
    before writing this fix). Uses AFTER_PREPARED (the cheapest barrier
    to reach) to keep this test fast; the full 6-barrier x 2-action x
    2-outcome matrix is the owner's own drill to run for real, per this
    module's own docstring."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    barrier_payload = _arm_wait_for_stop_then_kill(
        case_dir, barrier="AFTER_PREPARED", action="submit", broker_outcome="absent",
    )
    assert barrier_payload["barrier"] == "AFTER_PREPARED"
    assert barrier_payload["intent"]["status"] == "PREPARED"
    original_intent_id = barrier_payload["intent"]["intent_id"]

    recover = _run_drill_cli("--case-dir", str(case_dir), "--phase", "recover", "--broker-outcome", "absent", timeout=60.0)
    report_path = case_dir / "recovery_report.json"
    assert report_path.is_file(), f"recover stdout:\n{recover.stdout}\nstderr:\n{recover.stderr}"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["barrier"] == "AFTER_PREPARED"
    assert len(report["stray_intents_before_recovery"]) == 1
    assert report["stray_intents_before_recovery"][0]["status"] == "PREPARED"

    # The CLI's own exit code -- independent-audit finding, 2026-08-24:
    # `--phase recover` used to always exit 0 regardless of whether
    # recovery actually raised, so a human/CI gate checking only the exit
    # code (not parsing recovery_report.json by hand) would see a false
    # PASS. A clean recovery must exit 0.
    assert recover.returncode == 0, f"recover stdout:\n{recover.stdout}\nstderr:\n{recover.stderr}"

    # RECOVERY ACTUALLY SUCCEEDED -- the acceptance test's own real
    # criteria, not just "a stray was found":
    assert report["execute_raised"] is None, (
        f"recovery raised instead of completing: {report['execute_raised']}"
    )
    assert report["execute_outcome"] == "RUN_OK"
    # The SAME intent_id was reused (RESUMABLE_PREPARED, finding #1's own
    # fix) -- never a second, divergent record for the same candidate --
    # and it resolved past PREPARED as part of the same successful run.
    stray_after = report["stray_intents_after_recovery"]
    assert stray_after == [] or all(i["intent_id"] != original_intent_id for i in stray_after), (
        "the original intent must not still be stray after a successful run"
    )
    # At most ONE real broker submission for this ticker's ENTRY_MARKET_BUY
    # -- never a duplicate caused by the old deadlock's own retry/conflict.
    entry_actions = [
        record for key, record in report["final_position_state"]["submitted_actions"].items()
        if record.get("kind") == "ENTRY_MARKET_BUY"
    ]
    assert len(entry_actions) <= 1
    if entry_actions:
        assert entry_actions[0]["client_order_id"] == barrier_payload["intent"]["client_order_id"]


@pytest.mark.skipif(sys.platform == "win32", reason="SIGSTOP/ps -o state= are POSIX-only")
def test_submitting_then_broker_acknowledged_crash_fails_closed_with_the_specific_diagnosis(tmp_path):
    """Reboot-drill finding #2's own acceptance scenario -- the most
    dangerous window: SUBMITTING was durably recorded, the (fake) broker
    genuinely acknowledged the order, and the process crashes BEFORE
    local state (`submitted_actions`) is updated to reflect it. Recovery
    must (a) NOT prematurely claim TERMINAL/done for the journal entry,
    and (b) still halt fail-closed, but with the specific
    `JournalAcknowledgedButLocalStateMissingError` diagnosis rather than
    the generic "mystery order" one -- confirmed by hand before writing
    this fix (the old code raised `UnknownBrokerOrderError` while the
    journal had already, wrongly, closed TERMINAL)."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    barrier_payload = _arm_wait_for_stop_then_kill(
        case_dir, barrier="AFTER_SUBMITTING", action="submit", broker_outcome="present",
    )
    assert barrier_payload["barrier"] == "AFTER_SUBMITTING"
    assert barrier_payload["intent"]["status"] == "SUBMITTING"

    recover = _run_drill_cli("--case-dir", str(case_dir), "--phase", "recover", "--broker-outcome", "present", timeout=60.0)
    report_path = case_dir / "recovery_report.json"
    assert report_path.is_file(), f"recover stdout:\n{recover.stdout}\nstderr:\n{recover.stderr}"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    # Fail-closed, with the SPECIFIC diagnosis -- never the generic one,
    # and never silently succeeding.
    assert report["execute_raised"] is not None
    assert "JournalAcknowledgedButLocalStateMissingError" in report["execute_raised"]
    assert "UnknownBrokerOrderError" not in report["execute_raised"].split(":")[0]
    assert report["execute_outcome"] is None

    # The CLI's own exit code must reflect this halt too -- independent-
    # audit finding, 2026-08-24 (see the sibling success test's own
    # comment): before the fix, `--phase recover` exited 0 even here,
    # which would have let a human/CI gate checking only the exit code
    # falsely believe recovery from THIS exact dangerous window succeeded.
    assert recover.returncode != 0, (
        f"recover exited 0 despite execute_raised being set -- exit code must reflect the halt. "
        f"stdout:\n{recover.stdout}\nstderr:\n{recover.stderr}"
    )

    # The journal must be left HONESTLY at BROKER_ACKNOWLEDGED, never
    # prematurely claimed TERMINAL -- the real bug finding #2 closes.
    stray_after = report["stray_intents_after_recovery"]
    assert len(stray_after) == 1
    assert stray_after[0]["status"] == "BROKER_ACKNOWLEDGED"
    assert stray_after[0]["broker_order_id"] is not None

    # And local state must NOT have been fabricated to paper over the gap
    # -- zero submitted_actions entries recorded for this run's own
    # candidate, matching the honestly-still-open journal above.
    entry_actions = [
        record for key, record in report["final_position_state"]["submitted_actions"].items()
        if record.get("kind") == "ENTRY_MARKET_BUY"
    ]
    assert entry_actions == []


@pytest.mark.skipif(sys.platform == "win32", reason="SIGSTOP/ps -o state= are POSIX-only")
def test_stray_intent_for_a_different_signal_is_not_resumed_and_recovers_deterministically(tmp_path):
    """Independent-audit finding, 2026-08-24 -- the two follow-up gaps
    found in the fix for finding #1, both closed here in one real,
    subprocess-based scenario:

    (a) `_stray_intent_matches_live_candidate`'s REAL bug: the original
    version recomputed a stray intent's own `client_order_id` from its
    OWN stored fields and compared it to its OWN stored `client_order_id`
    -- tautological, always true given a same-ticker pending candidate,
    regardless of whether that candidate was genuinely the SAME signal.
    Reproduced here for real: between the crash and recovery, the
    on-disk `pending_signal_metadata` for the armed candidate is
    mutated to a DIFFERENT `target_execution_session_date` (simulating
    "the pending signal changed underneath the crashed run" -- e.g. a
    TTL re-evaluation or an operator action) -- recovery must NOT resume
    the stray intent into this unrelated candidate.

    (b) Because `_write_blind_prepared_intents` still needs to write a
    fresh PREPARED intent for AA's own (still-pending, just different)
    candidate under the SAME client_order_id (both dates are today's
    real session), this run ALSO exercises the exact TERMINAL+PREPARED
    duplicate-record scenario for real -- and a subsequent lookup inside
    this SAME recovery run (the write-ahead-evidence check, right before
    `rdd.run_daily_decision()`) must deterministically find the fresh
    ACTIVE record, never the old TERMINAL one, for the run to complete.
    """
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    barrier_payload = _arm_wait_for_stop_then_kill(
        case_dir, barrier="AFTER_PREPARED", action="submit", broker_outcome="absent",
    )
    assert barrier_payload["intent"]["status"] == "PREPARED"
    stale_client_order_id = barrier_payload["intent"]["client_order_id"]

    # Simulate "the pending signal changed underneath the crashed run":
    # mutate the on-disk pending_signal_metadata for AA|BUY to a
    # DIFFERENT target_execution_session_date than the stray intent's
    # own source_signal_timestamp -- a genuinely different signal
    # instance, not the same one the crashed run was working on. This
    # subprocess's own guard directory (order_intents, high_water_mark)
    # lives under case_dir/guard -- see the module-level
    # AI_STOCK_RADAR_GUARD_DIR isolation each drill subprocess sets up
    # for itself (module docstring safety point (1)).
    state_paths = drill._drill_paths(case_dir)
    guard_path = case_dir / "guard" / "high_water_mark.json"
    state = ps.load_position_state(state_paths["state_path"], guard_path=guard_path)
    assert "AA|BUY" in state.pending_signal_metadata
    # A FUTURE date, not a past one: `pending_signal_ttl.evaluate_pending_signals`
    # leaves a signal completely untouched whenever
    # `expected_session_date <= target_execution_session_date` (see that
    # function's own docstring) -- a PAST date would instead get pruned
    # as expired before `_write_blind_prepared_intents` ever runs, which
    # tests TTL pruning, not this fix. A future date keeps AA's candidate
    # genuinely live and pending today while still differing from the
    # stray intent's own `source_signal_timestamp` (today), which is
    # exactly the real-world shape of the bug: the SAME ticker has a
    # different, still-valid signal instance than the one the crashed
    # run was working on.
    state.pending_signal_metadata["AA|BUY"]["target_execution_session_date"] = "2099-01-01"
    ps.save_position_state(state, state_paths["state_path"], guard_path=guard_path)

    recover = _run_drill_cli("--case-dir", str(case_dir), "--phase", "recover", "--broker-outcome", "absent", timeout=60.0)
    report_path = case_dir / "recovery_report.json"
    assert report_path.is_file(), f"recover stdout:\n{recover.stdout}\nstderr:\n{recover.stderr}"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    # (a) The mismatch must be honored: the confirmed-404 stray closes
    # ABANDONED_NO_SUBMISSION (TERMINAL), never silently resumed into
    # the now-different pending candidate. `stray_intents_after_recovery`
    # only lists NON-terminal intents by construction, so its absence
    # there is expected; read the real on-disk journal directly (this
    # subprocess's own isolated guard dir, same filesystem) for the
    # authoritative proof, including the TERMINAL+PREPARED duplicate
    # pair bug #2's fix must now resolve correctly.
    intent_directory = case_dir / "guard" / "order_intents"
    on_disk_intents = [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted(intent_directory.glob("*.json"))
    ]
    matching_client_order_id = [i for i in on_disk_intents if i.get("client_order_id") == stale_client_order_id]
    assert len(matching_client_order_id) == 2, (
        f"expected exactly 2 on-disk records for {stale_client_order_id!r} (the abandoned original "
        f"+ a fresh one for today's still-pending, now-different candidate), got: {matching_client_order_id}"
    )
    # Both end up TERMINAL by the end of a fully successful run -- the
    # abandoned original AND the fresh record, which (bug #2's fix
    # letting the write-ahead-evidence check find IT, not the stale
    # one) went on to actually submit and fill for real. What matters
    # is that they are two DIFFERENT records, closed for two DIFFERENT
    # reasons -- never one masquerading as the other.
    abandoned = [i for i in matching_client_order_id if "ABANDONED_NO_SUBMISSION" in (i.get("last_error") or "")]
    filled = [i for i in matching_client_order_id if i.get("broker_status") == "filled"]
    assert len(abandoned) == 1, f"expected exactly one ABANDONED_NO_SUBMISSION record: {matching_client_order_id}"
    assert len(filled) == 1, f"expected exactly one genuinely-filled record: {matching_client_order_id}"
    assert abandoned[0]["intent_id"] != filled[0]["intent_id"]

    # (b) Recovery must still complete successfully -- proves the
    # duplicate-record lookup (bug #2's fix) resolved to the fresh
    # ACTIVE record, not the stale TERMINAL one, letting this run's own
    # write-ahead-evidence check and run_daily_decision() proceed.
    assert report["execute_raised"] is None, f"recovery raised: {report['execute_raised']}"
    assert report["execute_outcome"] == "RUN_OK"
    assert recover.returncode == 0

    # Exactly ONE real broker submission -- never a duplicate caused by
    # the TERMINAL+PREPARED pair sharing one client_order_id.
    entry_actions = [
        record for key, record in report["final_position_state"]["submitted_actions"].items()
        if record.get("kind") == "ENTRY_MARKET_BUY"
    ]
    assert len(entry_actions) == 1
