"""Tests for the 3 remaining gaps workspace-c flagged (2026-08-24)
before missing-session-replay's cron can be activated:

1. A normal, successful `run_daily_decision()` run (via `main()`) must
   ALSO stamp the session cursor, atomically, under the same
   `single_instance_lock` -- not just a replay run.
2. A dated PASS gate: the real 21:15 job (`main()`, with either real
   order-enabling flag set) must refuse to proceed (fail-closed, zero
   API calls) without a same-day gate written by
   `equity_session_orchestrator.run_missing_session_replay`'s own
   self-verifying check.
3. Three end-to-end scenarios, using the GENUINE `run_daily_decision()`
   and the GENUINE `run_missing_session_replay()` -- never fakes
   standing in for the functions under test (same "real, not fake"
   discipline independent-audit-round-3 finding #2 already
   established): a normal run persisting the cursor, the next
   morning's clean no-op, and a missed prior-day job being detected and
   replayed exactly once the following morning.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_daily_decision as runner  # noqa: E402

import src.live.equity_session_detection as detection  # noqa: E402
import src.live.equity_session_orchestrator as orchestrator  # noqa: E402
import src.live.session_replay_journal as session_journal  # noqa: E402
import src.live.session_replay_pass_gate as pass_gate  # noqa: E402
from src.live import position_state as ps  # noqa: E402


# --- shared fixtures / helpers ---------------------------------------------


def _integration_frame(*dates: str) -> pd.DataFrame:
    """EMA20 < EMA50 guarantees `_is_entry_setup` is always False -- no
    new position opens, so these tests are only about the session
    cursor/PASS-gate wiring, not entry/exit decision logic. Same trick
    `test_daily_decision_bars_today.py`'s own harness already uses."""
    index = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    n = len(dates)
    return pd.DataFrame(
        {
            "Open": [100.0] * n, "High": [101.0] * n, "Low": [99.0] * n, "Close": [100.0] * n,
            "EMA20": [95.0] * n, "EMA50": [100.0] * n, "RSI14": [50.0] * n, "RegimeAllowed": [True] * n,
        },
        index=index,
    )


class _FakeClient:
    """Satisfies every real call `run_daily_decision()` (reconciliation
    included, since these tests exercise the REAL, non-skip-able
    reconciliation path `main()` itself goes through) and
    `run_missing_session_replay()`'s own preflight make when orders are
    disabled/no positions exist locally."""

    def __init__(self, *, next_open_already_passed: bool = False) -> None:
        self._next_open_already_passed = next_open_already_passed

    def get_account(self):
        return SimpleNamespace(account_number="PA3HONFDTEST")

    def get_orders(self, *args, **kwargs):
        return []

    def get_all_positions(self):
        return []

    def get_clock(self):
        return SimpleNamespace(is_open=False, timestamp=datetime.now(timezone.utc))

    def get_calendar(self, request):
        offset = timedelta(hours=-2) if self._next_open_already_passed else timedelta(hours=2)
        open_utc = datetime.now(timezone.utc) + offset
        naive_eastern_open = open_utc.astimezone(detection._EASTERN).replace(tzinfo=None)
        entry = SimpleNamespace(
            date=request.start, open=naive_eastern_open, close=naive_eastern_open + timedelta(hours=6, minutes=30),
        )
        return [entry]


def _prepared(equity_dates: tuple[str, ...], crypto_dates: tuple[str, ...]) -> dict:
    return {
        **{t: _integration_frame(*equity_dates) for t in ("AAPL", "MSFT")},
        **{t: _integration_frame(*crypto_dates) for t in ("BTC-USD", "ETH-USD")},
    }


@pytest.fixture(autouse=True)
def _isolate_everything(tmp_path, monkeypatch):
    monkeypatch.setattr(session_journal, "SESSION_REPLAY_DIRECTORY", tmp_path / "session_replay_intents")
    monkeypatch.setattr(runner, "STOP_FLAG_PATH", tmp_path / "no-such-STOP-flag")
    monkeypatch.setattr(runner, "FREEZE_FLAG_PATH", tmp_path / "no-such-FREEZE-flag")
    monkeypatch.setenv("LIVE_ACCOUNT_NUMBER_SUFFIX", "TEST")
    monkeypatch.setattr(orchestrator.rdd, "_require_live_account_suffix", lambda: "TEST")

    from contextlib import contextmanager

    @contextmanager
    def _fake_lock(state_path):
        yield {}

    monkeypatch.setattr(runner, "single_instance_lock", _fake_lock)
    monkeypatch.setattr(orchestrator, "single_instance_lock", _fake_lock)


def _paths(tmp_path):
    return {
        "state_path": tmp_path / "position_state.json",
        "guard_path": tmp_path / "high_water_mark.json",
        "decision_log_directory": tmp_path / "decisions",
        "provenance_log_directory": tmp_path / "provenance",
        "pass_gate_directory": tmp_path / "pass_gate",
    }


# --- item 1: normal run also stamps the cursor ------------------------------


def test_main_stamps_the_session_cursor_after_a_normal_dry_run(tmp_path, monkeypatch):
    """A DRY-RUN (no order flags) invocation of main() must still stamp
    the cursor -- item 1's own fix applies to every successful run,
    since the stamp reflects "which session did this run genuinely
    process," true regardless of whether real orders were submitted.
    Dry-run also means the PASS-gate check (item 2, scoped to
    order-enabled invocations only) never engages -- isolates item 1
    from item 2 cleanly."""
    paths = _paths(tmp_path)
    prepared = _prepared(("2026-08-19", "2026-08-20"), ("2026-08-19", "2026-08-20"))
    monkeypatch.setattr(runner, "prepare_live_market_data", lambda tickers: prepared)
    monkeypatch.setattr(runner, "get_live_cash_balance", lambda client=None: 100_000.0)
    monkeypatch.setattr(runner.order_submission, "get_trading_client", lambda: _FakeClient())
    monkeypatch.setattr(runner, "send_telegram_message", lambda text: True)
    monkeypatch.setattr(
        sys, "argv",
        ["run_daily_decision.py", "--state-path", str(paths["state_path"]),
         "--decision-log-directory", str(paths["decision_log_directory"]),
         "--guard-path", str(paths["guard_path"])],
    )

    runner.main()

    reloaded = ps.load_position_state(paths["state_path"], guard_path=paths["guard_path"])
    assert reloaded.last_processed_equity_session_date == "2026-08-20"
    assert reloaded.last_processed_equity_date == datetime.now(timezone.utc).date().isoformat()


# --- item 2: dated PASS gate enforcement ------------------------------------


def test_main_refuses_real_orders_without_a_same_day_pass_gate(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    monkeypatch.setattr(pass_gate, "DEFAULT_PASS_GATE_DIRECTORY", paths["pass_gate_directory"])
    called = []
    monkeypatch.setattr(runner, "run_daily_decision", lambda **k: called.append(1))
    monkeypatch.setattr(
        sys, "argv",
        ["run_daily_decision.py", "--enable-equity-orders",
         "--state-path", str(paths["state_path"]), "--decision-log-directory", str(paths["decision_log_directory"]),
         "--guard-path", str(paths["guard_path"])],
    )

    with pytest.raises(runner.MissingPassGateError):
        runner.main()
    assert called == []  # zero API calls -- run_daily_decision() itself never reached


def test_main_proceeds_with_real_orders_when_a_same_day_pass_gate_exists(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    monkeypatch.setattr(pass_gate, "DEFAULT_PASS_GATE_DIRECTORY", paths["pass_gate_directory"])
    pass_gate.write_pass_gate(
        paths["pass_gate_directory"], last_processed_equity_session_date="2026-08-19", account_number_masked=None,
    )
    called = []
    fake_result = {"decision": {"as_of_bar_timestamp_equity": "2026-08-20"}, "log_path": tmp_path / "d.json"}
    monkeypatch.setattr(runner, "run_daily_decision", lambda **k: called.append(1) or fake_result)
    monkeypatch.setattr(runner, "send_telegram_message", lambda text: True)
    monkeypatch.setattr(
        sys, "argv",
        ["run_daily_decision.py", "--enable-equity-orders",
         "--state-path", str(paths["state_path"]), "--decision-log-directory", str(paths["decision_log_directory"]),
         "--guard-path", str(paths["guard_path"])],
    )

    runner.main()  # must not raise
    assert called == [1]


def test_main_dry_run_is_unaffected_by_a_missing_pass_gate(tmp_path, monkeypatch):
    """Scoping confirmation: dry-run invocations (every pre-existing
    test that calls main() without order flags included) never engage
    the gate check at all."""
    paths = _paths(tmp_path)
    monkeypatch.setattr(pass_gate, "DEFAULT_PASS_GATE_DIRECTORY", paths["pass_gate_directory"])
    assert pass_gate.has_valid_pass_gate_for_today(paths["pass_gate_directory"]) is False
    fake_result = {"decision": {"as_of_bar_timestamp_equity": "2026-08-20"}, "log_path": tmp_path / "d.json"}
    monkeypatch.setattr(runner, "run_daily_decision", lambda **k: fake_result)
    monkeypatch.setattr(runner, "send_telegram_message", lambda text: True)
    monkeypatch.setattr(
        sys, "argv",
        ["run_daily_decision.py", "--state-path", str(paths["state_path"]),
         "--decision-log-directory", str(paths["decision_log_directory"]),
         "--guard-path", str(paths["guard_path"])],
    )

    runner.main()  # must not raise MissingPassGateError


# --- item 3: the three required end-to-end scenarios ------------------------


def test_scenario_a_normal_2115_run_persists_the_cursor(tmp_path, monkeypatch):
    """"Normal 21:15 run -> cursor kalıcı." Real order flags set (the
    genuine 21:15 job's own shape), a valid PASS gate pre-exists (as it
    would after a real morning orchestrator pass), the REAL
    run_daily_decision() processes real (synthetic) market data end to
    end, and the cursor is durably stamped afterward."""
    paths = _paths(tmp_path)
    monkeypatch.setattr(pass_gate, "DEFAULT_PASS_GATE_DIRECTORY", paths["pass_gate_directory"])
    pass_gate.write_pass_gate(
        paths["pass_gate_directory"], last_processed_equity_session_date="2026-08-19", account_number_masked=None,
    )
    prepared = _prepared(("2026-08-19", "2026-08-20"), ("2026-08-19", "2026-08-20"))
    monkeypatch.setattr(runner, "prepare_live_market_data", lambda tickers: prepared)
    monkeypatch.setattr(runner, "get_live_cash_balance", lambda client=None: 100_000.0)
    fake_client = _FakeClient()
    monkeypatch.setattr(runner.order_submission, "get_trading_client", lambda: fake_client)
    monkeypatch.setattr(runner, "send_telegram_message", lambda text: True)
    monkeypatch.setattr(
        sys, "argv",
        ["run_daily_decision.py", "--enable-equity-orders", "--enable-crypto-orders",
         "--state-path", str(paths["state_path"]), "--decision-log-directory", str(paths["decision_log_directory"]),
         "--guard-path", str(paths["guard_path"])],
    )

    runner.main()

    reloaded = ps.load_position_state(paths["state_path"], guard_path=paths["guard_path"])
    assert reloaded.last_processed_equity_session_date == "2026-08-20"
    assert reloaded.last_processed_equity_bar_timestamp == "2026-08-20"
    assert reloaded.last_processed_equity_date == datetime.now(timezone.utc).date().isoformat()


def test_scenario_b_next_morning_is_a_clean_gap_free_no_op(tmp_path, monkeypatch):
    """"Ertesi sabah -> temiz, boşluksuz no-op." Starting from a cursor
    already at the latest real session (as scenario A leaves it), the
    REAL orchestrator finds nothing further expected and writes today's
    own PASS gate -- the cursor is left exactly as it was, no gap."""
    paths = _paths(tmp_path)
    state = ps.LiveRunnerState()
    state.last_processed_equity_session_date = "2026-08-20"
    ps.save_position_state(state, paths["state_path"], guard_path=paths["guard_path"])

    monkeypatch.setattr(orchestrator.detection, "detect_missing_equity_sessions", lambda *a, **k: detection.DetectionResult(
        expected_sessions=(), complete_sessions=(), missing_by_session={}, unexpected_future_bars={}, raw_bars_by_ticker={},
    ))
    monkeypatch.setattr(orchestrator.detection, "fetch_expected_equity_sessions", lambda *a, **k: [])
    monkeypatch.setattr(orchestrator.broker_reconciliation, "reconcile", lambda *a, **k: None)

    result = orchestrator.run_missing_session_replay(
        state_path=paths["state_path"], decision_log_directory=paths["decision_log_directory"],
        guard_path=paths["guard_path"], provenance_log_directory=paths["provenance_log_directory"],
        pass_gate_directory=paths["pass_gate_directory"], trading_client=_FakeClient(),
    )

    assert result.outcome == "CLEAN_NO_OP"
    reloaded = ps.load_position_state(paths["state_path"], guard_path=paths["guard_path"])
    assert reloaded.last_processed_equity_session_date == "2026-08-20"  # unchanged, no gap
    assert pass_gate.has_valid_pass_gate_for_today(paths["pass_gate_directory"]) is True


def test_scenario_c_a_missed_prior_day_job_is_detected_and_replayed_exactly_once(tmp_path, monkeypatch):
    """"Önceki günün job'ı kaçırılır -> ertesi sabah tam bir seans
    tespit edilip yalnız bir kez replay edilir." Cursor stuck at
    2026-08-19 (yesterday's own job never ran); the REAL orchestrator,
    using the REAL run_daily_decision(), detects exactly one missing
    session (2026-08-20), replays it exactly once, advances the cursor,
    and writes today's PASS gate."""
    paths = _paths(tmp_path)
    state = ps.LiveRunnerState()
    state.last_processed_equity_session_date = "2026-08-19"
    ps.save_position_state(state, paths["state_path"], guard_path=paths["guard_path"])

    prepared = _prepared(("2026-08-19", "2026-08-20"), ("2026-08-19", "2026-08-20"))
    monkeypatch.setattr(orchestrator.rdd, "prepare_live_market_data", lambda tickers: prepared)
    monkeypatch.setattr(orchestrator.rdd, "get_live_cash_balance", lambda client=None: 100_000.0)
    monkeypatch.setattr(orchestrator.broker_reconciliation, "reconcile", lambda *a, **k: None)
    monkeypatch.setattr(
        orchestrator.detection, "detect_missing_equity_sessions",
        lambda *a, **k: detection.DetectionResult(
            expected_sessions=("2026-08-20",), complete_sessions=("2026-08-20",), missing_by_session={},
            unexpected_future_bars={}, raw_bars_by_ticker={},
        ),
    )
    monkeypatch.setattr(orchestrator.detection, "fetch_expected_equity_sessions", lambda *a, **k: [])

    fake_client = _FakeClient()
    result = orchestrator.run_missing_session_replay(
        state_path=paths["state_path"], decision_log_directory=paths["decision_log_directory"],
        guard_path=paths["guard_path"], provenance_log_directory=paths["provenance_log_directory"],
        pass_gate_directory=paths["pass_gate_directory"], trading_client=fake_client,
    )

    assert result.outcome == "REPLAYED_ONE_SESSION"
    assert result.replayed_session_date == "2026-08-20"

    reloaded = ps.load_position_state(paths["state_path"], guard_path=paths["guard_path"])
    assert reloaded.last_processed_equity_session_date == "2026-08-20"
    assert pass_gate.has_valid_pass_gate_for_today(paths["pass_gate_directory"]) is True

    # Replayed EXACTLY once -- one session-replay journal intent, TERMINAL/SUCCEEDED.
    intents = session_journal.list_session_intents()
    assert len(intents) == 1
    assert intents[0].status == session_journal.TERMINAL
    assert intents[0].outcome == session_journal.SUCCEEDED
    assert intents[0].session_date == "2026-08-20"
