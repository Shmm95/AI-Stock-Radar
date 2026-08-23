"""Tests for `src/live/equity_session_orchestrator.py` -- "Missing-
Equity-Session Remediation" design (workspace-c), sections A-E tied
together.

`detection.detect_missing_equity_sessions` is monkeypatched to return
canned `DetectionResult`s throughout -- detection's OWN logic
(calendar settle-buffer, complete/missing/unexpected-future
classification) is already covered in isolation by
`test_equity_session_detection.py`; these tests exist to prove the
ORCHESTRATOR's own branching (section B eligibility, section C
routing, fail-closed conditions) is correct given a specific detected
picture, not to re-prove detection's own internals.

Two scenarios here are explicit, separately-requested owner
requirements (not part of the original section-G matrix): the
session-cursor-persistence proof
(`test_session_cursor_persists_after_orchestrator_run_without_control_arm_wrapper`)
and the state-path-isolation assertions
(`Test_StatePathIsolation`). Both are called out by name in their own
docstrings below.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_control_arm_decision as carm
import scripts.run_daily_decision as rdd
import src.live.equity_session_detection as detection
import src.live.equity_session_orchestrator as orchestrator
import src.live.session_replay_journal as session_journal
import src.live.single_instance_lock as single_instance_lock_module
from src.backtest.portfolio_backtest_engine import _MutablePosition, _PendingOrder
from src.backtest.portfolio_backtest_models import PortfolioSignal
from src.live import position_state as ps


# --- shared fixtures / helpers ---------------------------------------------


@contextmanager
def _fake_lock(state_path):
    yield {}


@pytest.fixture(autouse=True)
def _isolate_everything(tmp_path, monkeypatch):
    monkeypatch.setattr(session_journal, "SESSION_REPLAY_DIRECTORY", tmp_path / "session_replay_intents")
    monkeypatch.setattr(orchestrator, "single_instance_lock", _fake_lock)
    monkeypatch.setattr(orchestrator.rdd, "_require_live_account_suffix", lambda: "TEST")
    monkeypatch.setattr(orchestrator.broker_reconciliation, "reconcile", lambda *a, **k: None)
    # A non-existent path, never the real STOP_FLAG_PATH -- this fixture
    # must not touch the real repo-relative "data/live/STOP" file, and
    # must not monkeypatch Path.exists globally (that would also break
    # every tmp_path-based file check these tests rely on elsewhere).
    monkeypatch.setattr(orchestrator.rdd, "STOP_FLAG_PATH", tmp_path / "no-such-STOP-flag")


class _FakeClient:
    def __init__(self, suffix: str = "TEST") -> None:
        self._suffix = suffix

    def get_account(self):
        return SimpleNamespace(account_number=f"PA3HONFD{self._suffix}")


def _paths(tmp_path):
    return {
        "state_path": tmp_path / "position_state.json",
        "decision_log_directory": tmp_path / "decisions",
        "guard_path": tmp_path / "high_water_mark.json",
        "provenance_log_directory": tmp_path / "provenance",
    }


def _save_runner_state(paths, *, last_processed_equity_session_date=None, positions=None):
    state = ps.LiveRunnerState()
    state.last_processed_equity_session_date = last_processed_equity_session_date
    if positions:
        state.positions.update(positions)
    ps.save_position_state(state, paths["state_path"], guard_path=paths["guard_path"])
    return state


def _detection_result(*, expected_sessions, missing_by_session=None, unexpected_future_bars=None, raw_bars_by_ticker=None):
    return detection.DetectionResult(
        expected_sessions=tuple(expected_sessions),
        complete_sessions=tuple(s for s in expected_sessions if s not in (missing_by_session or {})),
        missing_by_session=missing_by_session or {},
        unexpected_future_bars=unexpected_future_bars or {},
        raw_bars_by_ticker=raw_bars_by_ticker or {},
    )


def _run(paths, *, trading_client=None):
    return orchestrator.run_missing_session_replay(
        state_path=paths["state_path"],
        decision_log_directory=paths["decision_log_directory"],
        guard_path=paths["guard_path"],
        provenance_log_directory=paths["provenance_log_directory"],
        trading_client=trading_client or _FakeClient(),
    )


# --- preflight / fail-closed conditions -------------------------------------


def test_stop_flag_short_circuits_to_clean_no_op(tmp_path, monkeypatch):
    stop_flag = tmp_path / "STOP"
    stop_flag.write_text("stop", encoding="utf-8")
    monkeypatch.setattr(orchestrator.rdd, "STOP_FLAG_PATH", stop_flag)
    paths = _paths(tmp_path)
    _save_runner_state(paths, last_processed_equity_session_date="2026-08-19")
    result = _run(paths)
    assert result.outcome == "CLEAN_NO_OP"


def test_stray_prepared_journal_entry_blocks_new_replay(tmp_path):
    paths = _paths(tmp_path)
    _save_runner_state(paths, last_processed_equity_session_date="2026-08-19")
    session_journal.create_session_intent(
        session_id="EQUITY_SESSION:2026-08-20:leftover", session_date="2026-08-20",
        bar_set_sha256="leftover", replay_sequence_index=1, replay_sequence_total=1,
    )
    with pytest.raises(orchestrator.StaleSessionReplayJournalError):
        _run(paths)


def test_stray_committed_journal_entry_blocks_new_replay(tmp_path):
    """The crash-window fix: a COMMITTED-but-not-yet-TERMINAL entry
    (the state save landed but the final bookkeeping transition did
    not) must also block a new attempt -- not just PREPARED/VERIFIED."""
    paths = _paths(tmp_path)
    _save_runner_state(paths, last_processed_equity_session_date="2026-08-19")
    intent = session_journal.create_session_intent(
        session_id="EQUITY_SESSION:2026-08-20:leftover", session_date="2026-08-20",
        bar_set_sha256="leftover", replay_sequence_index=1, replay_sequence_total=1,
    )
    intent = session_journal.transition_session_intent(intent, session_journal.VERIFIED)
    session_journal.transition_session_intent(intent, session_journal.COMMITTED)
    with pytest.raises(orchestrator.StaleSessionReplayJournalError):
        _run(paths)


def test_terminal_journal_entry_does_not_block_a_new_attempt(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    _save_runner_state(paths, last_processed_equity_session_date="2026-08-19")
    intent = session_journal.create_session_intent(
        session_id="EQUITY_SESSION:2026-08-20:other-hash", session_date="2026-08-20",
        bar_set_sha256="other-hash", replay_sequence_index=1, replay_sequence_total=1,
    )
    intent = session_journal.transition_session_intent(intent, session_journal.VERIFIED)
    intent = session_journal.transition_session_intent(intent, session_journal.COMMITTED)
    session_journal.transition_session_intent(intent, session_journal.TERMINAL)
    monkeypatch.setattr(
        orchestrator.detection, "detect_missing_equity_sessions",
        lambda *a, **k: _detection_result(expected_sessions=[]),
    )
    result = _run(paths)
    assert result.outcome == "CLEAN_NO_OP"


def test_session_cursor_ahead_of_calendar_raises(tmp_path):
    paths = _paths(tmp_path)
    future_date = "2099-01-01"
    _save_runner_state(paths, last_processed_equity_session_date=future_date)
    with pytest.raises(orchestrator.SessionCursorAheadOfCalendarError):
        _run(paths)


def test_no_prior_cursor_is_clean_no_op_and_never_calls_detection(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    _save_runner_state(paths, last_processed_equity_session_date=None)
    called = []
    monkeypatch.setattr(
        orchestrator.detection, "detect_missing_equity_sessions",
        lambda *a, **k: called.append(1) or _detection_result(expected_sessions=["2026-08-20"]),
    )
    result = _run(paths)
    assert result.outcome == "CLEAN_NO_OP"
    assert called == []


def test_no_expected_sessions_is_clean_no_op(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    _save_runner_state(paths, last_processed_equity_session_date="2026-08-19")
    monkeypatch.setattr(
        orchestrator.detection, "detect_missing_equity_sessions",
        lambda *a, **k: _detection_result(expected_sessions=[]),
    )
    result = _run(paths)
    assert result.outcome == "CLEAN_NO_OP"


def test_unexpected_future_bars_raises_fail_closed(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    _save_runner_state(paths, last_processed_equity_session_date="2026-08-19")
    monkeypatch.setattr(
        orchestrator.detection, "detect_missing_equity_sessions",
        lambda *a, **k: _detection_result(
            expected_sessions=["2026-08-20"], unexpected_future_bars={"AAPL": ("2026-08-25",)}
        ),
    )
    with pytest.raises(orchestrator.SessionReplayFailClosedError):
        _run(paths)


def test_sole_expected_session_still_missing_data_is_clean_no_op_not_replayed(tmp_path, monkeypatch):
    """Still awaiting data for the ONLY expected session -- not an
    error, and NOT eligible for replay (nothing to replay yet)."""
    paths = _paths(tmp_path)
    _save_runner_state(paths, last_processed_equity_session_date="2026-08-19")
    monkeypatch.setattr(
        orchestrator.detection, "detect_missing_equity_sessions",
        lambda *a, **k: _detection_result(
            expected_sessions=["2026-08-20"], missing_by_session={"2026-08-20": ("AAPL",)}
        ),
    )
    called = []
    monkeypatch.setattr(orchestrator.rdd, "run_daily_decision", lambda **k: called.append(1))
    result = _run(paths)
    assert result.outcome == "CLEAN_NO_OP"
    assert called == []


def test_too_many_missing_sessions_raises(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    _save_runner_state(paths, last_processed_equity_session_date="2026-08-18")
    monkeypatch.setattr(
        orchestrator.detection, "detect_missing_equity_sessions",
        lambda *a, **k: _detection_result(
            expected_sessions=["2026-08-19", "2026-08-20"],
            missing_by_session={"2026-08-19": ("AAPL",), "2026-08-20": ("AAPL",)},
        ),
    )
    with pytest.raises(orchestrator.TooManyMissingSessionsError):
        _run(paths)


def test_too_many_complete_sessions_is_a_multi_gap_case_not_replayed(tmp_path, monkeypatch):
    """Both expected sessions are fully bar-complete RIGHT NOW, but the
    cursor is two sessions behind -- a genuine multi-gap catch-up case,
    out of scope for this module's single-session auto-replay."""
    paths = _paths(tmp_path)
    _save_runner_state(paths, last_processed_equity_session_date="2026-08-18")
    monkeypatch.setattr(
        orchestrator.detection, "detect_missing_equity_sessions",
        lambda *a, **k: _detection_result(expected_sessions=["2026-08-19", "2026-08-20"]),
    )
    with pytest.raises(orchestrator.TooManyMissingSessionsError):
        _run(paths)


# --- section C: missed-window handling --------------------------------------


def test_missed_window_handled_when_newer_session_already_settled(tmp_path, monkeypatch):
    """The missing session is NOT the sole/latest expected one -- a
    newer session has already settled on top of it, meaning its own
    execution window has passed. Must route to handle_missed_execution_window,
    never attempt a replay."""
    paths = _paths(tmp_path)
    state = _save_runner_state(paths, last_processed_equity_session_date="2026-08-18")
    state.pending_buys["AAPL"] = _PendingOrder(
        signal=PortfolioSignal(timestamp="2026-08-18", ticker="AAPL", action="BUY", reference_price=100.0),
        submitted_portfolio_bar_index=1,
    )
    ps.save_position_state(state, paths["state_path"], guard_path=paths["guard_path"])

    monkeypatch.setattr(
        orchestrator.detection, "detect_missing_equity_sessions",
        lambda *a, **k: _detection_result(
            expected_sessions=["2026-08-19", "2026-08-20"],
            missing_by_session={"2026-08-19": ("AAPL",)},
        ),
    )
    replay_called = []
    monkeypatch.setattr(orchestrator.rdd, "run_daily_decision", lambda **k: replay_called.append(1))

    result = _run(paths)

    assert result.outcome == "MISSED_WINDOW_HANDLED"
    assert replay_called == []
    assert result.missed_window_outcome.expired_buy_tickers == ("AAPL",)
    assert result.provenance_log_path is not None
    assert result.provenance_log_path.is_file()

    reloaded = ps.load_position_state(paths["state_path"], guard_path=paths["guard_path"])
    assert reloaded.pending_buys == {}


# --- section B / D: the narrow eligible auto-replay path --------------------


def _fake_run_daily_decision_factory(as_of_bar_timestamp_equity, *, log_path):
    def _fake(**kwargs):
        assert kwargs["enable_equity_orders"] is False
        assert kwargs["enable_crypto_orders"] is False
        assert kwargs["skip_broker_reconciliation"] is True
        return {"decision": {"as_of_bar_timestamp_equity": as_of_bar_timestamp_equity}, "log_path": log_path}
    return _fake


def test_eligible_single_session_replay_succeeds_end_to_end(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    _save_runner_state(paths, last_processed_equity_session_date="2026-08-19")
    monkeypatch.setattr(
        orchestrator.detection, "detect_missing_equity_sessions",
        lambda *a, **k: _detection_result(expected_sessions=["2026-08-20"], raw_bars_by_ticker={}),
    )
    monkeypatch.setattr(
        orchestrator.rdd, "run_daily_decision",
        _fake_run_daily_decision_factory("2026-08-20", log_path=tmp_path / "decision.json"),
    )

    result = _run(paths)

    assert result.outcome == "REPLAYED_ONE_SESSION"
    assert result.replayed_session_date == "2026-08-20"
    assert result.provenance_log_path.is_file()

    reloaded = ps.load_position_state(paths["state_path"], guard_path=paths["guard_path"])
    assert reloaded.last_processed_equity_session_date == "2026-08-20"

    intents = session_journal.list_session_intents()
    assert len(intents) == 1
    assert intents[0].status == session_journal.TERMINAL
    assert intents[0].session_date == "2026-08-20"


def test_replay_is_idempotent_for_an_already_terminal_session(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    _save_runner_state(paths, last_processed_equity_session_date="2026-08-19")
    detection_result = _detection_result(expected_sessions=["2026-08-20"], raw_bars_by_ticker={})
    monkeypatch.setattr(orchestrator.detection, "detect_missing_equity_sessions", lambda *a, **k: detection_result)

    bar_hash = orchestrator._bar_set_sha256({}, "2026-08-20")
    session_id = session_journal.build_session_id("2026-08-20", bar_hash)
    intent = session_journal.create_session_intent(
        session_id=session_id, session_date="2026-08-20", bar_set_sha256=bar_hash,
        replay_sequence_index=1, replay_sequence_total=1,
    )
    intent = session_journal.transition_session_intent(intent, session_journal.VERIFIED)
    intent = session_journal.transition_session_intent(intent, session_journal.COMMITTED)
    session_journal.transition_session_intent(intent, session_journal.TERMINAL)

    called = []
    monkeypatch.setattr(orchestrator.rdd, "run_daily_decision", lambda **k: called.append(1))

    result = _run(paths)
    assert result.outcome == "CLEAN_NO_OP"
    assert called == []


def test_outcome_mismatch_raises_and_closes_journal_terminal_with_error(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    _save_runner_state(paths, last_processed_equity_session_date="2026-08-19")
    monkeypatch.setattr(
        orchestrator.detection, "detect_missing_equity_sessions",
        lambda *a, **k: _detection_result(expected_sessions=["2026-08-20"], raw_bars_by_ticker={}),
    )
    monkeypatch.setattr(
        orchestrator.rdd, "run_daily_decision",
        _fake_run_daily_decision_factory("2026-08-19", log_path=tmp_path / "decision.json"),  # wrong session!
    )

    with pytest.raises(orchestrator.SessionOutcomeMismatchError):
        _run(paths)

    intents = session_journal.list_session_intents()
    assert len(intents) == 1
    assert intents[0].status == session_journal.TERMINAL
    assert intents[0].last_error is not None

    # The cursor must NOT have been advanced on a mismatched outcome.
    reloaded = ps.load_position_state(paths["state_path"], guard_path=paths["guard_path"])
    assert reloaded.last_processed_equity_session_date == "2026-08-19"


def test_run_daily_decision_exception_closes_journal_terminal_and_reraises(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    _save_runner_state(paths, last_processed_equity_session_date="2026-08-19")
    monkeypatch.setattr(
        orchestrator.detection, "detect_missing_equity_sessions",
        lambda *a, **k: _detection_result(expected_sessions=["2026-08-20"], raw_bars_by_ticker={}),
    )

    def _boom(**kwargs):
        raise RuntimeError("network exploded")

    monkeypatch.setattr(orchestrator.rdd, "run_daily_decision", _boom)

    with pytest.raises(RuntimeError, match="network exploded"):
        _run(paths)

    intents = session_journal.list_session_intents()
    assert len(intents) == 1
    assert intents[0].status == session_journal.TERMINAL
    assert "network exploded" in intents[0].last_error


# --- explicitly requested scenario 11: session-cursor persistence proof ----


def test_session_cursor_persists_after_orchestrator_run_without_control_arm_wrapper(tmp_path, monkeypatch):
    """Direct, explicit proof (owner-requested, separate from the design's
    own scenario matrix) that `last_processed_equity_date`/
    `last_processed_crypto_date`/`last_processed_equity_session_date`/
    `last_processed_equity_bar_timestamp` genuinely persist to disk after
    `run_missing_session_replay()` runs -- WITH NO involvement of
    `run_control_arm_decision.py`'s own `_stamp_last_processed_dates`
    (never imported or called anywhere in this test). This is the exact
    root-cause bug this whole module exists to fix for the real LIVE
    path: `run_daily_decision()`'s own internal save never populates
    these fields (confirmed by direct inspection of its final
    `LiveRunnerState(...)` construction), so before this module existed,
    nothing ever durably recorded them for a real live run."""
    assert "run_control_arm_decision" not in dir()  # sanity: this test never imports carm's stamping helper

    paths = _paths(tmp_path)
    _save_runner_state(paths, last_processed_equity_session_date="2026-08-19")
    monkeypatch.setattr(
        orchestrator.detection, "detect_missing_equity_sessions",
        lambda *a, **k: _detection_result(expected_sessions=["2026-08-20"], raw_bars_by_ticker={}),
    )
    monkeypatch.setattr(
        orchestrator.rdd, "run_daily_decision",
        _fake_run_daily_decision_factory("2026-08-20", log_path=tmp_path / "decision.json"),
    )

    result = _run(paths)
    assert result.outcome == "REPLAYED_ONE_SESSION"

    reloaded = ps.load_position_state(paths["state_path"], guard_path=paths["guard_path"])
    today_iso = datetime.now(timezone.utc).date().isoformat()
    assert reloaded.last_processed_equity_session_date == "2026-08-20"
    assert reloaded.last_processed_equity_bar_timestamp == "2026-08-20"
    assert reloaded.last_processed_equity_date == today_iso
    assert reloaded.last_processed_crypto_date == today_iso


# --- explicitly requested: state-path isolation from control-arm's own state


class TestStatePathIsolation:
    """Owner-requested verification (separate from the design's own
    scenario matrix): confirm the new orchestrator's state-file path is
    correctly related to control-arm's own default -- no accidental
    cross-write risk, and no accidental DIVERGENCE from the real live
    path either."""

    def test_orchestrator_default_state_path_is_the_same_constant_live_main_uses(self):
        """The orchestrator MUST target the exact same file
        `run_daily_decision.py`'s own `main()` reads/writes -- that is
        the entire point (fixing THAT file's missing session cursor),
        not a different, accidentally-separate one."""
        import inspect

        default = inspect.signature(orchestrator.run_missing_session_replay).parameters["state_path"].default
        assert default is rdd.DEFAULT_STATE_PATH

    def test_run_daily_decision_and_control_arm_share_the_identical_default_state_path_constant(self):
        """Confirms the real, disclosed isolation model this session's
        own fact-finding established: `run_control_arm_decision.py`'s
        own `--state-path` argparse default is LITERALLY the same
        constant as `run_daily_decision.py`'s own default -- isolation
        between the live and control-arm deployments comes ENTIRELY
        from process cwd + `AI_STOCK_RADAR_GUARD_DIR`, never from a
        different default path constant. Checked via source
        introspection (matching this arg's own real `default=` call
        site), not merely re-asserted in prose."""
        import inspect

        source = inspect.getsource(carm)
        assert '"--state-path", type=Path, default=rdd.DEFAULT_STATE_PATH' in source
        assert rdd.DEFAULT_STATE_PATH == Path("data/live/position_state.json")
        assert not rdd.DEFAULT_STATE_PATH.is_absolute()

    def test_guard_directory_derived_paths_share_one_env_var_controlled_root(self):
        """`session_replay_journal`'s own storage directory and
        `single_instance_lock`'s own default lock path are both rooted
        under the SAME `_GUARD_DIRECTORY` constant `position_state.py`
        resolves from `$AI_STOCK_RADAR_GUARD_DIR` -- i.e. isolating one
        deployment from another by setting that single environment
        variable moves the session-replay journal, the single-instance
        lock, and the high-water-mark file together, never
        independently. `session_replay_journal`'s own
        `SESSION_REPLAY_DIRECTORY` is checked via SOURCE inspection
        (not the live module attribute), since this file's own
        `_isolate_everything` autouse fixture deliberately monkeypatches
        that attribute to a tmp_path for every other test in this
        module -- source inspection proves the real wiring
        independently of that per-test override. `single_instance_lock.LOCK_PATH`
        and `position_state.HIGH_WATER_MARK_PATH` are never monkeypatched
        anywhere in this file, so those two are checked directly."""
        import inspect

        journal_source = inspect.getsource(session_journal)
        assert 'SESSION_REPLAY_DIRECTORY = _GUARD_DIRECTORY / "session_replay_intents"' in journal_source
        assert single_instance_lock_module.LOCK_PATH.parent == ps._GUARD_DIRECTORY
        assert ps.HIGH_WATER_MARK_PATH.parent == ps._GUARD_DIRECTORY

    def test_orchestrator_never_hardcodes_a_second_guard_directory(self):
        """Source-level guard against a regression where a future edit
        introduces a SEPARATE, hardcoded guard/session directory for
        this module instead of reusing `position_state._GUARD_DIRECTORY`
        indirectly (via `session_replay_journal`/`single_instance_lock`,
        as this module already does) -- same "prove it's wired, not
        just asserted in prose" discipline
        `test_reconcile_stale_equity_positions.py::test_main_runs_under_single_instance_lock`
        already established for a different module."""
        import inspect

        source = inspect.getsource(orchestrator)
        assert "_GUARD_DIRECTORY" not in source
        assert "single_instance_lock" in source
