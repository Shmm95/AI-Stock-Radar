"""Tests for `src/live/session_replay_pass_gate.py` -- the dated PASS
gate between `equity_session_orchestrator.py` and the real 21:15 live
daily-decision job (missing-session-replay cron-activation finding #2,
2026-08-24). Pure mechanical read/write tests; the POLICY of when a
gate should be written lives in
`equity_session_orchestrator._cursor_is_current` (tested in
`test_equity_session_orchestrator.py`) and is not re-tested here.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import src.live.session_replay_pass_gate as pass_gate


def test_write_pass_gate_creates_a_file_named_for_today(tmp_path):
    path = pass_gate.write_pass_gate(
        tmp_path, last_processed_equity_session_date="2026-08-20", account_number_masked="...1234",
    )
    today = datetime.now(timezone.utc).date().isoformat()
    assert path == tmp_path / f"PASS_{today}.json"
    assert path.is_file()


def test_write_pass_gate_payload_contents(tmp_path):
    path = pass_gate.write_pass_gate(
        tmp_path, last_processed_equity_session_date="2026-08-20", account_number_masked="...1234",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["last_processed_equity_session_date"] == "2026-08-20"
    assert payload["broker_account_masked"] == "...1234"
    assert payload["date"] == datetime.now(timezone.utc).date().isoformat()
    assert payload["verified_at_utc"]


def test_has_valid_pass_gate_for_today_false_when_directory_empty(tmp_path):
    assert pass_gate.has_valid_pass_gate_for_today(tmp_path) is False


def test_has_valid_pass_gate_for_today_false_when_directory_does_not_exist(tmp_path):
    assert pass_gate.has_valid_pass_gate_for_today(tmp_path / "does-not-exist") is False


def test_has_valid_pass_gate_for_today_true_after_write(tmp_path):
    pass_gate.write_pass_gate(
        tmp_path, last_processed_equity_session_date="2026-08-20", account_number_masked=None,
    )
    assert pass_gate.has_valid_pass_gate_for_today(tmp_path) is True


def test_has_valid_pass_gate_for_today_false_for_a_stale_yesterday_gate(tmp_path):
    """A gate file for a DIFFERENT date must never satisfy today's
    check -- confirms the filename (not just directory non-emptiness)
    is what gates the real 21:15 job."""
    stale_path = tmp_path / "PASS_2020-01-01.json"
    stale_path.write_text("{}", encoding="utf-8")
    assert pass_gate.has_valid_pass_gate_for_today(tmp_path) is False


def test_write_pass_gate_is_idempotent_for_repeated_calls_same_day(tmp_path):
    first = pass_gate.write_pass_gate(
        tmp_path, last_processed_equity_session_date="2026-08-20", account_number_masked=None,
    )
    second = pass_gate.write_pass_gate(
        tmp_path, last_processed_equity_session_date="2026-08-21", account_number_masked=None,
    )
    assert first == second  # same file, overwritten -- never two files for the same real day
    payload = json.loads(second.read_text(encoding="utf-8"))
    assert payload["last_processed_equity_session_date"] == "2026-08-21"  # latest write wins
